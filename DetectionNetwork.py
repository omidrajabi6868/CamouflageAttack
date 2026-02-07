import os
import torch
import numpy as np
import segmentation_models_pytorch as smp

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator.rpn import RPN
from detectron2.modeling.roi_heads import StandardROIHeads, select_foreground_proposals, ROI_HEADS_REGISTRY
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.modeling.box_regression import Box2BoxTransform, _dense_box_regression_loss
from detectron2.config import LazyConfig, instantiate
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.engine import SimpleTrainer, hooks, launch, default_writers, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import transforms as T

from fvcore.nn import giou_loss, smooth_l1_loss
from detectron2.layers import cat, diou_loss
from pathlib import Path
from YOLO import YOLO
from utils import load_yaml, load_pretrained_yolo

@PROPOSAL_GENERATOR_REGISTRY.register()
class CustomRPN(RPN):
    def losses(self, anchors,
               pred_objectness_logits,
               gt_labels,
               pred_anchor_deltas,
               gt_boxes):
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi*Wi*A) representing
                the predicted objectness logits for all anchors.
            gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            gt_boxes (list[Tensor]): Output of :meth:`label_and_sample_anchors`.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)

        valid_mask = gt_labels >= 0

        localization_loss = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )

        adv_localization_loss = torch.log1p(1 + (1/(localization_loss + 1e-6)))

        # adv_objectness_loss = F.binary_cross_entropy_with_logits(
        #     cat(pred_objectness_logits, dim=1),
        #     torch.zeros_like(cat(pred_objectness_logits, dim=1)),
        #     reduction="mean")

        adv_objectness_loss = torch.sigmoid(cat(pred_objectness_logits, dim=1)).mean()        

        losses = {
            "loss_rpn_cls": adv_objectness_loss,
            # The original Faster R-CNN paper uses a slightly different normalizer
            # for loc loss. But it doesn't matter in practice
            "loss_rpn_loc": adv_localization_loss,
        }

        self.loss_weight['loss_rpn_cls'] = 1.
        self.loss_weight['loss_rpn_loc'] = 1.
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

        return losses

@ROI_HEADS_REGISTRY.register()
class CustomStandardROIHeads(StandardROIHeads):
    def _forward_box(self, features, proposals):
        """
              Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
                  the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

              Args:
                  features (dict[str, Tensor]): mapping from feature map names to tensor.
                      Same as in :meth:`ROIHeads.forward`.
                  proposals (list[Instances]): the per-image object proposals with
                      their matching ground truth.
                      Each has fields "proposal_boxes", and "objectness_logits",
                      "gt_classes", "gt_boxes".

              Returns:
                  In training, a dict of losses.
                  In inference, a list of `Instances`, the predicted instances.
              """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    def _forward_mask(self, features, instances):
        """
                Forward logic of the mask prediction branch.

                Args:
                    features (dict[str, Tensor]): mapping from feature map names to tensor.
                        Same as in :meth:`ROIHeads.forward`.
                    instances (list[Instances]): the per-image instances to train/predict masks.
                        In training, they can be the proposals.
                        In inference, they can be the boxes predicted by R-CNN box head.

                Returns:
                    In training, a dict of losses.
                    In inference, update `instances` with new fields "pred_masks" and return it.
                """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f].to(device) for f in self.mask_in_features}
        return self.mask_head(features, instances)

class LazyPredictor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model
        self.model = instantiate(cfg.model)
        self.model.to(self.device)
        self.model.eval()

        DetectionCheckpointer(self.model).load(cfg.train.init_checkpoint)

        # Input format
        self.input_format = "BGR"
        try:
            self.input_format = cfg.dataloader.test.mapper.img_format
        except Exception:
            pass

        # Augmentation / resize pipeline
        aug_cfg = cfg.dataloader.test.mapper.augmentations[0]
        self.aug = T.ResizeShortestEdge(
            short_edge_length=getattr(aug_cfg, "short_edge_length", [800]),
            max_size=getattr(aug_cfg, "max_size", 1333)
        )

    @torch.no_grad()
    def __call__(self, image):

        original_height, original_width = image.shape[:2]

        # Convert color format if needed
        if self.input_format == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        if self.aug:
            transform = self.aug.get_transform(image)
            image = transform.apply_image(image)

        # Convert to tensor
        image_tensor = torch.as_tensor(image.transpose(2,0,1)).to(self.device)

        inputs = [{"image": image_tensor, "height": original_height, "width": original_width}]
        predictions  = self.model(inputs)[0]

        return predictions 

class DetectionNetwork:
    def __init__(self, output_dir, model_path, batch_size, learning_rate, epoch_num, num_classes,attack_name):
        self.output_dir = output_dir
        self.model_path = model_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epoch_num = epoch_num
        self.num_classes = num_classes
        self.attack_name = attack_name
        return

    def detectron_net(self, model_config, train_data_len):

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_config))
        cfg.DATASETS.TRAIN = ("train_data",)
        cfg.DATASETS.TEST = ("val_data",)
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)  # Let training initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = self.batch_size  # This is the real "batch size" commonly known to deep learning people
        cfg.SOLVER.BASE_LR = self.learning_rate  # pick a good LR
        cfg.SOLVER.MAX_ITER = int(train_data_len / self.batch_size ) * self.epoch_num 
        cfg.SOLVER.CHECKPOINT_PERIOD = int(train_data_len /self.batch_size ) * 5
        cfg.SOLVER.STEPS = []  # do not decay learning rate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14  # for bounding boxes, keep the default if it works well
        cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 28  # start with 28 and try increasing to 32 or 36 if necessary
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
        # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
        cfg.OUTPUT_DIR = self.output_dir
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        if Path(self.model_path).exists():
            cfg.MODEL.WEIGHTS = self.model_path # path to the model we just trained


        if self.attack_name == 'shapeAware':
            cfg.MODEL.ROI_HEADS.NAME = "CustomStandardROIHeads"
            cfg.MODEL.PROPOSAL_GENERATOR.NAME = "CustomRPN"

        detection_model = build_model(cfg)

        DetectionCheckpointer(detection_model).load(cfg.MODEL.WEIGHTS)

        return detection_model, cfg
    
    def segmentation_model(self, MODEL_SEG):

        run_id = MODEL_SEG

        if MODEL_SEG == 'IUNET':
            model = IUNet()
        elif MODEL_SEG == 'UNET':
            model = UNet()
        elif MODEL_SEG == 'UNET_RESNET34ImgNet':
            model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
        elif MODEL_SEG == 'FPN_RESNET34ImgNet':
            model = smp.FPN("resnet34", encoder_weights="imagenet", activation=None)
        elif MODEL_SEG == "UNET_RESNET50ImgNet":
            model = smp.Unet("resnet50", encoder_weights="imagenet", activation=None)
        elif MODEL_SEG == "DeepLabV3ImgNet":
            model = smp.DeepLabV3Plus("mobilenet_v2", encoder_weights="imagenet", activation=None)
        elif MODEL_SEG == "PANImgNet":
            model = smp.PAN("efficientnet-b1", encoder_weights="imagenet", activation=None)
        else:
            raise NameError("model not supported")
        
        model_path = Path('/home/oraja001/airbus_ship/AdversarialProject/trained_models/segmentation/model_{fold}.pt'.format(fold=run_id))
        state = torch.load(str(model_path))
        state = {key.replace('module.', ''): value for key, value in state['model'].items()}
        model.load_state_dict(state)

        return model
    
    def transformer_based_net(self, model_config, train_data_len):

        cfg = LazyConfig.load(model_config)
        cfg.dataloader.train.dataset.names = ("train_data",)
        cfg.dataloader.test.dataset.names = ("val_data",)

        cfg.model.roi_heads.num_classes = self.num_classes
        cfg.train.output_dir = self.output_dir
        os.makedirs(cfg.train.output_dir, exist_ok=True)

        cfg.train.max_iter = int(train_data_len / self.batch_size) * self.epoch_num
        cfg.train.eval_period = int(train_data_len / self.batch_size) * 5  
        cfg.dataloader.train.total_batch_size = self.batch_size

        cfg.optimizer.lr = self.learning_rate


        model = instantiate(cfg.model)
        cfg.train.init_checkpoint = f"{self.model_path}"

        DetectionCheckpointer(model).load(self.model_path)
        
        return model, cfg

    def yolo_model_net(self, model_config, train_data_len, pretrained="/home/oraja001/airbus_ship/AdversarialProject/trained_models/yolo/yolov8l.pt"):

        cfg = load_yaml(model_config)

        model = YOLO(
            model_size=cfg["model"]["size"],
            num_classes=self.num_classes,
            reg_max=cfg["model"]["reg_max"],
        )

        if pretrained:
            model = load_pretrained_yolo(model, pretrained)

        return model, cfg