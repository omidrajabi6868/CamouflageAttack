import numpy as np
import argparse
import math
import torch
import copy
from detectron2.config import LazyConfig, instantiate
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.engine import SimpleTrainer, hooks, launch, default_writers, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
from YOLO import YOLOv8Loss

from Dataset import Dataset
from DetectionNetwork import DetectionNetwork

import logging
from detectron2.utils.logger import setup_logger
setup_logger()
logger = logging.getLogger("detectron2")
logger.setLevel(logging.DEBUG)  # or INFO, WARNING, etc.

import warnings
warnings.filterwarnings('ignore')

class WarmupCosineLR:
    def __init__(self, optimizer, max_lr, min_lr, warmup_iters, total_iters):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.iter = 0

    def step(self):
        self.iter += 1

        if self.iter <= self.warmup_iters:
            lr = self.max_lr * self.iter / self.warmup_iters
        else:
            progress = (self.iter - self.warmup_iters) / (self.total_iters - self.warmup_iters)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


class ModelEMA:
    def __init__(self, model, decay=0.9999, tau=2000):
        self.ema = copy.deepcopy(model).eval()
        self.updates = 0
        self.decay = decay
        self.tau = tau

        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))

        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k], alpha=1 - d)
            else:
                v.copy_(msd[k])


class Train:
    def __init__(self, cfg):
        self.cfg = cfg

    def cnn_based_training(self):
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    def transformer_based_training(self, model):

        self.cfg.optimizer.params.model = model
        optimizer = instantiate(self.cfg.optimizer)

        lr_scheduler = instantiate(self.cfg.lr_multiplier)

        # Dataloaders
        train_loader = instantiate(self.cfg.dataloader.train)

        # -------------------------------
        #  Trainer
        # -------------------------------
        trainer = SimpleTrainer(model, train_loader, optimizer)
        checkpointer = DetectionCheckpointer(model, save_dir=self.cfg.train.output_dir)
        trainer.register_hooks([
            hooks.IterationTimer(),
            hooks.LRScheduler(optimizer=optimizer, scheduler=lr_scheduler),
            hooks.PeriodicWriter(default_writers(self.cfg.train.output_dir), period=20),
            hooks.PeriodicCheckpointer(checkpointer, period=500, max_to_keep=5),
        ])

        # -------------------------------
        # Train
        # -------------------------------
        trainer.train(0, self.cfg.train.max_iter)

    def transformer_based_evaluation(self, model, model_path):

        test_loader = instantiate(self.cfg.dataloader.test)

        DetectionCheckpointer(model).load(self.model_path)

        evaluator = COCOEvaluator("val_data", output_dir=self.cfg.train.output_dir)
        results = inference_on_dataset(model, test_loader, evaluator)
        print("Evaluation results:", results)

    def build_optimizer(self, model, lr=0.01, momentum=0.937, weight_decay=5e-4):
        g0, g1 = [], []  # no_decay, decay

        for v in model.modules():
            if hasattr(v, "bias") and isinstance(v.bias, torch.nn.Parameter):
                g0.append(v.bias)
            if isinstance(v, torch.nn.BatchNorm2d):
                g0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, torch.nn.Parameter):
                g1.append(v.weight)

        optimizer = torch.optim.SGD(
            [
                {"params": g0, "weight_decay": 0.0},
                {"params": g1, "weight_decay": weight_decay},
            ],
            lr=lr,
            momentum=momentum,
            nesterov=True,
        )
        return optimizer

    def yolo_model_training(self, model, dataset):
        train_loader, val_loader = dataset.yolo_data_loaders(self.cfg)

        device = "cuda"
        model.train()
        model.to(device)
        optimizer = self.build_optimizer(
                    model,
                    lr=self.cfg["learning_rate"],
                    momentum=0.937)
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable:,}")

        yolo_loss = YOLOv8Loss(self.cfg['model']['num_classes'], reg_max=16).to(device)
        train_epoch_stop = self.cfg['train_data_len']//self.cfg['batch_size']
        warmup_iters = int(0.05*(self.cfg['epoch_num']*train_epoch_stop))
        scheduler = WarmupCosineLR(
                        optimizer,
                        max_lr=self.cfg["learning_rate"],
                        min_lr=self.cfg["learning_rate"] * 0.01,
                        warmup_iters=warmup_iters,
                        total_iters=self.cfg['epoch_num']*train_epoch_stop)
        ema = ModelEMA(model)
        epoch= 0
        losses = []
        best_loss = float("inf")
        for iteration, batch in zip(range(self.cfg['epoch_num']*train_epoch_stop), train_loader):
            model.train()
            images = torch.stack([x["image"] for x in batch]).to(device)
            targets = []
            for x in batch:
                inst = x["instances"]
                targets.append({
                    "boxes": inst.gt_boxes.tensor.to(device),
                    "labels": inst.gt_classes.to(device)
                })

            preds = model(images)
            loss = yolo_loss(preds, targets, strides=[8, 16, 32])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema.update(model)
            losses.append(loss.item())
            if iteration == (train_epoch_stop*(epoch+1)):
                print(f'Epoch_num: {epoch} trainig_loss: {np.mean(losses)}')
                losses = []
                epoch += 1
                with torch.no_grad():
                   val_loss = self.yolo_model_validation(ema.ema, val_loader, yolo_loss)
                   print(f'val_loss: {val_loss}')
                   if val_loss < best_loss:
                        torch.save(ema.ema.state_dict(), '/home/oraja001/airbus_ship/AdversarialProject/trained_models/yolo/custom_yolo.pt')
                        best_loss = val_loss
    
        return model

    def yolo_model_validation(self, model, val_loader, yolo_loss):
        losses = []
        model.eval()
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
        device = 'cuda'
        val_stop_iteration = (self.cfg['val_data_len']//self.cfg['batch_size']) + 1
        for iteration, batch in zip(range(val_stop_iteration), val_loader):
            images = torch.stack([x["image"] for x in batch]).to(device)
            targets = []

            for x in batch:
                inst = x["instances"]
                targets.append({
                    "boxes": inst.gt_boxes.tensor.to(device),
                    "labels": inst.gt_classes.to(device)
                })

            preds = model(images)
            loss = yolo_loss(preds, targets, strides=[8, 16, 32])
            losses.append(loss.item())

        return np.mean(losses)



def main(args):
    print(args)
    dataset_name = args.dataset_name
    img_dir = args.img_dir

    dataset = Dataset(dataset_name, category_id=args.category_id, random_id=args.random_id)
    train_df, valid_df = dataset.airbus_df()

    DatasetCatalog.register('train_data', lambda: dataset.airbus_dicts(df=train_df[:5000], img_dir=img_dir))
    if args.multiclass:
        coco_classes = MetadataCatalog.get('coco_2017_train').thing_classes.copy()
        coco_classes[8] = 'ship'
        MetadataCatalog.get('train_data').set(thing_classes=coco_classes)
    else:
        MetadataCatalog.get('train_data').set(thing_classes=[args.category_name])

    train_data_len = len(DatasetCatalog.get('train_data'))
    train_metadata = MetadataCatalog.get("train_data")

    DatasetCatalog.register('val_data', lambda: dataset.airbus_dicts(df=valid_df[:], img_dir=img_dir))
    if args.multiclass:
        MetadataCatalog.get('val_data').set(thing_classes=coco_classes)
    else:
        MetadataCatalog.get('val_data').set(thing_classes=[args.category_name])

    val_data_len = len(DatasetCatalog.get('val_data'))
    val_metadata = MetadataCatalog.get("val_data")

    network = DetectionNetwork(output_dir=args.output_dir,
                               model_path=args.model_path,
                               batch_size=args.batch_size,
                               learning_rate=args.learning_rate,
                               epoch_num=args.epoch_num,
                               num_classes=args.num_classes,
                               attack_name=None)

    if args.yolo_training:
        model, cfg = network.yolo_model_net(model_config=args.model_config,
                                            train_data_len=train_data_len)
        cfg['batch_size'] = args.batch_size
        cfg['epoch_num'] = args.epoch_num
        cfg['learning_rate'] = args.learning_rate
        cfg['model']['num_classes'] = args.num_classes
        cfg['train_data_len'] = train_data_len
        cfg['val_data_len'] = val_data_len
        Train(cfg).yolo_model_training(model, dataset)
    
    elif args.transformer_based:
        model, cfg = network.transformer_based_net(model_config=args.model_config, 
                                                   train_data_len=train_data_len)
        Train(cfg).transformer_based_training(model)
    else:
        model, cfg = network.detectron_net(model_config=args.model_config, 
                                           train_data_len=train_data_len)
        Train(cfg).cnn_based_training()
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_name", type=str, default="airbus")
    parser.add_argument("--img_dir", type=str, default="/home/oraja001/airbus_ship/airbus/train_v2")
    parser.add_argument("--multiclass", type=bool, default=False)
    parser.add_argument("--epoch_num", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--category_id", type=int, default=0)
    parser.add_argument("--random_id", type=bool, default=False)
    parser.add_argument("--category_name", type=str, default='ship')
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="/home/oraja001/airbus_ship/AdversarialProject/outputs/mask_rcnn_vitdet/")
    parser.add_argument("--model_path", type=str, default="/home/oraja001/airbus_ship/AdversarialProject/trained_models/mask_rcnn_vitdet/model_final_61ccd1.pkl")
    parser.add_argument("--model_config", type=str, default="/home/oraja001/airbus_ship/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py")
    parser.add_argument("--transformer_based", type=bool, default=False)
    parser.add_argument("--yolo_training", type=bool, default=False)

    args = parser.parse_args()

    main(args)