import numpy as np
import cv2
import torch
import random
from torch.nn.parallel import parallel_apply, replicate

from detectron2.structures import ImageList
from detectron2.utils.events import EventStorage, get_event_storage

from Poison import Poison
from UViT import UViT
from Network import UNet, IUNet, ParameterRender, CustomLoss, TVLoss, NPSLoss
from PPO import PPO, RolloutBuffer

import logging
from detectron2.utils.logger import setup_logger
setup_logger()
logger = logging.getLogger("detectron2")
logger.setLevel(logging.DEBUG)  # or INFO, WARNING, etc.

class Attack:
    def __init__(self, name, poisoning_func, train_loader, val_loader, optimizer, epoch_num, attack_loss, save_name, mean, std, device_ids=None):
        self.name = name
        self.poisoning_func = poisoning_func
        self.optimizer = optimizer
        self.epoch_num = epoch_num
        self.train_loader = train_loader
        self.val_loader = val_loader
        if torch.cuda.is_available():
            available_device_ids = list(range(torch.cuda.device_count()))
            if device_ids is None:
                self.device_ids = available_device_ids[:1]
            else:
                requested_device_ids = [int(device_id) for device_id in device_ids]
                self.device_ids = [device_id for device_id in requested_device_ids if device_id in available_device_ids]
                if not self.device_ids:
                    raise ValueError(f"None of the requested CUDA devices are available: {requested_device_ids}")
            self.device_ids = sorted(dict.fromkeys(self.device_ids))
            self.rank = self.device_ids[0]
            self.device = torch.device(f'cuda:{self.rank}')
        else:
            self.device_ids = []
            self.rank = 0
            self.device = torch.device('cpu')
        self.multi_gpu = len(self.device_ids) > 1
        self.mean = mean.to(self.device)
        self.std = std.to(self.device)
        self.attack_loss = attack_loss
        self.save_name = save_name
        self.log_interval = 25
        # GradNorm is memory intensive because it relies on higher-order gradients.
        # Update GradNorm weights less frequently to reduce peak memory pressure.
        self.gradnorm_update_interval = 2

    def _active_device_ids(self, batch_len=None):
        if not self.device_ids:
            return []
        if batch_len is None:
            return self.device_ids
        return self.device_ids[:max(1, min(len(self.device_ids), batch_len))]

    def _split_batch_for_devices(self, batch_inputs):
        devices = self._active_device_ids(len(batch_inputs))
        if len(devices) <= 1:
            return [(self.device, batch_inputs)]

        chunks = []
        per_device = int(np.ceil(len(batch_inputs) / len(devices)))
        for chunk_index, device_id in enumerate(devices):
            start = chunk_index * per_device
            end = min(start + per_device, len(batch_inputs))
            if start < end:
                chunks.append((torch.device(f'cuda:{device_id}'), batch_inputs[start:end]))
        return chunks

    def _device_index_from_input(self, input_value):
        if isinstance(input_value, torch.Tensor):
            return input_value.device.index
        if isinstance(input_value, list) and input_value:
            return input_value[0]['image'].device.index
        return self.rank

    def _parallel_apply_module(self, module, inputs_by_device):
        if len(inputs_by_device) == 1:
            return [module(*inputs_by_device[0])]

        used_device_ids = [self._device_index_from_input(inputs[0]) for inputs in inputs_by_device]
        replicas = replicate(module, used_device_ids)
        return parallel_apply(replicas, inputs_by_device, devices=used_device_ids)

    def _model_forward(self, model, adversarial_chunks):
        nonempty_chunks = [chunk for _, chunk in adversarial_chunks if len(chunk) > 0]
        inputs_by_device = [(chunk,) for chunk in nonempty_chunks]
        outputs = self._parallel_apply_module(model, inputs_by_device)
        if isinstance(outputs[0], dict):
            return self._reduce_loss_dict(outputs, [len(chunk) for chunk in nonempty_chunks])
        return outputs[0]

    def _reduce_loss_dict(self, loss_dicts, weights):
        reduced = {}
        total_weight = sum(weights)
        for key in loss_dicts[0].keys():
            values = []
            for loss_dict, weight in zip(loss_dicts, weights):
                value = loss_dict[key]
                if value.ndim > 0:
                    value = value.mean()
                values.append(value.to(self.device) * (weight / total_weight))
            reduced[key] = sum(values)
        return reduced

    def _weighted_mean(self, weighted_values):
        if not weighted_values:
            return torch.zeros((), device=self.device)
        total_weight = sum(weight for _, weight in weighted_values)
        return sum(value.to(self.device) * (weight / total_weight) for value, weight in weighted_values)

    def _backbone_features_by_device(self, backbone, image_tensors):
        inputs_by_device = [(images,) for images in image_tensors]
        return self._parallel_apply_module(backbone, inputs_by_device)

    def _seg_outputs_by_device(self, seg_model, image_tensors):
        inputs_by_device = [(images,) for images in image_tensors]
        return self._parallel_apply_module(seg_model, inputs_by_device)

    def _feature_loss_from_chunks(self, clean_feature_chunks, adv_feature_chunks, weights):
        losses = []
        for clean_features, adv_features, weight in zip(clean_feature_chunks, adv_feature_chunks, weights):
            common_keys = [key for key in ['p2', 'p3', 'p4', 'p5', 'p6'] if key in clean_features and key in adv_features]
            if not common_keys:
                continue
            feature_loss = torch.zeros((), device=next(iter(adv_features.values())).device)
            for key in common_keys:
                feature_loss = feature_loss + torch.nn.MSELoss()(clean_features[key], adv_features[key])
            losses.append((feature_loss, weight))
        return self._weighted_mean(losses)

    def _seg_loss_from_chunks(self, seg_output_chunks, target_mask_chunks, weights):
        losses = []
        for seg_outputs, target_masks, weight in zip(seg_output_chunks, target_mask_chunks, weights):
            losses.append((self.segmentation_loss(seg_outputs, target_masks), weight))
        return self._weighted_mean(losses)

    def go_loss(self, dict_losses):
        adv_loss = dict_losses['loss_cls']*(-1)
        return adv_loss
    
    def ss_loss(self, dict_losses):
        adv_loss = (-1)*dict_losses['loss_rpn_cls'] - dict_losses['loss_cls']
        return adv_loss
    
    def ss_multiclass_loss(self, dict_losses):
        adv_loss = (-1)*(dict_losses['loss_rpn_cls']) + (-1)*(dict_losses['loss_cls'])
        return adv_loss    
    
    def as_loss(self, dict_losses):
        adv_loss = (-1)*dict_losses['loss_rpn_cls']
        return adv_loss
    
    def shipCamou_loss(self, dict_losses, patch):
        adv_bbox_loss = (-1)*(dict_losses['loss_box_reg'])
        adv_objectness_loss = (-1)*(dict_losses['loss_cls'])
        loss_tv = TVLoss()(patch.unsqueeze(0)/57.3750)
        loss_nps= NPSLoss()(patch.unsqueeze(0)/57.3750)
        adv_loss = 0.2*adv_bbox_loss + adv_objectness_loss + 0.01*loss_tv + 0.01*loss_nps
        return adv_loss

    def segmentation_loss(self, seg_outputs, target_masks):
        adv_loss = CustomLoss(alpha=0.25, gamma=2, focal_coef=1e-2, bce_coef=1,
                               dice_coef=1e-2, logit_penalty_coef=1e-2)(seg_outputs, target_masks)
        
        return adv_loss

    def _compute_attack_terms(self, dict_losses, clean_features=None, adv_features=None, seg_outputs=None, target_masks=None, feature_loss=None, seg_loss=None):
        if feature_loss is None:
            feature_loss = torch.nn.MSELoss()(clean_features['p2'], adv_features['p2'])
            for key in ['p3', 'p4', 'p5', 'p6']:
                feature_loss += torch.nn.MSELoss()(clean_features[key], adv_features[key])
        if seg_loss is None:
            seg_loss = self.segmentation_loss(seg_outputs, target_masks)
        return {
            "rpn_cls": dict_losses['loss_rpn_cls'],
            "rpn_loc": dict_losses['loss_rpn_loc'],
            "roi_cls": torch.log1p(1 + (1/(dict_losses['loss_cls'] + 1e-6))),
            "roi_loc": torch.log1p(1 + (1/(dict_losses['loss_box_reg'] + 1e-6))),
            "mask": torch.log1p(1 + (1/(dict_losses['loss_mask'] + 1e-6))),
            "feature": torch.log1p(1 + (1/(feature_loss + 1e-6))),
            "seg": seg_loss,
        }

    def equally_weighted_loss(self, dict_losses, clean_features=None, adv_features=None, seg_outputs=None, target_masks=None, feature_loss=None, seg_loss=None):
        terms = self._compute_attack_terms(dict_losses, clean_features, adv_features, seg_outputs, target_masks, feature_loss, seg_loss)
        adv_loss = sum(terms.values())
        return adv_loss

    def fixed_weighted_loss(self, dict_losses, clean_features=None, adv_features=None, lambdas=None, seg_outputs=None, target_masks=None, feature_loss=None, seg_loss=None):
        terms = self._compute_attack_terms(dict_losses, clean_features, adv_features, seg_outputs, target_masks, feature_loss, seg_loss)
        adv_loss = sum(terms[k] * lambdas[k] for k in terms.keys())
        return adv_loss
    
    def get_loss_weights(self, epoch, cycle_length=20):
        """
        Returns dynamic weights for sampling one loss type at each iteration.
        """
        phase = epoch % cycle_length

        if phase < 3:  # Normal phase
            return {
                "rpn_cls": .0,
                "roi_cls": .8,
                "segmentation": .1,
                "feature": .1,
                "mask": .1,
                "box_reg": .0,
                "rpn_loc": .0,
            }
        elif phase < 6:  
            return {
                "rpn_cls": .0,
                "roi_cls": .5,
                "segmentation": .2,
                "feature": .2,
                "mask": .2,
                "box_reg": .0,
                "rpn_loc": .0,
            }
        else:  
            return {
                "rpn_cls": .14,
                "roi_cls": .14,
                "segmentation": .14,
                "feature": .14,
                "mask": .14,
                "box_reg": .14,
                "rpn_loc": .16,
            }

    def sample_loss_type(self, epoch):
        weights = self.get_loss_weights(epoch)
        return random.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]

    def random_sampling_loss(self, epoch, dict_losses, clean_features=None, adv_features=None, seg_outputs=None, target_masks=None, feature_loss=None, seg_loss=None):

        selected_loss = self.sample_loss_type(epoch)
        terms = self._compute_attack_terms(dict_losses, clean_features, adv_features, seg_outputs, target_masks, feature_loss, seg_loss)
        choice_to_term = {
            "rpn_cls": "rpn_cls",
            "roi_cls": "roi_cls",
            "segmentation": "seg",
            "feature": "feature",
            "mask": "mask",
            "box_reg": "roi_loc",
            "rpn_loc": "rpn_loc",
        }
        term_name = choice_to_term.get(selected_loss, None)
        if term_name is None:
            return torch.tensor(0.0, device=self.device), selected_loss
        return terms[term_name], selected_loss
  
    def gradnorm_penalty(self, task_losses, loss_weights, patch_params, L0, alpha=0.5):
        """
        Returns a scalar GradNorm penalty.  No tensor is modified in-place and
        the gradient graph is preserved so the loss weights can be updated.
        """
        g_norm = []

        for i, Li in enumerate(task_losses):
            gi = torch.autograd.grad(
                loss_weights[i] * Li,
                patch_params,
                retain_graph=True,
                create_graph=True,
                allow_unused=True
            )
            # allow_unused = True handles rare params not touched by a task.
            # Avoid torch.cat on full flattened gradients to keep memory lower.
            sq_norm = torch.zeros((), device=self.device)
            for g in gi:
                if g is not None:
                    sq_norm = sq_norm + g.pow(2).sum()
            g_norm.append(torch.sqrt(sq_norm + 1e-12))
            del gi, sq_norm  # free right away

        g_norm = torch.stack(g_norm)                 # (N_TASKS,)
        g_avg  = g_norm.mean().detach()

        # target ĝᵢ = ḡ · rᵢ^α, where rᵢ is the relative inverse training rate.
        rates = task_losses.detach() / (L0 + 1e-8)
        rates = rates / (rates.mean() + 1e-8)
        target = g_avg * (rates ** alpha)

        # L1 penalty  Σ |gᵢ – ĝᵢ|
        return torch.nn.functional.l1_loss(g_norm, target, reduction='sum')

    def grad_norm_loss(self, epoch, patch_param, L0, dict_losses, loss_weights, clean_features=None, adv_features=None, seg_outputs=None, target_masks=None, alpha=1.5, training=True, compute_gpen=True, feature_loss=None, seg_loss=None):
        terms = self._compute_attack_terms(dict_losses, clean_features, adv_features, seg_outputs, target_masks, feature_loss, seg_loss)
        task_losses = torch.stack([
            terms["rpn_cls"],
            terms["rpn_loc"],
            terms["roi_cls"],
            terms["roi_loc"],
            terms["mask"],
            terms["feature"],
            terms["seg"],
        ])

        if epoch == 0 and torch.count_nonzero(L0) == 0:
            L0.copy_(task_losses.detach())
        
        if training == True:
            gpen = None
            if compute_gpen:
                gpen = self.gradnorm_penalty(task_losses, loss_weights, patch_param, L0, alpha=alpha)
            weighted_patch_loss = (loss_weights.detach() * task_losses).sum()
            return weighted_patch_loss, gpen
        else:
            return (loss_weights.detach() * task_losses).sum()

    def conduct_attack(self, victim_model, detection_net=None):

        victim_model = victim_model.to(self.device)
        for parameter in victim_model.parameters():
            parameter.requires_grad_(False)
        if self.multi_gpu:
            logger.info(f"Using manual multi-GPU attack execution on CUDA devices: {self.device_ids}")

        if self.name == 'shapeShifter' or self.name == 'google':
            patch_param = torch.randn(size=(3, 128, 128), device=self.device)
        elif self.name == 'Dpatch':
            patch_param = torch.randn(size=(3, 128, 128), device=self.device)
        elif self.name == 'scaleAdaptive':
            patch_param = torch.randn(size=(3, 128, 128), device=self.device)
        elif self.name == 'shipCamou':
            patch_param = torch.randn(size=(3, 128, 128), device=self.device)
        elif self.name == 'chunLiu':
            patch_param = torch.randn(size=(3, 128, 128), device=self.device)
        elif self.name == 'shapeAware':
            patch_param = torch.randn(size=(3, 768, 768), device=self.device)
        else:
            patch_param = None

        patch_param = torch.nn.Parameter(patch_param, requires_grad=True)
        
        parameters_count = patch_param.numel()
        parameter_render = None
        seg_model = None
        if self.name == 'shapeAware':
            parameter_render = UNet().to(self.device)
            params_to_optimize = list(parameter_render.parameters()) + [patch_param]
            parameters_count += sum(p.numel() for p in parameter_render.parameters())
        else:
            params_to_optimize = [patch_param]

        if self.attack_loss in ['equally_weighted', 'fixed_weighted', 'random_sampling', 'grad_norm', 'rl_optimization']:
            if detection_net is None:
                raise ValueError("detection_net is required for attack losses that use segmentation terms")
            seg_model = detection_net.segmentation_model(MODEL_SEG='UNET_RESNET34ImgNet').to(self.device).eval()
            for parameter in seg_model.parameters():
                parameter.requires_grad_(False)

        if self.attack_loss == 'grad_norm':
            # --- 1‑D learnable weights w_i, initialised to 1 ---------------------
            loss_weights = torch.nn.Parameter(torch.ones(7, device=self.device))
            # --- an optimiser that updates *only* the weights --------------------
            optim_w = torch.optim.Adam([loss_weights], lr=1e-1)
            # --- store the first‑epoch (un‑weighted) losses as L0_i --------------
            L0 = torch.zeros(7, device=self.device)    # will be filled after 1st step

        if self.attack_loss == "rl_optimization":
            ppo = PPO(device=self.device)
            ppo_optimizer = torch.optim.Adam(ppo.controller.parameters(), lr=1e-2)

        print(f"Number of parameters to be trained is: {parameters_count}")
        optimizer = self.optimizer
        optimizer.param_groups = [] # Empty the list of parameter groups
        optimizer.add_param_group({'params': params_to_optimize})
        poison = Poison(prob=1)
        
        best_loss = np.inf

        def make_adversarial_examples(examples, patch, device=None):
            device = self.device if device is None else device
            patch = (torch.tanh(patch)*103).to(device)
            mean = self.mean.to(device)
            adversarial_data = []
            for inp in examples:
                adversarial_example = inp.copy()
                polygons = inp['instances'].gt_masks

                binary_masks = []
                for polygon in polygons:
                    binary_mask = np.zeros((inp['image'].shape[1], inp['image'].shape[2]), dtype=np.uint8)
                    polygon = polygon[0].reshape((-1, 1, 2))
                    binary_mask = cv2.fillPoly(binary_mask, [np.array(polygon, dtype=np.int32)], 1)
                    binary_masks.append(binary_mask)
                
                image = (inp['image'].to(device) - mean[0])
                if self.poisoning_func == 'Dpatch':
                    adv_image = poison.dpatch_poisoning(image.to(device), patch=patch, masks=binary_masks, training=True)
                elif self.poisoning_func in ['google', 'shapeShifter']:
                    adv_image = poison.google_poisoning(image.to(device), patch=patch, percentage=random.uniform(.2, .6), masks=binary_masks, training=True)
                elif self.poisoning_func == 'scaleAdaptive':
                    adv_image = poison.scaleAdaptive_poisoning(image.to(device), patch=patch, alpha=2.1, masks=binary_masks, training=True)
                elif self.poisoning_func == 'shapeAware':
                    adv_image = poison.shapeAware_poisoning(image.to(device), patch=patch, shape='ellipse', percentage=random.uniform(.2, .7), masks=binary_masks, training=True)
                elif self.poisoning_func == "pieceWise":
                    adv_image = poison.pieceWise_poisoning(image.to(device), patch=patch, shape='ellipse', percentage=0.6, masks=binary_masks, training=True)
                elif self.poisoning_func == "shipCamou":
                    adv_image = poison.shipCamou_poisoning(image.to(device), patch=patch, shape=None, percentage=1., masks=binary_masks, training=True)
                elif self.poisoning_func == "chunLiu":
                    adv_image = poison.chunLiu_poisoning(image.to(device), patch=patch, shape=None, percentage=1., masks=binary_masks, training=True)
                else:
                    adv_image = None

                adversarial_example['image'] = (adv_image.to(device) + mean[0]).clamp(0, 255).requires_grad_(True)
                adversarial_example['height'] = adv_image.shape[1]
                adversarial_example['width'] = adv_image.shape[2]
                
                adversarial_data.append(adversarial_example)
                
            return adversarial_data, patch

        def polygons_to_binary_mask(polygons, height, width):
            mask = np.zeros((height, width), dtype=np.float32)
            rr = []
            cc = []
            for polygon in polygons:
                for i in range(len(polygon[0])):
                    if i % 2 == 0:
                        rr.append(int(polygon[0][i]) - 1)
                    else:
                        cc.append(int(polygon[0][i]) - 1)
                mask[np.array(cc), np.array(rr)] = 1
            return mask
            

        with (EventStorage(0) as storage):
            train_loss = []
            val_loss = []
            lambdas = {
                "rpn_cls": 1,
                "rpn_loc": 1e-3,
                "feature": 1,
                "seg": 1e-3,
                "roi_cls": 1e-2,
                "roi_loc": 1e-3,
                "mask": 1e-2,
            }
            for epoch in range(self.epoch_num):
                losses = []
                iteration = 0
                for batch_inputs in self.train_loader:
                    victim_model.train()
                    if self.name == 'shapeAware':
                        parameter_render.train()
                        patch = parameter_render(patch_param.unsqueeze(0)).squeeze()
                    else:
                        patch = patch_param*1
                    batch_chunks = self._split_batch_for_devices(batch_inputs)
                    adversarial_chunks = [(device, make_adversarial_examples(chunk, patch, device=device)[0]) for device, chunk in batch_chunks]
                    adversarial_data = [item for _, chunk in adversarial_chunks for item in chunk]
                    if len(adversarial_data) == 0:
                        continue

                    dict_losses = self._model_forward(victim_model, adversarial_chunks)


                    if self.attack_loss == 'go':
                        loss = self.go_loss(dict_losses)
                    elif self.attack_loss == 'ss':
                        loss = self.ss_loss(dict_losses)
                    elif self.attack_loss == 'ss_multiclass':
                        loss = self.ss_multiclass_loss(dict_losses)
                    elif self.attack_loss == 'sa':
                        loss = self.as_loss(dict_losses)
                    elif self.attack_loss == 'shipCamou':
                        loss = self.shipCamou_loss(dict_losses, patch)
                    elif self.attack_loss in ['equally_weighted', 'fixed_weighted', 'random_sampling', 'grad_norm', 'rl_optimization']:
                        chunk_weights = [len(chunk) for _, chunk in adversarial_chunks]
                        target_mask_chunks = [
                            torch.tensor(
                                [polygons_to_binary_mask(d['instances'].gt_masks.polygons, d['image'].shape[1], d['image'].shape[2]) for d in clean_chunk]
                            ).unsqueeze(1).to(device)
                            for device, clean_chunk in batch_chunks
                        ]

                        adv_image_tensors = [
                            (torch.stack([adv_d['image'].requires_grad_(True) for adv_d in adv_chunk]).to(device) - self.mean.to(device)).requires_grad_(True)
                            for device, adv_chunk in adversarial_chunks
                        ]
                        clean_image_tensors = [
                            torch.stack([clean_d['image'].float() for clean_d in clean_chunk]).to(device) - self.mean.to(device)
                            for device, clean_chunk in batch_chunks
                        ]

                        adv_features_chunks = self._backbone_features_by_device(victim_model.backbone, adv_image_tensors)
                        seg_input_tensors = [images / self.std.to(images.device) for images in adv_image_tensors]
                        seg_outputs_chunks = self._seg_outputs_by_device(seg_model, seg_input_tensors)
                        clean_features_chunks = self._backbone_features_by_device(victim_model.backbone, clean_image_tensors)
                        feature_loss = self._feature_loss_from_chunks(clean_features_chunks, adv_features_chunks, chunk_weights)
                        seg_loss = self._seg_loss_from_chunks(seg_outputs_chunks, target_mask_chunks, chunk_weights)

                        adv_inputs_for_detection = ImageList.from_tensors([adv_d['image'].requires_grad_(True) for adv_d in adversarial_chunks[0][1]])
                        gt_instances = [x['instances'].to(self.device) for x in adversarial_chunks[0][1]]
                        adv_features = adv_features_chunks[0]
                        del adversarial_data, batch_inputs, adv_image_tensors, clean_image_tensors, seg_input_tensors
                        if self.attack_loss == 'equally_weighted':
                            loss = self.equally_weighted_loss(dict_losses, feature_loss=feature_loss, seg_loss=seg_loss)
                        elif self.attack_loss == "fixed_weighted":
                            loss = self.fixed_weighted_loss(dict_losses, lambdas=lambdas, feature_loss=feature_loss, seg_loss=seg_loss)
                            del adv_features_chunks, clean_features_chunks, target_mask_chunks, seg_outputs_chunks
                        elif self.attack_loss == "random_sampling":
                            loss, sampled_loss_name = self.random_sampling_loss(epoch, dict_losses, feature_loss=feature_loss, seg_loss=seg_loss)
                            del adv_features_chunks, clean_features_chunks, target_mask_chunks, seg_outputs_chunks
                        elif self.attack_loss == "grad_norm":
                            should_update_gradnorm = (iteration % self.gradnorm_update_interval == 0)
                            patch_loss, gradnorm_loss = self.grad_norm_loss(
                                epoch,
                                [patch_param],
                                L0,
                                dict_losses,
                                loss_weights,
                                alpha=1.5,
                                training=True,
                                compute_gpen=should_update_gradnorm,
                                feature_loss=feature_loss,
                                seg_loss=seg_loss
                            )
                            loss = patch_loss
                            del adv_features_chunks, clean_features_chunks, target_mask_chunks, seg_outputs_chunks
                        elif self.attack_loss == "rl_optimization":
                            if epoch%2==0:
                                if iteration == 0:
                                    proposals, _ = victim_model.proposal_generator(adv_inputs_for_detection,
                                                                               adv_features, gt_instances)
                                    ppo.buffer.clear()
                                    state = ppo.initial_state(proposals, gt_instances, patch_param)

                                logits = ppo.controller(state)
                                actions, log_probs = ppo.sample_actions(logits)
                                optimizer.param_groups[0]["lr"] = float((0.001 * ppo.action_values[actions["lr"]]).item())
                                lambdas["rpn_cls"] = float((1 * ppo.action_values[actions["rpn_cls"]]).item())
                                lambdas["rpn_loc"] = float((0.1 * ppo.action_values[actions["rpn_loc"]]).item())
                                lambdas["roi_cls"] = float((1 * ppo.action_values[actions["roi_cls"]]).item())
                                lambdas["roi_loc"] = float((0.1 * ppo.action_values[actions["roi_loc"]]).item())
                                lambdas["feature"] = float((1 * ppo.action_values[actions["feature"]]).item())
                                lambdas["seg"] = float((0.1 * ppo.action_values[actions["seg"]]).item())
                                lambdas["mask"] = float((1 * ppo.action_values[actions["mask"]]).item())
                            loss = self.fixed_weighted_loss(dict_losses, lambdas=lambdas, feature_loss=feature_loss, seg_loss=seg_loss)
                        else:
                            loss = None
                    else:
                        loss = None

                    if self.attack_loss == "grad_norm":
                        if gradnorm_loss is not None:
                            optim_w.zero_grad(set_to_none=True)
                            gradnorm_loss.backward(retain_graph=True)
                            optim_w.step() 
                            with torch.no_grad():
                                loss_weights.data.clamp_(min=1e-8)
                                loss_weights.data *= (loss_weights.numel() / loss_weights.data.sum())
                                print(loss_weights.data)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward(retain_graph=False)
                        optimizer.step()
                    else:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward(retain_graph=False)
                        # print(f"Memory Usage: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GiB")
                        optimizer.step()

                    if iteration % self.log_interval == 0 and self.attack_loss in ['equally_weighted', 'fixed_weighted', 'random_sampling', 'grad_norm']:
                        log_message = f"[train] epoch={epoch} iter={iteration} mode={self.attack_loss} loss={loss.item():.6f}"
                        if self.attack_loss == "fixed_weighted":
                            log_message += f" lambdas={{{', '.join([f'{k}:{v:.3e}' for k, v in lambdas.items()])}}}"
                        elif self.attack_loss == "random_sampling":
                            log_message += f" sampled={sampled_loss_name}"
                        elif self.attack_loss == "grad_norm":
                            gradnorm_value = float("nan") if gradnorm_loss is None else gradnorm_loss.item()
                            log_message += f" gradnorm={gradnorm_value:.6f} weights={loss_weights.detach().cpu().numpy().round(4).tolist()}"
                        logger.info(log_message)
                    
                    if self.attack_loss == "rl_optimization" and epoch%2==0:
                        proposals, _ = victim_model.proposal_generator(adv_inputs_for_detection,
                                                                               adv_features, gt_instances)
                        state, reward, done = ppo.compute_rewards(proposals, gt_instances, patch_param)
                        ppo.buffer.states.append(state)
                        ppo.buffer.rewards.append(float(reward.item()))
                        ppo.buffer.dones.append(done)
                        ppo.buffer.values.append(float(logits["value"].item()))

                        for k in actions:
                            ppo.buffer.actions[k].append(int(actions[k].item()))
                            ppo.buffer.log_probs[k].append(float(log_probs[k].item()))

                        del proposals
                                            
                    losses.append(loss.item())
                    iteration += 1
                
                if self.attack_loss == "rl_optimization" and epoch%2 == 0:
                    ppo.ppo_update(ppo.controller, ppo_optimizer)
                    print(lambdas)
                   
                
                train_loss.append(np.mean(losses))
                logger.info('Epoch {}  train loss: {:.5f}'.format(epoch, np.mean(losses)))

                if epoch%1==0:
                    losses = []
                    with torch.no_grad():
                        for batch_inputs in self.val_loader:
                            if self.name == 'shapeAware':
                                parameter_render.eval()
                                patch = parameter_render(patch_param.unsqueeze(0)).squeeze()
                            else:
                                patch = patch_param*1
                            batch_chunks = self._split_batch_for_devices(batch_inputs)
                            adversarial_chunks = [(device, make_adversarial_examples(chunk, patch, device=device)[0]) for device, chunk in batch_chunks]
                            adversarial_data = [item for _, chunk in adversarial_chunks for item in chunk]
                            if len(adversarial_data) == 0:
                                continue

                            dict_losses = self._model_forward(victim_model, adversarial_chunks)
                            if self.attack_loss == 'go':
                                loss = self.go_loss(dict_losses)
                            elif self.attack_loss == 'ss':
                                loss = self.ss_loss(dict_losses)
                            elif self.attack_loss == 'ss_multiclass':
                                loss = self.ss_multiclass_loss(dict_losses)
                            elif self.attack_loss == 'sa':
                                loss = self.as_loss(dict_losses)
                            elif self.attack_loss == 'shipCamou':
                                loss = self.shipCamou_loss(dict_losses, patch)
                            elif self.attack_loss in ['equally_weighted', 'fixed_weighted', 'random_sampling', 'grad_norm', 'rl_optimization']:
                                chunk_weights = [len(chunk) for _, chunk in adversarial_chunks]
                                target_mask_chunks = [
                                    torch.tensor(
                                        [polygons_to_binary_mask(d['instances'].gt_masks.polygons, d['image'].shape[1], d['image'].shape[2]) for d in clean_chunk]
                                    ).unsqueeze(1).to(device)
                                    for device, clean_chunk in batch_chunks
                                ]

                                adv_image_tensors = [
                                    (torch.stack([adv_d['image'] for adv_d in adv_chunk]).to(device) - self.mean.to(device)).requires_grad_(True)
                                    for device, adv_chunk in adversarial_chunks
                                ]
                                clean_image_tensors = [
                                    torch.stack([clean_d['image'].float() for clean_d in clean_chunk]).to(device) - self.mean.to(device)
                                    for device, clean_chunk in batch_chunks
                                ]

                                adv_features_chunks = self._backbone_features_by_device(victim_model.backbone, adv_image_tensors)
                                seg_input_tensors = [images / self.std.to(images.device) for images in adv_image_tensors]
                                seg_outputs_chunks = self._seg_outputs_by_device(seg_model, seg_input_tensors)
                                clean_features_chunks = self._backbone_features_by_device(victim_model.backbone, clean_image_tensors)
                                feature_loss = self._feature_loss_from_chunks(clean_features_chunks, adv_features_chunks, chunk_weights)
                                seg_loss = self._seg_loss_from_chunks(seg_outputs_chunks, target_mask_chunks, chunk_weights)
                                del adversarial_data, batch_inputs, adv_image_tensors, clean_image_tensors, seg_input_tensors
                                if self.attack_loss == 'equally_weighted':
                                    loss = self.equally_weighted_loss(dict_losses, feature_loss=feature_loss, seg_loss=seg_loss)
                                elif self.attack_loss == "fixed_weighted":
                                    loss = self.fixed_weighted_loss(dict_losses, lambdas=lambdas, feature_loss=feature_loss, seg_loss=seg_loss)
                                    del adv_features_chunks, clean_features_chunks, target_mask_chunks, seg_outputs_chunks
                                elif self.attack_loss == "random_sampling":
                                    loss, _ = self.random_sampling_loss(epoch, dict_losses, feature_loss=feature_loss, seg_loss=seg_loss)
                                    del adv_features_chunks, clean_features_chunks, target_mask_chunks, seg_outputs_chunks
                                elif self.attack_loss == "grad_norm":
                                    loss = self.grad_norm_loss(epoch, [patch_param], L0, dict_losses, loss_weights, alpha=1.5, training=False, feature_loss=feature_loss, seg_loss=seg_loss)
                                    del adv_features_chunks, clean_features_chunks, target_mask_chunks, seg_outputs_chunks
                                elif self.attack_loss == "rl_optimization":
                                    loss = self.fixed_weighted_loss(dict_losses, lambdas=lambdas, feature_loss=feature_loss, seg_loss=seg_loss)
                            else:
                                loss = None

                            losses.append(loss.item())

                        val_loss.append(np.mean(losses))
                        logger.info('val loss: {:.5f}'.format(np.mean(losses)))

                        torch.save(patch.cpu(), f'/home/oraja001/airbus_ship/AdversarialProject/outputs/{self.save_name}.pt')
                        if val_loss[-1] < best_loss:
                            best_loss = val_loss[-1]
                            torch.save(patch.cpu(), f'/home/oraja001/airbus_ship/AdversarialProject/outputs/{self.save_name}_best.pt')

        return patch
