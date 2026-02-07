import numpy as np
import cv2
import torch
import random

from detectron2.structures import ImageList
from detectron2.utils.events import EventStorage, get_event_storage

from Poison import Poison
from UViT import UViT
from Network import UNet, IUNet, ParameterRender, CustomLoss
from PPO import PPO, RolloutBuffer

import logging
from detectron2.utils.logger import setup_logger
setup_logger()
logger = logging.getLogger("detectron2")
logger.setLevel(logging.DEBUG)  # or INFO, WARNING, etc.

class Attack:
    def __init__(self, name, poisoning_func, train_loader, val_loader, optimizer, epoch_num, attack_loss, save_name, mean, std):
        self.name = name
        self.poisoning_func = poisoning_func
        self.optimizer = optimizer
        self.epoch_num = epoch_num
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.rank = 0
        self.device = torch.device(f'cuda:{self.rank}')
        self.mean = mean.to(self.device)
        self.std = std.to(self.device)
        self.attack_loss = attack_loss
        self.save_name = save_name

    def go_loss(self, dict_losses):
        adv_loss = dict_losses['loss_cls']*(-1)
        return adv_loss
    
    def ss_loss(self, dict_losses):
        adv_loss = (-1)*dict_losses['loss_rpn_cls'] - dict_losses['loss_cls']
        return adv_loss
    
    def ss_multiclass_loss(self, dict_losses):
        adv_loss = (dict_losses['loss_rpn_cls']) + (dict_losses['loss_cls'])
        return adv_loss    
    
    def as_loss(self, dict_losses):
        adv_loss = (-1)*dict_losses['loss_rpn_cls']
        return adv_loss
    
    def segmentation_loss(self, seg_outputs, target_masks):
        adv_loss = CustomLoss(alpha=0.25, gamma=2, focal_coef=1e-2, bce_coef=1,
                               dice_coef=1e-2, logit_penalty_coef=1e-2)(seg_outputs, target_masks)
        
        return adv_loss

    def equally_weighted_loss(self, dict_losses, clean_features, adv_features, seg_outputs=None, target_masks=None):
        adv_rpn_cls = dict_losses['loss_rpn_cls']
        adv_rpn_loc = dict_losses['loss_rpn_loc']
        adv_roi_cls = torch.log1p(1 + (1/(dict_losses['loss_cls'] + 1e-6)))
        adv_roi_loc = torch.log1p(1 + (1/(dict_losses['loss_box_reg'] + 1e-6)))
        adv_loss_mask = torch.log1p(1 + (1/(dict_losses['loss_mask'] + 1e-6)))

        feature_loss = torch.nn.MSELoss()(clean_features['p2'], adv_features['p2'])
        for key in ['p3', 'p4', 'p5', 'p6']:
            feature_loss += torch.nn.MSELoss()(clean_features[key], adv_features[key])
        
        adv_feature_loss = torch.log1p(1 + (1/(feature_loss + 1e-6)))

        adv_seg_loss = self.segmentation_loss(seg_outputs, target_masks)

        adv_loss = adv_rpn_cls + adv_rpn_loc + adv_feature_loss + adv_seg_loss + adv_roi_cls + adv_roi_loc + adv_loss_mask

        return adv_loss

    def fixed_weighted_loss(self, dict_losses, clean_features, adv_features, lambdas, seg_outputs=None, target_masks=None):
        adv_rpn_cls = (dict_losses['loss_rpn_cls'])*lambdas['rpn_cls']
        adv_rpn_loc = dict_losses['loss_rpn_loc']*lambdas['rpn_loc']
        adv_roi_cls = torch.log1p(1 + (1/(dict_losses['loss_cls'] + 1e-6)))*lambdas['roi_cls']
        adv_roi_loc = torch.log1p(1 + (1/(dict_losses['loss_box_reg'] + 1e-6)))*lambdas['roi_loc']
        adv_loss_mask = torch.log1p(1 + (1/(dict_losses['loss_mask'] + 1e-6)))*lambdas['mask']

        feature_loss = torch.nn.MSELoss()(clean_features['p2'], adv_features['p2'])
        for key in ['p3', 'p4', 'p5', 'p6']:
            feature_loss += torch.nn.MSELoss()(clean_features[key], adv_features[key])
        
        adv_feature_loss = torch.log1p(1 + (1/(feature_loss + 1e-6)))*lambdas['feature']

        adv_seg_loss = self.segmentation_loss(seg_outputs, target_masks)*lambdas['seg']

        adv_loss = adv_rpn_cls + adv_rpn_loc + adv_feature_loss + adv_seg_loss + adv_roi_cls + adv_roi_loc + adv_loss_mask
        # adv_loss = adv_roi_cls + adv_roi_loc + adv_feature_loss + adv_seg_loss + adv_loss_mask

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
                "box_reg": .0,
                "rpn_loc": .0,
            }
        elif phase < 6:  
            return {
                "rpn_cls": .0,
                "roi_cls": .5,
                "segmentation": .2,
                "feature": .3,
                "box_reg": .0,
                "rpn_loc": .0,
            }
        else:  
            return {
                "rpn_cls": .1,
                "roi_cls": .1,
                "segmentation": .2,
                "feature": .2,
                "box_reg": .2,
                "rpn_loc": .2,
            }

    def sample_loss_type(self, epoch):
        weights = self.get_loss_weights(epoch)
        return random.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]

    def random_sampling_loss(self, epoch, dict_losses, clean_features, adv_features, seg_outputs=None, target_masks=None):

        selected_loss = self.sample_loss_type(epoch)
           
        if selected_loss == "rpn_cls":
            loss = dict_losses['loss_rpn_cls']

        elif selected_loss == "roi_cls":
            loss = torch.log1p(1 + (1/(dict_losses['loss_cls'] + 1e-6)))

        elif selected_loss == "segmentation":
            loss = self.segmentation_loss(seg_outputs, target_masks)

        elif selected_loss == "feature":
            feature_loss = torch.nn.MSELoss()(clean_features['p2'], adv_features['p2'])
            for key in ['p3', 'p4', 'p5', 'p6']:
                feature_loss += torch.nn.MSELoss()(clean_features[key], adv_features[key])
            loss = torch.log1p(1 + (1/(feature_loss + 1e-6)))

        elif selected_loss == "box_reg":
            loss = torch.log1p(1 + (1/(dict_losses['loss_box_reg'] + 1e-6)))

        elif selected_loss == "rpn_loc":
            loss = dict_losses['loss_rpn_loc']
        
        else:
            loss = torch.tensor(0.0)

        return loss
  
    def gradnorm_penalty(self, task_losses, loss_weights, patch_params, L0, alpha=0.5):
        """
        Returns a scalar GradNorm penalty.  No tensor is modified in-place and
        `create_graph=False` so ReLU-in-place inside the detector is harmless.
        """
        g_norm = []

        # We only need first‑order gradients; do NOT build higher‑order graph.
        for i, Li in enumerate(task_losses):
            gi = torch.autograd.grad(
                loss_weights[i] * Li,
                patch_params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )
            # allow_unused = True handles rare params not touched by a task
            flat = torch.cat([g.reshape(-1) for g in gi if g is not None])
            g_norm.append(torch.norm(flat, p=2))
            del gi, flat  # free right away

        g_norm = torch.stack(g_norm)                 # (N_TASKS,)
        g_avg  = g_norm.mean().detach()

        # target ĝᵢ  =  ḡ · (Lᵢ/L₀ᵢ)^α   (no grad through ratios)
        target = g_avg * ((task_losses.detach() / L0)**alpha)

        # L1 penalty  Σ |gᵢ – ĝᵢ|
        return torch.nn.functional.l1_loss(g_norm, target, reduction='sum')

    def grad_norm_loss(self, epoch, patch_param, L0, dict_losses, loss_weights, clean_features, adv_features, seg_outputs=None, target_masks=None, alpha=1.5, training=True):
        adv_rpn_cls = dict_losses['loss_rpn_cls']
        adv_rpn_loc = dict_losses['loss_rpn_loc']
        adv_roi_cls = torch.log1p(1 + (1/(dict_losses['loss_cls'] + 1e-6)))
        adv_roi_loc = torch.log1p(1 + (1/(dict_losses['loss_box_reg'] + 1e-6)))
        adv_loss_mask = torch.log1p(1 + (1/(dict_losses['loss_mask'] + 1e-6)))

        feature_loss = torch.nn.MSELoss()(clean_features['p2'], adv_features['p2'])
        for key in ['p3', 'p4', 'p5', 'p6']:
            feature_loss += torch.nn.MSELoss()(clean_features[key], adv_features[key])
        
        adv_feature_loss = torch.log1p(1 + (1/(feature_loss + 1e-6)))

        adv_seg_loss = self.segmentation_loss(seg_outputs, target_masks)

        task_losses = torch.stack([
            adv_rpn_cls,              
            adv_rpn_loc,                              
            adv_roi_cls,                                
            adv_roi_loc,       
            adv_loss_mask,  
            adv_feature_loss,
            adv_seg_loss
        ])

        if epoch == 0:
            L0.copy_(task_losses.detach())
        
        if training == True:
            gpen = self.gradnorm_penalty(task_losses, loss_weights, patch_param, L0, alpha=alpha)
        else:
            gpen = torch.tensor(0)

        loss = (loss_weights * task_losses).sum() + gpen

        return loss

    def conduct_attack(self, victim_model, detection_net=None):

        if self.name == 'shapeShifter' or self.name == 'google':
            patch_param = torch.randn(size=(3, 32, 32), device=self.device)
        elif self.name == 'Dpatch':
            patch_param = torch.randn(size=(3, 32, 32), device=self.device)
        elif self.name == 'scaleAdaptive':
            patch_param = torch.randn(size=(3, 30, 30), device=self.device)
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
            seg_model = detection_net.segmentation_model(MODEL_SEG='UNET_RESNET34ImgNet').to(self.device).eval()
        else:
            params_to_optimize = [patch_param]

        if self.attack_loss == 'grad_norm':
            # --- 1‑D learnable weights w_i, initialised to 1 ---------------------
            loss_weights = torch.nn.Parameter(torch.ones(7, device=self.device))
            # --- an optimiser that updates *only* the weights --------------------
            optim_w = torch.optim.Adam([loss_weights], lr=1e-1)
            # --- store the first‑epoch (un‑weighted) losses as L0_i --------------
            L0 = torch.zeros(7, device=self.device)    # will be filled after 1st step

        if self.attack_loss == "rl_optimization":
            ppo = PPO()
            ppo_optimizer = torch.optim.Adam(ppo.controller.parameters(), lr=1e-3)

        print(f"Number of parameters to be trained is: {parameters_count}")
        optimizer = self.optimizer
        optimizer.param_groups = [] # Empty the list of parameter groups
        optimizer.add_param_group({'params': params_to_optimize})
        poison = Poison(prob=1)
        
        best_loss = np.inf

        def make_adversarial_examples(examples, patch):
            patch = torch.tanh(patch)*103
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
                
                image = (inp['image'].to(self.device) - self.mean[0])
                if self.poisoning_func == 'Dpatch':
                    adv_image = poison.dpatch_poisoning(image.to(self.device), patch=patch, masks=binary_masks, training=True)
                elif self.poisoning_func in ['google', 'shapeShifter']:
                    adv_image = poison.google_poisoning(image.to(self.device), patch=patch, percentage=random.uniform(.2, .6), masks=binary_masks, training=True)
                elif self.poisoning_func == 'scaleAdaptive':
                    adv_image = poison.scaleAdaptive_poisoning(image.to(self.device), patch=patch, alpha=2.1, masks=binary_masks, training=True)
                elif self.poisoning_func == 'shapeAware':
                    adv_image = poison.shapeAware_poisoning(image.to(self.device), patch=patch, shape='ellipse', percentage=random.uniform(.2, .6), masks=binary_masks, training=True)
                elif self.poisoning_func == "pieceWise":
                    adv_image = poison.pieceWise_poisoning(image.to(self.device), patch=patch, shape='ellipse', percentage=0.8, masks=binary_masks, training=True)
                else:
                    adv_image = None

                adversarial_example['image'] = (adv_image.to(self.device) + self.mean[0].to(self.device)).clamp(0, 255).requires_grad_(True)
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
                    adversarial_data, patch = make_adversarial_examples(batch_inputs, patch)
                    if len(adversarial_data) == 0:
                        continue

                    dict_losses = victim_model(adversarial_data)


                    if self.attack_loss == 'go':
                        loss = self.go_loss(dict_losses)
                    elif self.attack_loss == 'ss':
                        loss = self.ss_loss(dict_losses)
                    elif self.attack_loss == 'ss_multiclass':
                        loss = self.ss_multiclass_loss(dict_losses)
                    elif self.attack_loss == 'sa':
                         loss = self.as_loss(dict_losses)
                    elif self.attack_loss in ['equally_weighted', 'fixed_weighted', 'random_sampling', 'grad_norm', 'rl_optimization']:
                        target_masks = torch.tensor(
                            [polygons_to_binary_mask(d['instances'].gt_masks.polygons, d['image'].shape[1], d['image'].shape[2]) for d in batch_inputs]
                        ).unsqueeze(1).to(self.device)

                        adversarial_images = [adv_d['image'].requires_grad_(True) for adv_d in adversarial_data]
                        gt_instances = [x['instances'].to(self.device) for x in adversarial_data]
                        adv_inputs_for_detection = ImageList.from_tensors(adversarial_images)
                        clean_images = [clean_d['image'].float() for clean_d in batch_inputs]
                        del adversarial_data, batch_inputs

                        adversarial_images = ((torch.stack(adversarial_images).to(self.device) - self.mean)).requires_grad_(True)
                        adv_features = victim_model.backbone(adversarial_images)
                        adversarial_images = adversarial_images/self.std
                        seg_outputs = seg_model(adversarial_images)
                        del adversarial_images

                        clean_images = (torch.stack(clean_images).to(self.device) - self.mean)
                        clean_features = victim_model.backbone(clean_images)
                        del clean_images
                        if self.attack_loss == 'equally_weighted':
                            loss = self.equally_weighted_loss(dict_losses, clean_features, adv_features, seg_outputs, target_masks)
                        elif self.attack_loss == "fixed_weighted":
                            loss = self.fixed_weighted_loss(dict_losses, clean_features, adv_features, lambdas=lambdas, seg_outputs=seg_outputs, target_masks=target_masks)
                            del seg_outputs, adv_features, clean_features, target_masks
                        elif self.attack_loss == "random_sampling":
                            loss = self.random_sampling_loss(epoch, dict_losses, clean_features, adv_features, seg_outputs, target_masks)
                            del seg_outputs, adv_features, clean_features, target_masks
                        elif self.attack_loss == "grad_norm":
                            loss = self.grad_norm_loss(epoch, [patch_param], L0, dict_losses, loss_weights, clean_features, adv_features, seg_outputs, target_masks, alpha=1.5, training=True)
                            del seg_outputs, adv_features, clean_features, target_masks
                            optim_w.zero_grad(set_to_none=True)
                        elif self.attack_loss == "rl_optimization":
                            if epoch%2==0:
                                ppo.controller.eval()
                                if iteration == 0:
                                    proposals, _ = victim_model.proposal_generator(adv_inputs_for_detection,
                                                                               adv_features, gt_instances)
                                    ppo.buffer.clear()
                                    state = ppo.initial_state(proposals, gt_instances, patch_param)

                                logits = ppo.controller(state)
                                actions, log_probs = ppo.sample_actions(logits)
                                optimizer.param_groups[0]["lr"] = 0.001*ppo.action_values[actions["lr"]]
                                lambdas["rpn_cls"] = 1*ppo.action_values[actions["rpn_cls"]]
                                lambdas["rpn_loc"] = 0.1*ppo.action_values[actions["rpn_loc"]]
                                lambdas["roi_cls"] = 1*ppo.action_values[actions["roi_cls"]]
                                lambdas["roi_loc"] = 0.1*ppo.action_values[actions["roi_loc"]]
                                lambdas["feature"] = 1*ppo.action_values[actions["feature"]]
                                lambdas["seg"] = 0.1*ppo.action_values[actions["seg"]]
                                lambdas["mask"] = 1*ppo.action_values[actions["mask"]]
                            loss = self.fixed_weighted_loss(dict_losses, clean_features, adv_features, lambdas=lambdas, seg_outputs=seg_outputs, target_masks=target_masks)
                        else:
                            loss = None
                    else:
                        loss = None

                    optimizer.zero_grad()
                    loss.backward(retain_graph=False)
                    # print(f"Memory Usage: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GiB")
                    optimizer.step()

                    if self.attack_loss == "grad_norm":
                        optim_w.step() 
                        with torch.no_grad():
                            loss_weights.data.clamp_(min=1e-8)
                            loss_weights.data /= loss_weights.data.sum()
                            print(loss_weights.data)
                    
                    if self.attack_loss == "rl_optimization" and epoch%2==0:
                        proposals, _ = victim_model.proposal_generator(adv_inputs_for_detection,
                                                                               adv_features, gt_instances)
                        state, reward, done = ppo.compute_rewards(proposals, gt_instances, patch_param)
                        ppo.buffer.states.append(state)
                        ppo.buffer.rewards.append(reward)
                        ppo.buffer.dones.append(done)
                        ppo.buffer.values.append(logits["value"].item())

                        for k in actions:
                            ppo.buffer.actions[k].append(actions[k])
                            ppo.buffer.log_probs[k].append(log_probs[k])

                        if done:
                            break
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
                            adversarial_data, patch = make_adversarial_examples(batch_inputs, patch)
                            if len(adversarial_data) == 0:
                                continue

                            dict_losses = victim_model(adversarial_data)
                            if self.attack_loss == 'go':
                                loss = self.go_loss(dict_losses)
                            elif self.attack_loss == 'ss':
                                loss = self.ss_loss(dict_losses)
                            elif self.attack_loss == 'ss_multiclass':
                                loss = self.ss_multiclass_loss(dict_losses)
                            elif self.attack_loss == 'sa':
                                loss = self.as_loss(dict_losses)
                            elif self.attack_loss in ['equally_weighted', 'fixed_weighted', 'random_sampling', 'grad_norm', 'rl_optimization']:
                                target_masks = torch.tensor(
                                    [polygons_to_binary_mask(d['instances'].gt_masks.polygons, d['image'].shape[1], d['image'].shape[2]) for d in batch_inputs]
                                ).unsqueeze(1).to(self.device)

                                adversarial_images = [adv_d['image'] for adv_d in adversarial_data]
                                clean_images = [clean_d['image'].float() for clean_d in batch_inputs]
                                del adversarial_data, batch_inputs

                                adversarial_images = ((torch.stack(adversarial_images).to(self.device) - self.mean)).requires_grad_(True)
                                adv_features = victim_model.backbone(adversarial_images)
                                adversarial_images = adversarial_images/self.std
                                seg_outputs = seg_model(adversarial_images)
                                del adversarial_images

                                clean_images = (torch.stack(clean_images).to(self.device) - self.mean)
                                clean_features = victim_model.backbone(clean_images)
                                del clean_images
                                if self.attack_loss == 'equally_weighted':
                                    loss = self.equally_weighted_loss(dict_losses, clean_features, adv_features, seg_outputs, target_masks)
                                elif self.attack_loss == "fixed_weighted":
                                    loss = self.fixed_weighted_loss(dict_losses, clean_features, adv_features, lambdas=lambdas, seg_outputs=seg_outputs, target_masks=target_masks)
                                    del seg_outputs, adv_features, clean_features, target_masks
                                elif self.attack_loss == "random_sampling":
                                    loss = self.fixed_weighted_loss(dict_losses, clean_features, adv_features, lambdas=lambdas, seg_outputs=seg_outputs, target_masks=target_masks)
                                    del seg_outputs, adv_features, clean_features, target_masks
                                elif self.attack_loss == "grad_norm":
                                    loss = self.grad_norm_loss(epoch, [patch_param], L0, dict_losses, loss_weights, clean_features, adv_features, seg_outputs, target_masks, alpha=1.5, training=False)
                                    del seg_outputs, adv_features, clean_features, target_masks
                                elif self.attack_loss == "rl_optimization":
                                    loss = self.fixed_weighted_loss(dict_losses, clean_features, adv_features, lambdas=lambdas, seg_outputs=seg_outputs, target_masks=target_masks)
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