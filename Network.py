import numpy as np
import math
import cv2
import os
import torch
import random
import torch.nn as nn
import time
import logging

import torch.nn.functional as F
import json
import torch.optim as optim
import torch

from skimage.draw import ellipse, disk, polygon
from skimage.measure import label, regionprops
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from detectron2.structures import Boxes, pairwise_iou

torch.autograd.set_detect_anomaly(True)


def compute_metrics(pred, true, batch_size=16, threshold=0.5):
    pred = pred.view(batch_size, -1)
    true = true.view(batch_size, -1)

    pred = (pred > threshold).float()
    true = (true > threshold).float()

    pred_sum = pred.sum(-1)
    true_sum = true.sum(-1)

    neg_index = torch.nonzero(true_sum == 0)
    pos_index = torch.nonzero(true_sum >= 1)

    dice_neg = (pred_sum == 0).float()
    dice_pos = 2 * ((pred * true).sum(-1)) / ((pred + true).sum(-1))

    dice_neg = dice_neg[neg_index]
    dice_pos = dice_pos[pos_index]

    dice = torch.cat([dice_pos, dice_neg])
    jaccard = dice / (2 - dice)

    return dice.numpy(), jaccard.numpy()


class metrics:
    def __init__(self, batch_size=16, threshold=0.5):
        self.threshold = threshold
        self.batchsize = batch_size
        self.dice = []
        self.jaccard = []

    def collect(self, pred, true):
        pred = torch.sigmoid(pred)
        dice, jaccard = compute_metrics(pred, true, batch_size=self.batchsize, threshold=self.threshold)
        self.dice.extend(dice)
        self.jaccard.extend(jaccard)

    def get(self):
        dice = np.nanmean(self.dice)
        jaccard = np.nanmean(self.jaccard)
        return dice, jaccard


class BCEJaccardWithLogitsLoss(nn.Module):
    def __init__(self, jaccard_weight=1, smooth=1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight
        self.smooth = smooth

    def forward(self, outputs, targets):
        if outputs.size() != targets.size():
            raise ValueError("size mismatch, {} != {}".format(outputs.size(), targets.size()))

        loss = self.bce(outputs, targets)

        if self.jaccard_weight:
            targets = (targets == 1.0).float()
            targets = targets.view(-1)
            outputs = torch.sigmoid(outputs)
            outputs = outputs.view(-1)

            intersection = (targets * outputs).sum()
            union = outputs.sum() + targets.sum() - intersection

            loss -= self.jaccard_weight * torch.log(
                (intersection + self.smooth) / (union + self.smooth))  # try with 1-dice
        return loss


class BCEDiceWithLogitsLoss(nn.Module):
    def __init__(self, dice_weight=1, smooth=1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.smooth = smooth

    def __call__(self, outputs, targets):
        if outputs.size() != targets.size():
            raise ValueError("size mismatch, {} != {}".format(outputs.size(), targets.size()))

        loss = self.bce(outputs, targets)

        targets = (targets == 1.0).float()
        targets = targets.view(-1)
        outputs = F.sigmoid(outputs)
        outputs = outputs.view(-1)

        intersection = (outputs * targets).sum()
        dice = 2.0 * (intersection + self.smooth) / (targets.sum() + outputs.sum() + self.smooth)

        loss -= self.dice_weight * torch.log(dice)  # try with 1- dice

        return loss


class FocalWithLogitsLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, outputs, targets):
        if outputs.size() != targets.size():
            raise ValueError("size mismatch, {} != {}".format(outputs.size(), targets.size()))

        targets = (targets == 1.0).float()

        # Sigmoid activation to get probabilities
        outputs = torch.sigmoid(outputs)

        # Clamp probabilities to avoid log(0)
        outputs = torch.clamp(outputs, 1e-6, 1.0 - 1e-6)

        # Calculate the loss for each element
        bce_loss = -targets * torch.log(outputs) - (1 - targets) * torch.log(1 - outputs)
        focal = self.alpha * (1 - outputs) ** self.gamma * targets * bce_loss + \
                (1 - self.alpha) * outputs ** self.gamma * (1 - targets) * bce_loss

        return focal.mean()


def IoU_function(output, target):
    target = (target > 0).float()
    target = target.view(-1)
    output = (output > 0).float()
    output = output.view(-1)

    if target.sum() > 0:
        intersection = (output * target).sum()
        precision = intersection / output.sum()
        recall = intersection / target.sum()
        f1_score = (2 * precision * recall) / (precision + recall)
        iou = intersection / (target.sum() + output.sum() - intersection)
        return precision, recall, f1_score, iou
    else:
        iou = torch.tensor(-1)
        return iou, iou, iou, iou


def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1e-8

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - (((2.0 * intersection) + smooth) / (iflat.sum() + tflat.sum() + smooth))


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha * self.focal(input, target) + dice_loss(input, target)
        return loss.mean()


class TVLoss(nn.Module):
    def __init__(self, tv_weight=1.0):
        super(TVLoss, self).__init__()
        self.tv_weight = tv_weight

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        return self.tv_weight * (h_tv + w_tv) / (batch_size * channels * height * width)


class NPSLoss(nn.Module):
    def __init__(self):
        super(NPSLoss, self).__init__()
        self.printer_colors = torch.tensor([
            [0.0, 0.0, 0.0],  # Black
            [1.0, 1.0, 1.0],  # White
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
            [0.5, 0.5, 0.5],  # Gray
            [0.75, 0.75, 0.75],  # Light Gray
            [0.5, 0.0, 0.0],  # Dark Red
            [0.0, 0.5, 0.0],  # Dark Green
            [0.0, 0.0, 0.5],  # Dark Blue
            [0.5, 0.5, 0.0],  # Olive
            [0.5, 0.0, 0.5],  # Purple
            [0.0, 0.5, 0.5],  # Teal
            [0.25, 0.25, 0.25],  # Dark Gray
            [0.75, 0.0, 0.0],  # Bright Red
            [0.0, 0.75, 0.0],  # Bright Green
            [0.0, 0.0, 0.75],  # Bright Blue
        ], dtype=torch.float32, device='cuda')

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        assert channels == 3, "Input image must have 3 channels (RGB)"

        # Reshape input image to (batch_size * height * width, 3)
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, 3)

        # Calculate distances to printer colors
        distances = torch.cdist(x_flat.unsqueeze(1), self.printer_colors.unsqueeze(0))

        # Find the minimum distance for each pixel
        min_distances, _ = distances.min(dim=2)

        # Reshape back to the original image shape
        min_distances = min_distances.view(batch_size, height, width)

        # Sum the distances to get the NPS loss
        nps_loss = min_distances.sum() / (batch_size * height * width)

        return nps_loss


class CustomLoss(nn.Module):
    def __init__(self, alpha, gamma, focal_coef, bce_coef, dice_coef, logit_penalty_coef=1e-5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.focal_coef = focal_coef
        self.focal = FocalWithLogitsLoss(alpha, gamma=gamma)
        self.bce_coef = bce_coef
        self.dice_coef = dice_coef
        self.logit_penalty_coef = logit_penalty_coef  # Penalty for high logits

    def forward(self, input, target, trigger=None):

        valid_mask = target > 0

        # Target with reduced confidence
        new_target = target * (0)  # Scaling target to lower confidence

        # Focal loss
        adv_focal_loss = (-1)*self.focal(input, target)

        # BCE loss
        adv_bce_loss = self.bce(input[valid_mask], new_target[valid_mask])

        # Dice loss
        adv_dice_loss = (-1)*dice_loss(input, target)

        logits_penalty = torch.mean(torch.nn.functional.relu(input[valid_mask]))

        # Combined loss with regularization for logits
        seg_loss = (self.focal_coef * adv_focal_loss +
                    self.bce_coef * adv_bce_loss +
                    self.dice_coef * adv_dice_loss +
                    self.logit_penalty_coef * logits_penalty)

        return seg_loss


"""Improved U-Net"""


# Implementation from https://github.com/timctho/unet-pytorch/
class IUNet_down_block(torch.nn.Module):
    def __init__(self, input_channel, output_channel, down_size):
        super(IUNet_down_block, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.conv3 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(output_channel)
        self.max_pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()
        self.down_size = down_size

    def forward(self, x):
        if self.down_size:
            x = self.max_pool(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class IUNet_up_block(torch.nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel):
        super(IUNet_up_block, self).__init__()
        self.up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = torch.nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.conv3 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(output_channel)
        self.relu = torch.nn.ReLU()

    def forward(self, prev_feature_map, x):
        x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class IUNet(torch.nn.Module):
    def __init__(self):
        super(IUNet, self).__init__()
        self.down_block1 = IUNet_down_block(3, 16, False)
        self.down_block2 = IUNet_down_block(16, 32, True)
        self.down_block3 = IUNet_down_block(32, 64, True)
        self.down_block4 = IUNet_down_block(64, 128, True)
        self.down_block5 = IUNet_down_block(128, 256, True)
        self.down_block6 = IUNet_down_block(256, 512, True)
        self.down_block7 = IUNet_down_block(512, 1024, True)

        self.mid_conv1 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(1024)
        self.mid_conv2 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(1024)
        self.mid_conv3 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(1024)

        self.up_block1 = IUNet_up_block(512, 1024, 512)
        self.up_block2 = IUNet_up_block(256, 512, 256)
        self.up_block3 = IUNet_up_block(128, 256, 128)
        self.up_block4 = IUNet_up_block(64, 128, 64)
        self.up_block5 = IUNet_up_block(32, 64, 32)
        self.up_block6 = IUNet_up_block(16, 32, 16)

        self.last_conv1 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = torch.nn.BatchNorm2d(16)
        self.last_conv2 = torch.nn.Conv2d(16, 3, 1, padding=0)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        self.x6 = self.down_block6(self.x5)
        self.x7 = self.down_block7(self.x6)
        self.x7 = self.relu(self.bn1(self.mid_conv1(self.x7)))
        self.x7 = self.relu(self.bn2(self.mid_conv2(self.x7)))
        self.x7 = self.relu(self.bn3(self.mid_conv3(self.x7)))
        x = self.up_block1(self.x6, self.x7)
        x = self.up_block2(self.x5, x)
        x = self.up_block3(self.x4, x)
        x = self.up_block4(self.x3, x)
        x = self.up_block5(self.x2, x)
        x = self.up_block6(self.x1, x)
        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        return x


"""Original U-Net"""


class UNet_down_block(torch.nn.Module):
    def __init__(self, input_channel, output_channel, down_size):
        super(UNet_down_block, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.max_pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()
        self.down_size = down_size

    def forward(self, x):
        if self.down_size:
            x = self.max_pool(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNet_up_block(torch.nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel):
        super(UNet_up_block, self).__init__()
        self.up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = torch.nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.relu = torch.nn.ReLU()

    def forward(self, prev_feature_map, x):
        x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.down_block1 = UNet_down_block(3, 64, False)
        self.down_block2 = UNet_down_block(64, 128, True)
        self.down_block3 = UNet_down_block(128, 256, True)
        self.down_block4 = UNet_down_block(256, 512, True)
        self.down_block5 = UNet_down_block(512, 1024, True)

        self.up_block1 = UNet_up_block(512, 1024, 512)
        self.up_block2 = UNet_up_block(256, 512, 256)
        self.up_block3 = UNet_up_block(128, 256, 128)
        self.up_block4 = UNet_up_block(64, 128, 64)

        self.last_conv = torch.nn.Conv2d(64, 3, 1, padding=0)

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        x = self.up_block1(self.x4, self.x5)
        x = self.up_block2(self.x3, x)
        x = self.up_block3(self.x2, x)
        x = self.up_block4(self.x1, x)
        x = self.last_conv(x)
        return x


class ParameterRender(torch.nn.Module):
    def __init__(self):
        super(ParameterRender, self).__init__()

        self.down_block1 = UNet_down_block(3, 16, False)
        self.down_block2 = UNet_down_block(16, 32, True)
        self.down_block3 = UNet_down_block(32, 32, True)
        self.down_block4 = UNet_down_block(32, 64, True)

        self.up_block1 = UNet_up_block(32, 64, 64)
        self.up_block2 = UNet_up_block(32, 64, 32)
        self.up_block3 = UNet_up_block(16, 32, 16)

        self.last_conv = torch.nn.Conv2d(16, 3, 1, padding=0)

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        x = self.up_block1(self.x3, self.x4)
        x = self.up_block2(self.x2, x)
        x = self.up_block3(self.x1, x)
        x = self.last_conv(x)

        return x


class VisualTransformerEncoder(nn.Module):
    def __init__(self, input_size, patch_size, num_patches, emb_dim, num_heads, num_layers, channel):
        super(VisualTransformerEncoder, self).__init__()
        self.input_size = input_size
        self.channel = channel
        self.patch_size = patch_size
        self.num_patches = num_patches
        # self.patch_emb = nn.Conv2d(1, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_emb = nn.Linear((patch_size ** 2) * channel, emb_dim)
        self.cls_emb = nn.Parameter(torch.rand(1, 1, emb_dim))
        self.positional_emb = nn.Parameter(torch.rand(1, num_patches + 1, emb_dim))

        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = x.view(len(x), x.shape[2] // self.patch_size, x.shape[3] // self.patch_size, self.channel, self.patch_size,
                   self.patch_size)
        x = x.view(len(x), self.num_patches, (self.patch_size ** 2) * self.channel)
        x = self.patch_emb(x)
        cls_token = self.cls_emb.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.positional_emb
        for block in self.transformer_blocks:
            x = block(x)
        x = x[:, :, :]
        return x


class VisualTransformerDecoder(nn.Module):
    def __init__(self, patch_size, num_patches,
                 emb_dim, num_heads, num_layers, channel):
        super(VisualTransformerDecoder, self).__init__()
        self.patch_size = patch_size
        self.channel = channel

        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads)
            for _ in range(num_layers)
        ])

        self.fc2 = nn.Linear(emb_dim, (self.patch_size ** 2) * self.channel)

    def forward(self, encoder_output):
        for block in self.transformer_blocks:
            encoder_output = block(encoder_output)
        x = self.fc2(encoder_output)

        return x[:, :, :]


class VisualTransformerAutoencoder(nn.Module):
    def __init__(self, image_size, patch_size, emb_dim, num_heads, num_layers, num_patches, output_channels):
        super(VisualTransformerAutoencoder, self).__init__()
        self.num_patches = num_patches
        self.encoder = VisualTransformerEncoder(image_size, patch_size, num_patches, emb_dim, num_heads, num_layers,
                                                output_channels)
        self.decoder = VisualTransformerDecoder(patch_size, num_patches, emb_dim, num_heads, num_layers,
                                                output_channels)

    def forward(self, x):
        encoder_output = self.encoder(x)
        x = encoder_output + self.encoder.positional_emb
        reconstruction = self.decoder(x)[:, 1:, :]
        reconstruction = reconstruction.view(len(reconstruction), self.encoder.input_size // self.encoder.patch_size,
                                             self.encoder.input_size // self.encoder.patch_size, self.encoder.channel,
                                             self.encoder.patch_size, self.encoder.patch_size)
        reconstruction = reconstruction.view(
            (len(reconstruction), self.encoder.channel, self.encoder.input_size, self.encoder.input_size))
        return reconstruction