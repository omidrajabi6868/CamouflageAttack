import numpy as np
import cv2
import yaml
import torch
import os
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

from ultralytics import YOLO as ULYOLO
from ultralytics.nn.modules.conv import Conv as ULConv
from detectron2.structures import Boxes, Instances

def rle_decode(mask_rle, shape=(768, 768)):

    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def mask_to_bbox(mask):
    """
    Convert binary mask to bounding box [x_min, y_min, x_max, y_max]
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None  # empty mask
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

def mask_to_polygons(mask):
    """
    Convert binary mask to polygons in COCO format
    (list of lists of x,y coordinates).
    """
    # Ensure mask is uint8
    mask = mask.astype(np.uint8)
    
    # Find contours (external polygons)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) >= 6 and len(contour) % 2 == 0:  # need at least 3 points (x,y)
            polygons.append(contour)

    return polygons

def load_yaml(path):
    """
    Load a YAML file and return it as a Python dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)

def collect_my_conv_blocks(model):
    from YOLO import Conv as MyConv
    blocks = []
    for m in model.modules():
        if isinstance(m, MyConv):
            blocks.append(m)
    return blocks

def collect_ultralytics_conv_blocks(ul_model):
    blocks = []
    for m in ul_model.modules():
        if isinstance(m, ULConv):
            blocks.append(m)
    return blocks

def extract_ultralytics_modules(weight_path):
    ul_model = ULYOLO(weight_path).model
    ul_model.eval()
    return collect_conv_bn_modules(ul_model)

def load_pretrained_yolo(model, weight_path):
    from ultralytics import YOLO

    ul_model = YOLO(weight_path).model.eval()

    my_blocks = collect_my_conv_blocks(model)
    ul_blocks = collect_ultralytics_conv_blocks(ul_model)

    used_ul = set()
    loaded = 0

    for my in my_blocks:
        for i, ul in enumerate(ul_blocks):
            if i in used_ul:
                continue

            # --- Conv weight shape must match ---
            if my.conv.weight.shape == ul.conv.weight.shape:
                my.conv.weight.data.copy_(ul.conv.weight.data)

                # --- BN must exist in both ---
                my.bn.weight.data.copy_(ul.bn.weight.data)
                my.bn.bias.data.copy_(ul.bn.bias.data)
                my.bn.running_mean.data.copy_(ul.bn.running_mean.data)
                my.bn.running_var.data.copy_(ul.bn.running_var.data)

                used_ul.add(i)
                loaded += 1
                break

    print(used_ul)
    total = len(my_blocks)
    print(f"Loaded Conv blocks: {loaded}/{total}")
    print(f"Coverage: {100 * loaded / total:.1f}%")

    return model

def dist2bbox(dist, anchor_points, stride):
    """
    dist: [N,4] or [B,N,4]   (l,t,r,b) in feature-map units
    anchor_points: [N,2] or [B,N,2]  (cx,cy) in feature-map units
    stride: scalar or [N,1]
    returns: boxes in xyxy image coords
    """
    x1 = anchor_points[..., 0] - dist[..., 0]
    y1 = anchor_points[..., 1] - dist[..., 1]
    x2 = anchor_points[..., 0] + dist[..., 2]
    y2 = anchor_points[..., 1] + dist[..., 3]

    boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    return boxes * stride

def box2dist(anchor_points, gt_boxes, stride, reg_max):
    """
    anchor_points: [N,2] (cx,cy) in feature-map units
    gt_boxes: [N,4] (x1,y1,x2,y2) in image coords
    stride: [N,1] or scalar
    returns: [N,4] distances in feature-map units
    """
    x1y1 = gt_boxes[:, :2] / stride
    x2y2 = gt_boxes[:, 2:] / stride

    dist = torch.cat(
        (
            anchor_points - x1y1,
            x2y2 - anchor_points
        ),
        dim=1
    )

    return dist.clamp(0, reg_max - 1e-4)
    
def bbox_ciou(pred, target, eps=1e-7):
    """
    pred:   [...,4]
    target: [...,4]
    """
    px1, py1, px2, py2 = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]
    tx1, ty1, tx2, ty2 = target[..., 0], target[..., 1], target[..., 2], target[..., 3]

    xi1 = torch.max(px1, tx1)
    yi1 = torch.max(py1, ty1)
    xi2 = torch.min(px2, tx2)
    yi2 = torch.min(py2, ty2)

    inter = (xi2 - xi1).clamp(0) * (yi2 - yi1).clamp(0)

    area_p = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    area_t = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)

    union = area_p + area_t - inter + eps
    iou = inter / union

    return iou

def yolo_to_instances(boxes, scores, classes, image_shape):
    """
    boxes: [N,4] xyxy
    image_shape: (H, W)
    """
    instances = Instances(image_shape)

    instances.pred_boxes = Boxes(boxes)
    instances.scores = scores
    instances.pred_classes = classes

    return instances

def make_grid(H, W, device):
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    return torch.stack((x, y), dim=-1).view(-1, 2).float()

def make_anchors(feats, strides, device):
    anchor_points, stride_tensor = [], []
    for feat, stride in zip(feats, strides):
        _, _, h, w = feat.shape
        sy, sx = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij"
        )
        points = torch.stack((sx, sy), dim=-1).view(-1, 2)
        anchor_points.append(points)
        stride_tensor.append(torch.full((points.shape[0], 1), stride, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def decode_boxes(pred_dist, anchor_points, stride_tensor, reg_max):
    """
    pred_dist: [N, 4*reg_max]
    anchor_points: [N,2]  (in grid coords)
    stride_tensor: [N] or [N,1]
    """
    N = pred_dist.shape[0]

    pred_dist = pred_dist.view(N, 4, reg_max)
    pred_dist = pred_dist.softmax(dim=2)

    proj = torch.arange(reg_max, device=pred_dist.device, dtype=pred_dist.dtype)
    dist = pred_dist @ proj  # [N,4]

    x1y1 = anchor_points - dist[:, :2]
    x2y2 = anchor_points + dist[:, 2:]

    boxes = torch.cat([x1y1, x2y2], dim=1)

    # critical fix: force stride to [N,1]
    if stride_tensor.ndim == 1:
        stride_tensor = stride_tensor.unsqueeze(1)

    return boxes * stride_tensor