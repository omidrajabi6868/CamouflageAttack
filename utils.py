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
    device = gt_boxes.device

    if not torch.is_tensor(stride):
        stride = torch.tensor(stride, device=device)

    if stride.ndim == 1:
        stride = stride.unsqueeze(1)

    # convert GT boxes to feature-map space
    x1y1 = gt_boxes[:, :2] / stride
    x2y2 = gt_boxes[:, 2:] / stride

    # L, T, R, B distances
    dist = torch.cat(
        (
            anchor_points - x1y1,
            x2y2 - anchor_points
        ),
        dim=1
    )

    eps = 1e-6
    return dist.clamp(min=0.0, max=reg_max - eps)
    
def bbox_ciou(pred, target, eps=1e-7):
    """
    pred, target: [..., 4] in (x1, y1, x2, y2), image space
    returns: CIoU in [-1, 1]
    """
    px1, py1, px2, py2 = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]
    tx1, ty1, tx2, ty2 = target[..., 0], target[..., 1], target[..., 2], target[..., 3]

    # widths & heights
    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    tw = (tx2 - tx1).clamp(min=eps)
    th = (ty2 - ty1).clamp(min=eps)

    # centers
    pcx = (px1 + px2) * 0.5
    pcy = (py1 + py2) * 0.5
    tcx = (tx1 + tx2) * 0.5
    tcy = (ty1 + ty2) * 0.5

    # intersection
    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)

    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    area_p = pw * ph
    area_t = tw * th
    union = area_p + area_t - inter + eps

    iou = inter / union

    # enclosing box
    ex1 = torch.min(px1, tx1)
    ey1 = torch.min(py1, ty1)
    ex2 = torch.max(px2, tx2)
    ey2 = torch.max(py2, ty2)

    cw = (ex2 - ex1).clamp(min=eps)
    ch = (ey2 - ey1).clamp(min=eps)

    # center distance
    rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
    c2 = cw ** 2 + ch ** 2 + eps

    # aspect ratio penalty
    v = (4 / (torch.pi ** 2)) * (
        torch.atan(tw / th) - torch.atan(pw / ph)
    ) ** 2

    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (rho2 / c2) - alpha * v
    return ciou

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
    anchor_points = []
    stride_tensor = []

    for feat, stride in zip(feats, strides):
        _, _, h, w = feat.shape

        sy, sx = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij"
        )

        points = torch.stack(
            (sx + 0.5, sy + 0.5), dim=-1
        ).reshape(-1, 2).float()

        anchor_points.append(points)

        stride_tensor.append(
            torch.full(
                (points.shape[0], 1),
                float(stride),
                device=device
            )
        )

    return (
        torch.cat(anchor_points, dim=0),
        torch.cat(stride_tensor, dim=0),
    )

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

@torch.no_grad()
def iou_and_counts_detectron2(pred_boxes: Boxes,
                   pred_scores: torch.Tensor,
                   gt_boxes: Boxes,
                   iou_thresh: float = 0.50):
    """
    Return mean IoU & detected (recall proxy) **plus** TP / FP / FN
    calculated with greedy one-to-one matching at `iou_thresh`.

    Args
    ----
    pred_boxes   : detectron2.structures.Boxes  (N, 4)
    pred_scores  : 1-D tensor (N,)  confidence per prediction
    gt_boxes     : Boxes  (M, 4)
    iou_thresh   : float  IoU threshold for TP/FP

    Returns
    -------
    dict { mean_iou, detected, tp, fp, fn }
    """
    device = pred_boxes.tensor.device
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        mean_iou   = 0.0
        detected   = 0
        tp, fp, fn = 0, len(pred_boxes), len(gt_boxes)
        return dict(mean_iou=mean_iou, detected=detected,
                    tp=tp, fp=fp, fn=fn)

    ious = pairwise_iou(pred_boxes, gt_boxes)

    best_iou_per_gt, _ = ious.max(dim=0)         
    mean_iou = best_iou_per_gt.mean().item()
    detected = (best_iou_per_gt > 0).sum().item()

    _, order = pred_scores.sort(descending=True)
    ious = ious[order]

    matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool, device=device)
    tp = fp = 0

    for row in ious:
        max_iou, gt_idx = row.max(dim=0)
        if max_iou >= iou_thresh and not matched_gt[gt_idx]:
            tp += 1
            matched_gt[gt_idx] = True
        else:
            fp += 1

    fn = int((~matched_gt).sum())

    return dict(mean_iou=mean_iou,
                detected=detected,
                tp=tp, fp=fp, fn=fn)