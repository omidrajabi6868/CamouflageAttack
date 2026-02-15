import torch
import torchvision
import torch.nn.functional as F

from torch import nn
from utils import *
from detectron2.structures import Boxes, Instances

# YOLOv8 scaling coefficients
YOLOV8_SCALES = {
    "n": dict(depth=0.33, width=0.25),
    "s": dict(depth=0.33, width=0.50),
    "m": dict(depth=0.67, width=0.75),
    "l": dict(depth=1.00, width=1.00),
    "x": dict(depth=1.33, width=1.25),
}

def make_divisible(x, divisor=8):
    return int((x + divisor / 2) // divisor * divisor)

def yolo_v8_postprocess(preds, strides=(8, 16, 32),
                        conf_thres=0.25, iou_thres=0.7, max_det=300):

    device = preds[0].device
    bs = preds[0].shape[0]
    reg_max = 16
    nc = preds[0].shape[1] - 4 * reg_max

    outputs = []
    anchor_points, stride_tensor = make_anchors(preds, strides, device)

    for b in range(bs):
        pred_dist, pred_cls = [], []

        for p in preds:
            p = p[b]
            pred_dist.append(
                p[:4 * reg_max].permute(1, 2, 0).reshape(-1, 4 * reg_max)
            )
            pred_cls.append(
                p[4 * reg_max:].permute(1, 2, 0).reshape(-1, nc)
            )

        pred_dist = torch.cat(pred_dist)
        pred_cls = torch.cat(pred_cls).sigmoid()

        assert pred_dist.shape[0] == anchor_points.shape[0]

        boxes = decode_boxes(pred_dist, anchor_points, stride_tensor, reg_max)

        scores, labels = pred_cls.max(dim=1)
        mask = scores > conf_thres

        boxes = boxes[mask].float()
        scores = scores[mask]
        labels = labels[mask]

        if boxes.numel() == 0:
            outputs.append(torch.zeros((0, 6), device=device))
            continue

        keep = torchvision.ops.batched_nms(
            boxes, scores, labels, iou_thres
        )[:max_det]

        outputs.append(torch.cat([
            boxes[keep],
            scores[keep, None],
            labels[keep, None].float()
        ], dim=1))

    return outputs

# ---------------------------
# Yolo Predictor
# ---------------------------
class YOLOPredictor:
    def __init__(
        self,
        model,
        strides=(8, 16, 32),
        class_names=None,
        conf_thres=0.25,
        iou_thres=0.7,
        device="cuda",
    ):
        self.model = model.to(device).eval()
        self.strides = strides
        self.class_names = class_names
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device

    @torch.no_grad()
    def __call__(self, image_bgr):
        h, w = image_bgr.shape[:2]

        # --- preprocess (YOLOv8 style) ---
        image = (
            torch.from_numpy(image_bgr)
            .to(self.device)
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            / 255.0
        )

        # --- forward ---
        preds = self.model(image)

        # --- YOLOv8 postprocess ---
        dets = yolo_v8_postprocess(
            preds,
            strides=self.strides,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
        )

        if dets[0].numel() == 0:
            return {"instances": Instances((h, w))}

        det = dets[0]

        boxes = det[:, :4]
        scores = det[:, 4]
        classes = det[:, 5].long()

        instances = yolo_to_instances(
            boxes, scores, classes, (h, w)
        )

        return {"instances": instances}
# ---------------------------
# Basic Convolution Layer
# ---------------------------
class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p or k//2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# ---------------------------
# Bottleneck
# ---------------------------
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

# ---------------------------
# C2f Block
# ---------------------------
class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


# ---------------------------
# SPPF Block
# ---------------------------
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c_*4, c2, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))

# ---------------------------
# Backbone
# ---------------------------
class Backbone(nn.Module):
    def __init__(self, model_size="l"):
        super().__init__()
        s = YOLOV8_SCALES[model_size]
        d, w = s["depth"], s["width"]

        def c(ch): return make_divisible(ch * w)
        def n(rep): return max(1, int(rep * d))

        # Stem
        self.stem = Conv(3, c(64), 3, 2)

        self.stage1 = nn.Sequential(
            Conv(c(64), c(128), 3, 2),
            C2f(c(128), c(128), n(3), shortcut=True)
        )

        # Stage2 (P3)
        self.stage2 = nn.Sequential(
            Conv(c(128), c(256), 3, 2),
            C2f(c(256), c(256), n(6), shortcut=True)
        )

        # Stage3 (P4)
        self.stage3 = nn.Sequential(
            Conv(c(256), c(512), 3, 2),
            C2f(c(512), c(512), n(6), shortcut=True)
        )

         # Stage4 
        self.stage4 = nn.Sequential(
            Conv(c(512), c(1024), 3, 2),
            C2f(c(1024), c(1024), n(3), shortcut=True)
        )

        # (P5)
        self.sppf = SPPF(c(1024), c(1024))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.sppf(self.stage4(p4))
        return p3, p4, p5

# ---------------------------
# Neck
# ---------------------------
class Neck(nn.Module):
    def __init__(self, model_size="l"):
        super().__init__()
        w = YOLOV8_SCALES[model_size]["width"]
        c = lambda x: make_divisible(x * w)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Top-down
        self.reduce_p5 = Conv(c(1024), c(512), 1)
        self.c2f_p4 = C2f(c(512)+c(512), c(512), 3)

        self.reduce_p4 = Conv(c(512), c(256), 1)
        self.c2f_p3 = C2f(c(256)+c(256), c(256), 3)

        # Bottom-up
        self.down_p3 = Conv(c(256), c(256), 3, 2)
        self.c2f_n4 = C2f(c(256)+c(512), c(512), 3)

        self.down_p4 = Conv(c(512), c(512), 3, 2)
        self.c2f_n5 = C2f(c(512)+c(1024), c(1024), 3)

    def forward(self, feats):
        p3, p4, p5 = feats

        # Top-down pathway
        p5_up = self.upsample(self.reduce_p5(p5))         # 512 channels, 48x48
        p4_td = self.c2f_p4(torch.cat([p5_up, p4], dim=1)) # 512+512=1024 → 512

        p4_up = self.upsample(self.reduce_p4(p4_td))       # 256 channels, 96x96
        p3_td = self.c2f_p3(torch.cat([p4_up, p3], dim=1)) # 256+256=512 → 256

        # Bottom-up pathway
        p3_down = self.down_p3(p3_td)                      # 256 → 256, 48x48
        p4_out = self.c2f_n4(torch.cat([p3_down, p4_td], dim=1)) # 256+512=768 → 512

        p4_down = self.down_p4(p4_out)                     # 512 → 512, 24x24
        p5_out = self.c2f_n5(torch.cat([p4_down, p5], dim=1))     # 512+1024=1536 →1024

        return p3_td, p4_out, p5_out

# ---------------------------
# Detection Head
# ---------------------------
class Detect(nn.Module):
    def __init__(self, ch, nc=80, reg_max=16):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.no = nc + 4 * reg_max

        self.cv = nn.ModuleList(
            nn.Conv2d(c, self.no, 1) for c in ch
        )

    def forward(self, x):
        return [self.cv[i](x[i]) for i in range(len(x))]

# ---------------------------
# YOLO Model
# ---------------------------
class YOLO(nn.Module):

    def __init__(self, model_size="l", num_classes=80, reg_max=16):
        super().__init__()
        self.backbone = Backbone(model_size)
        self.neck = Neck(model_size)

        w = YOLOV8_SCALES[model_size]["width"]
        ch = [
            make_divisible(256*w),
            make_divisible(512*w),
            make_divisible(1024*w),
        ]
        self.detect = Detect(ch, num_classes, reg_max)

    def forward(self, x):
        return self.detect(self.neck(self.backbone(x)))

# -----------------------------
# DFL Loss
# -----------------------------
class DFLLoss(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred, target):
        """
        pred:   [N, 4, reg_max] logits
        target: [N, 4] float distances (feature-map units)
        """
        target = target.clamp(0, self.reg_max - 1e-4)

        left = target.floor().long()
        right = (left + 1).clamp(max=self.reg_max - 1)

        wl = right.float() - target
        wr = target - left.float()

        pred = pred.view(-1, self.reg_max)
        left = left.view(-1)
        right = right.view(-1)

        loss = (
            F.cross_entropy(pred, left, reduction="none") * wl.view(-1) +
            F.cross_entropy(pred, right, reduction="none") * wr.view(-1)
        )

        return loss.mean()


# -----------------------------
# YOLOv8 Task Aligned Assigner
# -----------------------------
class TaskAlignedAssigner:
    def __init__(self, topk=10, alpha=1.0, beta=6.0):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def assign(self, pred_boxes, pred_scores, gt_boxes, gt_labels, anchor_points, stride_tensor):
        device = pred_boxes.device
        N = pred_boxes.shape[0]
        M = gt_boxes.shape[0]

        if stride_tensor.ndim == 1:
            stride_tensor = stride_tensor.unsqueeze(1)

        anchor_points_img = anchor_points * stride_tensor

        if M == 0:
            return (
                torch.zeros(N, dtype=torch.bool, device=device),
                torch.zeros(N, dtype=torch.long, device=device),
                torch.zeros(N, 4, device=device),
                torch.zeros(N, device=device),
            )

        in_gt = (
            (anchor_points_img[:, None, 0] >= gt_boxes[None, :, 0]) &
            (anchor_points_img[:, None, 0] <= gt_boxes[None, :, 2]) &
            (anchor_points_img[:, None, 1] >= gt_boxes[None, :, 1]) &
            (anchor_points_img[:, None, 1] <= gt_boxes[None, :, 3])
        ) 

        ious = bbox_ciou(
            pred_boxes[:, None, :],
            gt_boxes[None, :, :]
        ) 
        assert gt_labels.max() < pred_scores.shape[1]
        cls_scores = pred_scores[:, gt_labels.long()]  # [N, M]

        ious = ious.clamp(min=0)
        alignment = (cls_scores ** self.alpha) * (ious ** self.beta)
        alignment = alignment * in_gt

        eps = 1e-9
        alignment = alignment / (alignment.max(dim=0, keepdim=True).values + eps)
        alignment = alignment * (ious > 0.1)

        fg_mask = torch.zeros(N, dtype=torch.bool, device=device)
        assigned_gt = torch.full((N,), -1, device=device)
        assigned_score = torch.zeros(N, device=device)

        topk = min(self.topk, N)

        for gt_idx in range(M):
            scores = alignment[:, gt_idx]
            topk_scores, idx = scores.topk(topk)

            valid = topk_scores > 0
            idx = idx[valid]

            better = scores[idx] > assigned_score[idx]
            idx = idx[better]

            fg_mask[idx] = True
            assigned_gt[idx] = gt_idx
            assigned_score[idx] = scores[idx]

        # ---------------------------
        # Safe label & box gathering
        # ---------------------------
        labels = torch.zeros(N, dtype=torch.long, device=device)
        boxes = torch.zeros(N, 4, device=device)

        pos = assigned_gt >= 0
        labels[pos] = gt_labels[assigned_gt[pos]]
        boxes[pos] = gt_boxes[assigned_gt[pos]]

        return fg_mask, labels, boxes, assigned_score


# -----------------------------
# YOLOv8 Loss (DFL + BCE + CIoU)
# -----------------------------
class YOLOv8Loss(nn.Module):
    def __init__(self, nc, reg_max=16):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max

        self.box_weight = 7.5
        self.cls_weight = 0.5
        self.dfl_weight = 1.5

        self.assigner = TaskAlignedAssigner()
        self.dfl = DFLLoss(reg_max)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        self.register_buffer("proj", torch.arange(reg_max).float())

    def forward(self, preds, targets, strides):
        device = preds[0].device
        total_box = torch.zeros(1, device=device)
        total_cls = torch.zeros(1, device=device)
        total_dfl = torch.zeros(1, device=device)

        anchor_points, stride_tensor = make_anchors(preds, strides, device)

        for b, target in enumerate(targets):
            gt_boxes = target["boxes"].to(device)
            gt_labels = target["labels"].to(device)

            pred_dist, pred_cls = [], []

            for p in preds:
                p = p[b]
                pred_dist.append(
                    p[:4*self.reg_max].view(4, self.reg_max, -1).permute(2,0,1)
                )
                pred_cls.append(
                    p[4*self.reg_max:].permute(1,2,0).reshape(-1, self.nc)
                )

            pred_dist = torch.cat(pred_dist)
            pred_cls = torch.cat(pred_cls)

            pred_boxes = decode_boxes(pred_dist, anchor_points, stride_tensor, self.reg_max)

            fg_mask, labels, boxes, scores = self.assigner.assign(
                pred_boxes,
                pred_cls.sigmoid(),
                gt_boxes,
                gt_labels,
                anchor_points,
                stride_tensor
                )

            num_fg = fg_mask.sum().clamp(min=1)

            if fg_mask.sum() == 0:
                total_box += pred_boxes.sum() * 0.0
                total_cls += pred_cls.sum() * 0.0
                total_dfl += pred_dist.sum() * 0.0
                continue

            total_box += (1 - bbox_ciou(
                pred_boxes[fg_mask],
                boxes[fg_mask]
            )).mean()

            dist_targets = box2dist(
                anchor_points[fg_mask],
                boxes[fg_mask],
                stride_tensor[fg_mask],
                self.reg_max
            )

            total_dfl += self.dfl(pred_dist[fg_mask], dist_targets)

            cls_target = torch.zeros_like(pred_cls)
            cls_target[fg_mask, labels[fg_mask]] = scores[fg_mask]

            cls_loss = self.bce(pred_cls, cls_target)
            cls_loss = cls_loss.sum(dim=1)          # sum over classes
            cls_loss = cls_loss.mean()     # only positives
            total_cls += cls_loss

        return (
            self.box_weight * total_box +
            self.cls_weight * total_cls +
            self.dfl_weight * total_dfl
        ).sum()


