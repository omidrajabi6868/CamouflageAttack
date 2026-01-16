import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import math


from skimage.draw import ellipse, disk, polygon
from skimage.measure import label, regionprops

class Poison:
    def __init__(self, prob):
        self.prob = prob

    def _apply_affine_torch(self, patch, angle_deg=0.0, translate=(0, 0), scale=1.0):
        """
        Apply affine transform (scale -> rotate -> translate) to a patch using affine_grid+grid_sample.
        patch: (C, h, w)
        angle_deg: rotation in degrees (positive = counter-clockwise)
        translate: (tx_pixels, ty_pixels) in patch pixel coords
        scale: uniform scaling factor
        Returns transformed patch (C, h, w) with same output size as input patch (padding with zeros).
        Fully differentiable.
        """
        C, H, W = patch.shape
        device = patch.device
        # build theta (1,2,3)
        angle = torch.tensor(angle_deg * np.pi / 180.0, device=device)
        cos = torch.cos(angle)
        sin = torch.sin(angle)

        # linear part: rotation * scale
        a = scale * cos
        b = -scale * sin
        c = scale * sin
        d = scale * cos

        theta = torch.zeros((1, 2, 3), dtype=torch.float, device=device)
        theta[0, 0, 0] = a
        theta[0, 0, 1] = b
        theta[0, 1, 0] = c
        theta[0, 1, 1] = d

        # translate: convert pixel translation to normalized coords in [-1, 1]
        if translate is None:
            tx = ty = 0.0
        else:
            tx_pixels, ty_pixels = translate
            # normalized: 2*tx/(W-1), 2*ty/(H-1) (note: x corresponds to width axis)
            if W > 1:
                tx = 2.0 * float(tx_pixels) / (W - 1)
            else:
                tx = 0.0
            if H > 1:
                ty = 2.0 * float(ty_pixels) / (H - 1)
            else:
                ty = 0.0
        theta[0, 0, 2] = tx
        theta[0, 1, 2] = ty

        # sample grid
        patch_b = patch.unsqueeze(0)  # (1,C,H,W)
        grid = F.affine_grid(theta, patch_b.size(), align_corners=False)  # (1,H,W,2)
        out = F.grid_sample(patch_b, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        return out.squeeze(0)  # (C, H, W)
    
    def google_poisoning(self, image, patch, percentage, masks, training=True):
        """
        Places adversarial patch inside an ellipse-shaped region aligned with object orientation.
        Differentiable transforms so gradients flow to patch.
        """
        device = image.device
        C, H, W = image.shape

        mask_one = torch.zeros((C, H, W), device=device, dtype=image.dtype)
        mask_zero = torch.ones((C, H, W), device=device, dtype=image.dtype)

        if random.random() < self.prob:
            for mask_np in masks:
                lbl = label(mask_np)
                regions = regionprops(lbl)

                for region in regions:
                    x0, y0 = region.centroid
                    orientation = region.orientation - (np.pi / 2)

                    # ellipse radii
                    r_radius = int(percentage * (0.5 * region.axis_minor_length + 1))
                    c_radius = int(percentage * (0.5 * region.axis_major_length + 1))

                    if r_radius <= 0 or c_radius <= 0:
                        continue

                    # ellipse coordinates in image space
                    rr_main, cc_main = ellipse(
                        int(x0), int(y0), r_radius, c_radius,
                        shape=(H, W), rotation=orientation
                    )

                    # --- patch transform ---
                    transform_type = random.choice(["rotate", "translate", "scale", "none"])
                    if training and transform_type == "rotate":
                        angle = random.uniform(-30, 30)
                        patch_t = self._apply_affine_torch(patch, angle_deg=angle)
                    elif training and transform_type == "translate":
                        max_dx = max(1, patch.shape[2] // 10)
                        max_dy = max(1, patch.shape[1] // 10)
                        tx = random.randint(-max_dx, max_dx)
                        ty = random.randint(-max_dy, max_dy)
                        patch_t = self._apply_affine_torch(patch, angle_deg=0.0, translate=(tx, ty))
                    elif training and transform_type == "scale":
                        scale = random.uniform(0.8, 1.2)
                        patch_t = self._apply_affine_torch(patch, angle_deg=0.0, scale=scale)
                    else:
                        patch_t = patch

                    # resize patch to fit ellipse region
                    target_h = max(1, len(rr_main))
                    target_w = max(1, len(cc_main))
                    patch_resized = F.interpolate(
                        patch_t.unsqueeze(0), size=(target_h, target_w),
                        mode="bilinear", align_corners=False
                    ).squeeze(0)

                    # restrict to ellipse indices
                    rr_patch, cc_patch = ellipse(
                        patch_resized.shape[1] // 2,
                        patch_resized.shape[2] // 2,
                        r_radius, c_radius,
                        shape=(patch_resized.shape[1], patch_resized.shape[2]),
                        rotation=orientation
                    )

                    # make sure lengths match
                    min_len = min(len(rr_main), len(rr_patch), len(cc_main), len(cc_patch))
                    rr_main = rr_main[:min_len]
                    cc_main = cc_main[:min_len]
                    rr_patch = rr_patch[:min_len]
                    cc_patch = cc_patch[:min_len]

                    mask_one[:, rr_main, cc_main] = patch_resized[:, rr_patch, cc_patch]
                    mask_zero[:, rr_main, cc_main] = 0.0

        poisoned_image = image * mask_zero + mask_one
        return poisoned_image
    
    def dpatch_poisoning(self, image, patch, masks, training=True):
        """
        image: tensor (C, H, W)
        patch: tensor (C, h0, w0) -- trainable/unconstrained param should be transformed BEFORE passing here
        masks: list of binary masks (numpy arrays) or mask-like objects used previously

        Returns poisoned image (C, H, W). Gradients flow back into patch.
        """
        device = image.device
        C, H, W = image.shape

        # initialize placement maps
        mask_one = torch.zeros((C, H, W), device=device, dtype=image.dtype)
        mask_zero = torch.ones((C, H, W), device=device, dtype=image.dtype)

        if random.random() < self.prob:
            for mask_np in masks:
                # convert mask to labeled regions (you used label + regionprops previously)
                lbl = label(mask_np)
                regions = regionprops(lbl)

                for region in regions:
                    # region.centroid returns (row, col)
                    centroid_row, centroid_col = region.centroid
                    # center pixel where patch will be placed
                    center_r = int(round(centroid_row))
                    center_c = int(round(centroid_col))

                    # random transform parameters
                    angle = random.uniform(-30.0, 30.0)
                    translate = (random.randint(-5, 5), random.randint(-5, 5))
                    scale = random.uniform(0.8, 1.2)

                    transform_type = random.choice(["rotate", "translate", "scale", "none"])


                    if training and transform_type == "rotate":
                        # small random jitter for rotation
                        angle = random.uniform(-30.0, 30.0)
                        patch_t = self._apply_affine_torch(patch, angle_deg=angle, translate=(0, 0), scale=1.0)

                    elif training and transform_type == "translate":
                        # translate in pixels (x,y) where x is width axis
                        max_dx = max(1, patch.shape[2] // 10)
                        max_dy = max(1, patch.shape[1] // 10)
                        tx = random.randint(-max_dx, max_dx)
                        ty = random.randint(-max_dy, max_dy)
                        patch_t = self._apply_affine_torch(patch, angle_deg=0.0, translate=(tx, ty), scale=1.0)

                    elif training and transform_type == "scale":
                        scale = random.uniform(0.8, 1.2)
                        # If you want to actually resize the patch to different spatial dims while preserving gradient,
                        # use interpolate to new size (simpler) OR apply affine with scale and keep same output size.
                        # Here we will use affine with scale (keeps output size same) and then optionally crop when placing.
                        patch_t = self._apply_affine_torch(patch, angle_deg=0.0, translate=(0, 0), scale=scale)

                    else:
                        # no transform
                        patch_t = patch

                    # Now paste patch_t into mask_one at centroid location.
                    # patch_t has shape (C, ph, pw)
                    ph, pw = patch_t.shape[1], patch_t.shape[2]

                    # compute bounding box in image coordinates
                    top = center_r - ph // 2
                    left = center_c - pw // 2
                    bottom = top + ph
                    right = left + pw

                    # compute overlap with image (clamp)
                    top_img = max(0, top)
                    left_img = max(0, left)
                    bottom_img = min(H, bottom)
                    right_img = min(W, right)

                    if top_img >= bottom_img or left_img >= right_img:
                        # no overlap
                        continue

                    # corresponding patch region indices
                    top_patch = top_img - top
                    left_patch = left_img - left
                    bottom_patch = top_patch + (bottom_img - top_img)
                    right_patch = left_patch + (right_img - left_img)

                    # slice and paste
                    patch_slice = patch_t[:, top_patch:bottom_patch, left_patch:right_patch]  # (C, h', w')
                    h_slice, w_slice = patch_slice.shape[1], patch_slice.shape[2]

                    # place into mask_one and zero-out mask_zero at the same region
                    mask_one[:, top_img:bottom_img, left_img:right_img] = patch_slice
                    mask_zero[:, top_img:bottom_img, left_img:right_img] = 0.0

        poisoned_image = image * mask_zero + mask_one
        return poisoned_image
    
    def scaleAdaptive_poisoning(self, image, patch, alpha, masks, training=True):

        mask_one = torch.zeros((3, image.shape[1], image.shape[2])).to('cuda')
        mask_zero = torch.ones((3, image.shape[1], image.shape[2])).to('cuda')

        if random.random() < self.prob:
            for mask in masks:
                lbl = label(mask)
                regions = regionprops(lbl)
                for region in regions:
                    x0, y0 = region.centroid

                    min_row, min_col, max_row, max_col = region.bbox
                    height = max_row - min_row
                    width = max_col - min_col

                    scale_h = (((height*(1/4))**alpha)**.5)/patch.shape[1]
                    scale_w = (((width*(1/4))**alpha)**.5)/patch.shape[2]

                    new_h = max(1, int(round(patch.shape[1] * scale_h)))
                    new_w = max(1, int(round(patch.shape[2] * scale_w)))
                    patch_t = F.interpolate(patch.unsqueeze(0), size=(new_h, new_w),
                                            mode='bilinear', align_corners=False).squeeze(0)

                    row_slice = patch_t.shape[1] // 2
                    column_slice = patch_t.shape[2] // 2
                    mask_one_shape = mask_one[:, int(x0) - row_slice:  int(x0) + row_slice, int(y0) - column_slice:int(y0) + column_slice]
                    mask_one[:, int(x0) - row_slice:  int(x0) + row_slice, int(y0) - column_slice:int(y0) + column_slice] = patch_t[:, :mask_one_shape.shape[1], :mask_one_shape.shape[2]]
                    mask_zero[:, int(x0) - row_slice:  int(x0) + row_slice, int(y0) - column_slice:int(y0) + column_slice] = 0

        poisoned_image = image * mask_zero + mask_one
        
        return poisoned_image
    
    def shapeAware_poisoning(self, img, patch, shape, percentage, masks, training=True):

        mask_one = torch.zeros((3, img.shape[1], img.shape[2])).to('cuda')
        mask_zero = torch.ones((3, img.shape[1], img.shape[2])).to('cuda')

        if patch.shape[1] != img.shape[1] or patch.shape[2] != img.shape[2]:
            patch = F.interpolate(patch.unsqueeze(0), size=(img.shape[1], img.shape[2]),
                            mode='bilinear', align_corners=False).squeeze(0)

        for mask in masks:
            lbl = label(mask)
            regions = regionprops(lbl)
            if random.random() < self.prob:
                for region in regions:
                    x0, y0 = region.centroid
                    orientation = region.orientation
                    bbox = region.bbox
                    orientation = orientation - (np.pi / 2)
                    r_radius = int(percentage * (int(.5 * region.axis_minor_length) + 1))
                    c_radius = int(percentage * (int(.5 * region.axis_major_length) + 1))
                    if shape in ['ellipse', 'base']:
                        rr_main, cc_main = ellipse(int(x0), int(y0), r_radius, c_radius,
                                                shape=(img.shape[1], img.shape[2]), rotation=orientation)
                        r = x0 - c_radius
                        c = y0 - c_radius

                        if r > 0 and c > 0 and x0 < patch.shape[1] and y0 < patch.shape[2] and training:
                            rr, cc = ellipse(int(patch.shape[1] / 2),
                                             int(patch.shape[2] / 2),
                                             r_radius, c_radius,
                                             shape=(patch.shape[1], patch.shape[1]),
                                             rotation=orientation)
                        else:
                            rr, cc = ellipse(int(patch.shape[1] / 2),
                                             int(patch.shape[2] / 2),
                                             r_radius, c_radius,
                                             shape=(patch.shape[1], patch.shape[1]),
                                             rotation=orientation)

                    if shape == 'disk':
                        rr_main, cc_main = disk((int(x0), int(y0)), min(r_radius, c_radius),
                                                shape=(img.shape[1], img.shape[1]))
                        r = x0 - c_radius
                        c = y0 - c_radius
                        if r > 0 and c > 0 and x0 < patch.shape[1] and y0 < patch.shape[1]:
                            rr, cc = disk((int(patch.shape[1] / 2), int(patch.shape[1] / 2)),
                                        min(r_radius, c_radius),
                                        shape=(patch.shape[1], patch.shape[1]))
                        else:
                            rr, cc = disk((int(patch.shape[1] / 2), int(patch.shape[1] / 2)),
                                        min(r_radius, c_radius),
                                        shape=(patch.shape[1], patch.shape[1]))

                    if shape == 'rectangle':
                        def get_rotated_rectangle(center_x, center_y, height, width, angle):
                            dx = width / 2.0
                            dy = height / 2.0
                            # Corners relative to center (unrotated)
                            corners = np.array([
                                [-dx, -dy],
                                [dx, -dy],
                                [dx, dy],
                                [-dx, dy]
                            ])
                            # Rotation matrix
                            rotation_matrix = np.array([
                                [math.cos(angle), -math.sin(angle)],
                                [math.sin(angle), math.cos(angle)]
                            ])
                            # Rotate and translate corners
                            rotated_corners = np.dot(corners, rotation_matrix.T)
                            rotated_corners[:, 0] += center_x
                            rotated_corners[:, 1] += center_y
                            return rotated_corners

                        # Rectangle dimensions
                        rect_height = int(self.percentage * region.axis_minor_length) + 1
                        rect_width = int(self.percentage * region.axis_major_length) + 1
                        # Compute the corners of the rectangle (rotated)
                        corners = get_rotated_rectangle(x0, y0, rect_height, rect_width, region.orientation)
                        # Draw the polygon
                        rr_main, cc_main = polygon(corners[:, 0], corners[:, 1], shape=(img.shape[1], img.shape[1]))

                        st = (rect_width // 2)
                        en = patch.shape[1] - (rect_width // 2)

                        if en > st and x0 < patch.shape[1] and y0 < patch.shape[1]:
                            # rand_row = random.randint(int(st), int(en))
                            # rand_col = random.randint(int(st), int(en))
                            center_row, center_col = int(x0), int(y0)
                            rect_corners = get_rotated_rectangle(center_row, center_col, rect_height, rect_width,
                                                                    region.orientation)
                            rr, cc = polygon(rect_corners[:, 0], rect_corners[:, 1],
                                            shape=(patch.shape[1], patch.shape[1]))
                        else:
                            center_row, center_col = patch.shape[1] // 2, patch.shape[1] // 2
                            rect_corners = get_rotated_rectangle(center_row, center_col, rect_height, rect_width,
                                                                    region.orientation)
                            rr, cc = polygon(rect_corners[:, 0], rect_corners[:, 1],
                                            shape=(patch.shape[1], patch.shape[1]))

                    if shape == 'base':
                        row_slice = patch.shape[1] // 2
                        column_slice = patch.shape[2] // 2
                        mask_one_shape = mask_one[:, int(x0) - row_slice:  int(x0) + row_slice,
                                        int(y0) - column_slice:int(y0) + column_slice]
                        mask_one[:, int(x0) - row_slice:  int(x0) + row_slice,
                        int(y0) - column_slice:int(y0) + column_slice] = patch[:, :mask_one_shape.shape[1],
                                                                        :mask_one_shape.shape[2]].cpu()
                        mask_zero[:, int(x0) - row_slice:  int(x0) + row_slice,
                        int(y0) - column_slice:int(y0) + column_slice] = 0
                    elif shape in ['ellipse', 'disk']:
                        mask_one[:, rr_main, cc_main] = 1
                        mask_zero[:, rr_main, cc_main] = 0
                        mask_one[:, rr_main, cc_main] = patch[:, rr[:len(rr_main)], cc[:len(cc_main)]]
                    else:
                        mask_one[:, rr_main, cc_main] = 1
                        mask_zero[:, rr_main, cc_main] = 0
                        mask_one[:, rr_main, cc_main] = patch[:, rr_main, cc_main]

        poisoned_image = img * mask_zero + mask_one

        return poisoned_image

    def pieceWise_poisoning(self, img, patch, shape, percentage, masks, training=True):

        mask_one = torch.zeros((3, img.shape[1], img.shape[2])).to('cuda')
        mask_zero = torch.ones((3, img.shape[1], img.shape[2])).to('cuda')

        if patch.shape[1] != img.shape[1] or patch.shape[2] != img.shape[2]:
            patch = F.interpolate(patch.unsqueeze(0), size=(img.shape[1], img.shape[2]), mode='bilinear', align_corners=False).squeeze(0)
        
        for mask in masks:
            lbl = label(mask)
            regions = regionprops(lbl)
            if random.random() < self.prob:
                for region in regions:
                    x0, y0 = region.centroid
                    orientation = region.orientation
                    bbox = region.bbox
                    orientation = orientation - (np.pi / 2)
                    r_radius = int(.5 * region.axis_minor_length)
                    c_radius = int(.5 * region.axis_major_length)
                    # percentage = (3/10) * math.sqrt((2*region.axis_major_length)/region.axis_minor_length)
                    percentage = (1/5) * math.sqrt((3*region.axis_major_length)/region.axis_minor_length)

                    x1 = x0 + (c_radius/2)*torch.cos(torch.tensor(orientation) + (np.pi / 2))
                    y1 = y0 + (c_radius/2)*torch.sin(torch.tensor(orientation) + (np.pi / 2))

                    x2 = x0 - (c_radius/2)*torch.cos((torch.tensor(orientation) + (np.pi / 2)))
                    y2 = y0 - (c_radius/2)*torch.sin((torch.tensor(orientation) + (np.pi / 2)))

                    rr_main0, cc_main0 = disk((int(x0), int(y0)),
                        int(percentage*(r_radius)),  
                        shape=(img.shape[1], img.shape[2]))

                    rr_main1, cc_main1 = disk((int(x1), int(y1)),
                                                int(percentage*(r_radius)), 
                                                shape=(img.shape[1], img.shape[2]))

                    rr_main2, cc_main2 = disk((int(x2), int(y2)),
                                                int(percentage*(r_radius)),  
                                                shape=(img.shape[1], img.shape[2]))

                    rr0, cc0 = disk((int(patch.shape[1]/2),
                                    int(patch.shape[2]/2)),
                                    int(percentage*(r_radius)), 
                                    shape=(patch.shape[1], patch.shape[1]))
                    
                    
                    rr1, cc1 = disk((int(patch.shape[1]/2),
                                    int(patch.shape[2]/2)),
                                    int(percentage*(r_radius)), 
                                    shape=(patch.shape[1], patch.shape[1]))

                    rr2, cc2 = disk((int(patch.shape[1]/2),
                                    int(patch.shape[2]/2)),
                                    int(percentage*(r_radius)), 
                                    shape=(patch.shape[1], patch.shape[1]))

                    mask_one[:, rr_main0, cc_main0] = 1
                    mask_one[:, rr_main1, cc_main1] = 1
                    mask_one[:, rr_main2, cc_main2] = 1
                  
                    mask_zero[:, rr_main0, cc_main0] = 0
                    mask_zero[:, rr_main1, cc_main1] = 0
                    mask_zero[:, rr_main2, cc_main2] = 0

                    mask_one[:, rr_main0, cc_main0] = patch[:, rr2[:len(rr_main0)], cc2[:len(cc_main0)]]
                    mask_one[:, rr_main1, cc_main1] = patch[:, rr1[:len(rr_main1)], cc1[:len(cc_main1)]]
                    mask_one[:, rr_main2, cc_main2] = patch[:, rr2[:len(rr_main2)], cc2[:len(cc_main2)]]

        poisoned_image = img * mask_zero + mask_one
        
        return poisoned_image


                    
                    


                    
                    
                    














