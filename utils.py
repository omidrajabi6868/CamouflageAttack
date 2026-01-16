import numpy as np
import cv2

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