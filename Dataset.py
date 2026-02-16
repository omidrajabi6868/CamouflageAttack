import numpy as np
import pandas as pd
import torch
import cv2
import detectron2.data.transforms as T
import random

from sklearn.model_selection import train_test_split
from utils import rle_decode, mask_to_bbox, mask_to_polygons
from Poison import Poison

from detectron2.structures import BoxMode, Instances, Boxes
from detectron2.config import get_cfg, instantiate
from detectron2.data import build_detection_train_loader, build_detection_test_loader, DatasetMapper, DatasetFromList, MapDataset
from detectron2.data import get_detection_dataset_dicts, detection_utils 
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

class CustomPoisonMapperCNN(DatasetMapper):
    def __init__(self, cfg, is_train=True, patch=None, percentage=0.6, poisoning_func=None):
        self.percentage = percentage
        self.poisoning_func = poisoning_func
        self.poison = Poison(prob=1.0)
        self.patch = patch
        self.mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, 3, 1, 1)
        super().__init__(cfg, is_train, augmentations=[])

    def __call__(self, dataset_dict):
        dataset_dict = super().__call__(dataset_dict)
        image = dataset_dict["image"]
        polygons = dataset_dict['instances'].gt_masks

        binary_masks = []
        for polygon in polygons:
            binary_mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)
            polygon = polygon[0].reshape((-1, 1, 2))
            binary_mask = cv2.fillPoly(binary_mask, [np.array(polygon, dtype=np.int32)], 1)
            binary_masks.append(binary_mask)
        
        self.mean = self.mean.to(image.device)
        self.patch = self.patch.to(image.device)
        image = (image - self.mean[0])
        if self.poisoning_func in ['google', 'shapeShifter']:
            adv_img = self.poison.google_poisoning(image, patch=self.patch, percentage=self.percentage, masks=binary_masks, training=False)
        elif self.poisoning_func == 'Dpatch':
            adv_img = self.poison.dpatch_poisoning(image, patch=self.patch, masks=binary_masks, training=False)
        elif self.poisoning_func == 'scaleAdaptive':
            adv_img = self.poison.scaleAdaptive_poisoning(image, patch=self.patch, alpha=self.percentage, masks=binary_masks, training=False)
        elif self.poisoning_func == 'shapeAware':
            adv_img = self.poison.shapeAware_poisoning(image, patch=self.patch, shape='ellipse', percentage=self.percentage, masks=binary_masks, training=False)
        elif self.poisoning_func == 'pieceWise':
            adv_img = self.poison.pieceWise_poisoning(image, patch=self.patch, shape='ellipse', percentage=self.percentage, masks=binary_masks, training=False)
        else:
            adv_img = image

        self.mean = self.mean.to(adv_img.device)
        adv_img = (adv_img + self.mean[0]).clamp(0, 255)
        adv_img = torch.tensor(adv_img, dtype=torch.uint8)
    
        dataset_dict["image"] = adv_img

        return dataset_dict

class CustomPoisonMapperTransformer:
    def __init__(self, cfg, is_train=True, patch=None, percentage=0.6, poisoning_func=None):
        self.percentage = percentage
        self.poisoning_func = poisoning_func
        self.poison = Poison(prob=1.0)
        self.patch = patch
        self.aug = []
        self.cfg = cfg
        self.cfg.dataloader.test.mapper.is_train = is_train  # force annotation loading
        self.cfg.dataloader.test.mapper.use_instance_mask = is_train
        self.cfg.dataloader.test.mapper.recompute_boxes = is_train  # useful for masks->boxes
        self.cfg.dataloader.test.num_workers = 0
        val_loader = instantiate(self.cfg.dataloader.test)
        self.default_mapper = val_loader.dataset._map_func  # the default DatasetMapper
        self.mean = torch.tensor(cfg.model.pixel_mean).view(1, 3, 1, 1)

    def __call__(self, dataset_dict):
        dataset_dict = self.default_mapper(dataset_dict)
        image = dataset_dict["image"]
        polygons = dataset_dict['instances'].gt_masks

        binary_masks = []
        for polygon in polygons:
            binary_mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)
            polygon = polygon[0].reshape((-1, 1, 2))
            binary_mask = cv2.fillPoly(binary_mask, [np.array(polygon, dtype=np.int32)], 1)
            binary_masks.append(binary_mask)
        
        self.mean = self.mean.to(image.device)
        self.patch = self.patch.to(image.device)
        image = (image - self.mean[0])
        if self.poisoning_func in ['google', 'shapeShifter']:
            adv_img = self.poison.google_poisoning(image, patch=self.patch, percentage=self.percentage, masks=binary_masks, training=False)
        elif self.poisoning_func == 'Dpatch':
            adv_img = self.poison.dpatch_poisoning(image, patch=self.patch, masks=binary_masks, training=False)
        elif self.poisoning_func == 'scaleAdaptive':
            adv_img = self.poison.scaleAdaptive_poisoning(image, patch=self.patch, alpha=self.percentage, masks=binary_masks, training=False)
        elif self.poisoning_func == 'shapeAware':
            adv_img = self.poison.shapeAware_poisoning(image, patch=self.patch, shape='ellipse', percentage=self.percentage, masks=binary_masks, training=False)
        elif self.poisoning_func == 'pieceWise':
            adv_img = self.poison.pieceWise_poisoning(image, patch=self.patch, shape='ellipse', percentage=self.percentage, masks=binary_masks, training=False)
        else:
            adv_img = image

        self.mean = self.mean.to(adv_img.device)
        adv_img = (adv_img + self.mean[0]).clamp(0, 255)
        adv_img = torch.tensor(adv_img, dtype=torch.uint8)
    
        dataset_dict["image"] = adv_img

        return dataset_dict

class CustomPoisonMapperYOLO:
    def __init__(self, cfg=None, is_train=True, patch=None, percentage=0.6, poisoning_func=None):
        self.percentage = percentage
        self.poisoning_func = poisoning_func
        self.poison = Poison(prob=1.0)
        self.patch = patch
        self.is_train = is_train
        pixel_mean = [103.53, 116.28, 123.675]
        self.mean = torch.tensor(pixel_mean, dtype=torch.float32).view(1, 3, 1, 1)

    def _build_binary_masks(self, dataset_dict, h, w):
        binary_masks = []

        for anno in dataset_dict.get('annotations', []):
            if anno.get('iscrowd', 0):
                continue

            mask = np.zeros((h, w), dtype=np.uint8)
            segs = anno.get('segmentation', [])

            if segs:
                for seg in segs:
                    poly = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
                    if poly.shape[0] < 3:
                        continue
                    cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
            else:
                x1, y1, x2, y2 = np.asarray(anno['bbox'], dtype=np.float32)
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 1

            if mask.any():
                binary_masks.append(mask)

        return binary_masks

    def __call__(self, dataset_dict):
        dataset_dict = dataset_dict.copy()
        image = detection_utils.read_image(dataset_dict['file_name'], format='BGR')
        h, w = image.shape[:2]

        image = torch.as_tensor(image.copy().transpose(2, 0, 1), dtype=torch.float32)
        binary_masks = self._build_binary_masks(dataset_dict, h, w)

        self.mean = self.mean.to(image.device)
        image = image - self.mean[0]

        if self.patch is not None:
            self.patch = self.patch.to(image.device)

        if self.poisoning_func in ['google', 'shapeShifter']:
            adv_img = self.poison.google_poisoning(image, patch=self.patch, percentage=self.percentage, masks=binary_masks, training=False)
        elif self.poisoning_func == 'Dpatch':
            adv_img = self.poison.dpatch_poisoning(image, patch=self.patch, masks=binary_masks, training=False)
        elif self.poisoning_func == 'scaleAdaptive':
            adv_img = self.poison.scaleAdaptive_poisoning(image, patch=self.patch, alpha=self.percentage, masks=binary_masks, training=False)
        elif self.poisoning_func == 'shapeAware':
            adv_img = self.poison.shapeAware_poisoning(image, patch=self.patch, shape='ellipse', percentage=self.percentage, masks=binary_masks, training=False)
        elif self.poisoning_func == 'pieceWise':
            adv_img = self.poison.pieceWise_poisoning(image, patch=self.patch, shape='ellipse', percentage=self.percentage, masks=binary_masks, training=False)
        else:
            adv_img = image

        adv_img = (adv_img + self.mean[0]).clamp(0, 255).to(torch.float32)

        boxes = []
        classes = []
        for anno in dataset_dict.get('annotations', []):
            if anno.get('iscrowd', 0):
                continue
            boxes.append(anno['bbox'])
            classes.append(anno['category_id'])

        instances = Instances((h, w))
        if boxes:
            box_tensor = torch.tensor(boxes, dtype=torch.float32)
        else:
            box_tensor = torch.zeros((0, 4), dtype=torch.float32)
        instances.gt_boxes = Boxes(box_tensor)
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

        dataset_dict['image'] = adv_img
        dataset_dict['instances'] = instances
        dataset_dict['height'] = h
        dataset_dict['width'] = w
        return dataset_dict
        
class Dataset:
    def __init__(self, name, category_id, random_id=False):
        self.name = name
        self.category_id = category_id
        self.random_id = random_id

    def airbus_df(self):
        masks = pd.read_csv('/home/oraja001/airbus_ship/train_ship_segmentations_v2.csv')
        print('Total number of images (original): %d' % masks['ImageId'].value_counts().shape[0])

        masks = masks[~masks['ImageId'].isin(['6384c3e78.jpg'])]  # remove corrupted file
        unique_img_ids = masks.groupby('ImageId').size().reset_index(name='counts')
        print('Total number of images (after removing corrupted images): %d' % masks['ImageId'].value_counts().shape[0])

        # Count number of ships per image
        df_wships = masks.dropna()
        df_wships = df_wships.groupby('ImageId').size().reset_index(name='counts')
        df_woships = masks[masks['EncodedPixels'].isna()]

        print('Number of images with ships : %d | Number of images without ships : %d  (x%0.1f)' \
          % (df_wships.shape[0], df_woships.shape[0], df_woships.shape[0] / df_wships.shape[0]))

        masks = masks.dropna()
        df_woships = masks[masks['EncodedPixels'].isna()]

        df_w15ships = df_wships.loc[df_wships['counts'] == 15]
        list_w15ships = df_w15ships.values.tolist()

        # Split dataset into training and validation sets
        # statritify : same histograms of numbe of ships
        masks = masks[~masks['EncodedPixels'].isna()]
        masks = masks[masks['EncodedPixels'].apply(lambda x: len(x.split(' ')) > 20)]
        unique_img_ids = masks[~masks['EncodedPixels'].isna()].groupby('ImageId').size().reset_index(name='counts')
        unique_img_ids = unique_img_ids[unique_img_ids['counts'] >= 1]

        train_ids, val_ids = train_test_split(unique_img_ids, test_size=0.1,
                                            random_state=42)
        train_df = pd.merge(masks, train_ids)
        valid_df = pd.merge(masks, val_ids)

        train_df['counts'] = train_df.apply(lambda c_row: c_row['counts'] if isinstance(c_row['EncodedPixels'], str) else 0, 1)
        valid_df['counts'] = valid_df.apply(lambda c_row: c_row['counts'] if isinstance(c_row['EncodedPixels'], str) else 0, 1)

        return train_df, valid_df

    def airbus_dicts(self, df, img_dir):
        dataset_dicts = []
        ImageIds = []
        for item in df.iterrows():
            if self.random_id:
                self.category_id = random.choice([i for i in range(80) if i != 8])

            record = {}
            filename = item[1]['ImageId']
            if filename in ImageIds:
                continue
            ImageIds.append(filename)
            img = cv2.imread(f'{img_dir}/{filename}')
            height, width = img.shape[:2]
            record['file_name'] = f'{img_dir}/{filename}'
            record['image_id'] = len(dataset_dicts)
            record['height'] = height
            record['width'] = width

            # Get binary mask
            masks_val = df.loc[df['ImageId'] == str(filename), 'EncodedPixels'].tolist()

            objs = []
            for rle_mask in masks_val:
                obj = {}
                mask = rle_decode(rle_mask)
                bbox = mask_to_bbox(mask)
                obj['bbox'] = bbox
                obj['bbox_mode'] = BoxMode.XYXY_ABS
                obj["category_id"] = self.category_id

                polygons = mask_to_polygons(mask)
                obj['segmentation'] = polygons
                objs.append(obj)
            record['annotations'] = objs

            dataset_dicts.append(record)

        return dataset_dicts

    def airbus_df_loaders(self, train_df, valid_df, BATCH_SZ_TRAIN, BATCH_SZ_VALID):
    
        # Data augmentation
        train_transform = DualCompose([HorizontalFlip(), VerticalFlip()])
        val_transform = DualCompose([])
        val_poison_transform = DualCompose([Val_Poison(parameters_name), CenterCrop((512, 512, 3))])

        # Initialize dataset
        train_dataset = AirbusDataset(train_df, transform=train_transform, mode='train')
        val_dataset = AirbusDataset(valid_df, transform=val_transform, mode='validation')
        poison_dataset = AirbusDataset(valid_df, transform=val_poison_transform, mode='validation')

        print('Train samples : %d | Validation samples : %d' % (len(train_dataset), len(val_dataset)))

        # Get loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SZ_TRAIN, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SZ_VALID, shuffle=False, num_workers=0)
        poison_loader = torch.utils.data.DataLoader(poison_dataset, batch_size=BATCH_SZ_VALID, shuffle=True, num_workers=0)

        return train_loader, val_loader, poison_loader

    def build_finite_loader(self, cfg):

        train_dataset_dicts = get_detection_dataset_dicts([cfg.DATASETS.TRAIN[0]], filter_empty=True)
        val_dataset_dicts = get_detection_dataset_dicts([cfg.DATASETS.TEST[0]], filter_empty=True)

        # Define augmentations
        augmentations = [
            T.RandomBrightness(0.8, 1.2),
            T.RandomContrast(0.8, 1.2),
            T.RandomSaturation(0.8, 1.2),
            T.RandomFlip(horizontal=True, vertical=False),
        ]

        # Mapper for augmentation/preprocessing
        train_mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentations)
        val_mapper = DatasetMapper(cfg, is_train=True, augmentations=[])

        train_dataset = DatasetFromList(train_dataset_dicts, copy=False)
        train_dataset = MapDataset(train_dataset, train_mapper)

        val_dataset = DatasetFromList(val_dataset_dicts, copy=False)
        val_dataset = MapDataset(val_dataset, val_mapper)

        # Choose finite sampler
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=train_sampler,
            collate_fn=lambda x: x  # detectron2 expects list of dicts
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=val_sampler,
            collate_fn=lambda x: x  # detectron2 expects list of dicts
        )
        
        return train_loader, val_loader

    def dota_dicts(self, img_dir, imgids):
        dataset_dicts = []
        ImageIds = []
        for imgid in imgids:
            record ={}
            filename = imgid
            if filename in ImageIds:
                continue
            ImageIds.append(filename)
            img = imread(f'{img_dir}/{filename}.png')
            height, width = img.shape[:2]
            record['file_name'] = f'{img_dir}/{filename}.png'
            record['image_id'] = len(dataset_dicts)
            record['height'] = height
            record['width'] = width

            if 'train' in basepath:
                anns = examplesplit_train.loadAnns(imgId=imgid)
            else:
                anns = examplesplit_valid.loadAnns(imgId=imgid)
            objs = []
            for ann in anns:
                obj={}
                if ann['name'] != category:
                    continue
                poly = TuplePoly2Poly(ann['poly'])
                xmin, ymin, xmax, ymax = min(poly[0::2]), min(poly[1::2]), max(poly[0::2]), max(poly[1::2])
                width, height = xmax - xmin, ymax - ymin
                obj['bbox'] = xmin, ymin, xmax, ymax
                obj['bbox_mode'] = BoxMode.XYXY_ABS
                obj["category_id"] = self.category_id
                obj['segmentation'] = [poly]
                objs.append(obj)
            record['annotations'] = objs

            # Collect and draw all polygons
            polygons = []
            for ann in record['annotations']:
                segmentation = ann['segmentation']  # Flat list of 8 floats (4 points)
                poly = np.array(segmentation, dtype=np.int32).reshape((-1, 2))     # (4, 2)
                poly = poly.reshape((-1, 1, 2))  # (4, 1, 2) required by OpenCV
                polygons.append(poly)

            record['polygons'] = polygons

            dataset_dicts.append(record)
        
        return dataset_dicts
    
    def visdrone_dicts(self, img_paths):

        dataset_dicts = []
        ImageIds = []
        for img_path in img_paths:
            record ={}
            filename = img_path.split('/')[-1][:-4]
            if filename in ImageIds:
                continue
            ImageIds.append(filename)
            img = cv2.imread(img_path)
            height, width = img.shape[:2]
            record['file_name'] = img_path
            record['image_id'] = len(dataset_dicts)
            record['height'] = height
            record['width'] = width

            with open('/'.join((img_path.split('/')[:-2] + ['annotations', f'{filename}.txt'])), 'r') as f:
                anns = [line.strip() for line in f if line.strip()]

            objs = []
            for ann in anns:
                obj={}
                ann = ann.split(',')
                x, y, bw, bh, cls = map(float, ann[:4] + [ann[5]])
                cls = int(cls)
                if cls != 4:
                    continue
                # poly = TuplePoly2Poly(ann['poly'])
                # xmin, ymin, xmax, ymax = min(poly[0::2]), min(poly[1::2]), max(poly[0::2]), max(poly[1::2])
                # width, height = xmax - xmin, ymax - ymin
                seg, xyxy, mode = bbox_to_polygon([x, y, bw, bh], BoxMode.XYWH_ABS)
                obj['bbox'] = xyxy
                obj['bbox_mode'] = mode
                obj["category_id"] = self.category_id
                obj['segmentation'] = [seg]
                objs.append(obj)
            record['annotations'] = objs

            # Collect and draw all polygons
            polygons = []
            for ann in record['annotations']:
                segmentation = ann['segmentation']  # Flat list of 8 floats (4 points)
                poly = np.array(segmentation, dtype=np.int32).reshape((-1, 2))     # (4, 2)
                poly = poly.reshape((-1, 1, 2))  # (4, 1, 2) required by OpenCV
                polygons.append(poly)

            record['polygons'] = polygons

            dataset_dicts.append(record)
        
        return dataset_dicts

    def yolo_data_loaders(self, cfg, split='both'):
        train_loader = None
        val_loader = None
        if split == 'both'or split == 'train':
            train_dataset_dicts = get_detection_dataset_dicts(['train_data'], filter_empty=True)
            train_loader = build_detection_train_loader(
                dataset=train_dataset_dicts,
                mapper=self.yolo_mapper,
                total_batch_size=cfg['batch_size'])

        if split == 'both' or split == 'val':
            val_dataset_dicts = get_detection_dataset_dicts(['val_data'], filter_empty=True)
            val_loader = build_detection_train_loader(
                dataset=val_dataset_dicts,
                mapper=self.yolo_mapper,
                total_batch_size=cfg['batch_size'])

        return train_loader, val_loader

    def yolo_poison_data_loaders(self, cfg, patch, percentage=0.6, poisoning_func=None, split='val'):
        train_loader = None
        val_loader = None

        mapper = CustomPoisonMapperYOLO(cfg=cfg, is_train=False, patch=patch, percentage=percentage, poisoning_func=poisoning_func)

        if split == 'both' or split == 'train':
            train_dataset_dicts = get_detection_dataset_dicts(['train_data'], filter_empty=True)
            train_loader = build_detection_train_loader(
                dataset=train_dataset_dicts,
                mapper=mapper,
                total_batch_size=cfg['batch_size'])

        if split == 'both' or split == 'val':
            val_dataset_dicts = get_detection_dataset_dicts(['val_data'], filter_empty=True)
            val_loader = build_detection_train_loader(
                dataset=val_dataset_dicts,
                mapper=mapper,
                total_batch_size=cfg['batch_size'])

        return train_loader, val_loader

    def yolo_mapper(self, dataset_dict):
        dataset_dict = dataset_dict.copy()

        # --- Load image ---
        image = detection_utils.read_image(dataset_dict["file_name"], format="BGR")
        h, w = image.shape[:2]

        image = torch.as_tensor(image.copy().transpose(2, 0, 1), dtype=torch.float32)

        # --- Load annotations ---
        boxes = []
        classes = []

        for anno in dataset_dict.get("annotations", []):
            if anno.get("iscrowd", 0):
                continue
            boxes.append(anno["bbox"])         
            classes.append(anno["category_id"])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.int64)

        instances = Instances((h, w))
        instances.gt_boxes = Boxes(boxes)
        instances.gt_classes = classes

        # --- Required by Detectron2 ---
        dataset_dict["image"] = image
        dataset_dict["instances"] = instances
        dataset_dict["height"] = h
        dataset_dict["width"] = w

        return dataset_dict