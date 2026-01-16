from detectron2.config import LazyConfig, instantiate
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.engine import SimpleTrainer, hooks, launch, default_writers, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer


from Dataset import Dataset
from DetectionNetwork import DetectionNetwork

import logging
from detectron2.utils.logger import setup_logger
setup_logger()
logger = logging.getLogger("detectron2")
logger.setLevel(logging.DEBUG)  # or INFO, WARNING, etc.

import warnings
warnings.filterwarnings('ignore')

import argparse

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



def main(args):
    
    dataset_name = args.dataset_name
    img_dir = args.img_dir

    dataset = Dataset(dataset_name, category_id=args.category_id, random_id=args.random_id)
    train_df, valid_df = dataset.airbus_df()

    DatasetCatalog.register('train_data', lambda: dataset.airbus_dicts(df=train_df[:], img_dir=img_dir))
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
    
    if args.transformer_based:
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
    parser.add_argument("--epoch_num", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--category_id", type=int, default=0)
    parser.add_argument("--random_id", type=bool, default=False)
    parser.add_argument("--category_name", type=str, default='ship')
    parser.add_argument("--num_classes", type=int, default=1)

    parser.add_argument("--output_dir", type=str, default="/home/oraja001/airbus_ship/AdversarialProject/outputs/mask_rcnn_vitdet/")
    parser.add_argument("--model_path", type=str, default="/home/oraja001/airbus_ship/AdversarialProject/trained_models/mask_rcnn_vitdet/model_final_61ccd1.pkl")
    parser.add_argument("--model_config", type=str, default="/home/oraja001/airbus_ship/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py")
    parser.add_argument("--transformer_based", type=bool, default=True)

    args = parser.parse_args()

    main(args)