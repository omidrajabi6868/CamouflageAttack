import torch
import argparse
from Dataset import Dataset
from DetectionNetwork import DetectionNetwork
from Attack import Attack
import warnings
warnings.filterwarnings('ignore')

from detectron2.data import MetadataCatalog, DatasetCatalog

def main(args):

    dataset_name = args.dataset_name
    img_dir = args.img_dir

    attack_name = args.attack_name
    multiclass = args.multiclass

    epoch_num = args.epoch_num
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    category_id = args.category_id
    category_name = args.category_name
    random_id = args.random_id
    dataset = Dataset(dataset_name, category_id=category_id, random_id=random_id)

    train_data_len = 0
    val_data_len = 0
    if dataset.name == 'airbus':
        train_df, valid_df = dataset.airbus_df()
        DatasetCatalog.register('train_data', lambda: dataset.airbus_dicts(df=train_df[:20000], img_dir=img_dir))
        if multiclass:
            coco_classes = MetadataCatalog.get('coco_2017_train').thing_classes.copy()
            coco_classes[8] = category_name
            MetadataCatalog.get('train_data').set(thing_classes=coco_classes)
        else:
            MetadataCatalog.get('train_data').set(thing_classes=[category_name])

        train_data_len = len(DatasetCatalog.get('train_data'))
        train_metadata = MetadataCatalog.get("train_data")

        DatasetCatalog.register('val_data', lambda: dataset.airbus_dicts(df=valid_df[:], img_dir=img_dir))
        if multiclass:
            MetadataCatalog.get('val_data').set(thing_classes=coco_classes)
        else:
            MetadataCatalog.get('val_data').set(thing_classes=[category_name])
        
        val_data_len = len(DatasetCatalog.get('val_data'))
        val_metadata = MetadataCatalog.get("val_data")


    # COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x | COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x | COCO-Detection/faster_rcnn_R_50_C4_3x |
    # COCO-Detection/retinanet_R_50_FPN_1x| COCO-Detection/rpn_R_50_FPN_1x | COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x
    output_dir = args.output_dir
    model_path = args.model_path
    model_config = args.model_config
    transformer_based = args.transformer_based
    if multiclass:
        num_classes=80
    else:
        num_classes=1

    detection_network = DetectionNetwork(output_dir=output_dir,
                                         model_path=model_path,
                                         batch_size=batch_size,
                                         learning_rate=learning_rate,
                                         epoch_num=epoch_num,
                                         num_classes=num_classes,
                                         attack_name=attack_name)
    if transformer_based:
        detection_model, cfg = detection_network.transformer_based_net(model_config=model_config, 
                                                                       train_data_len=train_data_len)
    else:
        detection_model, cfg = detection_network.detectron_net(model_config=model_config, 
                                                               train_data_len=train_data_len)


    mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, 3, 1, 1)
    std = torch.tensor([57.3750, 57.1200, 58.3950]).view(1, 3, 1, 1)
    
    train_loader, val_loader = dataset.build_finite_loader(cfg)
    
    attack_loss = args.attack_loss
    save_name = args.save_name
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.empty(0))], lr=learning_rate)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.empty(0))], lr=learning_rate)
    elif args.optimizer == 'ASGD':
        optimizer = torch.optim.ASGD([torch.nn.Parameter(torch.empty(0))], lr=learning_rate)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop([torch.nn.Parameter(torch.empty(0))], lr=learning_rate)
    elif args.optimizer == 'Adamax':
        optimizer = torch.optim.Adamax([torch.nn.Parameter(torch.empty(0))], lr=learning_rate)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.empty(0))], lr=learning_rate)
        
    attack = Attack(name=attack_name,
                    poisoning_func=args.poisoning_func, 
                    train_loader=train_loader, 
                    val_loader=val_loader,
                    optimizer = optimizer,
                    epoch_num=epoch_num,
                    attack_loss=attack_loss,
                    save_name=save_name,
                    mean=mean, std=std)

    trained_patch = attack.conduct_attack(detection_model, detection_network)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_name", type=str, default="airbus")
    parser.add_argument("--img_dir", type=str, default="/home/oraja001/airbus_ship/airbus/train_v2")
    parser.add_argument("--attack_name", type=str, default="shapeAware")
    parser.add_argument("--poisoning_func", type=str, default="shapeAware")
    parser.add_argument("--multiclass", type=bool, default=False)
    parser.add_argument("--epoch_num", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--category_id", type=int, default=0)
    parser.add_argument("--random_id", type=bool, default=False)
    parser.add_argument("--category_name", type=str, default='ship')
    parser.add_argument("--num_classes", type=int, default=1)

    parser.add_argument("--output_dir", type=str, default="/home/oraja001/airbus_ship/AdversarialProject/outputs/mask_rcnn_R_101_FPN_3x/")
    parser.add_argument("--model_path", type=str, default="/home/oraja001/airbus_ship/AdversarialProject/trained_models/mask_rcnn_R_101_FPN_3x/model_final_ship.pth")
    parser.add_argument("--model_config", type=str, default="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    parser.add_argument("--transformer_based", type=bool, default=False)

    parser.add_argument("--attack_loss", type=str, default='fixed_weighted')
    parser.add_argument("--save_name", type=str, default='shapeAware')
    parser.add_argument("--optimizer", type=str, default='Adam')


    args = parser.parse_args()
    main(args)