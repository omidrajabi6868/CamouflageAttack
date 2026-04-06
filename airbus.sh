#!/bin/bash
#SBATCH --job-name=grad_norm
#SBATCH --error=outputs/grad_norm.txt
#SBATCH --output=outputs/grad_norm.txt
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --partition=high-gpu-mem
#SBATCH --gres=gpu:1
#SBATCH --time=6-12:00:00  

usr/bin/true
enable_lmod
module load container_env pytorch-gpu/2.2.0
crun python TheMain.py \
    --dataset_name airbus \
    --attack_name shapeAware \
    --batch_size 16 \
    --optimizer Adam \
    --learning_rate 1e-4 \
    --output_dir /home/oraja001/airbus_ship/AdversarialProject/outputs/mask_rcnn_R_101_FPN_3x_one_class/ \
    --model_path /home/oraja001/airbus_ship/AdversarialProject/trained_models/mask_rcnn_R_101_FPN_3x/model_final_ship.pth \
    --model_config COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml \
    --attack_loss grad_norm \
    --save_name grad_norm_ship