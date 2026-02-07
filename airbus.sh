#!/bin/bash
#SBATCH --job-name=rl_optimization
#SBATCH --error=outputs/rl_optimization.txt
#SBATCH --output=outputs/rl_optimization.txt
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --partition=high-gpu-mem
#SBATCH --gres=gpu:1
#SBATCH --time=6-12:00:00  

usr/bin/true
enable_lmod
module load container_env pytorch-gpu/2.2.0
crun python TheMain.py --dataset_name airbus --attack_name shapeAware --batch_size 16 --optimizer Adam --learning_rate 1e-3 --output_dir /home/oraja001/airbus_ship/AdversarialProject/outputs/mask_rcnn_R_101_FPN_3x_one_class/ --model_path /home/oraja001/airbus_ship/AdversarialProject/trained_models/mask_rcnn_R_101_FPN_3x/model_final_ship.pth --model_config COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml --attack_loss rl_optimization --save_name rl_optimization
