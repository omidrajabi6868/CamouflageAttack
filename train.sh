#!/bin/bash
#SBATCH --job-name=yolo_training
#SBATCH --error=outputs/yolo_training.txt
#SBATCH --output=outputs/yolo_training.txt
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --partition=high-gpu-mem
#SBATCH --gres=gpu:1
#SBATCH --time=6-12:00:00  

enable_lmod
module load container_env pytorch-gpu/2.2.0
crun python -u Train.py --dataset_name airbus --num_classes 1 --batch_size 32 --learning_rate 1e-3 --output_dir /home/oraja001/airbus_ship/AdversarialProject/trained_models/yolo/ --model_path /home/oraja001/airbus_ship/AdversarialProject/trained_models/yolo/yolov8l.pt --model_config /home/oraja001/airbus_ship/AdversarialProject/yolo_configs/yolov8_l.yaml --transformer_based "False" --yolo_training "True"