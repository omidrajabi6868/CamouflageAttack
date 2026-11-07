# CamouflageAttack

Research code for learning adversarial camouflage patterns against object detectors and instance-segmentation models in remote-sensing imagery. The current experiments focus on ship detection in the Airbus Ship Detection dataset and support Detectron2, YOLO, CNN, and transformer-based detector configurations.

## Highlights

- Learnable adversarial patches/camouflage patterns
- Shape-aware poisoning and patch placement
- Detectron2 detector support
- ViTDet, MViTv2, Swin, and YOLO experiment paths
- Single- and multi-GPU attack optimization
- Multiple attack objectives, including fixed/equal weighting, random sampling, GradNorm, and reinforcement-learning optimization
- Regularization terms such as total variation and non-printability score

## Repository structure

```text
TheMain.py         # Main attack CLI
Attack.py          # Attack optimization and loss aggregation
Poison.py          # Patch rendering and poisoning operations
Dataset.py         # Airbus dataset registration and loaders
DetectionNetwork.py# Detectron2 model construction
Train.py           # Detector training entry point
Network.py         # Patch networks and custom losses
PPO.py             # Reinforcement-learning components
UViT.py            # Transformer-based generator components
YOLO.py            # YOLO integration
yolo_configs/      # YOLO model configurations
run.txt             # Experiment command examples
```

## Requirements

The project requires a CUDA-compatible PyTorch environment for the intended workloads. Install PyTorch for your CUDA version, then install the remaining dependencies used by your selected detector:

```bash
pip install numpy opencv-python pandas matplotlib scikit-learn ultralytics
```

Detectron2 and its optional transformer projects must be installed from versions compatible with your PyTorch/CUDA environment. ViTDet, MViTv2, or Swin configurations may require the corresponding Detectron2 project modules.

## Dataset

Download the Airbus Ship Detection dataset and provide the image directory with `--img_dir`. Dataset registration, category filtering, and train/validation construction are handled in `Dataset.py`.

## Run an attack

The following example trains a shape-aware camouflage attack against a Detectron2 Mask R-CNN model:

```bash
python TheMain.py \
  --dataset_name airbus \
  --img_dir /path/to/airbus/train_v2 \
  --attack_name shapeAware \
  --poisoning_func shipCamou \
  --batch_size 16 \
  --epoch_num 500 \
  --optimizer Adam \
  --learning_rate 1e-4 \
  --output_dir /path/to/outputs \
  --model_path /path/to/model_final.pth \
  --model_config COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml \
  --attack_loss equally_weighted \
  --save_name ship_camouflage
```

For explicit multi-GPU execution, add device IDs:

```bash
--device_ids 0 1
```

See `run.txt` for additional detector and loss configurations.

## Train a detector

`Train.py` contains training paths for Detectron2 and YOLO models. `train.sh` and `airbus.sh` are SLURM examples and contain machine-specific paths that must be updated before use.

## Configuration notes

- `--multiclass` controls one-class versus COCO-style category handling.
- `--transformer_based` selects transformer detector construction.
- `--attack_loss` selects the optimization strategy implemented in `Attack.py`.
- Default paths in the CLI are environment-specific; pass explicit paths for portable runs.

## Citation

If this repository supports published work, add the final paper citation and DOI here so experiments can be attributed correctly.
