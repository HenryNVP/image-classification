
# Image Classification with RegNet

This project trains a RegNetY-016 model (via [timm](https://github.com/huggingface/pytorch-image-models)) on the Oxford-IIIT Pet dataset. The repository includes scripts for downloading the data, preprocessing into flattened splits with CSV annotations, training/validation pipelines, and a demo subset.

## Project Structure

```
configs/              # YAML configs for model, training, data, and augmentation
scripts/              # Utility scripts (download data, preprocess, demo subset)
src/                  # Training code: data loaders, engine, models, utils
train.py              # CLI entry point for training
validate.py           # CLI entry point for evaluation
demo/                 # Optional mini dataset (created via scripts/create_demo_dataset.py)
notebooks/            # Jupyter notebooks
```

## Setup

1. Install dependencies (PyTorch build of your choice, plus utility packages):

```bash
pip install -r requirements.txt
```

2. Download the dataset and prepare splits:

```bash
python scripts/get_data.py
python scripts/split_dataset.py
```

3. (Optional) Create a tiny demo split (first two classes, 20/5/5 samples):

```bash
python scripts/create_demo_dataset.py
```

## Training

Run training with defaults (RegNetY-016, full dataset):

```bash
python train.py
```

For Vision Transformer models that require 224x224 inputs:

```bash
python train.py --model-config configs/model/vit_base_patch16_224.yaml --aug-config configs/aug_vit.yaml
```

Key options:

- `--model-config`, `--data-config`, `--train-config`, `--aug-config`: override YAML paths
- `--epochs`, `--batch-size`, `--device`: override hyperparameters from train config
- `--resume`: path to `last.pt` or another checkpoint to resume
- `--amp`: `on`/`off`/`auto` for mixed precision (default `auto`)
- `--plot`: save `training_curves.png` alongside `history.json`


Early stopping defaults are defined in `configs/train.yaml` (`early_stopping.patience`, `min_delta`). Set `patience` to `null` (or remove the block) to disable it.

Checkpoints are stored under `checkpoints/` (customizable via config or `--output-dir`).

## Validation

Evaluate a checkpoint on a split (val/test/demo):

```bash
python validate.py --checkpoint checkpoints/regnety_016/best.pth --split val
```

You can point `--data-config` to a custom YAML (for example, one targeting the `demo/` dataset) to evaluate on different splits.

## Exporting

Convert a trained model to TorchScript and ONNX (TorchScript embeds a center crop; ONNX expects pre-cropped inputs):

```bash
python scripts/export_models.py --checkpoint checkpoints/regnety_016/best.pth \
    --output-dir exports/regnety_016
```

For Vision Transformer models (requires ONNX opset ≥14, default is 17):

```bash
python scripts/export_models.py \
    --model-config configs/model/vit_small_patch16_224.yaml \
    --aug-config configs/aug_vit.yaml \
    --checkpoint checkpoints/vit_small_patch16_224/best.pth \
    --output-dir exports/vit_small_patch16_224 \
    --onnx-opset 17
```

**Note:** The default ONNX opset version is 17. ViT models require opset ≥14 due to `scaled_dot_product_attention`. Use `--onnx-opset` to specify a different version if needed.

TorchScript exports include an in-graph center crop (size inferred from `configs/aug.yaml`), while the ONNX export expects inputs already cropped to that size.

Benchmark raw PyTorch, TorchScript, and ONNX runtime performance (FPS, latency):

```bash
python scripts/benchmark_models.py --checkpoint checkpoints/regnety_016/best.pth \
    --torchscript exports/regnety_016/model_scripted.pt \
    --onnx exports/regnety_016/model.onnx --device cuda
```

## Notebook

`notebooks/image_classification.ipynb` demonstrates the end-to-end workflow:

- Environment setup & dependency installation
- Dataset download and preprocessing
- Launching training via the CLI
- Evaluating and inspecting predictions

Open it in Jupyter or VS Code for an interactive walkthrough.
