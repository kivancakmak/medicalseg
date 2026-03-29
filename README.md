# Camera-Based Intelligent Surgical Light Prototype

This repository contains a software-first prototype for a simplified intelligent surgical light system. The current version focuses on:

- simulated surgical scenes for immediate testing
- classical OpenCV-based segmentation for a lightweight baseline
- an optional TensorFlow U-Net path for later CholecSeg training
- rule-based light adjustment logic with mock hardware output
- webcam and video processing hooks for real-time experiments

## Project Layout

- `main.py`: CLI entry point for synthetic demos, single-image runs, videos, and webcam input
- `image_processor.py`: image I/O, preprocessing, synthetic scene generation, and visualization helpers
- `segmentation_model.py`: classical segmentation plus optional U-Net model construction/training utilities
- `light_controller.py`: intensity decision logic and mock hardware adapter
- `requirements.txt`: Python dependencies
- `data/README.md`: expected dataset layout

## Quick Start

1. Create a virtual environment.
2. Install dependencies:

```powershell
py -3.7 -m venv .venv
.venv\Scripts\Activate.ps1
.venv\Scripts\pip.exe install -r requirements.txt
```

3. Run a synthetic demo:

```powershell
.venv\Scripts\python.exe main.py --mode synthetic --frames 5 --save-dir outputs
```

4. Run on a local image:

```powershell
.venv\Scripts\python.exe main.py --mode image --input path\to\image.png --save-dir outputs
```

5. Run with a webcam:

```powershell
.venv\Scripts\python.exe main.py --mode webcam --display --max-frames 300
```

## Training With Your Dataset

The training entry point is `train.py`.

Expected layout:

```text
data/
  cholecseg/
    images/
      frame_001.png
      frame_002.png
    masks/
      frame_001.png
      frame_002.png
```

Rules:

- each image and mask must share the same filename stem
- masks should be single-channel class-id images using `0=background`, `1=tissue`, `2=organ`, `3=tumor`
- grayscale masks using `0, 85, 170, 255` are also accepted
- color masks that match the prototype palette in the code are also accepted

Train with:

```powershell
.venv\Scripts\python.exe train.py --images-dir data\cholecseg\images --masks-dir data\cholecseg\masks --epochs 10 --batch-size 4 --output-weights models\unet_weights.h5
```

The saved training history now includes `accuracy`, `dice_coefficient`, and `mean_iou` for both training and validation.

If your images and masks are in the same folder with names like `frame_00_endo.png` and `frame_00_endo_color_mask.png`, train with:

```powershell
.venv\Scripts\python.exe train.py --dataset-dir data\cholecseg --mask-suffix _color_mask --epochs 10 --batch-size 4 --output-weights models\unet_weights.h5
```

If the same folder also contains alternatives like `frame_00_endo_mask.png` and `frame_00_endo_watershed_mask.png`, the trainer now prefers `_color_mask` first and then falls back to `_mask` and `_watershed_mask` automatically:

```powershell
.venv\Scripts\python.exe train.py --dataset-dir data\cholecseg --mask-suffix _color_mask --extra-mask-suffixes _mask,_watershed_mask --epochs 10 --batch-size 4 --output-weights models\unet_weights.h5
```

Then run inference with the trained weights:

```powershell
.venv\Scripts\python.exe main.py --mode image --model-mode ml --weights models\unet_weights.h5 --input "C:\path\to\image.png" --save-dir outputs
```

## CholecSeg Dataset Placement

Place dataset files under `data/cholecseg/` using a structure like:

```text
data/
  cholecseg/
    images/
    masks/
```

The current codebase is ready for that layout, but training scripts for full dataset ingestion and experiment tracking are intentionally kept lightweight for this prototype stage.

## Roadmap Alignment

- Weeks 1-2: environment setup, dataset prep, synthetic data, and classical segmentation
- Weeks 3-4: U-Net training path and light adjustment logic
- Weeks 5-6: simulation, testing, and webcam processing
- Weeks 7-10: Raspberry Pi camera + PWM/LED hardware integration

## Notes

- The prototype is designed to be useful even before machine learning weights are available.
- If TensorFlow or trained weights are missing, the app automatically falls back to classical segmentation.
- Python 3.7 is workable with the pinned dependencies below, but Python 3.9 or 3.10 is a better long-term target for future package support.
