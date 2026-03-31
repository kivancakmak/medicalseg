import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from image_processor import CLASS_LABELS, DatasetSample, SurgicalImageProcessor
from segmentation_model import SegmentationModel
from train import discover_samples_in_single_directory, load_mask, parse_mask_suffixes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare classical OpenCV segmentation against a trained U-Net on a labeled dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        help="Single directory containing both images and masks.",
    )
    parser.add_argument(
        "--images-dir",
        default="data/cholecseg/images",
        help="Directory containing evaluation images.",
    )
    parser.add_argument(
        "--masks-dir",
        default="data/cholecseg/masks",
        help="Directory containing evaluation masks.",
    )
    parser.add_argument(
        "--mask-suffix",
        default="_color_mask",
        help="Preferred mask suffix for single-directory datasets.",
    )
    parser.add_argument(
        "--extra-mask-suffixes",
        default="_mask,_watershed_mask",
        help="Comma-separated fallback mask suffixes for single-directory datasets.",
    )
    parser.add_argument(
        "--weights",
        default="models/unet_best_weights.h5",
        help="Path to trained U-Net weights for the ML comparison.",
    )
    parser.add_argument("--width", type=int, default=256, help="Model input width.")
    parser.add_argument("--height", type=int, default=256, help="Model input height.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit how many labeled samples are evaluated. Use 0 for all.",
    )
    parser.add_argument(
        "--report-path",
        default="outputs/comparison_report.json",
        help="Where to save the aggregate metrics report.",
    )
    parser.add_argument(
        "--preview-dir",
        default="outputs/comparison_previews",
        help="Where to save side-by-side preview images.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=5,
        help="How many comparison preview images to save.",
    )
    return parser.parse_args()


def discover_samples(args) -> List[DatasetSample]:
    processor = SurgicalImageProcessor()
    if args.dataset_dir:
        mask_suffixes = parse_mask_suffixes(args.mask_suffix, args.extra_mask_suffixes)
        samples = discover_samples_in_single_directory(args.dataset_dir, mask_suffixes)
    else:
        samples = processor.discover_dataset(args.images_dir, args.masks_dir)
    return [sample for sample in samples if sample.mask_path is not None]


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    intersection = float(np.logical_and(y_true, y_pred).sum())
    union = float(np.logical_or(y_true, y_pred).sum())
    true_sum = float(y_true.sum())
    pred_sum = float(y_pred.sum())

    if true_sum == 0.0 and pred_sum == 0.0:
        return {"dice": 1.0, "iou": 1.0}

    dice = (2.0 * intersection + 1e-6) / (true_sum + pred_sum + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    return {"dice": dice, "iou": iou}


def compute_mask_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    metrics: Dict[str, object] = {
        "pixel_accuracy": float((y_true == y_pred).mean()),
        "per_class": {},
    }

    foreground_true = y_true > 0
    foreground_pred = y_pred > 0
    metrics.update(
        {
            f"foreground_{name}": value
            for name, value in compute_binary_metrics(foreground_true, foreground_pred).items()
        }
    )

    class_dice_values = []
    class_iou_values = []
    for class_id, class_name in CLASS_LABELS.items():
        if class_id == 0:
            continue
        per_class = compute_binary_metrics(y_true == class_id, y_pred == class_id)
        metrics["per_class"][class_name] = per_class
        class_dice_values.append(per_class["dice"])
        class_iou_values.append(per_class["iou"])

    metrics["mean_class_dice"] = float(np.mean(class_dice_values))
    metrics["mean_class_iou"] = float(np.mean(class_iou_values))
    return metrics


def append_metric(store: Dict[str, List[float]], prefix: str, metrics: Dict[str, object]) -> None:
    store.setdefault(f"{prefix}_pixel_accuracy", []).append(metrics["pixel_accuracy"])
    store.setdefault(f"{prefix}_foreground_dice", []).append(metrics["foreground_dice"])
    store.setdefault(f"{prefix}_foreground_iou", []).append(metrics["foreground_iou"])
    store.setdefault(f"{prefix}_mean_class_dice", []).append(metrics["mean_class_dice"])
    store.setdefault(f"{prefix}_mean_class_iou", []).append(metrics["mean_class_iou"])


def summarize_metric_store(store: Dict[str, List[float]]) -> Dict[str, float]:
    return {name: float(np.mean(values)) for name, values in store.items() if values}


def create_preview(
    processor: SurgicalImageProcessor,
    image: np.ndarray,
    ground_truth: np.ndarray,
    classical_mask: np.ndarray,
    ml_mask: np.ndarray,
) -> np.ndarray:
    gt_overlay = processor.overlay_segmentation(image, ground_truth)
    classical_overlay = processor.overlay_segmentation(image, classical_mask)
    ml_overlay = processor.overlay_segmentation(image, ml_mask)

    frames = [image.copy(), gt_overlay, classical_overlay, ml_overlay]
    titles = ["Input", "Ground Truth", "Classical", "U-Net"]

    for frame, title in zip(frames, titles):
        cv2.putText(
            frame,
            title,
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return np.hstack(frames)


def main():
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Trained weights not found: {args.weights}")

    processor = SurgicalImageProcessor()
    samples = discover_samples(args)
    if not samples:
        raise ValueError("No labeled samples found for evaluation.")

    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    classical_model = SegmentationModel(mode="classical", input_size=(args.width, args.height))
    ml_model = SegmentationModel(mode="ml", input_size=(args.width, args.height), weights_path=str(weights_path))
    if ml_model.mode != "ml":
        raise RuntimeError(f"Could not initialize the trained model. Details: {ml_model.note}")

    report_samples = []
    aggregate_store: Dict[str, List[float]] = {}

    preview_dir = Path(args.preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(samples):
        image = processor.load_image(str(sample.image_path))
        ground_truth = load_mask(sample.mask_path, target_size=(image.shape[1], image.shape[0]))

        classical_mask = classical_model.segment(image).mask
        ml_mask = ml_model.segment(image).mask

        classical_metrics = compute_mask_metrics(ground_truth, classical_mask)
        ml_metrics = compute_mask_metrics(ground_truth, ml_mask)

        append_metric(aggregate_store, "classical", classical_metrics)
        append_metric(aggregate_store, "ml", ml_metrics)

        report_samples.append(
            {
                "image": sample.image_path.name,
                "mask": sample.mask_path.name,
                "classical": classical_metrics,
                "ml": ml_metrics,
            }
        )

        if index < args.preview_count:
            preview = create_preview(processor, image, ground_truth, classical_mask, ml_mask)
            preview_path = preview_dir / f"{sample.image_path.stem}_comparison.png"
            processor.save_image(str(preview_path), preview)

    summary = summarize_metric_store(aggregate_store)
    report = {
        "num_samples": len(samples),
        "weights": str(weights_path),
        "summary": summary,
        "samples": report_samples,
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Saved comparison report to {report_path}")
    print(f"Saved preview images to {preview_dir}")
    print("Summary metrics:")
    for metric_name, value in summary.items():
        print(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
