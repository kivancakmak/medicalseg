import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from image_processor import CLASS_COLORS, CLASS_LABELS, DatasetSample, SurgicalImageProcessor
from segmentation_model import SegmentationModel


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare classical OpenCV segmentation against a trained U-Net."
    )
    parser.add_argument(
        "--dataset-dir",
        help="Single folder containing images and masks together.",
    )
    parser.add_argument(
        "--images-dir",
        default="data/cholecseg/images",
        help="Folder containing original images when masks are stored separately.",
    )
    parser.add_argument(
        "--masks-dir",
        default="data/cholecseg/masks",
        help="Folder containing ground-truth masks when masks are stored separately.",
    )
    parser.add_argument(
        "--mask-suffix",
        default="_color_mask",
        help="Preferred suffix for masks in a single-folder dataset.",
    )
    parser.add_argument(
        "--extra-mask-suffixes",
        default="_mask,_watershed_mask",
        help="Comma-separated fallback mask suffixes.",
    )
    parser.add_argument(
        "--weights",
        default="models/weights.h5",
        help="Path to trained U-Net Keras weights.",
    )
    parser.add_argument("--width", type=int, default=256, help="U-Net input width.")
    parser.add_argument("--height", type=int, default=256, help="U-Net input height.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of samples to evaluate. Use 0 for all samples.",
    )
    parser.add_argument(
        "--report-json",
        default="outputs/evaluation_report.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--report-csv",
        default="outputs/evaluation_report.csv",
        help="Output per-image CSV report path.",
    )
    parser.add_argument(
        "--preview-dir",
        default="outputs/evaluation_previews",
        help="Folder for side-by-side preview images.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=8,
        help="Number of preview images to save.",
    )
    return parser.parse_args()


def normalize_suffixes(primary_suffix: str, extra_suffixes: str) -> List[str]:
    suffixes = []
    raw_suffixes = [primary_suffix]
    raw_suffixes.extend(
        suffix.strip() for suffix in extra_suffixes.split(",") if suffix.strip()
    )

    for suffix in raw_suffixes:
        if not suffix.startswith("_"):
            suffix = f"_{suffix}"
        if suffix not in suffixes:
            suffixes.append(suffix)
    return suffixes


def discover_single_folder_samples(dataset_dir: str, mask_suffixes: Iterable[str]) -> List[DatasetSample]:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    mask_suffixes = list(mask_suffixes)
    image_files = sorted(
        path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    file_lookup = {path.name.lower(): path for path in image_files}
    samples: List[DatasetSample] = []

    for image_path in image_files:
        stem = image_path.stem
        if any(stem.endswith(suffix) for suffix in mask_suffixes):
            continue

        mask_path = find_mask_for_stem(stem, file_lookup, mask_suffixes)
        if mask_path is not None:
            samples.append(DatasetSample(image_path=image_path, mask_path=mask_path))

    return samples


def find_mask_for_stem(
    stem: str,
    file_lookup: Dict[str, Path],
    mask_suffixes: Iterable[str],
) -> Optional[Path]:
    for suffix in mask_suffixes:
        for extension in IMAGE_EXTENSIONS:
            candidate = f"{stem}{suffix}{extension}".lower()
            if candidate in file_lookup:
                return file_lookup[candidate]
    return None


def discover_separate_folder_samples(images_dir: str, masks_dir: str) -> List[DatasetSample]:
    processor = SurgicalImageProcessor()
    samples = processor.discover_dataset(images_dir, masks_dir, IMAGE_EXTENSIONS)
    return [sample for sample in samples if sample.mask_path is not None]


def load_ground_truth_mask(mask_path: Path, target_size: Tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Unable to load mask: {mask_path}")

    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    if mask.ndim == 2:
        return normalize_grayscale_mask(mask)

    return color_mask_to_class_ids(mask)


def normalize_grayscale_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)
    unique_values = set(np.unique(mask).tolist())

    if unique_values.issubset(set(CLASS_LABELS.keys())):
        return mask

    if unique_values.issubset({0, 85, 170, 255}):
        converted = np.zeros_like(mask, dtype=np.uint8)
        for source_value, class_id in {0: 0, 85: 1, 170: 2, 255: 3}.items():
            converted[mask == source_value] = class_id
        return converted

    raise ValueError(
        "Unsupported grayscale mask values. Expected 0-3 or 0,85,170,255. "
        f"Found sample values: {sorted(unique_values)[:12]}"
    )


def color_mask_to_class_ids(mask: np.ndarray) -> np.ndarray:
    if mask.shape[2] == 4:
        mask = mask[:, :, :3]

    pixels = mask.astype(np.float32)
    class_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    best_distance = np.full(mask.shape[:2], np.inf, dtype=np.float32)

    for class_id, color in CLASS_COLORS.items():
        prototype = np.array(color, dtype=np.float32)
        distance = np.linalg.norm(pixels - prototype, axis=2)
        update = distance < best_distance
        class_mask[update] = class_id
        best_distance[update] = distance[update]

    return class_mask


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = float(np.logical_and(y_true, y_pred).sum())
    tn = float(np.logical_and(~y_true, ~y_pred).sum())
    fp = float(np.logical_and(~y_true, y_pred).sum())
    fn = float(np.logical_and(y_true, ~y_pred).sum())

    total = tp + tn + fp + fn
    union = tp + fp + fn
    pred_sum = tp + fp
    true_sum = tp + fn

    if pred_sum == 0.0 and true_sum == 0.0:
        dice = 1.0
        iou = 1.0
    else:
        dice = (2.0 * tp + 1e-6) / (pred_sum + true_sum + 1e-6)
        iou = (tp + 1e-6) / (union + 1e-6)

    return {
        "pixel_accuracy": (tp + tn) / total if total else 0.0,
        "dice": dice,
        "iou": iou,
        "support_pixels": true_sum,
    }


def segmentation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    per_class = {}
    dice_values = []
    iou_values = []

    for class_id, class_name in CLASS_LABELS.items():
        metrics = binary_metrics(y_true == class_id, y_pred == class_id)
        per_class[class_name] = metrics
        dice_values.append(metrics["dice"])
        iou_values.append(metrics["iou"])

    foreground = binary_metrics(y_true > 0, y_pred > 0)

    return {
        "pixel_accuracy": float((y_true == y_pred).mean()),
        "mean_dice": float(np.mean(dice_values)),
        "mean_iou": float(np.mean(iou_values)),
        "foreground_dice": foreground["dice"],
        "foreground_iou": foreground["iou"],
        "per_class": per_class,
    }


def flatten_sample_metrics(image_name: str, method_name: str, metrics: Dict[str, object]) -> Dict[str, object]:
    row = {
        "image": image_name,
        "method": method_name,
        "pixel_accuracy": metrics["pixel_accuracy"],
        "mean_dice": metrics["mean_dice"],
        "mean_iou": metrics["mean_iou"],
        "foreground_dice": metrics["foreground_dice"],
        "foreground_iou": metrics["foreground_iou"],
    }

    for class_name, class_metrics in metrics["per_class"].items():
        row[f"{class_name}_pixel_accuracy"] = class_metrics["pixel_accuracy"]
        row[f"{class_name}_dice"] = class_metrics["dice"]
        row[f"{class_name}_iou"] = class_metrics["iou"]
        row[f"{class_name}_support_pixels"] = class_metrics["support_pixels"]

    return row


def summarize(rows: List[Dict[str, object]], method_name: str) -> Dict[str, float]:
    method_rows = [row for row in rows if row["method"] == method_name]
    if not method_rows:
        return {}

    summary = {}
    metric_names = [key for key in method_rows[0].keys() if key not in {"image", "method"}]
    for metric_name in metric_names:
        values = [float(row[metric_name]) for row in method_rows]
        summary[metric_name] = float(np.mean(values))
    return summary


def save_preview(
    processor: SurgicalImageProcessor,
    preview_path: Path,
    image: np.ndarray,
    ground_truth: np.ndarray,
    classical_mask: np.ndarray,
    unet_mask: np.ndarray,
) -> None:
    panels = [
        image.copy(),
        processor.overlay_segmentation(image, ground_truth),
        processor.overlay_segmentation(image, classical_mask),
        processor.overlay_segmentation(image, unet_mask),
    ]
    labels = ["Input", "Ground Truth", "OpenCV", "U-Net"]

    for panel, label in zip(panels, labels):
        cv2.putText(
            panel,
            label,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    processor.save_image(str(preview_path), np.hstack(panels))


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Trained U-Net weights not found: {weights_path}. "
            "Use --weights to point at your .h5 file."
        )

    if args.dataset_dir:
        mask_suffixes = normalize_suffixes(args.mask_suffix, args.extra_mask_suffixes)
        samples = discover_single_folder_samples(args.dataset_dir, mask_suffixes)
    else:
        samples = discover_separate_folder_samples(args.images_dir, args.masks_dir)

    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    if not samples:
        raise ValueError(
            "No labeled samples found. Check --dataset-dir or --images-dir/--masks-dir "
            "and confirm image/mask names match."
        )

    processor = SurgicalImageProcessor()
    classical_segmenter = SegmentationModel(mode="classical", input_size=(args.width, args.height))
    unet_segmenter = SegmentationModel(
        mode="ml",
        input_size=(args.width, args.height),
        weights_path=str(weights_path),
    )
    if unet_segmenter.mode != "ml":
        raise RuntimeError(f"U-Net weights could not be loaded. Details: {unet_segmenter.note}")

    preview_dir = Path(args.preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    sample_reports = []

    for index, sample in enumerate(samples):
        image = processor.load_image(str(sample.image_path))
        ground_truth = load_ground_truth_mask(
            sample.mask_path,
            target_size=(image.shape[1], image.shape[0]),
        )

        classical_mask = classical_segmenter.segment(image).mask
        unet_mask = unet_segmenter.segment(image).mask

        classical_metrics = segmentation_metrics(ground_truth, classical_mask)
        unet_metrics = segmentation_metrics(ground_truth, unet_mask)

        rows.append(flatten_sample_metrics(sample.image_path.name, "classical_opencv", classical_metrics))
        rows.append(flatten_sample_metrics(sample.image_path.name, "unet", unet_metrics))
        sample_reports.append(
            {
                "image": sample.image_path.name,
                "mask": sample.mask_path.name,
                "classical_opencv": classical_metrics,
                "unet": unet_metrics,
            }
        )

        if index < args.preview_count:
            preview_path = preview_dir / f"{sample.image_path.stem}_comparison.png"
            save_preview(processor, preview_path, image, ground_truth, classical_mask, unet_mask)

    report = {
        "num_samples": len(samples),
        "weights": str(weights_path),
        "class_labels": CLASS_LABELS,
        "summary": {
            "classical_opencv": summarize(rows, "classical_opencv"),
            "unet": summarize(rows, "unet"),
        },
        "samples": sample_reports,
    }

    report_json = Path(args.report_json)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    with report_json.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(report), handle, indent=2)

    write_csv(Path(args.report_csv), rows)

    print(f"Evaluated {len(samples)} samples")
    print(f"JSON report: {report_json}")
    print(f"CSV report: {args.report_csv}")
    print(f"Previews: {preview_dir}")
    print("Summary:")
    for method_name, metrics in report["summary"].items():
        print(
            f"  {method_name}: "
            f"pixel_acc={metrics['pixel_accuracy']:.4f}, "
            f"mean_dice={metrics['mean_dice']:.4f}, "
            f"mean_iou={metrics['mean_iou']:.4f}"
        )


if __name__ == "__main__":
    main()
