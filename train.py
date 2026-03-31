import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from image_processor import CLASS_COLORS, DatasetSample, SurgicalImageProcessor
from segmentation_model import SegmentationModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the surgical segmentation U-Net on a local image/mask dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        help="Single directory containing both images and masks.",
    )
    parser.add_argument(
        "--images-dir",
        default="data/cholecseg/images",
        help="Directory containing training images.",
    )
    parser.add_argument(
        "--masks-dir",
        default="data/cholecseg/masks",
        help="Directory containing segmentation masks with matching file stems.",
    )
    parser.add_argument(
        "--mask-suffix",
        default="_color_mask",
        help="Preferred mask filename suffix when images and masks live in the same directory.",
    )
    parser.add_argument(
        "--extra-mask-suffixes",
        default="_mask,_watershed_mask",
        help="Comma-separated fallback mask suffixes for single-directory datasets.",
    )
    parser.add_argument(
        "--output-weights",
        default="models/unet_weights.h5",
        help="Where to save trained model weights.",
    )
    parser.add_argument(
        "--history-path",
        default="outputs/training_history.json",
        help="Where to save the Keras training history JSON.",
    )
    parser.add_argument("--width", type=int, default=256, help="Training image width.")
    parser.add_argument("--height", type=int, default=256, help="Training image height.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of data reserved for validation.",
    )
    parser.add_argument(
        "--pretrained-weights",
        default="models/unet_weights.h5",
        help="Path to pre-trained weights for fine-tuning. Set to empty string to train from scratch.",
    )
    return parser.parse_args()


def discover_samples_in_single_directory(
    dataset_dir: str,
    mask_suffixes: List[str],
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
) -> List[DatasetSample]:
    root = Path(dataset_dir)
    if not root.exists():
        return []

    files = sorted(path for path in root.rglob("*") if path.suffix.lower() in extensions)
    file_map = {path.name.lower(): path for path in files}
    samples: List[DatasetSample] = []

    for image_path in files:
        stem = image_path.stem
        if any(stem.endswith(suffix) for suffix in mask_suffixes):
            continue

        mask_path: Optional[Path] = None
        for suffix in mask_suffixes:
            for extension in extensions:
                candidate_name = f"{stem}{suffix}{extension}".lower()
                if candidate_name in file_map:
                    mask_path = file_map[candidate_name]
                    break
            if mask_path is not None:
                break

        samples.append(DatasetSample(image_path=image_path, mask_path=mask_path))

    return samples


def parse_mask_suffixes(primary_suffix: str, extra_suffixes: str) -> List[str]:
    ordered_suffixes: List[str] = []
    raw_values = [primary_suffix] + [value.strip() for value in extra_suffixes.split(",") if value.strip()]

    for suffix in raw_values:
        if not suffix.startswith("_"):
            suffix = f"_{suffix}"
        if suffix not in ordered_suffixes:
            ordered_suffixes.append(suffix)
    return ordered_suffixes


def load_mask(mask_path: Path, target_size: Tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Unable to load mask: {mask_path}")

    resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    if resized.ndim == 2:
        return normalize_grayscale_mask(resized)

    return convert_color_mask_to_class_ids(resized)


def normalize_grayscale_mask(mask: np.ndarray) -> np.ndarray:
    normalized = mask.astype(np.uint8)
    unique_values = set(np.unique(normalized).tolist())

    if unique_values.issubset({0, 1, 2, 3}):
        return normalized

    if unique_values.issubset({0, 85, 170, 255}):
        mapping = {0: 0, 85: 1, 170: 2, 255: 3}
        converted = np.zeros_like(normalized, dtype=np.uint8)
        for source_value, class_id in mapping.items():
            converted[normalized == source_value] = class_id
        return converted

    raise ValueError(
        "Grayscale masks must use class ids 0-3 or values {0,85,170,255}. "
        f"Found values: {sorted(unique_values)[:10]}"
    )


def convert_color_mask_to_class_ids(mask: np.ndarray) -> np.ndarray:
    if mask.shape[2] == 4:
        mask = mask[:, :, :3]

    mask = mask.astype(np.float32)
    class_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    best_distance = np.full(mask.shape[:2], np.inf, dtype=np.float32)

    for class_id, color in CLASS_COLORS.items():
        prototype = np.array(color, dtype=np.float32)
        distance = np.linalg.norm(mask - prototype, axis=2)
        update = distance < best_distance
        class_mask[update] = class_id
        best_distance[update] = distance[update]

    return class_mask


def load_training_arrays(
    samples: List[DatasetSample],
    target_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    processor = SurgicalImageProcessor()
    images = []
    masks = []

    for sample in samples:
        if sample.mask_path is None:
            continue

        image = processor.load_image(str(sample.image_path))
        mask = load_mask(sample.mask_path, target_size=target_size)

        images.append(
            processor.preprocess_frame(
                image,
                target_size=target_size,
                normalize=True,
            )
        )
        masks.append(mask)

    if not images:
        raise ValueError("No image/mask pairs were found to train on.")

    return np.asarray(images, dtype=np.float32), np.asarray(masks, dtype=np.uint8)


def main():
    args = parse_args()
    target_size = (args.width, args.height)

    if args.dataset_dir:
        mask_suffixes = parse_mask_suffixes(args.mask_suffix, args.extra_mask_suffixes)
        samples = discover_samples_in_single_directory(args.dataset_dir, mask_suffixes)
    else:
        processor = SurgicalImageProcessor()
        samples = processor.discover_dataset(args.images_dir, args.masks_dir)
    paired_samples = [sample for sample in samples if sample.mask_path is not None]

    if not paired_samples:
        if args.dataset_dir:
            inspected_files = sorted(
                path.name for path in Path(args.dataset_dir).glob("*") if path.is_file()
            )[:12]
            raise ValueError(
                "No paired samples found. Checked suffixes "
                f"{mask_suffixes}. Example files seen: {inspected_files}"
            )
        raise ValueError(
            "No paired samples found. Make sure images and masks share the same filename stem."
        )

    print(f"Found {len(paired_samples)} paired samples.")

    images, masks = load_training_arrays(paired_samples, target_size=target_size)
    print(f"Training data shapes: images={images.shape}, masks={masks.shape}")

    # Load pre-trained weights if specified (for fine-tuning)
    weights_to_load = args.pretrained_weights if args.pretrained_weights else None
    if weights_to_load and not Path(weights_to_load).exists():
        print(f"Warning: Pre-trained weights not found at {weights_to_load}. Training from scratch.")
        weights_to_load = None
    elif weights_to_load:
        print(f"Loading pre-trained weights from {weights_to_load} for fine-tuning...")

    model = SegmentationModel(mode="ml", input_size=target_size, weights_path=weights_to_load)
    history = model.train_on_arrays(
        images=images,
        masks=masks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
    )

    output_weights = Path(args.output_weights)
    output_weights.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(output_weights))
    print(f"Saved weights to {output_weights}")

    history_path = Path(args.history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history.history, handle, indent=2)
    print(f"Saved history to {history_path}")


if __name__ == "__main__":
    main()
