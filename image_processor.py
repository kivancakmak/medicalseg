from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np


CLASS_LABELS = {
    0: "background",
    1: "tissue",
    2: "organ",
    3: "tumor",
}

CLASS_COLORS = {
    0: (30, 30, 30),
    1: (170, 110, 200),
    2: (60, 90, 220),
    3: (40, 215, 245),
}


@dataclass
class DatasetSample:
    image_path: Path
    mask_path: Optional[Path] = None


class SurgicalImageProcessor:
    def load_image(self, path: str) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to load image: {path}")
        return image

    def save_image(self, path: str, image: np.ndarray) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)

    def preprocess_frame(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        frame = image.copy()
        if target_size is not None:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        if normalize:
            frame = frame.astype(np.float32) / 255.0
        return frame

    def prepare_mask(
        self,
        mask: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        prepared = mask.copy().astype(np.uint8)
        if target_size is not None:
            prepared = cv2.resize(prepared, target_size, interpolation=cv2.INTER_NEAREST)
        return prepared

    def mask_to_color(self, mask: np.ndarray) -> np.ndarray:
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for class_id, color in CLASS_COLORS.items():
            color_mask[mask == class_id] = color
        return color_mask

    def overlay_segmentation(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.38,
    ) -> np.ndarray:
        base = image.copy()
        color_mask = self.mask_to_color(mask)
        blended = base.copy().astype(np.float32)
        mask_pixels = mask > 0
        blended[mask_pixels] = (
            base[mask_pixels].astype(np.float32) * (1.0 - alpha)
            + color_mask[mask_pixels].astype(np.float32) * alpha
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    def discover_dataset(
        self,
        images_dir: str,
        masks_dir: Optional[str] = None,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
    ) -> List[DatasetSample]:
        image_root = Path(images_dir)
        mask_root = Path(masks_dir) if masks_dir else None
        samples: List[DatasetSample] = []

        if not image_root.exists():
            return samples

        image_files = sorted(
            path for path in image_root.rglob("*") if path.suffix.lower() in extensions
        )
        for image_path in image_files:
            mask_path = None
            if mask_root is not None:
                candidates = [
                    mask_root / f"{image_path.stem}{suffix}"
                    for suffix in extensions
                ]
                matching = [candidate for candidate in candidates if candidate.exists()]
                if matching:
                    mask_path = matching[0]
            samples.append(DatasetSample(image_path=image_path, mask_path=mask_path))
        return samples

    def generate_synthetic_scene(
        self,
        width: int = 640,
        height: int = 480,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)

        y_grid = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        x_grid = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]

        base = np.zeros((height, width, 3), dtype=np.float32)
        base[..., 0] = 18 + 20 * y_grid + 7 * x_grid
        base[..., 1] = 16 + 26 * y_grid
        base[..., 2] = 20 + 30 * y_grid + 10 * (1.0 - x_grid)

        illumination = 0.75 + 0.35 * np.exp(-((x_grid - 0.58) ** 2 + (y_grid - 0.45) ** 2) / 0.18)
        image = np.clip(base * illumination[..., None], 0, 255).astype(np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)

        tissue_center = (
            int(width * rng.uniform(0.42, 0.56)),
            int(height * rng.uniform(0.45, 0.58)),
        )
        tissue_axes = (
            int(width * rng.uniform(0.24, 0.34)),
            int(height * rng.uniform(0.18, 0.28)),
        )
        tissue_angle = int(rng.integers(-20, 21))

        cv2.ellipse(mask, tissue_center, tissue_axes, tissue_angle, 0, 360, 1, -1)
        cv2.ellipse(image, tissue_center, tissue_axes, tissue_angle, 0, 360, CLASS_COLORS[1], -1)

        organ_center = (
            tissue_center[0] + int(width * rng.uniform(-0.06, 0.07)),
            tissue_center[1] + int(height * rng.uniform(-0.03, 0.03)),
        )
        organ_axes = (
            int(tissue_axes[0] * rng.uniform(0.35, 0.5)),
            int(tissue_axes[1] * rng.uniform(0.35, 0.52)),
        )
        organ_angle = int(rng.integers(-35, 36))

        cv2.ellipse(mask, organ_center, organ_axes, organ_angle, 0, 360, 2, -1)
        cv2.ellipse(image, organ_center, organ_axes, organ_angle, 0, 360, CLASS_COLORS[2], -1)

        tumor_radius = int(min(width, height) * rng.uniform(0.03, 0.05))
        tumor_center = (
            organ_center[0] + int(organ_axes[0] * rng.uniform(-0.15, 0.15)),
            organ_center[1] + int(organ_axes[1] * rng.uniform(-0.18, 0.18)),
        )
        cv2.circle(mask, tumor_center, tumor_radius, 3, -1)
        cv2.circle(image, tumor_center, tumor_radius, CLASS_COLORS[3], -1)

        highlight_center = (
            int(width * rng.uniform(0.25, 0.75)),
            int(height * rng.uniform(0.20, 0.55)),
        )
        highlight_radius = int(min(width, height) * rng.uniform(0.10, 0.18))
        highlight = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(highlight, highlight_center, highlight_radius, 255, -1)
        highlight = cv2.GaussianBlur(highlight, (0, 0), sigmaX=31)
        image = np.clip(
            image.astype(np.float32) + 0.08 * highlight[..., None],
            0,
            255,
        ).astype(np.uint8)

        noise = rng.normal(0.0, 6.0, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        image = cv2.GaussianBlur(image, (5, 5), sigmaX=1.1)

        return image, mask

    def compose_panel(
        self,
        original: np.ndarray,
        mask: np.ndarray,
        overlay: np.ndarray,
        titles: Optional[Iterable[str]] = None,
    ) -> np.ndarray:
        color_mask = self.mask_to_color(mask)
        frames = [original.copy(), color_mask, overlay.copy()]
        labels = list(titles or ["Input", "Mask", "Overlay"])

        for frame, label in zip(frames, labels):
            cv2.putText(
                frame,
                label,
                (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return np.hstack(frames)

