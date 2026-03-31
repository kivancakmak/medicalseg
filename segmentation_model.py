from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from image_processor import CLASS_COLORS, CLASS_LABELS, SurgicalImageProcessor

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - optional dependency fallback
    tf = None


def _prepare_sparse_targets(y_true):
    y_true = tf.cast(y_true, tf.int32)
    if tf.keras.backend.ndim(y_true) == 4:
        y_true = tf.squeeze(y_true, axis=-1)
    return y_true


@dataclass
class SegmentationResult:
    mask: np.ndarray
    class_area_ratios: Dict[str, float]
    mode_used: str
    note: str = ""


class SegmentationModel:
    def __init__(
        self,
        mode: str = "classical",
        input_size: Tuple[int, int] = (256, 256),
        weights_path: Optional[str] = None,
    ) -> None:
        self.requested_mode = mode
        self.mode = "classical"
        self.input_size = input_size
        self.processor = SurgicalImageProcessor()
        self.model = None
        self.note = ""

        self.prototype_colors = {
            class_id: np.array(color, dtype=np.float32)
            for class_id, color in CLASS_COLORS.items()
            if class_id != 0
        }
        self.kernel = np.ones((5, 5), dtype=np.uint8)

        if mode == "synthetic":
            self.mode = "synthetic"
        elif mode == "ml":
            self._configure_ml(weights_path)

    def _configure_ml(self, weights_path: Optional[str]) -> None:
        if tf is None:
            self.note = "TensorFlow is unavailable, so the app is using classical segmentation."
            return

        self.model = self.build_unet_model(
            input_size=self.input_size,
            num_classes=len(CLASS_LABELS),
        )
        if weights_path and Path(weights_path).exists():
            try:
                self.model.load_weights(str(weights_path))
                self.mode = "ml"
                self.note = "Loaded trained U-Net weights."
            except Exception as error:  # pragma: no cover - depends on user weights
                self.note = f"Could not load weights ({error}); using classical segmentation."
                self.model = None
        else:
            self.note = "No trained weights were supplied, so the app is using classical segmentation."
            self.model = None

    def build_unet_model(self, input_size: Tuple[int, int], num_classes: int):
        if tf is None:
            raise RuntimeError("TensorFlow is required to build the U-Net model.")

        def dice_coefficient(y_true, y_pred):
            y_true_sparse = _prepare_sparse_targets(y_true)
            y_true_one_hot = tf.one_hot(y_true_sparse, depth=num_classes, dtype=tf.float32)
            y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
            y_pred_one_hot = tf.one_hot(y_pred_labels, depth=num_classes, dtype=tf.float32)

            # Ignore background so the metric reflects the surgical target regions.
            y_true_fg = y_true_one_hot[..., 1:]
            y_pred_fg = y_pred_one_hot[..., 1:]

            intersection = tf.reduce_sum(y_true_fg * y_pred_fg, axis=[1, 2])
            denominator = tf.reduce_sum(y_true_fg + y_pred_fg, axis=[1, 2])
            dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
            return tf.reduce_mean(dice)

        def mean_iou(y_true, y_pred):
            y_true_sparse = _prepare_sparse_targets(y_true)
            y_pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

            y_true_one_hot = tf.one_hot(y_true_sparse, depth=num_classes, dtype=tf.float32)
            y_pred_one_hot = tf.one_hot(y_pred_labels, depth=num_classes, dtype=tf.float32)

            y_true_fg = y_true_one_hot[..., 1:]
            y_pred_fg = y_pred_one_hot[..., 1:]

            intersection = tf.reduce_sum(y_true_fg * y_pred_fg, axis=[1, 2])
            union = tf.reduce_sum(y_true_fg + y_pred_fg, axis=[1, 2]) - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            return tf.reduce_mean(iou)

        dice_coefficient.__name__ = "dice_coefficient"
        mean_iou.__name__ = "mean_iou"

        inputs = tf.keras.Input(shape=(input_size[1], input_size[0], 3))

        def conv_block(x, filters):
            x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            return x

        c1 = conv_block(inputs, 32)
        p1 = tf.keras.layers.MaxPool2D()(c1)

        c2 = conv_block(p1, 64)
        p2 = tf.keras.layers.MaxPool2D()(c2)

        c3 = conv_block(p2, 128)
        p3 = tf.keras.layers.MaxPool2D()(c3)

        bridge = conv_block(p3, 256)

        u3 = tf.keras.layers.UpSampling2D()(bridge)
        u3 = tf.keras.layers.Concatenate()([u3, c3])
        c4 = conv_block(u3, 128)

        u2 = tf.keras.layers.UpSampling2D()(c4)
        u2 = tf.keras.layers.Concatenate()([u2, c2])
        c5 = conv_block(u2, 64)

        u1 = tf.keras.layers.UpSampling2D()(c5)
        u1 = tf.keras.layers.Concatenate()([u1, c1])
        c6 = conv_block(u1, 32)

        outputs = tf.keras.layers.Conv2D(num_classes, 1, activation="softmax")(c6)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy", dice_coefficient, mean_iou],
        )
        return model

    def train_on_arrays(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        epochs: int = 5,
        batch_size: int = 4,
        validation_split: float = 0.2,
        callbacks=None,
    ):
        if tf is None:
            raise RuntimeError("TensorFlow is required for training.")
        if self.model is None:
            self.model = self.build_unet_model(
                input_size=self.input_size,
                num_classes=len(CLASS_LABELS),
            )

        images = np.asarray(images, dtype=np.float32)
        masks = np.asarray(masks, dtype=np.uint8)
        if images.max() > 1.0:
            images = images / 255.0

        history = self.model.fit(
            images,
            masks,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
        )
        self.mode = "ml"
        self.note = "Model trained in-session and ready for inference."
        return history

    def save_weights(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("No model is available to save.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(path)

    def segment(self, image: np.ndarray) -> SegmentationResult:
        if self.mode == "ml" and self.model is not None:
            mask = self._segment_with_model(image)
            note = self.note
        elif self.mode == "synthetic":
            mask = self._segment_by_prototype(image, threshold=62.0)
            note = self.note or "Synthetic-friendly color prototype segmentation."
        else:
            mask = self._segment_classically(image)
            note = self.note or "Classical HSV and prototype-color segmentation."

        return SegmentationResult(
            mask=mask,
            class_area_ratios=self._compute_area_ratios(mask),
            mode_used=self.mode,
            note=note,
        )

    def _segment_with_model(self, image: np.ndarray) -> np.ndarray:
        resized = self.processor.preprocess_frame(
            image,
            target_size=self.input_size,
            normalize=True,
        )
        prediction = self.model.predict(np.expand_dims(resized, axis=0), verbose=0)[0]
        mask = np.argmax(prediction, axis=-1).astype(np.uint8)
        return cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    def _segment_classically(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        proto_mask = self._segment_by_prototype(image, threshold=88.0)

        tissue_hsv = cv2.inRange(hsv, (125, 20, 45), (178, 210, 255))
        organ_hsv_1 = cv2.inRange(hsv, (0, 35, 40), (12, 255, 255))
        organ_hsv_2 = cv2.inRange(hsv, (168, 35, 40), (179, 255, 255))
        tumor_hsv = cv2.inRange(hsv, (12, 50, 55), (40, 255, 255))

        tissue = cv2.bitwise_or(tissue_hsv, self._binary_from_label(proto_mask, 1))
        organ = cv2.bitwise_or(cv2.bitwise_or(organ_hsv_1, organ_hsv_2), self._binary_from_label(proto_mask, 2))
        tumor = cv2.bitwise_or(tumor_hsv, self._binary_from_label(proto_mask, 3))

        tissue = self._clean_binary_mask(tissue)
        organ = self._clean_binary_mask(organ)
        tumor = self._clean_binary_mask(tumor)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[tissue > 0] = 1
        mask[organ > 0] = 2
        mask[tumor > 0] = 3
        return mask

    def _segment_by_prototype(self, image: np.ndarray, threshold: float) -> np.ndarray:
        pixels = image.astype(np.float32)
        best_distance = np.full(image.shape[:2], fill_value=np.inf, dtype=np.float32)
        labels = np.zeros(image.shape[:2], dtype=np.uint8)

        for class_id, prototype in self.prototype_colors.items():
            distance = np.linalg.norm(pixels - prototype, axis=2)
            update_mask = distance < best_distance
            labels[update_mask] = class_id
            best_distance[update_mask] = distance[update_mask]

        labels[best_distance > threshold] = 0

        cleaned = np.zeros_like(labels)
        for class_id in (1, 2, 3):
            binary = self._binary_from_label(labels, class_id)
            binary = self._clean_binary_mask(binary)
            cleaned[binary > 0] = class_id
        return cleaned

    def _binary_from_label(self, label_map: np.ndarray, class_id: int) -> np.ndarray:
        return np.where(label_map == class_id, 255, 0).astype(np.uint8)

    def _clean_binary_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, self.kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel)
        return cleaned

    def _compute_area_ratios(self, mask: np.ndarray) -> Dict[str, float]:
        total_pixels = float(mask.size)
        return {
            CLASS_LABELS[class_id]: float(np.count_nonzero(mask == class_id) / total_pixels)
            for class_id in CLASS_LABELS
            if class_id != 0
        }
