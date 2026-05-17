import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from image_processor import SurgicalImageProcessor
from segmentation_model import SegmentationModel


try:
    import tensorrt as trt
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
except Exception:
    trt = None
    cuda = None

try:
    import tensorflow as tf
except Exception:
    tf = None


@dataclass
class InferenceOutput:
    mask: np.ndarray
    backend: str
    note: str = ""


class BaseInferenceEngine:
    backend_name = "base"

    def predict_mask(self, image_bgr: np.ndarray) -> InferenceOutput:
        raise NotImplementedError


class TensorRTEngine(BaseInferenceEngine):
    backend_name = "tensorrt"

    def __init__(self, engine_path: str, input_size: Tuple[int, int] = (256, 256)) -> None:
        if trt is None or cuda is None:
            raise RuntimeError("TensorRT or PyCUDA is unavailable.")

        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        self.input_size = input_size
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with self.engine_path.open("rb") as handle:
            self.engine = self.runtime.deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        self._bindings = []
        self._host_inputs = []
        self._device_inputs = []
        self._host_outputs = []
        self._device_outputs = []
        self._stream = cuda.Stream()
        self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        for binding_index in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(binding_index)
            shape = self.context.get_binding_shape(binding_index)
            if -1 in tuple(shape):
                shape = self.engine.get_binding_shape(binding_index)

            dtype = trt.nptype(self.engine.get_binding_dtype(binding_index))
            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self._bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding_index):
                self._host_inputs.append((binding_name, host_mem, shape))
                self._device_inputs.append(device_mem)
            else:
                self._host_outputs.append((binding_name, host_mem, shape))
                self._device_outputs.append(device_mem)

        if len(self._host_inputs) != 1 or len(self._host_outputs) != 1:
            raise RuntimeError("Expected one TensorRT input and one TensorRT output binding.")

    def predict_mask(self, image_bgr: np.ndarray) -> InferenceOutput:
        original_size = (image_bgr.shape[1], image_bgr.shape[0])
        input_tensor = preprocess_for_model(image_bgr, self.input_size)

        _, host_input, input_shape = self._host_inputs[0]
        flat_input = input_tensor.reshape(-1).astype(host_input.dtype)
        if flat_input.size != host_input.size:
            raise RuntimeError(
                f"Input size mismatch. Engine expects {input_shape}, "
                f"but prepared tensor has {input_tensor.shape}."
            )

        np.copyto(host_input, flat_input)
        cuda.memcpy_htod_async(self._device_inputs[0], host_input, self._stream)
        self.context.execute_async_v2(bindings=self._bindings, stream_handle=self._stream.handle)

        _, host_output, output_shape = self._host_outputs[0]
        cuda.memcpy_dtoh_async(host_output, self._device_outputs[0], self._stream)
        self._stream.synchronize()

        prediction = host_output.reshape(tuple(output_shape))
        mask = prediction_to_mask(prediction, original_size)
        return InferenceOutput(mask=mask, backend=self.backend_name)


class TFLiteEngine(BaseInferenceEngine):
    backend_name = "tflite"

    def __init__(self, model_path: str, input_size: Tuple[int, int] = (256, 256)) -> None:
        if tf is None:
            raise RuntimeError("TensorFlow is unavailable, so TFLite cannot run.")

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {self.model_path}")

        self.input_size = input_size
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        if len(self.input_details) != 1 or len(self.output_details) != 1:
            raise RuntimeError("Expected one TFLite input tensor and one output tensor.")

    def predict_mask(self, image_bgr: np.ndarray) -> InferenceOutput:
        original_size = (image_bgr.shape[1], image_bgr.shape[0])
        input_tensor = preprocess_for_model(image_bgr, self.input_size)

        input_detail = self.input_details[0]
        if input_detail["dtype"] == np.uint8:
            scale, zero_point = input_detail.get("quantization", (0.0, 0))
            if scale and scale > 0:
                input_tensor = input_tensor / scale + zero_point
            input_tensor = np.clip(input_tensor, 0, 255).astype(np.uint8)
        else:
            input_tensor = input_tensor.astype(input_detail["dtype"])

        self.interpreter.set_tensor(input_detail["index"], input_tensor)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(self.output_details[0]["index"])
        mask = prediction_to_mask(prediction, original_size)
        return InferenceOutput(mask=mask, backend=self.backend_name)


class KerasEngine(BaseInferenceEngine):
    backend_name = "keras"

    def __init__(self, weights_path: str, input_size: Tuple[int, int] = (256, 256)) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Keras weights not found: {self.weights_path}")

        self.model = SegmentationModel(
            mode="ml",
            input_size=input_size,
            weights_path=str(self.weights_path),
        )
        if self.model.mode != "ml" or self.model.model is None:
            raise RuntimeError(f"Could not load Keras weights. Details: {self.model.note}")

    def predict_mask(self, image_bgr: np.ndarray) -> InferenceOutput:
        result = self.model.segment(image_bgr)
        return InferenceOutput(mask=result.mask, backend=self.backend_name, note=result.note)


def preprocess_for_model(image_bgr: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(image_bgr, input_size, interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0)


def prediction_to_mask(prediction: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
    prediction = np.asarray(prediction)

    if prediction.ndim == 4:
        prediction = prediction[0]

    if prediction.ndim == 3 and prediction.shape[0] == 4 and prediction.shape[-1] != 4:
        prediction = np.transpose(prediction, (1, 2, 0))

    if prediction.ndim == 3:
        mask = np.argmax(prediction, axis=-1).astype(np.uint8)
    elif prediction.ndim == 2:
        mask = prediction.astype(np.uint8)
    else:
        raise RuntimeError(f"Unsupported model output shape: {prediction.shape}")

    return cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)


def select_inference_engine(
    trt_engine_path: Optional[str] = "models/segmentation.trt",
    tflite_model_path: Optional[str] = "models/segmentation.tflite",
    keras_weights_path: Optional[str] = "models/weights.h5",
    input_size: Tuple[int, int] = (256, 256),
    verbose: bool = True,
) -> BaseInferenceEngine:
    if trt_engine_path and Path(trt_engine_path).exists() and trt is not None and cuda is not None:
        try:
            engine = TensorRTEngine(trt_engine_path, input_size=input_size)
            if verbose:
                print(f"[Inference] backend=tensorrt path={trt_engine_path}")
            return engine
        except Exception as error:
            if verbose:
                print(f"[Inference] TensorRT unavailable, falling back. Reason: {error}")

    if tflite_model_path and Path(tflite_model_path).exists() and tf is not None:
        try:
            engine = TFLiteEngine(tflite_model_path, input_size=input_size)
            if verbose:
                print(f"[Inference] backend=tflite path={tflite_model_path}")
            return engine
        except Exception as error:
            if verbose:
                print(f"[Inference] TFLite unavailable, falling back. Reason: {error}")

    keras_errors = []
    for weights_path in keras_weight_candidates(keras_weights_path):
        if not weights_path.exists():
            continue
        try:
            engine = KerasEngine(str(weights_path), input_size=input_size)
            if verbose:
                print(f"[Inference] backend=keras path={weights_path}")
            return engine
        except Exception as error:
            keras_errors.append(f"{weights_path}: {error}")
            if verbose:
                print(f"[Inference] Keras weights unusable, trying fallback. Reason: {error}")

    raise FileNotFoundError(
        "No usable inference backend found. Provide one of: TensorRT .trt engine, "
        "TFLite .tflite model, or Keras .h5 weights. "
        f"Keras errors: {keras_errors}"
    )


def keras_weight_candidates(primary_weights_path: Optional[str]):
    seen = set()
    candidates = [
        primary_weights_path,
        "models/weights.h5",
        "models/unet_best_weights.h5",
        "models/unet_weights.h5",
    ]

    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        yield path


def resolve_default_weights() -> str:
    for candidate in (
        "models/weights.h5",
        "models/unet_best_weights.h5",
        "models/unet_weights.h5",
    ):
        if Path(candidate).exists():
            return candidate
    return "models/weights.h5"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backend selector for surgical segmentation inference: TensorRT -> TFLite -> Keras."
    )
    parser.add_argument("--input", help="Input image path for a one-image inference test.")
    parser.add_argument("--trt-engine", default="models/segmentation.trt", help="TensorRT engine path.")
    parser.add_argument("--tflite", default="models/segmentation.tflite", help="TFLite model path.")
    parser.add_argument("--weights", default=resolve_default_weights(), help="Keras weights fallback path.")
    parser.add_argument("--width", type=int, default=256, help="Model input width.")
    parser.add_argument("--height", type=int, default=256, help="Model input height.")
    parser.add_argument("--save-mask", default="outputs/jetson_inference_mask.png", help="Output mask image.")
    parser.add_argument("--save-overlay", default="outputs/jetson_inference_overlay.png", help="Output overlay image.")
    parser.add_argument("--backend-only", action="store_true", help="Only print selected backend, no image needed.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_size = (args.width, args.height)
    engine = select_inference_engine(
        trt_engine_path=args.trt_engine,
        tflite_model_path=args.tflite,
        keras_weights_path=args.weights,
        input_size=input_size,
        verbose=True,
    )

    if args.backend_only:
        print(f"Selected backend: {engine.backend_name}")
        return

    if not args.input:
        raise ValueError("--input is required unless --backend-only is used.")

    processor = SurgicalImageProcessor()
    image = processor.load_image(args.input)
    output = engine.predict_mask(image)
    overlay = processor.overlay_segmentation(image, output.mask)

    processor.save_image(args.save_mask, output.mask)
    processor.save_image(args.save_overlay, overlay)
    print(f"Selected backend: {output.backend}")
    print(f"Mask saved to: {args.save_mask}")
    print(f"Overlay saved to: {args.save_overlay}")


if __name__ == "__main__":
    main()
