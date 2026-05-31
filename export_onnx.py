"""Stage 1 of the Jetson export path: Keras weights -> ONNX.

Run this on the PC/training machine where TensorFlow works. It rebuilds the
U-Net, loads your trained weights, and writes an ONNX model with a fixed
(1, 256, 256, 3) float32 input. The resulting models/segmentation.onnx is
portable and gets copied to the Jetson, where stage 2 (trtexec) turns it into
models/segmentation.trt. See SETUP_JETSON.md for stage 2.

TensorRT engines are NOT portable across machines/TensorRT versions, which is
why the engine is built on-device and only the ONNX is produced here.
"""

import argparse
from pathlib import Path

import tensorflow as tf
import tf2onnx

from segmentation_model import SegmentationModel


INPUT_SIZE = (256, 256)  # (width, height) the inference pipeline expects.


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export trained U-Net Keras weights to ONNX for TensorRT conversion on the Jetson."
    )
    parser.add_argument(
        "--weights",
        default="models/unet_best_weights.h5",
        help="Path to the trained U-Net weights (.h5).",
    )
    parser.add_argument(
        "--output",
        default="models/segmentation.onnx",
        help="Output ONNX model path.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version (default 13, validated for TensorRT 8.x).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Trained weights not found: {weights_path}")

    segmentation = SegmentationModel(
        mode="ml",
        input_size=INPUT_SIZE,
        weights_path=str(weights_path),
    )
    if segmentation.mode != "ml" or segmentation.model is None:
        raise RuntimeError(
            f"Could not load Keras weights into the U-Net. Details: {segmentation.note}"
        )

    keras_model = segmentation.model
    # Fixed batch size of 1 so the TensorRT engine has a static input shape,
    # matching the single-frame bindings used by TensorRTEngine on the Jetson.
    input_signature = [
        tf.TensorSpec((1, INPUT_SIZE[1], INPUT_SIZE[0], 3), tf.float32, name="input")
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tf2onnx.convert.from_keras(
        keras_model,
        input_signature=input_signature,
        opset=args.opset,
        output_path=str(output_path),
    )

    print(f"Loaded weights: {weights_path}")
    print(f"Saved ONNX model (opset {args.opset}): {output_path}")
    print("Next: copy this file to the Jetson and run trtexec (see SETUP_JETSON.md).")


if __name__ == "__main__":
    main()
