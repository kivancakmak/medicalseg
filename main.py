import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from image_processor import SurgicalImageProcessor
from light_controller import IntelligentLightController, LightDecision, MockLightHardware
from segmentation_model import SegmentationModel, SegmentationResult


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prototype for camera-based intelligent surgical lighting with tissue segmentation."
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "image", "video", "webcam"],
        default="synthetic",
        help="Input mode for the prototype.",
    )
    parser.add_argument("--input", help="Path to an input image or video for image/video modes.")
    parser.add_argument(
        "--model-mode",
        choices=["classical", "synthetic", "ml"],
        default="classical",
        help="Segmentation backend to use.",
    )
    parser.add_argument("--weights", help="Path to trained TensorFlow weights for ML mode.")
    parser.add_argument("--save-dir", default="outputs", help="Directory for saved demo outputs.")
    parser.add_argument("--output-video", help="Optional path for saving annotated video output.")
    parser.add_argument("--display", action="store_true", help="Display frames using OpenCV windows.")
    parser.add_argument("--frames", type=int, default=5, help="Number of synthetic frames to generate.")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index.")
    parser.add_argument("--width", type=int, default=640, help="Synthetic frame width.")
    parser.add_argument("--height", type=int, default=480, help="Synthetic frame height.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Max frames for webcam/video processing. Use 0 for no limit.",
    )
    parser.add_argument(
        "--simulate-hardware",
        action="store_true",
        help="Apply light decisions to the mock hardware interface.",
    )
    return parser.parse_args()


def annotate_output(
    panel: np.ndarray,
    result: SegmentationResult,
    decision: LightDecision,
) -> np.ndarray:
    annotated = panel.copy()
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 92), (20, 20, 20), -1)
    cv2.putText(
        annotated,
        f"Segmentation mode: {result.mode_used}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        decision.message,
        (16, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 255, 200),
        2,
        cv2.LINE_AA,
    )
    if result.note:
        cv2.putText(
            annotated,
            result.note[:110],
            (16, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
    return annotated


def process_frame(
    frame: np.ndarray,
    processor: SurgicalImageProcessor,
    segmenter: SegmentationModel,
    controller: IntelligentLightController,
):
    result = segmenter.segment(frame)
    decision = controller.compute_adjustment(frame, result.mask)
    overlay = processor.overlay_segmentation(frame, result.mask)
    panel = processor.compose_panel(frame, result.mask, overlay)
    annotated = annotate_output(panel, result, decision)
    return annotated, result, decision


def run_synthetic_demo(
    args,
    processor: SurgicalImageProcessor,
    segmenter: SegmentationModel,
    controller: IntelligentLightController,
    hardware: Optional[MockLightHardware],
) -> None:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for index in range(args.frames):
        frame, _ = processor.generate_synthetic_scene(
            width=args.width,
            height=args.height,
            seed=index,
        )
        annotated, _, decision = process_frame(frame, processor, segmenter, controller)
        if hardware is not None:
            hardware.apply(decision)

        output_path = save_dir / f"synthetic_demo_{index:02d}.png"
        processor.save_image(str(output_path), annotated)
        print(f"[synthetic] saved {output_path}")

        if args.display:
            cv2.imshow("Surgical Light Prototype", annotated)
            if cv2.waitKey(300) & 0xFF == ord("q"):
                break


def run_single_image(
    args,
    processor: SurgicalImageProcessor,
    segmenter: SegmentationModel,
    controller: IntelligentLightController,
    hardware: Optional[MockLightHardware],
) -> None:
    if not args.input:
        raise ValueError("--input is required when --mode image is used.")
    input_path = Path(args.input)
    if not input_path.exists():
        hint = ""
        if args.input.lower() == r"path\to\image.png":
            hint = " Replace that example path with the real location of your image file."
        raise FileNotFoundError(f"Input image not found: {args.input}.{hint}")

    frame = processor.load_image(str(input_path))
    annotated, _, decision = process_frame(frame, processor, segmenter, controller)
    if hardware is not None:
        hardware.apply(decision)

    output_path = Path(args.save_dir) / f"{input_path.stem}_annotated.png"
    processor.save_image(str(output_path), annotated)
    print(f"[image] saved {output_path}")

    if args.display:
        cv2.imshow("Surgical Light Prototype", annotated)
        cv2.waitKey(0)


def run_stream(
    args,
    processor: SurgicalImageProcessor,
    segmenter: SegmentationModel,
    controller: IntelligentLightController,
    hardware: Optional[MockLightHardware],
) -> None:
    source = args.camera_index if args.mode == "webcam" else args.input
    if args.mode == "video" and not args.input:
        raise ValueError("--input is required when --mode video is used.")
    if args.mode == "video":
        input_path = Path(args.input)
        if not input_path.exists():
            hint = ""
            if args.input.lower() == r"path\to\video.mp4":
                hint = " Replace that example path with the real location of your video file."
            raise FileNotFoundError(f"Input video not found: {args.input}.{hint}")
        source = str(input_path)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open input source: {source}")

    writer = None
    frame_index = 0

    try:
        while args.max_frames <= 0 or frame_index < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            annotated, _, decision = process_frame(frame, processor, segmenter, controller)
            if hardware is not None:
                hardware.apply(decision)

            if args.output_video and writer is None:
                output_path = Path(args.output_video)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    20.0,
                    (annotated.shape[1], annotated.shape[0]),
                )

            if writer is not None:
                writer.write(annotated)

            if args.display:
                cv2.imshow("Surgical Light Prototype", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1

        print(f"[stream] processed {frame_index} frames")
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def main():
    args = parse_args()

    processor = SurgicalImageProcessor()
    segmenter = SegmentationModel(
        mode=args.model_mode,
        weights_path=args.weights,
    )
    controller = IntelligentLightController()
    hardware = MockLightHardware() if args.simulate_hardware else None

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == "synthetic":
        run_synthetic_demo(args, processor, segmenter, controller, hardware)
    elif args.mode == "image":
        run_single_image(args, processor, segmenter, controller, hardware)
    else:
        run_stream(args, processor, segmenter, controller, hardware)

    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
