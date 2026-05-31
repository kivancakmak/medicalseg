import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from camera import open_camera
from image_processor import SurgicalImageProcessor
from jetson_inference import resolve_default_weights, select_inference_engine
from light_controller import IntelligentLightController, LightDecision
from pwm_controller import JetsonPWMController, clamp_intensity


class SmoothedLightOutput:
    def __init__(self, alpha: float = 0.35, max_step: float = 12.0) -> None:
        self.alpha = max(0.0, min(1.0, float(alpha)))
        self.max_step = max(0.0, float(max_step))
        self._previous: Optional[Dict[str, float]] = None

    def update(self, target: Dict[str, int]) -> Dict[str, float]:
        if self._previous is None:
            self._previous = {zone: clamp_intensity(value) for zone, value in target.items()}
            return self._previous.copy()

        smoothed = {}
        for zone, target_value in target.items():
            previous_value = self._previous.get(zone, clamp_intensity(target_value))
            blended = previous_value + self.alpha * (float(target_value) - previous_value)
            delta = max(-self.max_step, min(self.max_step, blended - previous_value))
            smoothed[zone] = clamp_intensity(previous_value + delta)

        self._previous = smoothed
        return smoothed.copy()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Jetson real-time surgical lighting pipeline: camera -> segmentation -> lights -> GPIO."
    )
    parser.add_argument("--image", help="Run one still image through the full pipeline.")
    parser.add_argument("--input-video", help="Run a video file through the pipeline.")
    parser.add_argument("--camera-index", type=int, default=0, help="USB camera index, or CSI sensor-id when --csi is set.")
    parser.add_argument("--csi", action="store_true", help="Open a CSI camera (e.g. IMX219-77) via a GStreamer pipeline.")
    parser.add_argument("--flip-method", type=int, default=0, help="nvvidconv flip-method for the CSI pipeline (0=none).")
    parser.add_argument("--width", type=int, default=256, help="Segmentation model input width.")
    parser.add_argument("--height", type=int, default=256, help="Segmentation model input height.")
    parser.add_argument("--trt-engine", default="models/segmentation.trt", help="TensorRT engine path.")
    parser.add_argument("--tflite", default="models/segmentation.tflite", help="TFLite model path.")
    parser.add_argument("--weights", default=resolve_default_weights(), help="Keras weights fallback path.")
    parser.add_argument("--display", action="store_true", help="Display annotated frames.")
    parser.add_argument("--save-dir", default="outputs/jetson_main", help="Folder for still-image outputs.")
    parser.add_argument("--output-video", help="Optional annotated output video path.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum stream frames. Use 0 for no limit.")
    parser.add_argument("--fps", type=float, default=20.0, help="Output video FPS.")
    parser.add_argument("--pwm-frequency", type=int, default=50, help="PWM frequency in Hz.")
    parser.add_argument("--start-intensity", type=float, default=0.0, help="Initial PWM duty cycle.")
    parser.add_argument("--smooth-alpha", type=float, default=0.35, help="Light smoothing alpha.")
    parser.add_argument("--max-light-step", type=float, default=12.0, help="Max duty-cycle change per frame.")
    parser.add_argument("--no-lights", action="store_true", help="Run vision only without applying PWM.")
    parser.add_argument("--quiet-stub", action="store_true", help="Reduce Windows PWM stub print output.")
    return parser.parse_args()


def annotate_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    decision: LightDecision,
    backend_name: str,
    fps: float,
    processor: SurgicalImageProcessor,
) -> np.ndarray:
    overlay = processor.overlay_segmentation(frame, mask)
    panel = processor.compose_panel(frame, mask, overlay, titles=["Input", "Mask", "Overlay"])
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 104), (18, 18, 18), -1)

    lines = [
        f"backend={backend_name} fps={fps:.1f}",
        decision.message,
        format_zones(decision.zone_intensities),
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (16, 30 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (230, 255, 230),
            2,
            cv2.LINE_AA,
        )
    return panel


def format_zones(zone_intensities: Dict[str, int]) -> str:
    return (
        f"TL={zone_intensities.get('top_left', 0)}% "
        f"TR={zone_intensities.get('top_right', 0)}% "
        f"BL={zone_intensities.get('bottom_left', 0)}% "
        f"BR={zone_intensities.get('bottom_right', 0)}%"
    )


def process_frame(
    frame: np.ndarray,
    inference_engine,
    light_logic: IntelligentLightController,
    pwm_controller: Optional[JetsonPWMController],
    smoother: SmoothedLightOutput,
    processor: SurgicalImageProcessor,
    display_seconds: Optional[float] = None,
) -> Tuple[np.ndarray, LightDecision, float]:
    start_time = time.perf_counter()
    inference_output = inference_engine.predict_mask(frame)
    decision = light_logic.compute_adjustment(frame, inference_output.mask)
    smoothed_zones = smoother.update(decision.zone_intensities)

    if pwm_controller is not None:
        pwm_controller.apply_zone_intensities(smoothed_zones)

    elapsed_seconds = max(time.perf_counter() - start_time, 1e-6)
    # Label this frame with the carried-forward duration when one is supplied
    # (streaming), otherwise with this frame's own measured duration (stills).
    shown_seconds = display_seconds if display_seconds is not None else elapsed_seconds
    fps = 1.0 / shown_seconds if shown_seconds > 0 else 0.0
    annotated = annotate_frame(
        frame=frame,
        mask=inference_output.mask,
        decision=decision,
        backend_name=inference_output.backend,
        fps=fps,
        processor=processor,
    )
    return annotated, decision, elapsed_seconds


def run_image(args, inference_engine, light_logic, pwm_controller, smoother, processor) -> None:
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    frame = processor.load_image(str(image_path))
    annotated, decision, _elapsed = process_frame(
        frame=frame,
        inference_engine=inference_engine,
        light_logic=light_logic,
        pwm_controller=pwm_controller,
        smoother=smoother,
        processor=processor,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"{image_path.stem}_jetson_pipeline.png"
    processor.save_image(str(output_path), annotated)
    print(decision.message)
    print(f"Saved annotated output to: {output_path}")

    if args.display:
        cv2.imshow("Jetson Surgical Light Pipeline", annotated)
        cv2.waitKey(0)


def open_capture(args):
    if args.input_video:
        video_path = Path(args.input_video)
        if not video_path.exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {video_path}")
        return capture

    return open_camera(
        use_csi=args.csi,
        camera_index=args.camera_index,
        sensor_id=args.camera_index,
        flip_method=args.flip_method,
    )


def run_stream(args, inference_engine, light_logic, pwm_controller, smoother, processor) -> None:
    capture = open_capture(args)
    writer = None
    frame_index = 0
    last_elapsed: Optional[float] = None

    try:
        while args.max_frames <= 0 or frame_index < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            # Annotate with the previous frame's measured duration so the FPS
            # label reflects real inference time; the first frame uses its own.
            annotated, decision, last_elapsed = process_frame(
                frame=frame,
                inference_engine=inference_engine,
                light_logic=light_logic,
                pwm_controller=pwm_controller,
                smoother=smoother,
                processor=processor,
                display_seconds=last_elapsed,
            )

            if args.output_video and writer is None:
                output_path = Path(args.output_video)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    args.fps,
                    (annotated.shape[1], annotated.shape[0]),
                )

            if writer is not None:
                writer.write(annotated)

            if frame_index % 10 == 0:
                print(f"frame={frame_index} {decision.message}")

            if args.display:
                cv2.imshow("Jetson Surgical Light Pipeline", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    print(f"Processed {frame_index} frames")


def main():
    args = parse_args()
    input_size = (args.width, args.height)

    processor = SurgicalImageProcessor()
    inference_engine = select_inference_engine(
        trt_engine_path=args.trt_engine,
        tflite_model_path=args.tflite,
        keras_weights_path=args.weights,
        input_size=input_size,
        verbose=True,
    )
    light_logic = IntelligentLightController()
    smoother = SmoothedLightOutput(alpha=args.smooth_alpha, max_step=args.max_light_step)

    pwm_controller = None
    if not args.no_lights:
        pwm_controller = JetsonPWMController(
            frequency_hz=args.pwm_frequency,
            start_intensity=args.start_intensity,
            verbose_stub=not args.quiet_stub,
        )

    try:
        if pwm_controller is not None:
            pwm_controller.setup()

        if args.image:
            run_image(args, inference_engine, light_logic, pwm_controller, smoother, processor)
        else:
            run_stream(args, inference_engine, light_logic, pwm_controller, smoother, processor)
    finally:
        if pwm_controller is not None:
            pwm_controller.cleanup()


if __name__ == "__main__":
    main()
