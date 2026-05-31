"""Camera helpers for the Jetson surgical lighting pipeline.

CSI cameras such as the IMX219-77 cannot be opened with a plain integer index
(`cv2.VideoCapture(0)`). On the Jetson they are driven by the Argus stack and
must be opened through a GStreamer pipeline built around `nvarguscamerasrc`,
with OpenCV compiled against GStreamer (the JetPack system OpenCV is). USB
cameras and video files keep working through the usual integer/path source.
"""

from typing import Union

import cv2


def build_gstreamer_pipeline(
    sensor_id: int = 0,
    capture_width: int = 1920,
    capture_height: int = 1080,
    display_width: int = 960,
    display_height: int = 540,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
    """Build the standard CSI capture pipeline string for OpenCV.

    The pipeline is: nvarguscamerasrc -> nvvidconv (resize + flip) ->
    videoconvert -> appsink, ending in BGR so frames are ready for OpenCV.

    Args:
        sensor_id: CSI sensor index (the `sensor-id` of nvarguscamerasrc).
        capture_width: Sensor capture width in pixels.
        capture_height: Sensor capture height in pixels.
        display_width: Output frame width after nvvidconv scaling.
        display_height: Output frame height after nvvidconv scaling.
        framerate: Capture framerate in frames per second.
        flip_method: nvvidconv flip-method (0=none, 2=180deg, etc.).

    Returns:
        A GStreamer launch string for ``cv2.VideoCapture(..., cv2.CAP_GSTREAMER)``.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, "
        f"height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, "
        f"height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1"
    )


def open_camera(
    use_csi: bool,
    camera_index: int = 0,
    sensor_id: int = 0,
    capture_width: int = 1920,
    capture_height: int = 1080,
    display_width: int = 960,
    display_height: int = 540,
    framerate: int = 30,
    flip_method: int = 0,
) -> cv2.VideoCapture:
    """Open a camera, using a GStreamer CSI pipeline when requested.

    When ``use_csi`` is True the IMX219-77 (or any nvarguscamerasrc CSI sensor)
    is opened via :func:`build_gstreamer_pipeline` with ``cv2.CAP_GSTREAMER``.
    Otherwise a plain integer index is used for USB / UVC cameras.

    Args:
        use_csi: True for a CSI camera via GStreamer, False for a USB index.
        camera_index: Integer device index used when ``use_csi`` is False.
        sensor_id: CSI sensor index used when ``use_csi`` is True.
        capture_width: See :func:`build_gstreamer_pipeline`.
        capture_height: See :func:`build_gstreamer_pipeline`.
        display_width: See :func:`build_gstreamer_pipeline`.
        display_height: See :func:`build_gstreamer_pipeline`.
        framerate: See :func:`build_gstreamer_pipeline`.
        flip_method: See :func:`build_gstreamer_pipeline`.

    Returns:
        An opened ``cv2.VideoCapture``.

    Raises:
        RuntimeError: If the capture source cannot be opened.
    """
    if use_csi:
        pipeline = build_gstreamer_pipeline(
            sensor_id=sensor_id,
            capture_width=capture_width,
            capture_height=capture_height,
            display_width=display_width,
            display_height=display_height,
            framerate=framerate,
            flip_method=flip_method,
        )
        capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        source: Union[str, int] = f"CSI sensor-id={sensor_id}"
    else:
        capture = cv2.VideoCapture(camera_index)
        source = camera_index

    if not capture.isOpened():
        raise RuntimeError(f"Unable to open camera source: {source}")
    return capture
