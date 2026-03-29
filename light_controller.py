from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass
class LightDecision:
    global_intensity: int
    zone_intensities: Dict[str, int]
    target_class: str
    scene_brightness: float
    message: str


class IntelligentLightController:
    def __init__(
        self,
        base_intensity: int = 55,
        min_intensity: int = 20,
        max_intensity: int = 100,
    ) -> None:
        self.base_intensity = base_intensity
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity

    def compute_adjustment(self, image: np.ndarray, mask: np.ndarray) -> LightDecision:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scene_brightness = float(gray.mean())

        tissue_ratio = float(np.mean(mask == 1))
        organ_ratio = float(np.mean(mask == 2))
        tumor_ratio = float(np.mean(mask == 3))

        boost = 0.0
        if scene_brightness < 90:
            boost += 22
        elif scene_brightness < 120:
            boost += 12
        elif scene_brightness > 170:
            boost -= 10

        boost += tissue_ratio * 10
        boost += organ_ratio * 24
        boost += tumor_ratio * 36

        if tumor_ratio > 0.002:
            target_class = "tumor"
        elif organ_ratio > 0.01:
            target_class = "organ"
        elif tissue_ratio > 0.05:
            target_class = "tissue"
        else:
            target_class = "background"

        global_intensity = int(np.clip(self.base_intensity + boost, self.min_intensity, self.max_intensity))
        zone_intensities = self._compute_zone_intensities(gray, mask, global_intensity)
        message = (
            f"Brightness={scene_brightness:.1f}, focus={target_class}, "
            f"global_intensity={global_intensity}%"
        )

        return LightDecision(
            global_intensity=global_intensity,
            zone_intensities=zone_intensities,
            target_class=target_class,
            scene_brightness=scene_brightness,
            message=message,
        )

    def _compute_zone_intensities(
        self,
        gray: np.ndarray,
        mask: np.ndarray,
        global_intensity: int,
    ) -> Dict[str, int]:
        height, width = gray.shape[:2]
        half_h = height // 2
        half_w = width // 2

        zones = {
            "top_left": (slice(0, half_h), slice(0, half_w)),
            "top_right": (slice(0, half_h), slice(half_w, width)),
            "bottom_left": (slice(half_h, height), slice(0, half_w)),
            "bottom_right": (slice(half_h, height), slice(half_w, width)),
        }

        intensities: Dict[str, int] = {}
        for zone_name, (rows, cols) in zones.items():
            zone_gray = gray[rows, cols]
            zone_mask = mask[rows, cols]
            zone_brightness = float(zone_gray.mean())
            organ_ratio = float(np.mean(zone_mask == 2))
            tumor_ratio = float(np.mean(zone_mask == 3))

            zone_boost = 0.0
            zone_boost += max(0.0, 118.0 - zone_brightness) * 0.18
            zone_boost += organ_ratio * 22
            zone_boost += tumor_ratio * 42

            intensities[zone_name] = int(
                np.clip(global_intensity + zone_boost, self.min_intensity, self.max_intensity)
            )
        return intensities


class MockLightHardware:
    def __init__(self) -> None:
        self.last_command: Tuple[int, Dict[str, int]] = (0, {})

    def apply(self, decision: LightDecision) -> Dict[str, object]:
        self.last_command = (decision.global_intensity, decision.zone_intensities.copy())
        return {
            "global_intensity": decision.global_intensity,
            "zones": decision.zone_intensities.copy(),
            "target_class": decision.target_class,
        }

