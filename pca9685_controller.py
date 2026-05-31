"""PCA9685 (I2C) PWM backend for four-zone surgical lighting.

The Jetson Nano's onboard header exposes only two hardware PWM channels and has
no software PWM, so the four-zone design in pwm_controller.py cannot drive all
zones from the board pins. This module offloads PWM generation to an external
PCA9685 16-channel controller over I2C, while exposing the SAME public interface
as JetsonPWMController so light_controller.py and jetson_main.py barely change.

If adafruit_pca9685 (or the I2C bus) is unavailable - e.g. running on a PC with
no hardware attached - it falls back to a verbose print-only stub, mirroring
JetsonPWMController's is_stub behavior, so the control logic can be tested now.
"""

import argparse
import platform
import time
from typing import Dict, Mapping, Optional

from pwm_controller import clamp_intensity, normalize_zone_name, parse_zone_values


# Each light zone maps to one PCA9685 output channel.
ZONE_CHANNELS = {
    "top_left": 0,
    "top_right": 1,
    "bottom_left": 2,
    "bottom_right": 3,
}

# PCA9685 duty cycle is a 16-bit value; full on is 0xFFFF.
MAX_DUTY_16 = 0xFFFF


try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
except Exception:
    board = None
    busio = None
    PCA9685 = None


def intensity_to_duty16(intensity: float) -> int:
    """Convert a 0-100% intensity to a PCA9685 16-bit duty cycle (0..0xFFFF)."""
    fraction = clamp_intensity(intensity) / 100.0
    return int(round(fraction * MAX_DUTY_16))


class PCA9685LightController:
    def __init__(
        self,
        zone_channels: Optional[Mapping[str, int]] = None,
        frequency_hz: int = 1000,
        start_intensity: float = 0.0,
        verbose_stub: bool = True,
        i2c_address: int = 0x40,
    ) -> None:
        self.zone_channels = dict(zone_channels or ZONE_CHANNELS)
        self.frequency_hz = frequency_hz
        self.start_intensity = clamp_intensity(start_intensity)
        self.verbose_stub = verbose_stub
        self.i2c_address = i2c_address
        # Library missing => stub from the start. The I2C bus may still be
        # unavailable even when the library imports; that is caught in setup().
        self.is_stub = PCA9685 is None
        self._pca = None
        self._i2c = None
        self._last_intensities = {zone: self.start_intensity for zone in self.zone_channels}

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def setup(self) -> None:
        if self.is_stub:
            self._print_stub_setup("adafruit_pca9685 is unavailable")
            return

        try:
            self._i2c = busio.I2C(board.SCL, board.SDA)
            self._pca = PCA9685(self._i2c, address=self.i2c_address)
            self._pca.frequency = self.frequency_hz
        except Exception as error:
            # No I2C bus / no device on this machine: degrade to the stub so the
            # rest of the pipeline still runs.
            self.is_stub = True
            self._pca = None
            self._i2c = None
            self._print_stub_setup(f"I2C bus/PCA9685 unavailable ({error})")
            return

        for zone in self.zone_channels:
            self.set_zone(zone, self.start_intensity)

    def _print_stub_setup(self, reason: str) -> None:
        if not self.verbose_stub:
            return
        print(
            f"[PCA9685 STUB] {reason} on {platform.system()}; "
            "using print-only PWM simulation."
        )
        for zone, channel in self.zone_channels.items():
            duty16 = intensity_to_duty16(self.start_intensity)
            print(
                f"[PCA9685 STUB] setup zone={zone} channel={channel} "
                f"frequency={self.frequency_hz}Hz duty={self.start_intensity:.1f}% "
                f"duty16=0x{duty16:04X}"
            )

    def set_zone(self, zone_name: str, intensity: float) -> None:
        zone = normalize_zone_name(zone_name)
        duty_cycle = clamp_intensity(intensity)
        self._last_intensities[zone] = duty_cycle
        channel = self.zone_channels[zone]
        duty16 = intensity_to_duty16(duty_cycle)

        if self.is_stub:
            if self.verbose_stub:
                print(
                    f"[PCA9685 STUB] zone={zone} channel={channel} "
                    f"duty={duty_cycle:.1f}% duty16=0x{duty16:04X}"
                )
            return

        self._pca.channels[channel].duty_cycle = duty16

    def apply_zone_intensities(self, zone_intensities: Mapping[str, float]) -> Dict[str, float]:
        normalized = {}
        for zone_name, intensity in zone_intensities.items():
            zone = normalize_zone_name(zone_name)
            normalized[zone] = clamp_intensity(intensity)

        for zone in self.zone_channels:
            if zone in normalized:
                self.set_zone(zone, normalized[zone])

        return self.last_intensities

    def set_all(self, intensity: float) -> Dict[str, float]:
        duty_cycle = clamp_intensity(intensity)
        for zone in self.zone_channels:
            self.set_zone(zone, duty_cycle)
        return self.last_intensities

    @property
    def last_intensities(self) -> Dict[str, float]:
        return self._last_intensities.copy()

    def off(self) -> None:
        self.set_all(0.0)

    def cleanup(self) -> None:
        if self.is_stub:
            if self.verbose_stub:
                print("[PCA9685 STUB] cleanup")
            return

        for channel in self.zone_channels.values():
            self._pca.channels[channel].duty_cycle = 0
        self._pca.deinit()
        if self._i2c is not None:
            try:
                self._i2c.deinit()
            except Exception:
                pass
        self._pca = None
        self._i2c = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="PCA9685 I2C PWM controller for four-zone surgical lighting."
    )
    parser.add_argument(
        "--all",
        type=float,
        default=None,
        help="Set all zones to one duty cycle percentage.",
    )
    parser.add_argument(
        "--zone",
        action="append",
        default=[],
        help="Set one zone using zone=value. Example: --zone TL=70 --zone BR=40",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.0,
        help="How long to hold the test output before cleanup.",
    )
    parser.add_argument(
        "--frequency",
        type=int,
        default=1000,
        help="PWM frequency in Hz (1000 avoids flicker/camera beating).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    zone_values = parse_zone_values(args.zone)

    with PCA9685LightController(frequency_hz=args.frequency) as controller:
        if args.all is not None:
            controller.set_all(args.all)
        if zone_values:
            controller.apply_zone_intensities(zone_values)
        if args.all is None and not zone_values:
            controller.apply_zone_intensities(
                {
                    "top_left": 25,
                    "top_right": 50,
                    "bottom_left": 75,
                    "bottom_right": 100,
                }
            )
        time.sleep(max(0.0, args.hold_seconds))


if __name__ == "__main__":
    main()
