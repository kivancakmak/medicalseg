import argparse
import platform
import time
from typing import Dict, Mapping, Optional


ZONE_PINS = {
    "top_left": 32,
    "top_right": 33,
    "bottom_left": 15,
    "bottom_right": 18,
}

ZONE_ALIASES = {
    "tl": "top_left",
    "tr": "top_right",
    "bl": "bottom_left",
    "br": "bottom_right",
    "top_left": "top_left",
    "top_right": "top_right",
    "bottom_left": "bottom_left",
    "bottom_right": "bottom_right",
}


try:
    import Jetson.GPIO as GPIO
except Exception:
    GPIO = None


def normalize_zone_name(zone_name: str) -> str:
    key = zone_name.strip().lower()
    if key not in ZONE_ALIASES:
        valid = ", ".join(sorted(ZONE_ALIASES))
        raise KeyError(f"Unknown light zone '{zone_name}'. Valid zones: {valid}")
    return ZONE_ALIASES[key]


def clamp_intensity(intensity: float) -> float:
    return max(0.0, min(100.0, float(intensity)))


class JetsonPWMController:
    def __init__(
        self,
        zone_pins: Optional[Mapping[str, int]] = None,
        frequency_hz: int = 50,
        start_intensity: float = 0.0,
        verbose_stub: bool = True,
    ) -> None:
        self.zone_pins = dict(zone_pins or ZONE_PINS)
        self.frequency_hz = frequency_hz
        self.start_intensity = clamp_intensity(start_intensity)
        self.verbose_stub = verbose_stub
        self.is_stub = GPIO is None
        self._pwm = {}
        self._last_intensities = {zone: self.start_intensity for zone in self.zone_pins}

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def setup(self) -> None:
        if self.is_stub:
            if self.verbose_stub:
                print(
                    "[PWM STUB] Jetson.GPIO is unavailable on "
                    f"{platform.system()}; using print-only PWM simulation."
                )
                for zone, pin in self.zone_pins.items():
                    print(
                        f"[PWM STUB] setup zone={zone} pin={pin} "
                        f"frequency={self.frequency_hz}Hz duty={self.start_intensity:.1f}%"
                    )
            return

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        for zone, pin in self.zone_pins.items():
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            pwm = GPIO.PWM(pin, self.frequency_hz)
            pwm.start(self.start_intensity)
            self._pwm[zone] = pwm

    def set_zone(self, zone_name: str, intensity: float) -> None:
        zone = normalize_zone_name(zone_name)
        duty_cycle = clamp_intensity(intensity)
        self._last_intensities[zone] = duty_cycle

        if self.is_stub:
            if self.verbose_stub:
                print(
                    f"[PWM STUB] zone={zone} pin={self.zone_pins[zone]} "
                    f"duty={duty_cycle:.1f}%"
                )
            return

        self._pwm[zone].ChangeDutyCycle(duty_cycle)

    def apply_zone_intensities(self, zone_intensities: Mapping[str, float]) -> Dict[str, float]:
        normalized = {}
        for zone_name, intensity in zone_intensities.items():
            zone = normalize_zone_name(zone_name)
            normalized[zone] = clamp_intensity(intensity)

        for zone in self.zone_pins:
            if zone in normalized:
                self.set_zone(zone, normalized[zone])

        return self.last_intensities

    def set_all(self, intensity: float) -> Dict[str, float]:
        duty_cycle = clamp_intensity(intensity)
        for zone in self.zone_pins:
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
                print("[PWM STUB] cleanup")
            return

        for pwm in self._pwm.values():
            pwm.ChangeDutyCycle(0.0)
            pwm.stop()
        self._pwm.clear()
        GPIO.cleanup()


def parse_zone_values(raw_values) -> Dict[str, float]:
    if not raw_values:
        return {}

    parsed = {}
    for raw_value in raw_values:
        if "=" not in raw_value:
            raise ValueError(f"Expected zone=value format, got: {raw_value}")
        zone_name, value = raw_value.split("=", 1)
        parsed[normalize_zone_name(zone_name)] = clamp_intensity(float(value))
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Jetson.GPIO PWM controller for four-zone surgical lighting."
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
        default=50,
        help="PWM frequency in Hz.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    zone_values = parse_zone_values(args.zone)

    with JetsonPWMController(frequency_hz=args.frequency) as controller:
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
