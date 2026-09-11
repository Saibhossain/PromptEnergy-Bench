"""Energy Monitoring Subsystem for PromptEnergy-Bench.

Provides independent, non-faked energy tracking across diverse hardware.
Strictly distinguishes between direct hardware measurements, software models,
and unavailable meters.
"""

import os
import platform
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List


@dataclass
class EnergyReading:
    energy_total_j: Optional[float] = None
    energy_prefill_j: Optional[float] = None
    energy_decode_j: Optional[float] = None
    energy_overhead_j: Optional[float] = None
    energy_net_j: Optional[float] = None
    idle_power_w: Optional[float] = None
    active_power_w: Optional[float] = None
    energy_measurement_method: str = "unavailable"
    energy_quality: str = "unavailable"
    energy_measurement_level: str = "unknown"
    energy_status: str = "unavailable"  # measured, estimated, unavailable

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EnergyMonitor(ABC):
    """Abstract base class for energy monitors."""

    def __init__(self):
        self.is_monitoring = False
        self.idle_power_w: Optional[float] = None

    @abstractmethod
    def start(self, phase: str = "total") -> None:
        pass

    @abstractmethod
    def stop(self, phase: str = "total") -> EnergyReading:
        pass

    def measure_idle_power(self, duration_secs: float = 2.0) -> Optional[float]:
        """Measures system baseline idle power before workload execution."""
        return None


class NullEnergyMonitor(EnergyMonitor):
    """Fallback monitor when no reliable hardware energy meter is available.
    
    Never invents or fabricates fake numbers. Clearly outputs null.
    """

    def start(self, phase: str = "total") -> None:
        self.is_monitoring = True

    def stop(self, phase: str = "total") -> EnergyReading:
        self.is_monitoring = False
        return EnergyReading(
            energy_total_j=None,
            energy_prefill_j=None,
            energy_decode_j=None,
            energy_overhead_j=None,
            energy_net_j=None,
            idle_power_w=None,
            active_power_w=None,
            energy_measurement_method="unavailable",
            energy_quality="unavailable",
            energy_measurement_level="unknown",
            energy_status="unavailable"
        )


class CodeCarbonMonitor(EnergyMonitor):
    """CodeCarbon integration with Joules conversion (1 kWh = 3.6e6 Joules).
    
    Labels measurement as software_estimate or estimated_system according to platform capabilities.
    """

    def __init__(self, country_iso_code: str = "USA"):
        super().__init__()
        self.country_iso_code = country_iso_code
        self.tracker = None
        self.start_time = 0.0

    def start(self, phase: str = "total") -> None:
        try:
            from codecarbon import OfflineEmissionsTracker
            # Suppress excessive logging
            os.environ["CODECARBON_LOG_LEVEL"] = "WARNING"
            self.tracker = OfflineEmissionsTracker(
                country_iso_code=self.country_iso_code,
                measure_power_secs=0.1,
                save_to_file=False
            )
            self.tracker.start()
            self.start_time = time.perf_counter()
            self.is_monitoring = True
        except Exception as e:
            self.tracker = None
            self.is_monitoring = False

    def stop(self, phase: str = "total") -> EnergyReading:
        elapsed = time.perf_counter() - self.start_time
        if not self.is_monitoring or self.tracker is None:
            return NullEnergyMonitor().stop()

        try:
            emissions = self.tracker.stop()
            data = getattr(self.tracker, "final_emissions_data", None)
            energy_kwh = getattr(data, "energy_consumed", 0.0) if data else 0.0

            # Convert kWh to Joules
            energy_j = float(energy_kwh) * 3.6e6

            # Compute average active power: P = E / t
            active_power = (energy_j / elapsed) if elapsed > 0 else 0.0

            # Net energy if idle baseline is known
            net_energy = None
            if self.idle_power_w is not None and elapsed > 0:
                idle_energy = self.idle_power_w * elapsed
                net_energy = max(0.0, energy_j - idle_energy)

            self.is_monitoring = False
            return EnergyReading(
                energy_total_j=round(energy_j, 4),
                energy_prefill_j=None,
                energy_decode_j=None,
                energy_overhead_j=None,
                energy_net_j=round(net_energy, 4) if net_energy is not None else None,
                idle_power_w=round(self.idle_power_w, 2) if self.idle_power_w else None,
                active_power_w=round(active_power, 2),
                energy_measurement_method="codecarbon_estimated",
                energy_quality="software_estimate",
                energy_measurement_level="estimated_system",
                energy_status="estimated"
            )
        except Exception:
            self.is_monitoring = False
            return NullEnergyMonitor().stop()


class NvidiaGPUMonitor(EnergyMonitor):
    """High-frequency NVIDIA GPU power polling using NVML / nvidia-ml-py."""

    def __init__(self, device_index: int = 0, poll_interval_ms: int = 50):
        super().__init__()
        self.device_index = device_index
        self.poll_interval = poll_interval_ms / 1000.0
        self.power_samples: List[float] = []
        self.sample_times: List[float] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.nvml_handle = None

    def _init_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception:
            self.nvml_handle = None

    def _poll_power(self):
        import pynvml
        while not self._stop_event.is_set():
            try:
                # Returns milliwatts
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle)
                now = time.perf_counter()
                self.power_samples.append(power_mw / 1000.0)
                self.sample_times.append(now)
            except Exception:
                pass
            time.sleep(self.poll_interval)

    def start(self, phase: str = "total") -> None:
        self._init_nvml()
        if not self.nvml_handle:
            return

        self.power_samples = []
        self.sample_times = []
        self._stop_event.clear()
        self.is_monitoring = True
        self._thread = threading.Thread(target=self._poll_power, daemon=True)
        self._thread.start()

    def stop(self, phase: str = "total") -> EnergyReading:
        if not self.is_monitoring or not self.nvml_handle:
            return NullEnergyMonitor().stop()

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self.is_monitoring = False

        if len(self.power_samples) < 2:
            return NullEnergyMonitor().stop()

        # Numerical trapezoidal integration of power samples over time: E = integral P dt
        total_energy_j = 0.0
        for i in range(1, len(self.power_samples)):
            dt = self.sample_times[i] - self.sample_times[i - 1]
            avg_p = (self.power_samples[i] + self.power_samples[i - 1]) / 2.0
            total_energy_j += avg_p * dt

        avg_power = sum(self.power_samples) / len(self.power_samples)

        return EnergyReading(
            energy_total_j=round(total_energy_j, 4),
            energy_prefill_j=None,
            energy_decode_j=None,
            energy_overhead_j=None,
            energy_net_j=None,
            idle_power_w=round(self.idle_power_w, 2) if self.idle_power_w else None,
            active_power_w=round(avg_power, 2),
            energy_measurement_method="nvidia_nvml",
            energy_quality="hardware_reported",
            energy_measurement_level="gpu_only",
            energy_status="measured"
        )


class ApplePowerMonitor(CodeCarbonMonitor):
    """Adapter for Apple Silicon Macs using estimated power via CodeCarbon."""

    def stop(self, phase: str = "total") -> EnergyReading:
        reading = super().stop(phase=phase)
        if reading.energy_total_j is not None:
            reading.energy_measurement_method = "apple_estimated"
            reading.energy_quality = "software_estimate"
            reading.energy_measurement_level = "estimated_system"
        return reading


class PhysicalMeterMonitor(EnergyMonitor):
    """Interface for external smart power meters (Shelly, WattsUp, Tapo)."""

    def __init__(self, api_url: Optional[str] = None):
        super().__init__()
        self.api_url = api_url

    def start(self, phase: str = "total") -> None:
        self.is_monitoring = True

    def stop(self, phase: str = "total") -> EnergyReading:
        self.is_monitoring = False
        # External hardware reading
        return EnergyReading(
            energy_total_j=None,
            energy_measurement_method="physical_meter",
            energy_quality="direct_measurement",
            energy_measurement_level="whole_system",
            energy_status="unavailable"
        )


def get_energy_monitor(mode: str = "automatic") -> EnergyMonitor:
    """Factory to return appropriate energy monitor based on hardware and user choice."""
    mode = mode.lower().strip()

    if mode in ("none", "unavailable", "disabled"):
        return NullEnergyMonitor()
    elif mode in ("physical", "meter", "physical_meter"):
        return PhysicalMeterMonitor()
    elif mode in ("nvidia", "nvml", "nvidia_smi"):
        return NvidiaGPUMonitor()
    elif mode in ("apple", "apple_power", "apple_estimated"):
        return ApplePowerMonitor()
    elif mode in ("codecarbon", "software", "software_estimate"):
        return CodeCarbonMonitor()
    elif mode == "automatic":
        # Auto-detect best supported monitor
        if platform.system() == "Darwin":
            return ApplePowerMonitor()
        try:
            import pynvml
            pynvml.nvmlInit()
            return NvidiaGPUMonitor()
        except Exception:
            pass
        return CodeCarbonMonitor()
    else:
        return CodeCarbonMonitor()
