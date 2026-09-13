"""Resource utilization sampling and continuous monitoring (CPU, RAM, GPU)."""

import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import psutil


@dataclass
class ResourceSnapshot:
    cpu_percent: float
    ram_used_gb: float
    ram_percent: float
    gpu_util_percent: Optional[float] = None
    gpu_mem_used_gb: Optional[float] = None


@dataclass
class ResourceReading:
    cpu_percent_mean: Optional[float] = None
    cpu_percent_peak: Optional[float] = None
    ram_used_gb_mean: Optional[float] = None
    ram_used_gb_peak: Optional[float] = None
    ram_percent: Optional[float] = None
    cpu_cores_logical: int = 1
    cpu_cores_physical: int = 1
    gpu_util_percent: Optional[float] = None
    gpu_mem_used_gb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResourceSampler:
    """Samples instantaneous hardware resource consumption."""

    @staticmethod
    def sample() -> ResourceSnapshot:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        ram_used = round(mem.used / (1024 ** 3), 2)
        ram_pct = mem.percent

        gpu_pct = None
        gpu_mem = None

        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            gpu_pct = float(util.gpu)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(h)
            gpu_mem = round(mem_info.used / (1024 ** 3), 2)
        except Exception:
            pass

        return ResourceSnapshot(
            cpu_percent=cpu,
            ram_used_gb=ram_used,
            ram_percent=ram_pct,
            gpu_util_percent=gpu_pct,
            gpu_mem_used_gb=gpu_mem
        )


class BackgroundResourceMonitor:
    """Continuously samples CPU, RAM, and GPU usage in the background during inference."""

    def __init__(self, poll_interval_ms: int = 50):
        self.poll_interval = poll_interval_ms / 1000.0
        self.cpu_samples: List[float] = []
        self.ram_samples: List[float] = []
        self.ram_pct_samples: List[float] = []
        self.gpu_util_samples: List[float] = []
        self.gpu_mem_samples: List[float] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.is_monitoring = False

    def _poll(self):
        # Initial call to prime psutil
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

        while not self._stop_event.is_set():
            try:
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory()
                ram_gb = mem.used / (1024 ** 3)
                ram_pct = mem.percent

                self.cpu_samples.append(cpu)
                self.ram_samples.append(ram_gb)
                self.ram_pct_samples.append(ram_pct)

                try:
                    import pynvml
                    pynvml.nvmlInit()
                    h = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(h)
                    self.gpu_util_samples.append(float(util.gpu))
                    self.gpu_mem_samples.append(mem_info.used / (1024 ** 3))
                except Exception:
                    pass
            except Exception:
                pass
            time.sleep(self.poll_interval)

    def start(self) -> None:
        self.cpu_samples = []
        self.ram_samples = []
        self.ram_pct_samples = []
        self.gpu_util_samples = []
        self.gpu_mem_samples = []
        self._stop_event.clear()
        self.is_monitoring = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> ResourceReading:
        if not self.is_monitoring:
            snap = ResourceSampler.sample()
            return ResourceReading(
                cpu_percent_mean=round(snap.cpu_percent, 2),
                cpu_percent_peak=round(snap.cpu_percent, 2),
                ram_used_gb_mean=round(snap.ram_used_gb, 2),
                ram_used_gb_peak=round(snap.ram_used_gb, 2),
                ram_percent=round(snap.ram_percent, 2),
                cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
                cpu_cores_physical=psutil.cpu_count(logical=False) or 1,
                gpu_util_percent=snap.gpu_util_percent,
                gpu_mem_used_gb=snap.gpu_mem_used_gb
            )

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self.is_monitoring = False

        if not self.cpu_samples:
            snap = ResourceSampler.sample()
            return ResourceReading(
                cpu_percent_mean=round(snap.cpu_percent, 2),
                cpu_percent_peak=round(snap.cpu_percent, 2),
                ram_used_gb_mean=round(snap.ram_used_gb, 2),
                ram_used_gb_peak=round(snap.ram_used_gb, 2),
                ram_percent=round(snap.ram_percent, 2),
                cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
                cpu_cores_physical=psutil.cpu_count(logical=False) or 1,
                gpu_util_percent=snap.gpu_util_percent,
                gpu_mem_used_gb=snap.gpu_mem_used_gb
            )

        mean_cpu = sum(self.cpu_samples) / len(self.cpu_samples)
        peak_cpu = max(self.cpu_samples)
        mean_ram = sum(self.ram_samples) / len(self.ram_samples)
        peak_ram = max(self.ram_samples)
        mean_ram_pct = sum(self.ram_pct_samples) / len(self.ram_pct_samples)

        mean_gpu_util = (sum(self.gpu_util_samples) / len(self.gpu_util_samples)) if self.gpu_util_samples else None
        mean_gpu_mem = (sum(self.gpu_mem_samples) / len(self.gpu_mem_samples)) if self.gpu_mem_samples else None

        return ResourceReading(
            cpu_percent_mean=round(mean_cpu, 2),
            cpu_percent_peak=round(peak_cpu, 2),
            ram_used_gb_mean=round(mean_ram, 2),
            ram_used_gb_peak=round(peak_ram, 2),
            ram_percent=round(mean_ram_pct, 2),
            cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
            cpu_cores_physical=psutil.cpu_count(logical=False) or 1,
            gpu_util_percent=round(mean_gpu_util, 2) if mean_gpu_util is not None else None,
            gpu_mem_used_gb=round(mean_gpu_mem, 2) if mean_gpu_mem is not None else None
        )

