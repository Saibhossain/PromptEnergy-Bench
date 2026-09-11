"""Resource utilization sampling (CPU, RAM, GPU)."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import psutil


@dataclass
class ResourceSnapshot:
    cpu_percent: float
    ram_used_gb: float
    ram_percent: float
    gpu_util_percent: Optional[float] = None
    gpu_mem_used_gb: Optional[float] = None


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
