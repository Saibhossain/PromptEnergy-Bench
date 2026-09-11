"""Device profiling, questionnaire, and normalization subsystem."""

import os
import platform
import re
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import psutil


@dataclass
class DeviceProfile:
    name: str
    normalized_name: str
    os: str
    cpu: str
    gpu_available: bool
    gpu_count: int
    gpu_name: Optional[str]
    gpu_vram_gb: Optional[float]
    ram_gb: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_device_name(name: str) -> str:
    """Normalizes raw device string into safe filesystem slug.
    
    Rules:
    - lowercase
    - replace spaces with "_"
    - replace unsafe filesystem characters with "_"
    - collapse repeated underscores
    - never use raw user input directly as an unsafe path
    """
    if not name:
        return "unknown_device"
    clean = name.lower()
    # Replace unsafe characters and spaces with underscore
    clean = re.sub(r"[^\w\-.]", "_", clean)
    # Collapse multiple underscores
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "unknown_device"


def detect_device_specs() -> Dict[str, Any]:
    """Auto-detects host system specifications."""
    os_name = platform.system()
    if os_name == "Darwin":
        os_str = "macOS"
    elif os_name == "Windows":
        os_str = "Windows"
    else:
        os_str = "Linux"

    # CPU Brand Detection
    cpu_str = platform.processor() or "Unknown CPU"
    if os_name == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL)
            cpu_str = out.decode("utf-8").strip()
        except Exception:
            pass

    # RAM
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3))

    # GPU
    gpu_available = False
    gpu_count = 0
    gpu_name = None
    gpu_vram_gb = None

    # Check Apple Silicon GPU
    if os_name == "Darwin" and ("Apple" in cpu_str or platform.machine() == "arm64"):
        gpu_available = True
        gpu_count = 1
        gpu_name = f"{cpu_str} GPU (Metal Unified Memory)"
        gpu_vram_gb = ram_gb  # Unified memory architecture

    # Check NVIDIA
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count > 0:
            gpu_available = True
            gpu_count = count
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(h)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode("utf-8")
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            gpu_vram_gb = round(mem.total / (1024 ** 3), 1)
    except Exception:
        pass

    # Fallback device name suggestion
    if os_name == "Darwin" and "Apple" in cpu_str:
        dev_name = f"MacBook {cpu_str}" if "M" in cpu_str else f"Mac {cpu_str}"
    elif gpu_name and "NVIDIA" in str(gpu_name):
        dev_name = f"{os_str} {gpu_name} PC"
    else:
        dev_name = f"{os_str} {cpu_str} Machine"

    return {
        "device_name": dev_name,
        "os": os_str,
        "cpu": cpu_str,
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "gpu_vram_gb": gpu_vram_gb,
        "ram_gb": ram_gb
    }


def collect_device_info(interactive: bool = False, cli_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Collects complete device and execution configuration via questionnaire or CLI flags."""
    specs = detect_device_specs()
    cli = cli_args or {}

    def get_val(key: str, default: Any, prompt_text: str = "") -> Any:
        if cli.get(key) is not None:
            return cli[key]
        if not interactive:
            return default
        user_input = input(f"{prompt_text} [{default}]: ").strip()
        return user_input if user_input else default

    print("\n" + "=" * 60)
    print("DEVICE & EXPERIMENT QUESTIONNAIRE")
    print("=" * 60)

    # 1. Device name
    name = get_val("device_name", specs["device_name"], "1. Device name")
    norm_name = normalize_device_name(name)

    # 2. OS
    os_name = get_val("os", specs["os"], "2. Operating system")

    # 3. CPU
    cpu = get_val("cpu", specs["cpu"], "3. CPU")

    # 4. GPU Available
    gpu_avail_def = "yes" if specs["gpu_available"] else "no"
    gpu_avail_str = str(get_val("gpu", gpu_avail_def, "4. GPU available? (yes/no)")).lower()
    gpu_available = gpu_avail_str in ("yes", "y", "true", "1")

    # 5. GPU Details
    gpu_count = specs["gpu_count"]
    gpu_name = specs["gpu_name"]
    gpu_vram = specs["gpu_vram_gb"]
    if gpu_available:
        gpu_count = int(get_val("gpu_count", specs["gpu_count"] or 1, "5a. GPU count"))
        gpu_name = get_val("gpu_name", specs["gpu_name"] or "Generic GPU", "5b. GPU model(s)")
        gpu_vram = get_val("gpu_vram_gb", specs["gpu_vram_gb"], "5c. GPU VRAM (GB)")
        if gpu_vram is not None:
            try:
                gpu_vram = float(gpu_vram)
            except ValueError:
                gpu_vram = None

    # 6. System RAM
    ram_gb = int(get_val("ram_gb", specs["ram_gb"], "6. System RAM (GB)"))

    # 7. Model operator / backend
    operator = get_val("operator", "ollama", "7. Model operator/backend (ollama, openai, transformers, mlx, llama.cpp)").lower()

    # 8. Model format
    fmt = get_val("model_format", "mlx" if "Darwin" in platform.system() else "gguf", "8. Model format (mlx, gguf, f16, bf16, fp8, int8, int4, api, unknown)").lower()

    # 9. Model name
    model_name = get_val("model", "qwen3.5:0.8b-mlx", "9. Model name")

    # 10. Dataset evaluation size
    eval_size_raw = get_val("eval_size", "50", "10. Dataset evaluation size ('50' or 'full')")
    eval_size = 50 if str(eval_size_raw) == "50" else "full"

    # 11. Energy measurement mode
    energy_mode_def = "apple_estimated" if os_name == "macOS" else ("nvidia_nvml" if gpu_available and "NVIDIA" in str(gpu_name) else "codecarbon_estimated")
    energy_mode = get_val("energy_mode", energy_mode_def, "11. Energy mode (physical_meter, software_estimate, automatic, unavailable)")

    # 12. Warm-ups
    warmups = int(get_val("warmups", 3, "12. Warm-up runs"))

    # 13. Repetitions per condition
    repetitions = int(get_val("repetitions", 3, "13. Repetitions per condition"))

    print("=" * 60 + "\n")

    return {
        "device": {
            "name": name,
            "normalized_name": norm_name,
            "os": os_name,
            "cpu": cpu,
            "gpu_available": gpu_available,
            "gpu_count": gpu_count,
            "gpu_name": gpu_name,
            "gpu_vram_gb": gpu_vram,
            "ram_gb": ram_gb
        },
        "backend": {
            "operator": operator,
            "model_format": fmt,
            "model_name": model_name
        },
        "dataset": {
            "name": "GSM8K",
            "evaluation_split": "test",
            "context_source_split": "train",
            "evaluation_size": eval_size
        },
        "execution": {
            "warmups": warmups,
            "repetitions": repetitions,
            "energy_mode": energy_mode
        }
    }
