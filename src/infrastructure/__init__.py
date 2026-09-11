"""Infrastructure package: device configuration, metadata, checkpoints, and logging."""
from src.infrastructure.device import (
    DeviceProfile,
    normalize_device_name,
    detect_device_specs,
    collect_device_info
)
from src.infrastructure.metadata import (
    generate_timestamp,
    generate_run_id,
    build_experiment_paths,
    save_metadata_json,
    save_config_json,
    save_summary_json,
    get_git_metadata
)
from src.infrastructure.checkpoint import CheckpointManager
from src.infrastructure.logging import setup_experiment_logging

__all__ = [
    "DeviceProfile",
    "normalize_device_name",
    "detect_device_specs",
    "collect_device_info",
    "generate_timestamp",
    "generate_run_id",
    "build_experiment_paths",
    "save_metadata_json",
    "save_config_json",
    "save_summary_json",
    "get_git_metadata",
    "CheckpointManager",
    "setup_experiment_logging"
]
