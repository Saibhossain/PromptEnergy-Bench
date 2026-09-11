"""Metadata and directory management subsystem.

Enforces canonical path structure:
results/{experiment_name}/{normalized_device_name}/{timestamp}/
and generates complete provenance records.
"""

import datetime
import json
import os
import platform
import subprocess
import sys
from typing import Dict, Any, Optional, Tuple


def generate_timestamp() -> str:
    """Generates a standardized run timestamp: YYYY-MM-DD_HH-MM-SS."""
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def generate_run_id(experiment_name: str, normalized_device_name: str, timestamp: str) -> str:
    """Generates unique global run_id: {experiment_name}__{device_name}__{timestamp}."""
    return f"{experiment_name}__{normalized_device_name}__{timestamp}"


def get_git_metadata() -> Dict[str, Any]:
    """Extracts git commit hash and dirty status safely."""
    commit = None
    dirty = False
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        dirty = len(status_out) > 0
    except Exception:
        commit = None
        dirty = False
    return {"git_commit": commit, "git_dirty": dirty}


def get_software_metadata() -> Dict[str, Any]:
    """Gathers Python version, package versions, and platform details."""
    git_meta = get_git_metadata()
    packages: Dict[str, str] = {}
    
    for pkg in ["datasets", "codecarbon", "ollama", "pandas", "matplotlib", "seaborn", "psutil", "pyyaml", "scipy"]:
        try:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    ollama_ver = None
    try:
        out = subprocess.check_output(["ollama", "--version"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        ollama_ver = out.replace("ollama version is", "").strip()
    except Exception:
        pass

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "ollama_version": ollama_ver,
        "git_commit": git_meta["git_commit"],
        "git_dirty": git_meta["git_dirty"],
        "package_versions": packages
    }


def build_experiment_paths(
    experiment_name: str,
    normalized_device_name: str,
    timestamp: str,
    results_root: str = "results"
) -> Dict[str, str]:
    """Constructs and creates canonical directory hierarchy."""
    run_dir = os.path.join(results_root, experiment_name, normalized_device_name, timestamp)
    plots_dir = os.path.join(run_dir, "plots")
    logs_dir = os.path.join(run_dir, "logs")
    tables_dir = os.path.join(run_dir, "tables")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    return {
        "run_dir": run_dir,
        "metadata_file": os.path.join(run_dir, "metadata.json"),
        "config_file": os.path.join(run_dir, "config.json"),
        "results_file": os.path.join(run_dir, "results.jsonl"),
        "summary_file": os.path.join(run_dir, "summary.json"),
        "plots_dir": plots_dir,
        "logs_dir": logs_dir,
        "tables_dir": tables_dir,
        "log_file": os.path.join(logs_dir, "experiment.log")
    }


def save_metadata_json(file_path: str, metadata: Dict[str, Any]) -> None:
    """Atomically saves metadata.json."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def save_config_json(file_path: str, config: Dict[str, Any]) -> None:
    """Atomically saves config.json."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


def save_summary_json(file_path: str, summary: Dict[str, Any]) -> None:
    """Atomically saves summary.json."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
