"""Base Experiment Orchestrator for PromptEnergy-Bench.

Handles lifecycle management, pre-flight directory/metadata initialization,
device questionnaire integration, warmup execution, checkpoint deduplication,
robust error containment, and summary output.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple

from src.backends import get_backend
from src.backends.base import ModelBackend, InferenceOutput
from src.data.gsm8k import load_gsm8k, GSM8KRecord
from src.evaluation.gsm8k_evaluator import GSM8KEvaluator
from src.evaluation.metrics import compute_experiment_metrics
from src.infrastructure.checkpoint import (
    CheckpointManager,
    compute_condition_key,
    validate_resume_directory
)
from src.infrastructure.device import collect_device_info, normalize_device_name
from src.infrastructure.logging import setup_experiment_logging
from src.infrastructure.metadata import (
    generate_timestamp,
    generate_run_id,
    build_experiment_paths,
    save_metadata_json,
    save_config_json,
    save_summary_json,
    get_software_metadata
)
from src.monitoring.energy import get_energy_monitor, EnergyMonitor


class BaseExperiment(ABC):
    """Abstract base runner for all experiments."""

    def __init__(
        self,
        experiment_name: str,
        cli_args: Optional[Dict[str, Any]] = None,
        interactive: bool = False
    ):
        self.experiment_name = experiment_name
        self.cli_args = cli_args or {}
        self.interactive = interactive

        self.resume_dir = self.cli_args.get("resume")
        self.is_resumed = self.resume_dir is not None

        # Setup experiment configuration and directories
        self._initialize_run()

    def _initialize_run(self) -> None:
        """Sets up run directory, metadata, config, logging, and checkpoint manager."""
        if self.is_resumed:
            # Resuming an existing run
            self.paths = {
                "run_dir": self.resume_dir,
                "metadata_file": os.path.join(self.resume_dir, "metadata.json"),
                "config_file": os.path.join(self.resume_dir, "config.json"),
                "results_file": os.path.join(self.resume_dir, "results.jsonl"),
                "summary_file": os.path.join(self.resume_dir, "summary.json"),
                "plots_dir": os.path.join(self.resume_dir, "plots"),
                "logs_dir": os.path.join(self.resume_dir, "logs"),
                "log_file": os.path.join(self.resume_dir, "logs", "experiment.log")
            }
            # Verify and load config
            self.config, self.metadata = validate_resume_directory(
                self.resume_dir,
                expected_experiment_name=self.experiment_name,
                expected_model=self.cli_args.get("model", ""),
                expected_device_name=self.cli_args.get("device_name", "")
            )
            self.run_id = self.metadata["run_id"]
            self.normalized_device = self.metadata["device"]["normalized_name"]
            self.timestamp = self.metadata["timestamp"]
            self.eval_size = self.config["dataset"]["evaluation_size"]
            self.warmups = self.config["warmups"]
            self.repetitions = self.config["repetitions"]
            self.model_name = self.config["model"]["name"]
            self.operator = self.config["model"]["backend"]
            self.format = self.config["model"]["format"]
            self.energy_mode = self.metadata.get("measurement", {}).get("energy_method", "automatic")
        else:
            # Collect device and user info
            info = collect_device_info(interactive=self.interactive, cli_args=self.cli_args)
            self.device_info = info["device"]
            self.backend_info = info["backend"]
            self.dataset_info = info["dataset"]
            self.exec_info = info["execution"]

            self.normalized_device = self.device_info["normalized_name"]
            self.timestamp = generate_timestamp()
            self.run_id = generate_run_id(self.experiment_name, self.normalized_device, self.timestamp)

            self.eval_size = self.dataset_info["evaluation_size"]
            self.warmups = self.exec_info["warmups"]
            self.repetitions = self.exec_info["repetitions"]
            self.model_name = self.backend_info["model_name"]
            self.operator = self.backend_info["operator"]
            self.format = self.backend_info["model_format"]
            self.energy_mode = self.exec_info["energy_mode"]

            # Create paths
            self.paths = build_experiment_paths(
                experiment_name=self.experiment_name,
                normalized_device_name=self.normalized_device,
                timestamp=self.timestamp
            )

            # Build metadata and config
            software_meta = get_software_metadata()
            self.metadata = {
                "run_id": self.run_id,
                "experiment_name": self.experiment_name,
                "timestamp": self.timestamp,
                "device": self.device_info,
                "backend": self.backend_info,
                "dataset": self.dataset_info,
                "measurement": {
                    "energy_method": self.energy_mode,
                    "energy_quality": "estimated" if "estimated" in self.energy_mode else "hardware_reported",
                    "token_method": "backend_usage",
                    "latency_method": "perf_counter"
                },
                "software": software_meta
            }

            self.config = {
                "run_id": self.run_id,
                "experiment_name": self.experiment_name,
                "model": {
                    "name": self.model_name,
                    "backend": self.operator,
                    "format": self.format
                },
                "dataset": {
                    "name": "gsm8k",
                    "evaluation_split": "test",
                    "context_source_split": "train",
                    "evaluation_size": self.eval_size
                },
                "sampling": {
                    "temperature": float(self.cli_args.get("temperature", 0.0)),
                    "seed": int(self.cli_args.get("seed", 42)),
                    "top_p": float(self.cli_args.get("top_p", 1.0)),
                    "max_tokens": int(self.cli_args.get("max_tokens", 1024))
                },
                "warmups": self.warmups,
                "repetitions": self.repetitions,
                "validation": bool(self.cli_args.get("validation", False) or self.eval_size == 50)
            }

            # Save initial metadata and config BEFORE inference starts
            save_metadata_json(self.paths["metadata_file"], self.metadata)
            save_config_json(self.paths["config_file"], self.config)

        # Initialize logging and checkpointing
        self.logger = setup_experiment_logging(self.paths["log_file"])
        self.checkpoint_mgr = CheckpointManager(self.paths["results_file"])

        self.logger.info(f"Initialized Experiment: {self.experiment_name}")
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Run Directory: {self.paths['run_dir']}")
        self.logger.info(f"Model: {self.model_name} (Backend: {self.operator}, Format: {self.format})")
        self.logger.info(f"Device: {self.normalized_device}")
        self.logger.info(f"Evaluation Size: {self.eval_size} (Evaluation Split: TEST)")

        # Instantiate backend and energy monitor
        self.backend: ModelBackend = get_backend(operator=self.operator, model_name=self.model_name)
        self.energy_monitor: EnergyMonitor = get_energy_monitor(mode=self.energy_mode)

    def run_warmup(self, warmup_messages: List[Dict[str, str]]) -> None:
        """Executes warm-up requests without logging to results."""
        if self.warmups <= 0:
            return

        self.logger.info(f"Executing {self.warmups} warmup run(s)...")
        for w in range(1, self.warmups + 1):
            try:
                _ = self.backend.generate(
                    messages=warmup_messages,
                    temperature=0.0,
                    max_tokens=64,
                    stream=False
                )
                self.logger.info(f"Warmup {w}/{self.warmups} completed.")
            except Exception as e:
                self.logger.warning(f"Warmup {w} failed: {e}")

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Executes the specific experiment protocol."""
        pass

    def finalize(self) -> Dict[str, Any]:
        """Calculates final aggregate metrics and saves summary.json."""
        records = self.checkpoint_mgr.completed_records
        metrics = compute_experiment_metrics(records)

        summary = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "device": self.normalized_device,
            "model": self.model_name,
            "backend": self.operator,
            "evaluation_size": self.eval_size,
            "validation": bool(self.cli_args.get("validation", False) or self.eval_size == 50),
            "full_benchmark": self.eval_size == "full",
            "metrics": metrics
        }

        save_summary_json(self.paths["summary_file"], summary)
        self.logger.info(f"Experiment completed. Summary saved to {self.paths['summary_file']}")
        return summary
