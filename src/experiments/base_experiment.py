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
from src.monitoring.resources import BackgroundResourceMonitor, ResourceReading


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

            # Load generation configuration
            gen_cfg_path = self.cli_args.get("generation_config") or "configs/generation.yaml"
            gen_cfg = {}
            if os.path.exists(gen_cfg_path):
                import yaml
                try:
                    with open(gen_cfg_path, "r", encoding="utf-8") as f:
                        raw_data = yaml.safe_load(f)
                        if isinstance(raw_data, dict) and "generation" in raw_data:
                            gen_cfg = raw_data["generation"]
                        elif isinstance(raw_data, dict):
                            gen_cfg = raw_data
                except Exception as e:
                    self.logger.warning(f"Could not parse generation config {gen_cfg_path}: {e}")

            default_max_tok = gen_cfg.get("default_max_tokens", 512)
            if self.cli_args.get("max_output_tokens") is not None:
                default_max_tok = int(self.cli_args["max_output_tokens"])
            elif self.cli_args.get("max_tokens") is not None and self.cli_args.get("max_tokens") != 1024:
                default_max_tok = int(self.cli_args["max_tokens"])

            strategy_max_tok = gen_cfg.get("strategy_max_tokens", {
                "zero_shot_direct": 256,
                "few_shot_3": 256,
                "zero_shot_cot": 512,
                "short_cot": 512,
                "long_cot": 1024
            })

            generation_metadata = {
                "default_max_tokens": default_max_tok,
                "strategy_max_tokens": strategy_max_tok
            }

            self.metadata["generation"] = generation_metadata

            self.config = {
                "run_id": self.run_id,
                "experiment_name": self.experiment_name,
                "model": {
                    "name": self.model_name,
                    "backend": self.operator,
                    "format": self.format
                },
                "dataset": {
                    "name": str(self.cli_args.get("dataset") or self.cli_args.get("dataset_name") or "gsm8k"),
                    "evaluation_split": str(self.cli_args.get("evaluation_split") or "test"),
                    "context_source_split": str(self.cli_args.get("context_source_split") or "train"),
                    "evaluation_size": self.eval_size
                },
                "sampling": {
                    "temperature": float(self.cli_args.get("temperature", 0.0)),
                    "seed": int(self.cli_args.get("seed", 42)),
                    "top_p": float(self.cli_args.get("top_p", 1.0)),
                    "max_tokens": default_max_tok
                },
                "generation": generation_metadata,
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

        # Instantiate backend and energy/resource monitors
        self.backend: ModelBackend = get_backend(operator=self.operator, model_name=self.model_name)
        self.energy_monitor: EnergyMonitor = get_energy_monitor(mode=self.energy_mode)
        self.resource_monitor: BackgroundResourceMonitor = BackgroundResourceMonitor(poll_interval_ms=50)

    def run_warmup(self, warmup_messages: Any) -> None:
        """Executes warm-up requests without logging to results."""
        if self.warmups <= 0:
            return

        if isinstance(warmup_messages, str):
            warmup_messages = [{"role": "user", "content": warmup_messages}]

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

    execute_warmup = run_warmup

    def get_max_tokens_for_strategy(self, strategy_name: str) -> int:
        """Retrieves configured max_tokens for a given prompting strategy.
        
        Falls back to strategy_max_tokens dict, default_max_tokens, or sampling max_tokens.
        """
        # 1. Check generation metadata
        gen_meta = self.metadata.get("generation", {})
        if isinstance(gen_meta, dict):
            strat_map = gen_meta.get("strategy_max_tokens", {})
            if isinstance(strat_map, dict) and strategy_name in strat_map:
                return int(strat_map[strategy_name])
            if "default_max_tokens" in gen_meta:
                return int(gen_meta["default_max_tokens"])

        # 2. Check config generation
        if isinstance(self.config, dict):
            cfg_gen = self.config.get("generation", {})
            if isinstance(cfg_gen, dict):
                cfg_strat_map = cfg_gen.get("strategy_max_tokens", {})
                if isinstance(cfg_strat_map, dict) and strategy_name in cfg_strat_map:
                    return int(cfg_strat_map[strategy_name])
                if "default_max_tokens" in cfg_gen:
                    return int(cfg_gen["default_max_tokens"])

            # 3. Check sampling config
            sampling = self.config.get("sampling", {})
            if isinstance(sampling, dict) and "max_tokens" in sampling:
                return int(sampling["max_tokens"])

        return 512

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Executes the specific experiment protocol."""
        pass

    def finalize(self) -> Dict[str, Any]:
        """Calculates final aggregate metrics, generates publication tables, and saves summary.json."""
        from src.analysis.tables import generate_all_tables

        records = self.checkpoint_mgr.completed_records
        metrics = compute_experiment_metrics(records)

        # Detect energy interpretation
        meas_method = self.metadata.get("measurement", {}).get("energy_method", self.energy_mode)
        if "apple" in str(meas_method).lower() or "estimate" in str(meas_method).lower():
            energy_interp = "software-estimated system energy"
        else:
            energy_interp = "hardware-reported energy"

        num_examples = len(set(r.get("sample_id") for r in records if r.get("sample_id")))
        if num_examples == 0:
            num_examples = self.eval_size

        num_strategies = len(set(r.get("strategy") for r in records if r.get("strategy")))
        if num_strategies == 0 and hasattr(self, "strategies"):
            num_strategies = len(self.strategies)

        summary = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "device": self.normalized_device,
            "model": self.model_name,
            "backend": self.operator,
            "evaluation_split": "test",
            "evaluation_examples": num_examples,
            "strategies": num_strategies,
            "repetitions": self.repetitions,
            "validation": bool(self.cli_args.get("validation", False) or self.eval_size == 50),
            "full_benchmark": self.eval_size == "full",
            "generation": self.metadata.get("generation", {}),
            "energy_interpretation": energy_interp,
            "metrics": metrics,
            "strategy_metrics": metrics.get("strategy_summaries", {})
        }

        save_summary_json(self.paths["summary_file"], summary)
        self.logger.info(f"Experiment completed. Summary saved to {self.paths['summary_file']}")

        # Ensure raw_results.jsonl is present as alias/copy of results.jsonl (Section 2.D)
        raw_results_path = os.path.join(self.paths["run_dir"], "raw_results.jsonl")
        if os.path.exists(self.paths["results_file"]) and not os.path.exists(raw_results_path):
            try:
                import shutil
                shutil.copyfile(self.paths["results_file"], raw_results_path)
            except Exception:
                pass

        # Generate publication tables (CSV, Markdown, LaTeX)
        try:
            tables_dir = os.path.join(self.paths["run_dir"], "tables")
            generate_all_tables(records, self.metadata, self.config, summary, tables_dir)
            self.logger.info(f"Generated publication tables in {tables_dir}")
            # Ensure summary.csv exists in root of run directory as well
            root_summary_csv = os.path.join(self.paths["run_dir"], "summary.csv")
            src_csv = os.path.join(tables_dir, "strategy_comparison.csv")
            if os.path.exists(src_csv):
                import shutil
                shutil.copyfile(src_csv, root_summary_csv)
        except Exception as e:
            self.logger.warning(f"Failed to generate tables: {e}")

        # Generate publication plots (PNG @ 300 DPI, PDF, SVG)
        try:
            from visual.plot_results import generate_all_plots_for_run
            generate_all_plots_for_run(self.paths["run_dir"])
            self.logger.info(f"Generated publication plots in {self.paths['run_dir']}")
        except Exception as e:
            self.logger.warning(f"Failed to generate plots: {e}")

        return summary
