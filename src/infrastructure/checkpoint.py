"""Checkpointing and resume subsystem with serialized condition keys."""

import hashlib
import json
import os
from typing import Dict, Any, Set, List, Optional, Tuple


def compute_condition_key(
    experiment_name: str = "",
    sample_id: str = "",
    model: str = "",
    strategy: str = "",
    repetition: int = 1,
    context_type: str = "none",
    context_target_tokens: int = 0,
    retriever: str = "none",
    top_k: int = 0
) -> str:
    """Computes a unique SHA-256 condition key across all experimental dimensions."""
    elements = [
        str(experiment_name),
        str(sample_id),
        str(model),
        str(strategy),
        str(context_type),
        str(context_target_tokens),
        str(retriever),
        str(top_k),
        str(repetition)
    ]
    raw = "|".join(elements).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


class CheckpointManager:
    """Manages appending inference records and deduplicating resumable conditions."""

    def __init__(self, results_jsonl_path: str):
        self.results_path = results_jsonl_path
        self.completed_keys: Set[str] = set()
        self.completed_records: List[Dict[str, Any]] = []
        self._load_existing()

    def _load_existing(self) -> None:
        """Loads completed condition keys from existing results.jsonl."""
        if not os.path.exists(self.results_path):
            return

        with open(self.results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    cond_key = rec.get("condition_key")
                    if cond_key:
                        self.completed_keys.add(cond_key)
                    self.completed_records.append(rec)
                except json.JSONDecodeError:
                    pass

    def is_completed(self, condition_key: str) -> bool:
        """Checks whether a condition has already been successfully executed."""
        return condition_key in self.completed_keys

    def record_result(self, result_record: Dict[str, Any]) -> None:
        """Atomically appends one result row to results.jsonl."""
        cond_key = result_record.get("condition_key")
        if cond_key:
            self.completed_keys.add(cond_key)
        self.completed_records.append(result_record)

        with open(self.results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_record) + "\n")
            f.flush()
            os.fsync(f.fileno())


def validate_resume_directory(
    resume_dir: str,
    expected_experiment_name: str,
    expected_model: str,
    expected_device_name: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Validates that a resume directory matches the current experiment configuration."""
    if not os.path.isdir(resume_dir):
        raise FileNotFoundError(f"Resume directory does not exist: {resume_dir}")

    config_path = os.path.join(resume_dir, "config.json")
    metadata_path = os.path.join(resume_dir, "metadata.json")

    if not os.path.exists(config_path) or not os.path.exists(metadata_path):
        raise ValueError(f"Resume directory lacks config.json or metadata.json: {resume_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Validate identity
    cfg_exp = config.get("experiment_name")
    if cfg_exp != expected_experiment_name:
        raise ValueError(f"Experiment mismatch in resume dir. Expected '{expected_experiment_name}', found '{cfg_exp}'")

    cfg_model = config.get("model", {}).get("name")
    if expected_model and cfg_model != expected_model:
        raise ValueError(f"Model mismatch in resume dir. Expected '{expected_model}', found '{cfg_model}'")

    cfg_device = metadata.get("device", {}).get("normalized_name")
    if expected_device_name and cfg_device and cfg_device != expected_device_name:
        raise ValueError(f"Device mismatch in resume dir. Expected '{expected_device_name}', found '{cfg_device}'")

    return config, metadata


def find_latest_resumable_run(
    results_root: str,
    experiment_name: str,
    device_name: str,
    model_name: str,
    dataset_name: Optional[str] = None,
    eval_size: Optional[Any] = None
) -> Optional[str]:
    """Finds the best existing run directory that matches the experiment parameters to resume from."""
    device_dir = os.path.join(results_root, experiment_name, device_name)
    if not os.path.exists(device_dir):
        return None

    candidates = []
    for entry in sorted(os.listdir(device_dir), reverse=True):
        run_path = os.path.join(device_dir, entry)
        if not os.path.isdir(run_path):
            continue
        cfg_path = os.path.join(run_path, "config.json")
        res_path = os.path.join(run_path, "results.jsonl")
        meta_path = os.path.join(run_path, "metadata.json")

        if os.path.exists(cfg_path) and os.path.exists(res_path) and os.path.exists(meta_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                cfg_model = cfg.get("model", {}).get("name")
                cfg_dataset = cfg.get("dataset", {}).get("name")
                cfg_eval_size = cfg.get("dataset", {}).get("evaluation_size")

                # Match model
                if cfg_model != model_name:
                    continue
                # Match dataset if given
                if dataset_name and cfg_dataset and cfg_dataset.lower() != dataset_name.lower():
                    continue
                # Match eval_size if given
                if eval_size is not None and str(cfg_eval_size).lower() != str(eval_size).lower():
                    continue

                res_size = os.path.getsize(res_path)
                candidates.append((run_path, res_size, entry))
            except Exception:
                continue

    if candidates:
        # Prioritize directory with the most completed progress, then most recent timestamp
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return candidates[0][0]

    return None

