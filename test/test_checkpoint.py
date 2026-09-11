"""Unit tests for checkpoint condition keys, resume verification, and device normalization."""

import json
import os
import shutil
import tempfile
import unittest

from src.infrastructure.checkpoint import (
    compute_condition_key,
    CheckpointManager,
    validate_resume_directory
)
from src.infrastructure.device import normalize_device_name


class TestCheckpointAndResume(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.results_path = os.path.join(self.test_dir, "results.jsonl")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_device_name_normalization(self):
        self.assertEqual(normalize_device_name("MacBook Air M1"), "macbook_air_m1")
        self.assertEqual(normalize_device_name("Windows RTX 4070 PC"), "windows_rtx_4070_pc")
        self.assertEqual(normalize_device_name("Ubuntu NVIDIA A100 Server!"), "ubuntu_nvidia_a100_server")
        self.assertEqual(normalize_device_name("Special // Device :::: 1"), "special_device_1")

    def test_condition_key_uniqueness(self):
        k1 = compute_condition_key("exp1", "sample1", "modelA", "zero_shot", 1)
        k2 = compute_condition_key("exp1", "sample1", "modelA", "zero_shot", 2)  # different repetition
        k3 = compute_condition_key("exp1", "sample1", "modelA", "few_shot", 1)   # different strategy
        self.assertNotEqual(k1, k2)
        self.assertNotEqual(k1, k3)
        self.assertEqual(k1, compute_condition_key("exp1", "sample1", "modelA", "zero_shot", 1))

    def test_checkpoint_deduplication(self):
        mgr = CheckpointManager(self.results_path)
        k = compute_condition_key("exp1", "sample1", "modelA", "zero_shot", 1)
        self.assertFalse(mgr.is_completed(k))

        mgr.record_result({"condition_key": k, "sample_id": "sample1", "status": "success"})
        self.assertTrue(mgr.is_completed(k))

        # Re-instantiate to simulate restart
        mgr2 = CheckpointManager(self.results_path)
        self.assertTrue(mgr2.is_completed(k))
        self.assertEqual(len(mgr2.completed_records), 1)

    def test_validate_resume_directory_validation(self):
        run_dir = os.path.join(self.test_dir, "test_run")
        os.makedirs(run_dir)
        cfg_path = os.path.join(run_dir, "config.json")
        meta_path = os.path.join(run_dir, "metadata.json")

        with open(cfg_path, "w") as f:
            json.dump({"experiment_name": "primary_exp_gsm8k", "model": {"name": "qwen3.5:0.8b-mlx"}}, f)
        with open(meta_path, "w") as f:
            json.dump({"run_id": "run123"}, f)

        # Matching configuration
        cfg, meta = validate_resume_directory(
            resume_dir=run_dir,
            expected_experiment_name="primary_exp_gsm8k",
            expected_model="qwen3.5:0.8b-mlx",
            expected_device_name=""
        )
        self.assertEqual(cfg["experiment_name"], "primary_exp_gsm8k")

        # Mismatched model
        with self.assertRaises(ValueError):
            validate_resume_directory(
                resume_dir=run_dir,
                expected_experiment_name="primary_exp_gsm8k",
                expected_model="different-model",
                expected_device_name=""
            )


if __name__ == "__main__":
    unittest.main()
