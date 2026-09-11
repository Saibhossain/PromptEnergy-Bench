"""Structured logging subsystem for experiment runs."""

import logging
import os
import sys


def setup_experiment_logging(log_file_path: str, log_level: int = logging.INFO) -> logging.Logger:
    """Configures multi-handler logging to console and experiment log file."""
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logger = logging.getLogger("PromptEnergyBench")
    logger.setLevel(log_level)
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file_path, encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.propagate = False
    return logger
