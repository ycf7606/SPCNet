import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import ConfigManager, DATASET_LABELS
from src.runner import ExperimentRunner

DEFAULT_CONFIG_PATH = Path(PROJECT_ROOT) / "configs" / "houston2013.json"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SPCNet using a JSON experiment configuration."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the JSON configuration file.",
    )
    return parser.parse_args()

def load_base_config(config_path):
    manager = ConfigManager()
    manager.load_config(config_path)
    config = manager.config
    if config.model_name != "SPCNet":
        raise ValueError("This release supports SPCNet only.")
    config.enable_training = True
    config.enable_testing = True
    config._setup_paths()
    return config

if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config = load_base_config(config_path)
    print(f"Configuration: {config_path}")

    if config.dataset_type in DATASET_LABELS:
        print(f"Dataset: {DATASET_LABELS[config.dataset_type]} ({config.dataset_type})")

    runner = ExperimentRunner(config)
    runner.print_config()
    runner.run_experiment()
