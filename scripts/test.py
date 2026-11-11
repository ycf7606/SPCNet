import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigManager
from src.runner import ExperimentRunner

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "houston2013.json"

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an SPCNet checkpoint.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()

def main():
    args = parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    manager = ConfigManager()
    manager.load_config(args.config)
    config = manager.config
    if config.model_name != "SPCNet":
        raise ValueError("This release supports SPCNet only.")

    config.enable_training = False
    config.enable_testing = True
    config._setup_paths()

    runner = ExperimentRunner(config)
    runner.print_config()
    runner.run_experiment()

if __name__ == "__main__":
    main()
