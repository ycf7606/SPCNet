import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_LABELS = {
    0: "Houston2013",
    1: "Houston2018",
    2: "Trento",
    3: "Berlin",
    4: "Augsburg",
}


@dataclass
class ExperimentConfig:
    experiment_name: str = "SPCNet"
    dataset_type: int = 0
    model_name: str = "SPCNet"
    cuda_device: str = "cuda:0"
    learning_rate: float = 0.0002
    optimizer_name: str = "adam"
    weight_decay: float = 0.0
    epochs: int = 35
    batch_size: int = 128
    eval_batch_size: int = 512
    num_workers: int = 2
    random_seed: int = 6
    eval_start_epoch: int = 1
    show_progress: bool = False
    pause_file: str = ""
    checkpoint_snapshot_dir: str = ""
    channels: int = 30
    window_size: int = 11
    dropout: float = 0.05
    spcnet_params: Optional[Dict[str, Any]] = None
    enable_training: bool = True
    enable_testing: bool = True
    out_features: Optional[list] = None
    data_channels: Optional[list] = None
    lidar_or_sar_channels: Optional[list] = None
    model_save_path: Optional[list] = None
    log_path: Optional[list] = None
    report_path: Optional[list] = None
    resume_checkpoint: str = ""

    def __post_init__(self):
        if self.dataset_type not in DATASET_LABELS:
            raise ValueError(f"Unsupported dataset_type: {self.dataset_type}")

        self.out_features = self.out_features or [15, 20, 6, 8, 7]
        self.data_channels = self.data_channels or [144, 50, 63, 244, 180]
        self.lidar_or_sar_channels = self.lidar_or_sar_channels or [1, 1, 1, 4, 4]

        defaults = {
            "embed_dim": 64,
            "depth": 6,
            "block_m": 2,
            "num_heads": 4,
            "partial_channels": 16,
            "dk_hidden": 32,
            "lk_kernel": 11,
            "use_self_attn": True,
            "dropout": self.dropout,
            "lk_padding_mode": "reflect",
            "spatial_gate_gain": 0.25,
        }
        self.spcnet_params = {**defaults, **(self.spcnet_params or {})}
        self._setup_paths()

    def _setup_paths(self):
        suffix = (
            f"_{self.experiment_name}_pca={self.channels}"
            f"_window={self.window_size}_lr={self.learning_rate}"
            f"_epochs={self.epochs}"
        )
        model_dir = PROJECT_ROOT / "model" / "SPCNet"
        log_dir = PROJECT_ROOT / "log" / "SPCNet"
        report_dir = PROJECT_ROOT / "report" / "SPCNet"
        self.model_save_path = [
            str(model_dir / f"{name}_model{suffix}.pth")
            for name in DATASET_LABELS.values()
        ]
        self.log_path = [
            str(log_dir / f"{name}_log{suffix}.txt")
            for name in DATASET_LABELS.values()
        ]
        self.report_path = [
            str(report_dir / f"{name}_report{suffix}.txt")
            for name in DATASET_LABELS.values()
        ]


class ConfigManager:
    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self._refresh()

    def _refresh(self):
        config = self.config
        self._parameter_dict = {
            "channels": config.channels,
            "windowSize": config.window_size,
            "out_features": config.out_features,
            "spcnet_params": config.spcnet_params,
            "cuda": config.cuda_device,
            "lr": config.learning_rate,
            "optimizer_name": config.optimizer_name,
            "weight_decay": config.weight_decay,
            "epoch_nums": config.epochs,
            "batch_size": config.batch_size,
            "eval_batch_size": config.eval_batch_size,
            "num_workers": config.num_workers,
            "random_seed": config.random_seed,
            "dropout": config.dropout,
            "eval_start_epoch": config.eval_start_epoch,
            "show_progress": config.show_progress,
            "pause_file": config.pause_file,
            "checkpoint_snapshot_dir": config.checkpoint_snapshot_dir,
            "data_channels": config.data_channels,
            "lidar_or_sar_channels": config.lidar_or_sar_channels,
            "model_savepath": config.model_save_path,
            "log_path": config.log_path,
            "report_path": config.report_path,
            "resume_checkpoint": config.resume_checkpoint,
        }

    def get_value(self, key: str) -> Any:
        return self._parameter_dict[key]

    def get_taskInfo(self) -> str:
        config = self.config
        return (
            "-----------------------taskInfo-----------------------\n"
            f"lr:\t{config.learning_rate}\n"
            f"optimizer:\t{config.optimizer_name}\n"
            f"weight_decay:\t{config.weight_decay}\n"
            f"epoch_nums:\t{config.epochs}\n"
            f"batch_size:\t{config.batch_size}\n"
            f"eval_batch_size:\t{config.eval_batch_size}\n"
            f"window_size:\t{config.window_size}\n"
            f"spcnet_params:\t{config.spcnet_params}\n"
            "------------------------------------------------------"
        )

    def load_config(self, path):
        with open(path, "r", encoding="utf-8") as file:
            values = json.load(file)
        self.config = ExperimentConfig(**values)
        self._refresh()


_global_config_manager: Optional[ConfigManager] = None


def _init(config: Optional[ExperimentConfig] = None) -> None:
    global _global_config_manager
    _global_config_manager = ConfigManager(config)


def get_value(key: str) -> Any:
    if _global_config_manager is None:
        _init()
    return _global_config_manager.get_value(key)


def get_taskInfo() -> str:
    if _global_config_manager is None:
        _init()
    return _global_config_manager.get_taskInfo()


def get_task_info() -> str:
    return get_taskInfo()
