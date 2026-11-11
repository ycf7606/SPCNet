from src.config import DATASET_LABELS, ConfigManager


class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.config_manager = ConfigManager(config)

        import src.config as config_module

        config_module._init(config)

    def run_experiment(self):
        print(
            f"{self.config.experiment_name}: "
            f"{DATASET_LABELS[self.config.dataset_type]}, "
            f"device={self.config.cuda_device}"
        )
        if self.config.enable_training:
            from src.trainer import myTrain

            print(self.config_manager.get_taskInfo())
            if myTrain(self.config.dataset_type, self.config.model_name) is False:
                return False

        if self.config.enable_testing:
            from src.evaluator import myTest

            myTest(self.config.dataset_type, self.config.model_name)

        return True

    def print_config(self):
        config = self.config
        print(
            f"{config.experiment_name}: {DATASET_LABELS[config.dataset_type]}, "
            f"epochs={config.epochs}, lr={config.learning_rate}, "
            f"device={config.cuda_device}"
        )
