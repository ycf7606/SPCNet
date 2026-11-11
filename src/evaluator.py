from torch.backends import cudnn
from torch.utils.data import DataLoader, Subset

from .data import *
from .metrics import *

from . import config

import os
from .model_registry import get_model_adapter

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def _limit_dataloader_samples(loader, max_samples):
    if max_samples is None:
        return loader
    subset_size = min(len(loader.dataset), max_samples)
    subset = Subset(loader.dataset, range(subset_size))
    return create_evaluation_loader(
        subset,
        loader.batch_size,
        loader.num_workers,
        pin_memory=loader.pin_memory,
    )

def myTest(datasetType, model, max_test_samples=None):
    """Evaluate a trained model using the active experiment configuration."""
    cuda = config.get_value('cuda')
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    channels = config.get_value('channels')
    windowSize = config.get_value('windowSize')
    batch_size = config.get_value('batch_size')
    eval_batch_size = config.get_value('eval_batch_size')
    num_workers = config.get_value('num_workers')
    random_seed = config.get_value('random_seed')
    model_savepath = config.get_value('model_savepath')
    report_path = config.get_value('report_path')

    print(f"Evaluation device: {cuda}")

    model_adapter = get_model_adapter(model)
    model_bundle = model_adapter.build_model(config, datasetType, device)
    net = model_bundle["net"]
    try:
        checkpoint = torch.load(
            model_savepath[datasetType], map_location=device, weights_only=True
        )
    except TypeError:

        checkpoint = torch.load(model_savepath[datasetType], map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Unsupported checkpoint format. Retrain once to create a state_dict checkpoint."
        )
    if checkpoint.get("model_name") not in (None, model):
        raise ValueError(
            f"Checkpoint model {checkpoint['model_name']!r} does not match requested model {model!r}."
        )
    if checkpoint.get("dataset_type") not in (None, datasetType):
        raise ValueError(
            "Checkpoint dataset does not match the requested dataset."
        )
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(device)
    set_random_seed(random_seed)
    train_loader, test_loader, trntst_loader, all_loader, hsi_pca_wight = getMyData(datasetType, channels, windowSize, batch_size, num_workers)
    if model == "SPCNet":
        test_loader.dataset.hsi = None
    test_loader = create_evaluation_loader(
        test_loader.dataset,
        eval_batch_size,
        num_workers,
        pin_memory=test_loader.pin_memory,
    )
    test_loader = _limit_dataloader_samples(test_loader, max_test_samples)

    getMyReport(datasetType, model_bundle, model_adapter, test_loader, report_path[datasetType], device, model)
