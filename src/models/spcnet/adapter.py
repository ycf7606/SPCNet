from .model import SPCNet

MODEL_NAME = "SPCNet"

SPCNET_PARAM_NAMES = {
    "embed_dim",
    "depth",
    "block_m",
    "num_heads",
    "partial_channels",
    "dk_hidden",
    "lk_kernel",
    "use_self_attn",
    "dropout",
    "lk_padding_mode",
    "spatial_gate_gain",
}

def _get_model_params(config):
    params = dict(config.get_value("spcnet_params") or {})
    unknown = set(params) - SPCNET_PARAM_NAMES
    if unknown:
        raise ValueError(f"Unknown SPCNet parameters: {sorted(unknown)}")
    positive_integer_params = (
        "embed_dim",
        "depth",
        "block_m",
        "num_heads",
        "partial_channels",
        "dk_hidden",
        "lk_kernel",
    )
    for name in positive_integer_params:
        if not isinstance(params[name], int) or params[name] < 1:
            raise ValueError(f"SPCNet {name} must be a positive integer.")
    if params["embed_dim"] % params["num_heads"] != 0:
        raise ValueError("SPCNet embed_dim must be divisible by num_heads.")
    if params["lk_kernel"] % 2 == 0:
        raise ValueError("SPCNet lk_kernel must be a positive odd integer.")
    if not 0.0 <= params["dropout"] <= 1.0:
        raise ValueError("SPCNet dropout must be between 0 and 1.")
    if params["lk_padding_mode"] not in {
        "constant", "reflect", "replicate", "circular"
    }:
        raise ValueError("Unsupported SPCNet lk_padding_mode.")
    return params

def build_model(config, dataset_type, device):
    hsi_channels = config.get_value("channels")
    aux_channels = config.get_value("lidar_or_sar_channels")[dataset_type]
    num_classes = config.get_value("out_features")[dataset_type]
    model_params = _get_model_params(config)

    net = SPCNet(
        hsi_dim=hsi_channels,
        sar_dim=aux_channels,
        num_classes=num_classes,
        **model_params,
    )
    return {"net": net}

def forward_train(bundle, batch):
    return bundle["net"](batch["hsi_pca"], batch["aux"])

def forward_eval(bundle, batch):
    return bundle["net"](batch["hsi_pca"], batch["aux"])

def compute_loss(outputs, labels, criterion):
    """Retain SPCNet's auxiliary-supervision loss from the original trainer."""
    if not isinstance(outputs, dict):
        return criterion(outputs, labels)

    loss = criterion(outputs["logits_main"], labels)
    if "logits_hsi" in outputs:
        loss = loss + 0.1 * criterion(outputs["logits_hsi"], labels)
    if "logits_sar" in outputs:
        loss = loss + 0.1 * criterion(outputs["logits_sar"], labels)
    return loss
