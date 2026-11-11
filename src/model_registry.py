from importlib import import_module

MODEL_REGISTRY = {
    "SPCNet": "src.models.spcnet.adapter",
}

def get_model_adapter(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return import_module(MODEL_REGISTRY[model_name])
