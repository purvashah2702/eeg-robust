import os
import torch
from pytorch_lightning import seed_everything

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    seed_everything(seed, workers=True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
