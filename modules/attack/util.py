import torch
import os
import random
import numpy as np
import torch

def clamp_eps(delta: torch.Tensor, eps: float, norm: str = "linf") -> torch.Tensor:

    if norm == "linf":
        return torch.clamp(delta, -eps, eps)
    elif norm == "l2":
        shape = delta.shape
        pop = shape[0]
        flat = delta.view(pop, -1)
        norms = torch.norm(flat, dim=1, keepdim=True).clamp_min(1e-12)
        factor = (eps / norms).clamp_max(1.0)
        return (flat * factor).view_as(delta)
    else:
        raise ValueError("norm must be 'linf' or 'l2'")

def project_delta(delta: torch.Tensor,
                  x0: torch.Tensor,
                  eps: float,
                  norm: str = "linf",
                  clip_min: float = 0.0,
                  clip_max: float = 1.0) -> torch.Tensor:

    adv = (x0 + delta).clamp(clip_min, clip_max)
    delta = adv - x0
    delta = clamp_eps(delta, eps, norm)
    adv = (x0 + delta).clamp(clip_min, clip_max)
    delta = adv - x0
    return delta




def seed_everything(
    seed: int,
    cudnn_deterministic: bool = True,
    cudnn_benchmark: bool = False,
) -> None:

    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

