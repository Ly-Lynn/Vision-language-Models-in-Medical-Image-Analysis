import torch
import numpy as np
from typing import Any, Dict
from .util import clamp_eps, project_delta
from tqdm import tqdm
class BaseAttack:

    def __init__(self, evaluator, eps=8/255, norm="l2", device=None):
        self.evaluator = evaluator
        self.eps = float(eps)
        self.norm = norm
        self.device = device if device is not None else next(self.evaluator.model.parameters()).device

    def evaluate_population(self, deltas: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            margins = self.evaluator.evaluate_blackbox(deltas)  # torch (pop,)
            return margins.clone()
        
    def is_success(self, margin):
        if margin <= 0:
            return True
        return False


class RandomSearchAttack(BaseAttack):
    def __init__(self, evaluator, eps=8/255, norm="linf",
                 iterations=200, pop_size=50, device='cuda'):
        super().__init__(evaluator, eps, norm, device)
        self.iterations = iterations
        self.pop_size = pop_size

    def run(self) -> Dict[str, Any]:
        _, C, H, W = self.evaluator.img_tensor.shape
        best_delta = torch.randn((1, C, H, W), device=self.device) * self.eps
        best_delta = project_delta(best_delta, self.eps, self.norm)

        best_margin = self.evaluator.evaluate_blackbox(best_delta).item()
        history = [best_margin]
        pbar = tqdm(range(self.iterations), desc=f"Best Loss: {best_margin:.6f}")

        for iter in pbar:
            cand = torch.randn((self.pop_size, C, H, W), device=self.device) * self.eps
            cand = project_delta(cand, self.eps, self.norm)
            margins = self.evaluate_population(cand)  # (pop,)
            idx = margins.argmin()                    # minimize

            cand_margin = margins[idx].item()
            if cand_margin < best_margin:
                best_margin = float(cand_margin)
                best_delta = cand[idx].clone()

            history.append(best_margin)
            if self.is_success(best_margin):
                break
            
            pbar.set_description(f"Best Loss: {best_margin:.6f}")

        return {"best_delta": best_delta, "best_margin": best_margin, "history": history}


