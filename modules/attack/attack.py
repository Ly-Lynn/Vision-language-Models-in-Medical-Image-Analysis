import torch
import numpy as np
from typing import Any, Dict
from .util import clamp_eps, project_delta
from tqdm import tqdm
from time import time

class BaseAttack:

    def __init__(self, evaluator, eps=8/255, norm="l2", device=None):
        self.evaluator = evaluator
        self.eps = float(eps)
        self.norm = norm
        self.device = device if device is not None else next(self.evaluator.model.parameters()).device

    def evaluate_population(self, deltas: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            margins, l2s = self.evaluator.evaluate_blackbox(deltas)  # torch (pop,)
            return margins.clone(), l2s.clone()
        
    def is_success(self, margin):
        if margin <= 0:
            return True
        return False
    


class ES_1_Lambda_visual(BaseAttack):
    def __init__(self, evaluator, eps=8/255, norm="linf", max_evaluation=10000,
                 lam=64, c_inc=1.5, c_dec=0.9, device='cuda'):
        super().__init__(evaluator, eps, norm, device)
        # assert lam >= 2 and c_inc > 1.0 and 0.0 < c_dec < 1.0
        self.lam = int(lam)
        self.c_inc = float(c_inc)
        self.c_dec = float(c_dec)
        self.sigma = 1.1  # σ tuyệt đối
        self._bs_steps = 50
        self.visual_interval = 5
        self.max_evaluation = max_evaluation

    def optimize_visual(self, m, f_m, l2_m):

        left, right = 0.0, 1.0
        best_m = m.clone()
        best_margin = float(f_m)
        best_l2 = l2_m
        num_evaluation = 0
        for _ in range(self._bs_steps):
            alpha = 0.5 * (left + right)
            m_try = alpha * m
            margin_try, l2_try = self.evaluator.evaluate_blackbox(m_try)
            num_evaluation += 1
            if self.is_success(margin_try):
                right = alpha
                best_m = m_try
                best_margin = margin_try
                best_l2 = l2_try
            else:
                left = alpha

        return best_m, best_margin, num_evaluation, best_l2
            


    def run(self) -> Dict[str, Any]:
        sigma = self.sigma
        if self.evaluator.decoder:
            _, C, H, W = self.evaluator.lq_shape
        else:
            _, C, H, W = self.evaluator.img_tensor.shape
            
        if self.evaluator.decoder:
            m = torch.randn((1, C, H, W), device=self.device) * self.eps
            m = project_delta(m, self.eps, self.norm)
        else:
            m = torch.randn((1, C, H, W), device=self.device)


        f_m, l2_m = self.evaluator.evaluate_blackbox(m)
        history = [[float(f_m.item()), float(l2_m.item())]]
    
        visual_interval = None
        num_evaluation = 1
        iter = 0
        while num_evaluation < self.max_evaluation:
            noise = torch.randn((self.lam, C, H, W), device=self.device)
            X = m + sigma * noise
            if not self.evaluator.decoder:
                X = project_delta(X, self.eps, self.norm)

            margins, l2s = self.evaluate_population(X)
            num_evaluation += self.lam
            idx_best = torch.argmin(margins).item()
            x_best = X[idx_best].clone()
            f_best = float(margins[idx_best].item())
            l2_best = float(l2s[idx_best].item())

            if f_best < f_m:
                m = x_best.clone()
                f_m = f_best
                l2_m = l2_best
                sigma *= self.c_inc
                sigma = min(self.eps, sigma)
            else:
                sigma *= self.c_dec       
                sigma = max(1e-6, sigma)     
            
            history.append([float(f_m.item()), float(l2_m.item())])
            print("Best loss: ", f_m, " L2: ", l2_m )
            if self.is_success(f_m) and not visual_interval: 
                m, f_m, visual_evaluation, l2_m = self.optimize_visual(m, f_m, l2_m)
                visual_interval = self.visual_interval
                num_evaluation += visual_evaluation
                
            if visual_interval and iter % visual_interval:
                m, f_m, visual_evaluation, l2_m = self.optimize_visual(m, f_m, l2_m)
                num_evaluation += visual_evaluation
            
            iter += 1
        if self.evaluator.decoder:
            m = self.evaluator.decoder(m, self.evaluator.img_W, self.evaluator.img_H)
            m = project_delta(m, self.eps, self.norm)

            
        return {"best_delta": m, "best_margin": f_m, "history": history}







    
    



