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
    
    def z_to_delta(self, z):
        s = torch.tanh(z)           # s in (-1,1)
        return self.eps * s

class ES_1_Lambda(BaseAttack):
    def __init__(self, evaluator, eps=8/255, norm="linf",
                 max_evaluation=10000, lam=64, c_inc=1.5, c_dec=0.9, device='cuda'):
        super().__init__(evaluator, eps, norm, device)
        # assert lam >= 2 and c_inc > 1.0 and 0.0 < c_dec < 1.0
        self.lam = int(lam)
        self.c_inc = float(c_inc)
        self.c_dec = float(c_dec)
        self.sigma = 1.1  # σ tuyệt đối
        self.max_evaluation = max_evaluation

    def run(self) -> Dict[str, Any]:
        sigma = self.sigma
        if self.evaluator.decoder:
            _, C, H, W = self.evaluator.lq_shape
        else:
            _, C, H, W = self.evaluator.img_tensor.shape
        
        m = torch.randn((1, C, H, W), device=self.device)
        delta_m = self.z_to_delta(m)
        delta_m = project_delta(delta_m, self.eps, self.norm)

        f_m, l2_m = self.evaluator.evaluate_blackbox(delta_m)
        history = [[float(f_m.item()), float(l2_m.item())]]

        num_evaluation = 1
        while num_evaluation < self.max_evaluation:
            noise = torch.randn((self.lam, C, H, W), device=self.device)
            X = m + sigma * noise
            X_delta = self.z_to_delta(X)
            X_delta = project_delta(X_delta, self.eps, self.norm)

            margins, l2s = self.evaluate_population(X_delta)
            num_evaluation += self.lam
            idx_best = torch.argmin(margins).item()
            x_best = X[idx_best].clone()
            f_best = float(margins[idx_best].item())
            l2_best = float(l2s[idx_best].item())
            x_delta_best = X_delta[idx_best].clone()
            if f_best < f_m:
                m = x_best.clone()
                delta_m = x_delta_best.clone()
                l2_m = l2_best
                f_m = f_best
                sigma *= self.c_inc
                # sigma = min(self.eps, self.sigma)
            else:
                sigma *= self.c_dec            
                # sigma = max(1e-6, sigma)     
            
            print(f"[{num_evaluation} - attack phase] Best loss: ", f_m, " L2: ", l2_m )

            history.append([float(f_m), float(l2_m)])
            if self.is_success(f_m):
                break
            
        if self.evaluator.decoder:
            delta_m = self.evaluator.decoder(delta_m, self.evaluator.img_W, self.evaluator.img_H)
            delta_m = project_delta(delta_m, self.eps, self.norm)

            
        return {"best_delta": delta_m, "best_margin": f_m, "history": history}

class ES_1_Lambda_visual(BaseAttack):
    def __init__(self, evaluator, eps=8/255, norm="linf", max_evaluation=10000, _bs_steps=20, additional_eval=200,
                 lam=64, c_inc=1.5, c_dec=0.9, device='cuda'):
        super().__init__(evaluator, eps, norm, device)
        # assert lam >= 2 and c_inc > 1.0 and 0.0 < c_dec < 1.0
        self.lam = int(lam)
        self.c_inc = float(c_inc)
        self.c_dec = float(c_dec)
        self.sigma = 1.1  # σ tuyệt đối
        self._bs_steps = _bs_steps
        self.visual_interval = 5
        self.max_evaluation = max_evaluation
        self.additional_eval = additional_eval

    def optimize_visual(self, m, delta_m, f_m, l2_m):

        left, right = 0.0, 1.0
        best_m = m.clone()
        best_margin = float(f_m)
        best_l2 = l2_m
        best_delta_m = delta_m.clone()
        num_evaluation = 0
        for _ in range(self._bs_steps):
            alpha = 0.5 * (left + right)
            m_try = alpha * m
            m_delta_try = self.z_to_delta(m_try)
            m_delta_try = project_delta(m_delta_try, self.eps, self.norm)
            margin_try, l2_try = self.evaluator.evaluate_blackbox(m_delta_try)
            num_evaluation += 1
            if self.is_success(margin_try):
                right = alpha
                best_m = m_try
                best_margin = margin_try
                best_delta_m = m_delta_try
                best_l2 = l2_try
            else:
                left = alpha

        return best_m, best_delta_m, best_margin, num_evaluation, best_l2
            


    def run(self) -> Dict[str, Any]:
        sigma = self.sigma
        if self.evaluator.decoder:
            _, C, H, W = self.evaluator.lq_shape
        else:
            _, C, H, W = self.evaluator.img_tensor.shape
            
        m = torch.randn((1, C, H, W), device=self.device)
        delta_m = self.z_to_delta(m)
        delta_m = project_delta(delta_m, self.eps, self.norm)

        f_m, l2_m = self.evaluator.evaluate_blackbox(delta_m)
        history = [[float(f_m.item()), float(l2_m.item())]]
    
        success = False
        num_evaluation = 1
        while num_evaluation < self.max_evaluation:
            if success == True and num_evaluation > stop_num_evaluation:
                break
                
            noise = torch.randn((self.lam, C, H, W), device=self.device)
            X = m + sigma * noise
            X_delta = self.z_to_delta(X)
            X_delta = project_delta(X_delta, self.eps, self.norm)

            margins, l2s = self.evaluate_population(X_delta)
            num_evaluation += self.lam
            idx_best = torch.argmin(margins).item()
            x_best = X[idx_best].clone()
            f_best = float(margins[idx_best].item())
            l2_best = float(l2s[idx_best].item())
            x_delta_best = X_delta[idx_best].clone()

            if f_best < f_m:
                m = x_best.clone()
                delta_m = x_delta_best.clone()
                f_m = f_best
                l2_m = l2_best
                sigma *= self.c_inc
                # sigma = min(self.eps, sigma)
            else:
                sigma *= self.c_dec       
                # sigma = max(1e-6, sigma)     
            
            history.append([f_m, l2_m])
            
            print(f"[{num_evaluation} - attack phase] Best loss: ", f_m, " L2: ", l2_m )
            
            if self.is_success(f_m): # neus lần đầu success
                m, m_delta, f_m, visual_evaluation, l2_m = self.optimize_visual(m, delta_m, f_m, l2_m)
                delta_m = self.z_to_delta(m)
                delta_m = project_delta(delta_m, self.eps, self.norm)
                num_evaluation += visual_evaluation
                if success == False:
                    stop_num_evaluation = num_evaluation + self.additional_eval # chạy thêm 50 dòng nữa
                    success = True
                
                
        if self.evaluator.decoder:
            delta_m = self.evaluator.decoder(delta_m, self.evaluator.img_W, self.evaluator.img_H)
            delta_m = project_delta(delta_m, self.eps, self.norm)

            
        return {"best_delta": delta_m, "best_margin": f_m, "history": history, "num_evaluation": num_evaluation}
