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


class ES_1_1(BaseAttack):
    def __init__(
        self,
        evaluator,
        eps=8/255,
        norm="linf",
        iterations=200,       # T
        sigma=1.1,            # σ^(0)
        c_inc=1.5,            # c_inc > 1
        c_dec=0.9,            # 0 < c_dec < 1
        device='cuda'
    ):
        super().__init__(evaluator, eps, norm, device)
        assert c_inc > 1.0 and 0.0 < c_dec < 1.0
        self.iterations = int(iterations)
        self.sigma = float(sigma)
        self.c_inc = float(c_inc)
        self.c_dec = float(c_dec)

    def run(self) -> Dict[str, Any]:
        _, C, H, W = self.evaluator.img_tensor.shape
        m = (2 * torch.rand((1, C, H, W), device=self.device) - 1.0) * self.eps
        m = project_delta(m, self.eps, self.norm)

        # f(m^(0))
        m_margin = float(self.evaluator.evaluate_blackbox(m).item())
        best_delta = m.clone()
        best_margin = float(m_margin)
        history = [best_margin]

        pbar = tqdm(range(self.iterations), desc=f"Best Loss: {best_margin:.6f}")
        for _ in pbar:
            # ε ~ N(0, I_D); x^(t) = m^(t) + σ(t) ε
            noise = torch.randn_like(m)
            x = m + self.sigma * noise
            x = project_delta(x, self.eps, self.norm)

            # Evaluate child
            x_margin = float(self.evaluator.evaluate_blackbox(x).item())
            if x_margin < m_margin:
                m = x.clone()
                m_margin = x_margin
                self.sigma = self.c_inc * self.sigma

                if m_margin < best_margin:
                    best_margin = float(m_margin)
                    best_delta = m.clone()
            else:
                # keep parent, shrink step-size
                self.sigma = self.c_dec * self.sigma

            history.append(best_margin)
            pbar.set_description(f"Best Loss: {best_margin:.6f}")

            # Early stop if success (margin ≤ 0)
            if self.is_success(best_margin):
                break

        return {"best_delta": best_delta, "best_margin": best_margin, "history": history}

    


class ES_1_Lambda(BaseAttack):
    def __init__(
        self,
        evaluator,
        eps=8/255,
        norm="linf",
        iterations=200,
        lam=64,              # λ
        sigma=1.1,           # σ^(0)
        c_inc=1.5,           # > 1
        c_dec=0.9,           # (0,1)
        plus_selection=False,
        device='cuda'
    ):
        super().__init__(evaluator, eps, norm, device)
        assert lam >= 2 and c_inc > 1.0 and 0.0 < c_dec < 1.0
        self.iterations = int(iterations)
        self.lam = int(lam)
        self.sigma = float(sigma)
        self.c_inc = float(c_inc)
        self.c_dec = float(c_dec)
        self.plus_selection = bool(plus_selection)

    def run(self) -> Dict[str, Any]:
        _, C, H, W = self.evaluator.img_tensor.shape

        # m^(0) ~ Uniform in feasible set (epsilon-ball/box)
        m = (2 * torch.rand((1, C, H, W), device=self.device) - 1.0) * self.eps
        m = project_delta(m, self.eps, self.norm)

        f_m = float(self.evaluator.evaluate_blackbox(m).item())
        best_delta = m.clone()
        best_margin = float(f_m)
        history = [best_margin]

        pbar = tqdm(range(self.iterations), desc=f"Best Loss: {best_margin:.6f}")
        for _ in pbar:
            # Generate λ offspring: x_i = m + σ·ε_i, ε_i~N(0,I)
            noise = (2 * torch.rand((self.lam, C, H, W), device=self.device) - 1.0) * self.eps
            X = m + self.sigma * noise
            X = project_delta(X, self.eps, self.norm)

            margins = self.evaluate_population(X)  # torch (λ,)
            print("Fitness: ", margins)
            idx_best = torch.argmin(margins).item()
            x_best = X[idx_best].clone()
            f_best = float(margins[idx_best].item())

            if self.plus_selection:
                # (1+λ)-ES: chỉ nhận nếu tốt hơn
                if f_best < f_m:
                    m = x_best.clone()
                    f_m = f_best
                    self.sigma *= self.c_inc
                else:
                    self.sigma *= self.c_dec
            else:
                # (1,λ)-ES (comma): luôn thay parent = best_offspring
                # update σ dựa trên việc có cải thiện so với parent cũ không
                improved = f_best < f_m
                m = x_best
                f_m = f_best
                self.sigma *= (self.c_inc if improved else self.c_dec)

            if f_m < best_margin:
                best_margin = float(f_m)
                best_delta = m.clone()

            history.append(best_margin)
            pbar.set_description(f"Best Loss: {best_margin:.6f}")

            if self.is_success(best_margin):
                break

        return {"best_delta": best_delta, "best_margin": best_margin, "history": history}
    
    
class ES_Mu_Lambda(BaseAttack):
    """
    (μ, λ)-ES mặc định (comma): 
      - Recombination: m_t = mean của μ parents (hoặc giữ m_t trực tiếp)
      - Generate λ offspring quanh m_t
      - Chọn top-μ offspring làm parents mới (equal-weight recombination)
    plus_selection=True -> (μ+λ)-ES: top-μ chọn từ μ parents ∪ λ offspring.
    """
    def __init__(
        self,
        evaluator,
        eps=8/255,
        norm="linf",
        iterations=200,
        mu=16,               # μ
        lam=64,              # λ (λ >= μ)
        sigma=1.1,
        c_inc=1.5,
        c_dec=0.9,
        plus_selection=False,
        device='cuda'
    ):
        super().__init__(evaluator, eps, norm, device)
        assert lam >= mu >= 2 and c_inc > 1.0 and 0.0 < c_dec < 1.0
        self.iterations = int(iterations)
        self.mu = int(mu)
        self.lam = int(lam)
        self.sigma = float(sigma)
        self.c_inc = float(c_inc)
        self.c_dec = float(c_dec)
        self.plus_selection = bool(plus_selection)

    def run(self) -> Dict[str, Any]:
        _, C, H, W = self.evaluator.img_tensor.shape

        # Khởi tạo μ parents trong epsilon set
        parents = (2 * torch.rand((self.mu, C, H, W), device=self.device) - 1.0) * self.eps
        parents = project_delta(parents, self.eps, self.norm)

        # fitness parents
        f_parents = self.evaluate_population(parents)   # (μ,)
        f_best = float(torch.min(f_parents).item())
        idx_b = int(torch.argmin(f_parents).item())
        best_delta = parents[idx_b:idx_b+1].clone()
        best_margin = float(f_best)
        history = [best_margin]

        pbar = tqdm(range(self.iterations), desc=f"Best Loss: {best_margin:.6f}")
        for _ in pbar:
            # Recombination mean m_t (equal weights)
            m_t = parents.mean(dim=0, keepdim=True)

            # Generate λ offspring: x_i = m_t + σ·ε_i
            noise = (2 * torch.rand((self.lam, C, H, W), device=self.device) - 1.0)
            off = m_t + self.sigma * noise
            off = project_delta(off, self.eps, self.norm)

            f_off = self.evaluate_population(off)  # (λ,)

            if self.plus_selection:
                # (μ+λ): pool = parents ∪ off, pick top-μ
                pool = torch.cat([parents, off], dim=0)
                f_pool = torch.cat([f_parents, f_off], dim=0)
                top_vals, top_idx = torch.topk(-f_pool, k=self.mu)  # largest of (-f) ≡ smallest of f
                next_parents = pool[top_idx]
                next_f = f_pool[top_idx]
                improved = torch.min(f_off).item() < torch.min(f_parents).item()
            else:
                # (μ,λ): chọn top-μ từ offspring
                top_vals, top_idx = torch.topk(-f_off, k=self.mu)
                next_parents = off[top_idx]
                next_f = f_off[top_idx]
                improved = torch.min(next_f).item() < torch.min(f_parents).item()

            parents = next_parents
            f_parents = next_f

            self.sigma *= (self.c_inc if improved else self.c_dec)

            curr_best_val, curr_best_idx = torch.min(f_parents, dim=0)
            if float(curr_best_val.item()) < best_margin:
                best_margin = float(curr_best_val.item())
                best_delta = parents[int(curr_best_idx.item()):int(curr_best_idx.item())+1].clone()

            history.append(best_margin)
            pbar.set_description(f"Best Loss: {best_margin:.6f}")

            if self.is_success(best_margin):
                break

        return {"best_delta": best_delta, "best_margin": best_margin, "history": history}
    
    



