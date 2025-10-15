import torch
import numpy as np
from typing import Any, Dict

class BaseAttack:

    def __init__(self, evaluator, eps=8/255, norm="linf", device=None):
        self.evaluator = evaluator
        self.eps = float(eps)
        self.norm = norm
        self.device = device if device is not None else next(self.evaluator.model.parameters()).device

        self.x0 = self.evaluator.imgs.to(self.device)[0:1]  # (1,C,H,W)

    def evaluate_population(self, deltas: torch.Tensor) -> np.ndarray:

        deltas = deltas.to(self.device)
        with torch.no_grad():
            margins = self.evaluator.evaluate_blackbox(deltas)  # torch (pop,)
            return margins.detach().cpu().numpy().astype(float)


class RandomSearchAttack(BaseAttack):
    def __init__(self, evaluator, eps=8/255, norm="linf", device=None,
                 iterations=500, pop_size=128):
        super().__init__(evaluator, eps, norm, device)
        self.iterations = iterations
        self.pop_size = pop_size

    def run(self) -> Dict[str, Any]:
        _, C, H, W = self.x0.shape
        best_delta = torch.zeros((1, C, H, W), device=self.device)
        best_margin = float(self.evaluator.evaluate_blackbox(best_delta).item())
        history = [best_margin]

        for _ in range(self.iterations):
            if self.norm == "linf":
                cand = (torch.rand((self.pop_size, C, H, W), device=self.device) * 2 - 1) * self.eps
            else:
                cand = torch.randn((self.pop_size, C, H, W), device=self.device)
                flat = cand.view(self.pop_size, -1)
                norms = torch.norm(flat, dim=1, keepdim=True).clamp_min(1e-12)
                cand = (flat / norms).view_as(cand) * self.eps

            cand = project_delta(cand, self.x0, self.eps, self.norm)
            margins = self.evaluate_population(cand)  # (pop,)
            idx = margins.argmin()                    # minimize
            cand_margin = margins[idx]
            if cand_margin < best_margin:
                best_margin = float(cand_margin)
                best_delta = cand[idx:idx+1].clone()
            history.append(best_margin)

        return {"best_delta": best_delta, "best_margin": best_margin, "history": history}


class NESAttack(BaseAttack):
    def __init__(self, evaluator, eps=8/255, norm="linf", device=None,
                 nes_samples=64, sigma=0.5/255, alpha=1/255, iterations=200, antithetic=True,
                 use_sign=True):
        super().__init__(evaluator, eps, norm, device)
        self.nes_samples = nes_samples if not antithetic else (nes_samples // 2) * 2
        self.sigma = float(sigma)
        self.alpha = float(alpha)
        self.iterations = iterations
        self.antithetic = antithetic
        self.use_sign = use_sign  # True -> step by sign; False -> raw grad

    def run(self) -> Dict[str, Any]:
        _, C, H, W = self.x0.shape
        delta = torch.zeros((1, C, H, W), device=self.device)
        best_delta = delta.clone()
        best_margin = float(self.evaluator.evaluate_blackbox(best_delta).item())
        history = [best_margin]

        for _ in range(self.iterations):
            if self.antithetic:
                S = self.nes_samples // 2
                z_half = torch.randn((S, C, H, W), device=self.device)
                z = z_half
            else:
                z = torch.randn((self.nes_samples, C, H, W), device=self.device)
                S = z.shape[0]

            delta_pos = project_delta(delta + self.sigma * z, self.x0, self.eps, self.norm)
            delta_neg = project_delta(delta - self.sigma * z, self.x0, self.eps, self.norm)

            Xq = torch.cat([delta_pos, delta_neg], dim=0)  # (2S, C,H,W)
            margins = self.evaluator.evaluate_blackbox(Xq)  # (2S,)
            loss_pos = margins[:S].view(S, 1, 1, 1)
            loss_neg = margins[S:].view(S, 1, 1, 1)

            grad_est = ((loss_pos - loss_neg) * z).mean(dim=0) / (2.0 * self.sigma)  # (C,H,W)

            if self.use_sign:
                delta = delta - self.alpha * grad_est.sign().unsqueeze(0)
            else:
                delta = delta - self.alpha * grad_est.unsqueeze(0)

            delta = project_delta(delta, self.x0, self.eps, self.norm)

            cur_margin = float(self.evaluator.evaluate_blackbox(delta).item())
            if cur_margin < best_margin:
                best_margin = cur_margin
                best_delta = delta.clone()
            history.append(best_margin)

        return {"best_delta": best_delta, "best_margin": best_margin, "history": history}


class ESAttack(BaseAttack):
    def __init__(self, evaluator, eps=8/255, norm="linf", device=None,
                 pop_size=128, mu=32, sigma=0.05, lr=0.2, iterations=200):
        super().__init__(evaluator, eps, norm, device)
        assert mu <= pop_size
        self.pop_size = pop_size
        self.mu = mu
        self.sigma = float(sigma)  
        self.lr = float(lr)
        self.iterations = iterations

    def run(self) -> Dict[str, Any]:
        _, C, H, W = self.x0.shape
        mean = torch.zeros((1, C, H, W), device=self.device)  # mean delta
        best_delta = mean.clone()
        best_margin = float(self.evaluator.evaluate_blackbox(best_delta).item())
        history = [best_margin]

        # log-ranking weights
        ranks = np.arange(1, self.mu + 1)
        w = np.log(self.mu + 0.5) - np.log(ranks)
        w = (w / w.sum()).astype(np.float32)
        weights_t = torch.from_numpy(w).to(self.device)

        for _ in range(self.iterations):
            noises = torch.randn((self.pop_size, C, H, W), device=self.device)
            candidates = project_delta(mean + self.sigma * noises, self.x0, self.eps, self.norm)

            margins = self.evaluate_population(candidates) 
            idx_sorted = np.argsort(margins)             
            top_idx = idx_sorted[:self.mu]
            top_noises = noises[top_idx]  # (mu, C,H,W)

            # recombine towards better region
            step_dir = (weights_t.view(-1,1,1,1) * top_noises).sum(dim=0, keepdim=True)  # (1,C,H,W)
            mean = mean + (self.lr / self.sigma) * step_dir
            mean = project_delta(mean, self.x0, self.eps, self.norm)

            cur_margin = float(self.evaluator.evaluate_blackbox(mean).item())
            if cur_margin < best_margin:
                best_margin = cur_margin
                best_delta = mean.clone()
            history.append(best_margin)

        return {"best_delta": best_delta, "best_margin": best_margin, "history": history}


class CMAESAttack(BaseAttack):
    def __init__(self, evaluator, eps=8/255, norm="l2", device=None,
                 pop_size=64, mu=None, sigma_init=0.5, iterations=200, decay=0.999):
        super().__init__(evaluator, eps, norm, device)
        self.pop_size = pop_size
        self.mu = mu if mu is not None else pop_size // 2
        self.sigma = float(sigma_init)  # scale trong không gian delta flatten
        self.iterations = iterations
        self.decay = float(decay)

    def run(self) -> Dict[str, Any]:
        _, C, H, W = self.x0.shape
        dim = C * H * W

        mean = np.zeros(dim, dtype=np.float64)
        cov_diag = np.ones(dim, dtype=np.float64)

        best_delta = torch.zeros((1, C, H, W), device=self.device)
        best_margin = float(self.evaluator.evaluate_blackbox(best_delta).item())
        history = [best_margin]

        # recombination weights
        ranks = np.arange(1, self.mu + 1)
        w = np.log(self.mu + 0.5) - np.log(ranks)
        w = (w / w.sum()).astype(np.float64)

        for _ in range(self.iterations):
            # sample population in flat delta-space
            z = np.random.randn(self.pop_size, dim) * np.sqrt(cov_diag)  # (pop, dim)
            cand_flat = mean[None, :] + self.sigma * z                   # (pop, dim)

            cand = torch.from_numpy(cand_flat).float().to(self.device).view(self.pop_size, C, H, W)
            cand = project_delta(cand, self.x0, self.eps, self.norm)

            margins = self.evaluate_population(cand)  # (pop,)
            idx_sorted = np.argsort(margins)         # ascending (minimize)
            top_idx = idx_sorted[:self.mu]
            top_flat = cand_flat[top_idx]            # (mu, dim)

            # rank-μ update (diag)
            new_mean = (w[:, None] * top_flat).sum(axis=0)  # (dim,)
            diff = top_flat - mean[None, :]
            cov_mu = (w[:, None] * (diff ** 2)).sum(axis=0)
            cov_diag = 0.9 * cov_diag + 0.1 * (cov_mu + 1e-12)

            mean = new_mean
            self.sigma *= self.decay  

            # evaluate mean
            mean_t = torch.from_numpy(mean).float().to(self.device).view(1, C, H, W)
            mean_t = project_delta(mean_t, self.x0, self.eps, self.norm)
            cur_margin = float(self.evaluator.evaluate_blackbox(mean_t).item())
            if cur_margin < best_margin:
                best_margin = cur_margin
                best_delta = mean_t.clone()
            history.append(best_margin)

        return {"best_delta": best_delta, "best_margin": best_margin, "history": history}


