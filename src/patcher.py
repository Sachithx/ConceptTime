"""
patcher.py — Phase 3: Patch boundary detection from a frozen density model.

Two patchers with identical interface:
  EntropyPatcher  — entropy-guided DP segmentation
  StaticPatcher   — fixed-window baseline (for ablation)

Interface: patcher(signal_H_or_raw) → list of (start, end) index tuples
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple


# ── Patch scale library ───────────────────────────────────────────────────────
# Each scale sets K ≈ T * k_frac (number of patches scales with sequence length).
# L_min, L_max, and burn_in are then derived automatically from avg_len = T / K.
#
# Choosing a scale:
#   xs — very short patches, many of them  (K ≈ T/5)
#   s  — short patches                     (K ≈ T/8)
#   m  — medium patches                    (K ≈ T/12)
#   l  — long patches                      (K ≈ T/20)
#   xl — very long patches, few of them    (K ≈ T/50)
PATCH_SCALES: dict[str, float] = {
    "xs": 0.20,
    "s":  0.12,
    "m":  0.08,
    "l":  0.05,
    "xl": 0.02,
}

# Per-dataset scale selection — the only thing that needs tuning per dataset.
DATASET_PATCH_SCALE: dict[str, str] = {
    "HAR":                  "m",    # T~128  → K~10
    "Epilepsy":             "l",    # T~178  → K~10
    "SLeep-EDF":            "xl",   # T~3000 → K~60
    "FD-A":                 "xl",    # T~5120  → K~100
    "FD-B":                 "xl",    # T~5120  → K~100
    "FD-C":                 "xl",    # T~5120  → K~100
    "FD-D":                 "xl",    # T~5120  → K~100
    # ── UEA/UCR archive ───────────────────────────────────────────────────────
    "JapaneseVowels":       "xl",   # T=28   → K=2
    "FaceDetection":        "xs",   # T=61   → K=12
    "SpokenArabicDigits":   "m",    # T=92   → K=7
    "PEMS-SF":              "s",    # T=144  → K=17
    "Handwriting":          "xs",    # T=152  → K=8
    "UWaveGestureLibrary":  "l",    # T=315  → K=16
    "Heartbeat":            "l",    # T=405  → K=20
    "SelfRegulationSCP1":   "l",    # T=896  → K=45
    "SelfRegulationSCP2":   "xl",   # T=1152 → K=23
    "EthanolConcentration": "xl",   # T=1751 → K=35
}


def get_patch_config(dataset: str, T: int, scale: str = None) -> dict:
    """
    Derive {K, L_min, L_max, burn_in} from the dataset's patch scale and T.

    scale overrides DATASET_PATCH_SCALE when provided (e.g. from --patch_scale CLI arg).
    """
    k_frac  = PATCH_SCALES[scale or DATASET_PATCH_SCALE.get(dataset, "m")]
    K       = max(2, round(T * k_frac))
    avg_len = T / K
    L_min   = max(2, round(avg_len * 0.50))
    L_max   = max(L_min + 1, round(avg_len * 2.0))
    burn_in = max(2, L_min)
    # return {"K": K, "L_min": L_min, "L_max": L_max, "burn_in": burn_in}
    return {"K": 8, "L_min": 16, "L_max": 32, "burn_in": 10}


# Sentinel that keeps the name importable but raises on access, pointing
# callers to the new API: ppc = get_patch_config(dataset, T)
class _RemovedDict(dict):
    _MSG = "PATCH_CONFIGS removed — use get_patch_config(dataset, T) instead."
    def __getitem__(self, _): raise RuntimeError(self._MSG)
    def get(self, *_): raise RuntimeError(self._MSG)

PATCH_CONFIGS = _RemovedDict()


# ── DP segmentation core ──────────────────────────────────────────────────────

def dp_segment(score: np.ndarray, K: int, L_min: int,
               L_max: int = None) -> np.ndarray:
    """
    DP placing exactly K-1 internal boundaries.

    Each consecutive pair of boundaries (including signal start/end) satisfies
    L_min ≤ gap ≤ L_max.  This guarantees no patch is shorter than L_min or
    longer than L_max — critical for rest-class signals where entropy is flat
    (without L_max the DP degenerately clusters all boundaries at the start
    and leaves one huge tail patch).

    Uses a monotonic deque (sliding-window max) for O(T·K) time.

    score[t] = boundary desirability (higher → prefer boundary here).
    Returns sorted array of K-1 internal boundary indices.
    """
    from collections import deque

    T = len(score)
    if L_max is None:
        L_max = T                   # unconstrained — same as old behaviour
    L_max = min(L_max, T)
    NEG_INF = -1e18

    prev = np.full(T, NEG_INF)
    prev[0] = 0.0                   # sentinel: 0 boundaries, "last" at t=0
    back = np.zeros((K - 1, T), dtype=np.int32)

    for k in range(K - 1):
        curr = np.full(T, NEG_INF)
        dq: deque = deque()         # indices; best (highest prev) at front

        for t in range(T):
            # t′ = t − L_min just became a valid predecessor for position t
            t_prime = t - L_min
            if t_prime >= 0 and prev[t_prime] > NEG_INF:
                # maintain monotonically decreasing deque
                while dq and prev[dq[-1]] <= prev[t_prime]:
                    dq.pop()
                dq.append(t_prime)

            # evict predecessors that are now too far back (patch would exceed L_max)
            while dq and dq[0] < t - L_max:
                dq.popleft()

            if dq:
                curr[t] = prev[dq[0]] + score[t]
                back[k, t] = dq[0]

        prev = curr

    # Last patch (from last boundary to T) must also satisfy L_min ≤ len ≤ L_max,
    # so last boundary lives in [T − L_max, T − L_min].
    lo = max(0, T - L_max)
    hi = T - L_min                  # inclusive
    if lo > hi:
        return np.array([], dtype=np.int32)

    candidates = np.arange(lo, hi + 1)
    valid_mask  = prev[candidates] > NEG_INF
    if not valid_mask.any():
        return np.array([], dtype=np.int32)

    candidates  = candidates[valid_mask]
    last_idx    = int(candidates[np.argmax(prev[candidates])])

    # Back-track through K-1 rounds
    boundaries = [last_idx]
    cur = last_idx
    for k in range(K - 2, -1, -1):
        cur = int(back[k, cur])
        boundaries.append(cur)
    boundaries.reverse()
    boundaries = [b for b in boundaries if b > 0]   # drop sentinel 0
    return np.array(sorted(set(boundaries)), dtype=np.int32)


def boundaries_to_patches(boundaries: np.ndarray, T: int) -> List[Tuple[int, int]]:
    """Convert boundary indices to (start, end) non-overlapping patches covering [0, T)."""
    bndry = np.concatenate([[0], boundaries, [T]])
    return [(int(bndry[i]), int(bndry[i + 1])) for i in range(len(bndry) - 1)]


# ── Boundary signal computation ───────────────────────────────────────────────

@torch.no_grad()
def compute_boundary_signal(model, channel_mixer, x_norm: torch.Tensor,
                             y_norm: torch.Tensor, mode: str = "entropy") -> np.ndarray:
    """
    x_norm, y_norm: [T, C] already normalized.
    mode: 'entropy' | 'surprise' | 'kl_shift'
    Returns: [T] boundary desirability signal.
    """
    x = x_norm.unsqueeze(0)   # [1, T, C]
    y = y_norm.unsqueeze(0)   # [1, T, C]

    mu, log_var, _ = model(x)   # [1, T, C]
    mu     = mu.squeeze(0)      # [T, C]
    log_var = log_var.squeeze(0)

    if mode == "entropy":
        LOG2PIE = math.log(2 * math.pi * math.e)
        H = 0.5 * (LOG2PIE + log_var)   # [T, C]
        return H.mean(dim=-1).cpu().numpy()

    elif mode == "surprise":
        y_sq = y.squeeze(0)
        inv_var = torch.exp(-log_var)
        nll = 0.5 * (log_var + (y_sq - mu).pow(2) * inv_var)
        return nll.mean(dim=-1).cpu().numpy()

    elif mode == "kl_shift":
        # KL between consecutive predictive Gaussians (diagonal)
        # KL(N(mu_t,sig_t) || N(mu_{t+1},sig_{t+1})) summed over channels
        mu1    = mu[:-1]                          # [T-1, C]
        mu2    = mu[1:]
        lv1    = log_var[:-1]
        lv2    = log_var[1:]
        sig1sq = lv1.exp()
        sig2sq = lv2.exp()
        kl = 0.5 * ((lv2 - lv1) + sig1sq / sig2sq +
                     (mu2 - mu1).pow(2) / sig2sq - 1).sum(dim=-1)
        kl = kl.clamp(min=0).cpu().numpy()
        return np.concatenate([[0.0], kl])   # pad first timestep with 0

    elif mode == "residual":
        # Mahalanobis squared distance: (y-μ)² / σ²  (no log_var term).
        # Unlike NLL/surprise, this is NOT inflated by model uncertainty alone —
        # a timestep scores high only when the signal specifically deviates from
        # the prediction in units of the model's own confidence.
        # Result: boundaries land at class-specific events rather than at
        # generally uncertain regions, leaving surprise free to vary inside
        # patches so that residual_morphology / surprise_trajectory signatures
        # carry genuine discriminative information.
        y_sq   = y.squeeze(0)                              # [T, C]
        inv_var = torch.exp(-log_var).clamp(max=1e6)       # 1/σ²  [T, C]
        mahal  = (y_sq - mu).pow(2) * inv_var              # [T, C]
        return mahal.mean(dim=-1).cpu().numpy()            # [T]

    else:
        raise ValueError(f"Unknown boundary mode: {mode}")


def boundary_signal_from_precomputed(mu: torch.Tensor, log_var: torch.Tensor,
                                      y_norm: torch.Tensor, mode: str) -> np.ndarray:
    """
    Same as compute_boundary_signal but uses already-computed mu/log_var [T, C].
    Avoids a redundant model forward pass when outputs are already available.
    """
    if mode == "entropy":
        LOG2PIE = math.log(2 * math.pi * math.e)
        return (0.5 * (LOG2PIE + log_var)).mean(dim=-1).cpu().numpy()

    elif mode == "surprise":
        inv_var = torch.exp(-log_var)
        nll = 0.5 * (log_var + (y_norm - mu).pow(2) * inv_var)
        return nll.mean(dim=-1).cpu().numpy()

    elif mode == "kl_shift":
        mu1, mu2 = mu[:-1], mu[1:]
        lv1, lv2 = log_var[:-1], log_var[1:]
        kl = 0.5 * ((lv2 - lv1) + lv1.exp() / lv2.exp() +
                     (mu2 - mu1).pow(2) / lv2.exp() - 1).sum(dim=-1)
        return np.concatenate([[0.0], kl.clamp(min=0).cpu().numpy()])

    elif mode == "residual":
        inv_var = torch.exp(-log_var).clamp(max=1e6)
        return ((y_norm - mu).pow(2) * inv_var).mean(dim=-1).cpu().numpy()

    else:
        raise ValueError(f"Unknown boundary mode: {mode}")


# ── EntropyPatcher ────────────────────────────────────────────────────────────

class EntropyPatcher:
    """
    Entropy-guided patcher using DP segmentation.

    Usage:
        patcher = EntropyPatcher(model, channel_mixer, dataset_cfg)
        patches = patcher.patch_signal(x_norm, y_norm)  # [(start, end), ...]
    """

    def __init__(self, model, channel_mixer,
                 K: int = 8, L_min: int = 8, L_max: int = None,
                 burn_in: int = 10, mode: str = "entropy"):
        self.model          = model
        self.channel_mixer  = channel_mixer
        self.K              = K
        self.L_min          = L_min
        self.L_max          = L_max   # None = unconstrained (not recommended)
        self.burn_in        = burn_in
        self.mode           = mode

    @torch.no_grad()
    def patch_signal(self, x_norm: torch.Tensor,
                     y_norm: torch.Tensor) -> List[Tuple[int, int]]:
        """
        x_norm, y_norm: [T, C] normalized tensors (single signal).
        Returns: K patches as [(start, end), ...].
        """
        T = x_norm.shape[0]
        score = compute_boundary_signal(
            self.model, self.channel_mixer, x_norm, y_norm, self.mode
        )
        # zero out unreliable burn_in region
        score[:self.burn_in] = score.min()

        if self.K <= 1:
            return [(0, T)]

        boundaries = dp_segment(score, self.K, self.L_min, self.L_max)
        patches    = boundaries_to_patches(boundaries, T)

        # safety: if DP returns fewer than K patches, pad with last patch split
        while len(patches) < self.K and len(patches) > 0:
            last = patches[-1]
            mid  = (last[0] + last[1]) // 2
            if mid > last[0]:
                patches = patches[:-1] + [(last[0], mid), (mid, last[1])]
            else:
                break

        return patches

    @torch.no_grad()
    def patch_signal_precomputed(self, mu: torch.Tensor, log_var: torch.Tensor,
                                  y_norm: torch.Tensor) -> List[Tuple[int, int]]:
        """Use already-computed model outputs [T, C] to skip a redundant forward pass."""
        T = mu.shape[0]
        score = boundary_signal_from_precomputed(mu, log_var, y_norm, self.mode)
        score[:self.burn_in] = score.min()

        if self.K <= 1:
            return [(0, T)]

        boundaries = dp_segment(score, self.K, self.L_min, self.L_max)
        patches    = boundaries_to_patches(boundaries, T)

        while len(patches) < self.K and len(patches) > 0:
            last = patches[-1]
            mid  = (last[0] + last[1]) // 2
            if mid > last[0]:
                patches = patches[:-1] + [(last[0], mid), (mid, last[1])]
            else:
                break

        return patches

    @torch.no_grad()
    def patch_batch(self, x_norm: torch.Tensor,
                    y_norm: torch.Tensor) -> List[List[Tuple[int, int]]]:
        """x_norm, y_norm: [B, T, C]. Returns list of patch lists."""
        return [
            self.patch_signal(x_norm[i], y_norm[i])
            for i in range(x_norm.shape[0])
        ]


# ── StaticPatcher (ablation baseline) ────────────────────────────────────────

class StaticPatcher:
    """
    Fixed-window patcher. No model needed. Used as ablation baseline.
    Divides [0, T) into K equal-length non-overlapping patches.
    """

    def __init__(self, K: int = 8):
        self.K = K

    def patch_signal(self, T: int) -> List[Tuple[int, int]]:
        window = T // self.K
        patches = []
        for k in range(self.K):
            start = k * window
            end   = start + window if k < self.K - 1 else T
            patches.append((start, end))
        return patches

    def patch_batch(self, T: int, B: int) -> List[List[Tuple[int, int]]]:
        p = self.patch_signal(T)
        return [p for _ in range(B)]


# ── Normalise-then-patch helper ───────────────────────────────────────────────

@torch.no_grad()
def normalize_batch(x_batch: torch.Tensor, y_batch: torch.Tensor,
                    channel_mixer, channel_mean: torch.Tensor,
                    channel_std: torch.Tensor, device: torch.device):
    """x_batch, y_batch: [B, T, C]. Returns normalized versions."""
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    x_n = (x_batch - channel_mean) / channel_std
    y_n = (y_batch - channel_mean) / channel_std
    x_n = channel_mixer(x_n.permute(0, 2, 1)).permute(0, 2, 1)
    y_n = channel_mixer(y_n.permute(0, 2, 1)).permute(0, 2, 1)
    return x_n, y_n
