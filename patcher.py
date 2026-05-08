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


# ── Per-dataset defaults ──────────────────────────────────────────────────────
# L_max = 2x expected average patch length.  Guarantees no patch grows beyond
# this regardless of how flat the entropy signal is (rest-class fix).
PATCH_CONFIGS = {
    "HAR":       {"L_min": 8,   "L_max": 32,  "K": 8,  "burn_in": 10},
    "Epilepsy":  {"L_min": 10,  "L_max": 44,  "K": 8,  "burn_in": 10},
    "SLeep-EDF": {"L_min": 50,  "L_max": 375, "K": 16, "burn_in": 20},
    "FD":        {"L_min": 64,  "L_max": 640, "K": 16, "burn_in": 20},
}


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
    def patch_batch(self, x_norm: torch.Tensor,
                    y_norm: torch.Tensor) -> List[List[Tuple[int, int]]]:
        """x_norm, y_norm: [B, T, C]. Returns list of patch lists."""
        return [
            self.patch_signal(x_norm[i], y_norm[i])
            for i in range(x_norm.shape[0])
        ]


# ── StaticPatcher (ablation baseline) ────────────────────────────────────────

class GreedyDualThresholdPatcher:
    """
    EntroPE-style greedy dual-threshold patcher (matched-backbone ablation).

    Algorithm (per signal):
      1. Compute entropy derivative dH[t] = H[t] - H[t-1].
      2. θ_lo = quantile(dH, lo_q)  — "soft" boundary threshold.
         θ_hi = quantile(dH, hi_q)  — "hard" boundary threshold.
      3. Scan left-to-right:
           - Hard boundary:  dH[t] > θ_hi  →  always place boundary.
           - Soft boundary:  dH[t] > θ_lo  AND  gap since last ≥ L_min.
      4. If result has more than K patches: keep the K-1 highest-derivative ones.
         If fewer than K patches: split the longest patches at their midpoints.

    This directly mirrors the monotonicity-based greedy algorithm in EntroPE
    (patch_start_mask_from_entropy_with_monotonicity_adaptive in Patcher.py),
    applied to PRECEPT's density model entropy signal so the backbone is matched.
    """

    def __init__(self, model, channel_mixer,
                 K: int = 8, L_min: int = 8,
                 lo_quantile: float = 0.60,
                 hi_quantile: float = 0.90,
                 burn_in: int = 10,
                 mode: str = "entropy"):
        self.model         = model
        self.channel_mixer = channel_mixer
        self.K             = K
        self.L_min         = L_min
        self.lo_q          = lo_quantile
        self.hi_q          = hi_quantile
        self.burn_in       = burn_in
        self.mode          = mode

    @torch.no_grad()
    def patch_signal(self, x_norm: torch.Tensor,
                     y_norm: torch.Tensor) -> List[Tuple[int, int]]:
        T = x_norm.shape[0]
        H = compute_boundary_signal(self.model, self.channel_mixer,
                                    x_norm, y_norm, self.mode)
        H[:self.burn_in] = H.min()

        # entropy derivative (boundary desirability = jump in H)
        dH = np.diff(H, prepend=H[0])          # shape [T]

        theta_lo = float(np.quantile(dH, self.lo_q))
        theta_hi = float(np.quantile(dH, self.hi_q))

        # greedy scan
        boundaries = []
        last_b = 0
        for t in range(1, T):
            hard = dH[t] > theta_hi
            soft = dH[t] > theta_lo and (t - last_b) >= self.L_min
            if hard or soft:
                boundaries.append(t)
                last_b = t

        # enforce at most K-1 internal boundaries
        if len(boundaries) > self.K - 1:
            # keep K-1 boundaries with the largest derivative jumps
            scores = [(dH[b], b) for b in boundaries]
            scores.sort(reverse=True)
            boundaries = sorted([b for _, b in scores[:self.K - 1]])

        patches = boundaries_to_patches(np.array(boundaries, dtype=np.int32), T)

        # pad up to K if necessary
        while len(patches) < self.K and len(patches) > 0:
            longest = max(range(len(patches)), key=lambda i: patches[i][1] - patches[i][0])
            s, e = patches[longest]
            mid  = (s + e) // 2
            if mid > s:
                patches = patches[:longest] + [(s, mid), (mid, e)] + patches[longest + 1:]
            else:
                break

        return patches

    @torch.no_grad()
    def patch_batch(self, x_norm: torch.Tensor,
                    y_norm: torch.Tensor) -> List[List[Tuple[int, int]]]:
        return [self.patch_signal(x_norm[i], y_norm[i])
                for i in range(x_norm.shape[0])]


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
def normalize_signal(x: torch.Tensor, channel_mixer,
                     channel_mean: torch.Tensor,
                     channel_std: torch.Tensor,
                     device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: [T, C] raw signal.
    Returns x_norm, y_norm: [T-1, C] and [T-1, C] (shifted by 1).
    """
    x = x.to(device)
    xn = (x - channel_mean) / channel_std
    xn = channel_mixer(xn.unsqueeze(0).permute(0, 2, 1)).permute(0, 2, 1).squeeze(0)
    return xn[:-1], xn[1:]


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
