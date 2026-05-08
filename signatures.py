"""
signatures.py — Phase 4: Extract fixed-length feature vectors from patches.

Each patch [t1, t2] → signature vector. Four named, ablatable categories:

  ENTROPY (5 dims):
    mean_entropy, var_entropy, max_entropy, argmax_pos, log_patch_len

  MOMENTS (2C dims + 1):
    mean_mu [C], mean_sigma2 [C], log_patch_len

  DISTRIBUTIONAL (4 dims):
    drift, mean_surprise, max_surprise, kl_to_marginal

  MORPHOLOGICAL (8 dims, computed on raw signal y_patch):
    fft_peak_freq, autocorr_lag1/2/3,
    skewness, excess_kurtosis, zero_crossing_rate, peak_to_peak

Modes (pass as `mode=`):
  "entropy_only"        →  5
  "moments_only"        →  2C + 1
  "distributional_only" →  4          (drift moved here from moments)
  "morphological_only"  →  8          (NEW — raw-signal shape descriptors)
  "full"                →  17 + 2C    (all four categories concatenated)

All four categories are independent → any subset can be used for ablation by
setting sig_mode in PIPELINE_CONFIGS or --sig_mode CLI flag.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict


LOG2PIE = math.log(2 * math.pi * math.e)

MORPH_DIM = 8  # fft_peak_freq + autocorr×3 + skewness + kurtosis + zcr + p2p
SURPRISE_TRAJ_DIM = 4  # argmax_pos + spread + confident_surprise + residual_energy

LOG2PIE = math.log(2 * math.pi * math.e)
# ── Morphological feature extractor ──────────────────────────────────────────

def _morph_features(y_patch: torch.Tensor) -> torch.Tensor:
    """
    Shape descriptors computed on channel-averaged raw signal.

    y_patch: [L, C]  normalized observed values (yn[t1:t2])
    Returns: [8] — all scale-invariant, robust to short patches.

    Features:
      fft_peak_freq     — dominant oscillation frequency (normalized 0–1)
      autocorr_lag1/2/3 — normalized autocorrelation at lags 1, 2, 3
      skewness          — standardized 3rd moment (clamped ±10)
      excess_kurtosis   — standardized 4th moment − 3 (clamped ±50)
      zero_crossing_rate — fraction of consecutive sign changes (mean-centered)
      peak_to_peak      — (max−min) / std  (range in std units)
    """
    L, C = y_patch.shape
    device = y_patch.device
    s = y_patch.mean(dim=-1)          # [L] channel-averaged signal

    s_mean = s.mean()
    s_std  = s.std().clamp(min=1e-8)
    s_c    = s - s_mean               # mean-centered

    # 1. FFT peak frequency (normalized by Nyquist bin count; 0 for short patches)
    if L >= 4:
        fft_mag = torch.fft.rfft(s_c).abs()   # [L//2+1] complex → magnitude
        fft_mag[0] = 0.0                        # zero DC component
        peak_bin      = fft_mag.argmax().float()
        fft_peak_freq = peak_bin / max(float(len(fft_mag) - 1), 1.0)
    else:
        fft_peak_freq = torch.zeros((), device=device)

    # 2. Normalized autocorrelation at lags 1, 2, 3
    ac0 = (s_c * s_c).mean().clamp(min=1e-8)
    autocorr = []
    for lag in range(1, 4):
        if L > lag:
            r = (s_c[:-lag] * s_c[lag:]).mean() / ac0
        else:
            r = torch.zeros((), device=device)
        autocorr.append(r.clamp(-1.0, 1.0))
    ac1, ac2, ac3 = autocorr

    # 3. Skewness E[(s-μ)³] / σ³  (clamped to avoid explosions on short patches)
    skewness = (s_c.pow(3).mean() / s_std.pow(3)).clamp(-10.0, 10.0)

    # 4. Excess kurtosis E[(s-μ)⁴] / σ⁴ − 3
    excess_kurtosis = (s_c.pow(4).mean() / s_std.pow(4) - 3.0).clamp(-10.0, 50.0)

    # 5. Zero-crossing rate (mean-centered sign changes / (L-1))
    if L > 1:
        zcr = ((s_c[:-1] * s_c[1:]) < 0).float().mean()
    else:
        zcr = torch.zeros((), device=device)

    # 6. Peak-to-peak range in std units (scale-invariant amplitude)
    p2p = (s.max() - s.min()) / s_std

    return torch.stack([fft_peak_freq, ac1, ac2, ac3,
                        skewness, excess_kurtosis, zcr, p2p])


# ── Surprise-aware feature extractors ────────────────────────────────────────

def _residual_morph_features(y_patch: torch.Tensor,
                              mu_patch: torch.Tensor,
                              log_var_patch: torch.Tensor) -> torch.Tensor:
    """
    Shape descriptors of the standardized residual r(t) = (y−μ) / σ.

    Subtracting μ removes the predictable background dynamics the density model
    already knows.  Dividing by σ normalises so that surprises in low-uncertainty
    regions (confident model, unusual signal) are amplified relative to surprises
    in high-uncertainty regions (expected noise).  The result captures the
    class-specific deviation pattern rather than the raw signal shape.

    y_patch, mu_patch, log_var_patch: [L, C]
    Returns: [8]  — same layout as _morph_features.
    """
    sigma = log_var_patch.exp().sqrt().clamp(min=1e-8)   # [L, C]
    r = (y_patch - mu_patch) / sigma                      # standardized residual
    return _morph_features(r)


def _surprise_trajectory_features(mu_patch: torch.Tensor,
                                   log_var_patch: torch.Tensor,
                                   y_patch: torch.Tensor) -> torch.Tensor:
    """
    Temporal structure of prediction surprise within the patch.

    Returns: [4]
      surprise_argmax_pos  — normalised position of peak NLL (0=start, 1=end)
      surprise_spread      — std of per-timestep NLL (concentrated spike vs spread)
      confident_surprise   — NLL weighted by model confidence (1 − H/H_max);
                             amplifies surprise that happens where the model was sure
      residual_energy      — mean Mahalanobis distance (y−μ)²/σ²;
                             scale-invariant total surprise magnitude
    """
    L = mu_patch.shape[0]
    device = mu_patch.device
    sigma2 = log_var_patch.exp()                                    # [L, C]

    nll_t = 0.5 * (log_var_patch + (y_patch - mu_patch).pow(2) /
                   (sigma2 + 1e-8))
    nll_t = nll_t.mean(dim=-1)                                      # [L]

    H_t   = 0.5 * (LOG2PIE + log_var_patch).mean(dim=-1)           # [L]
    H_max = H_t.max().clamp(min=1e-8)

    surprise_argmax_pos = nll_t.argmax().float() / max(L - 1, 1)
    surprise_spread     = (nll_t.std()
                           if L > 1 else torch.zeros((), device=device))
    confident_surprise  = (nll_t * (1.0 - H_t / H_max)).mean()
    residual_energy     = ((y_patch - mu_patch).pow(2) /
                           (sigma2 + 1e-8)).mean()

    return torch.stack([surprise_argmax_pos, surprise_spread,
                        confident_surprise, residual_energy])


# ── Marginal distribution (computed once on training set) ─────────────────────

def compute_marginal(model, channel_mixer, train_dataset,
                     channel_mean, channel_std, device,
                     batch_size: int = 256):
    """
    Moment-matched Gaussian marginal: p_marg ~ N(mu_marg, sigma2_marg).
    mu_marg    = E[mu_t]        over all train timesteps
    sigma2_marg = E[sigma2_t] + Var[mu_t]

    Returns: mu_marg [C], sigma2_marg [C]
    """
    from torch.utils.data import DataLoader
    loader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0, drop_last=False)

    mu_sum    = None
    mu_sq_sum = None
    var_sum   = None
    n_steps   = 0

    model.eval()
    channel_mixer.eval()

    with torch.no_grad():
        for batch_x, batch_y, *_ in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            xn = (batch_x - channel_mean) / channel_std
            yn = (batch_y - channel_mean) / channel_std
            xn = channel_mixer(xn.permute(0, 2, 1)).permute(0, 2, 1)

            mu, log_var, _ = model(xn)   # [B, T, C]
            sigma2 = log_var.exp()

            B, T, C = mu.shape
            n = B * T

            mu_flat     = mu.reshape(n, C)
            sigma2_flat = sigma2.reshape(n, C)

            if mu_sum is None:
                mu_sum    = mu_flat.sum(0)
                mu_sq_sum = mu_flat.pow(2).sum(0)
                var_sum   = sigma2_flat.sum(0)
            else:
                mu_sum    += mu_flat.sum(0)
                mu_sq_sum += mu_flat.pow(2).sum(0)
                var_sum   += sigma2_flat.sum(0)
            n_steps += n

    mu_marg     = mu_sum / n_steps                                  # [C]
    var_mu      = mu_sq_sum / n_steps - mu_marg.pow(2)             # [C]
    sigma2_marg = var_sum / n_steps + var_mu.clamp(min=0)          # [C]
    return mu_marg.cpu(), sigma2_marg.cpu()


# ── Single patch → signature ──────────────────────────────────────────────────

def patch_signature(mu_patch: torch.Tensor,
                    log_var_patch: torch.Tensor,
                    y_patch: torch.Tensor,
                    mu_marg: torch.Tensor,
                    sigma2_marg: torch.Tensor,
                    mode = "full") -> torch.Tensor:
    """
    mu_patch, log_var_patch, y_patch: [L, C]  (patch length L, C channels)
    mu_marg, sigma2_marg: [C]  training marginals
    mode: str or list[str] —
          Existing: 'full' | 'entropy_only' | 'moments_only' |
                    'distributional_only' | 'morphological_only'
          New:      'residual_morphology' | 'surprise_trajectory' | 'surprise_full'
          Pass a list to concatenate multiple feature groups.

    Returns: signature tensor.
    Dim = 17 + 2C (full) — entropy(5) + moments(2C+1) + distrib(4) + morph(8).
    Dim = 2C + 13 (surprise_full) — moments(2C+1) + res_morph(8) + surprise_traj(4).
    log(patch_len) appears in both entropy_only and moments_only because each
    category must be self-sufficient for single-category ablations.
    """
    L, C = mu_patch.shape
    device = mu_patch.device

    sigma2_patch = log_var_patch.exp()     # [L, C]
    H_t = 0.5 * (LOG2PIE + log_var_patch) # [L, C]
    H_t_mean = H_t.mean(dim=-1)           # [L]  channel-averaged entropy

    # Entropy trajectory features
    mean_entropy = H_t_mean.mean()
    var_entropy  = H_t_mean.var() if L > 1 else torch.zeros(1, device=device).squeeze()
    max_entropy  = H_t_mean.max()
    argmax_pos   = (H_t_mean.argmax().float() / max(L - 1, 1))

    # Patch length (log-scaled): lets clustering distinguish short vs long patches.
    # Short patches (L ≈ L_min) have unreliable variance/drift; including log-length
    # lets the model implicitly down-weight those features for short patches.
    log_patch_len = torch.tensor(math.log(max(L, 1)), device=device)

    # Per-channel moment features
    mean_mu     = mu_patch.mean(dim=0)     # [C]
    mean_sigma2 = sigma2_patch.mean(dim=0) # [C]

    # Drift: norm of mean-prediction shift, first-quarter vs last-quarter.
    # Scaled by patch length so that a 2-step and a 20-step patch with the
    # same absolute drift get comparable scores.
    q = max(L // 4, 1)
    drift = (mu_patch[-q:].mean(0) - mu_patch[:q].mean(0)).norm() / max(L, 1)

    # Surprise = mean NLL on observed values
    inv_var       = (sigma2_patch + 1e-8).reciprocal()
    nll_t         = 0.5 * (log_var_patch + (y_patch - mu_patch).pow(2) * inv_var)
    nll_t         = nll_t.mean(dim=-1)     # [L]  channel-averaged
    mean_surprise = nll_t.mean()
    max_surprise  = nll_t.max()

    # KL(patch || marginal): KL(N(mu_patch, sig2_patch) || N(mu_marg, sig2_marg))
    # mu_marg / sigma2_marg are expected to already be on the correct device
    kl = 0.5 * (
        torch.log((sigma2_marg + 1e-8) / (mean_sigma2 + 1e-8)) +
        (mean_sigma2 + (mean_mu - mu_marg).pow(2)) /
        (sigma2_marg + 1e-8) - 1
    ).sum().clamp(min=0)

    morph = _morph_features(y_patch)   # [8]

    def _assemble(m: str) -> torch.Tensor:
        if m == "entropy_only":
            return torch.stack([mean_entropy, var_entropy, max_entropy, argmax_pos,
                                log_patch_len])
        if m == "moments_only":
            return torch.cat([mean_mu, mean_sigma2,
                              log_patch_len.unsqueeze(0)])
        if m == "distributional_only":
            return torch.stack([drift, mean_surprise, max_surprise, kl])
        if m == "morphological_only":
            return morph
        if m == "residual_morphology":
            return _residual_morph_features(y_patch, mu_patch, log_var_patch)
        if m == "surprise_trajectory":
            return _surprise_trajectory_features(mu_patch, log_var_patch, y_patch)
        if m == "surprise_full":
            # moments(2C+1) + residual_morphology(8) + surprise_trajectory(4) = 2C+13
            return torch.cat([
                mean_mu,
                mean_sigma2,
                log_patch_len.unsqueeze(0),
                _residual_morph_features(y_patch, mu_patch, log_var_patch),
                _surprise_trajectory_features(mu_patch, log_var_patch, y_patch),
            ])
        # full: entropy(5) + moments(2C+1) + distributional(4) + morphological(8)
        #     = 17 + 2C
        return torch.cat([
            torch.stack([mean_entropy, var_entropy, max_entropy, argmax_pos,
                         log_patch_len]),
            mean_mu,
            mean_sigma2,
            torch.stack([drift, mean_surprise, max_surprise, kl]),
            morph,
        ])

    if isinstance(mode, (list, tuple)):
        return torch.cat([_assemble(m) for m in mode])
    return _assemble(mode)


def signature_dim(C: int, mode = "full") -> int:
    if isinstance(mode, (list, tuple)):
        return sum(signature_dim(C, m) for m in mode)
    if mode == "entropy_only":        return 5                 # 5 entropy stats
    if mode == "moments_only":        return 2 * C + 1         # mean_mu + mean_sigma2 + log_len
    if mode == "distributional_only": return 4                 # drift + NLL×2 + KL
    if mode == "morphological_only":  return MORPH_DIM         # 8 raw-signal descriptors
    if mode == "residual_morphology": return MORPH_DIM         # 8 residual-shape descriptors
    if mode == "surprise_trajectory": return SURPRISE_TRAJ_DIM # 4 surprise trajectory stats
    if mode == "surprise_full":       return 2 * C + 13        # (2C+1) + 8 + 4
    return 17 + 2 * C   # full: 5 + (2C+1) + 4 + 8 - 1 duplicate log_len = 17+2C


# ── Batch signature extraction ────────────────────────────────────────────────

@torch.no_grad()
def extract_signatures_for_dataset(model, channel_mixer,
                                    dataset, patch_fn,
                                    channel_mean, channel_std,
                                    mu_marg, sigma2_marg,
                                    device, batch_size=64,
                                    mode="full") -> Tuple[torch.Tensor,
                                                          List[List[Tuple[int, int]]],
                                                          torch.Tensor]:
    """
    Run the full dataset through the frozen density model and extract signatures.

    patch_fn: callable(x_norm [T,C], y_norm [T,C]) → [(start, end), ...]
              OR StaticPatcher.patch_signal(T) → [(start, end), ...]

    Returns:
      all_sigs:    [N_patches_total, D]  all signatures
      all_patches: list[list[(start, end)]]  per-sample patch ranges
      sample_idx:  [N_patches_total]  which sample each patch came from
    """
    from torch.utils.data import DataLoader

    # pre-move marginals once — patch_signature would otherwise .to(device) every patch
    mu_marg_d     = mu_marg.to(device)
    sigma2_marg_d = sigma2_marg.to(device)

    # resolve patch_fn dispatch — determined lazily on first real call to avoid
    # running model on wrong device via a probe with dummy tensors
    _patch_signal = getattr(patch_fn, 'patch_signal', None)
    _use_len_only = None  # None = unknown, True/False cached after first call

    def _patch_dispatch(xn, yn):
        nonlocal _use_len_only
        if _patch_signal is None:
            return patch_fn(xn, yn)
        if _use_len_only is True:
            return _patch_signal(xn.shape[0])
        if _use_len_only is False:
            return _patch_signal(xn, yn)
        try:
            result = _patch_signal(xn, yn)
            _use_len_only = False
            return result
        except TypeError:
            _use_len_only = True
            return _patch_signal(xn.shape[0])

    # batch_size > 1: amortises channel_mixer + model launches across samples.
    # requires fixed-length samples (true for windowed HAR/Epilepsy datasets).
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, drop_last=False)

    all_sigs    = []
    all_patches = []
    sample_ids  = []
    sample_base = 0

    model.eval()
    channel_mixer.eval()

    for batch_x, batch_y, *_ in loader:
        B  = batch_x.shape[0]
        xb = batch_x.to(device)                                          # [B, T, C]
        yb = batch_y.to(device)

        xn_b = (xb - channel_mean) / channel_std
        yn_b = (yb - channel_mean) / channel_std
        xn_b = channel_mixer(xn_b.permute(0, 2, 1)).permute(0, 2, 1)   # [B, T, C']
        yn_b = channel_mixer(yn_b.permute(0, 2, 1)).permute(0, 2, 1)

        mu_b, log_var_b, _ = model(xn_b)                                 # [B, T, C]

        for b in range(B):
            xn = xn_b[b]; yn = yn_b[b]
            mu = mu_b[b]; lv = log_var_b[b]

            patches = _patch_dispatch(xn, yn)

            sigs = []
            for (t1, t2) in patches:
                if t2 <= t1:
                    continue
                sig = patch_signature(mu[t1:t2], lv[t1:t2], yn[t1:t2],
                                      mu_marg_d, sigma2_marg_d, mode=mode)
                sigs.append(sig)

            if sigs:
                all_sigs.extend(sigs)
                all_patches.append(patches)
                sample_ids.extend([sample_base + b] * len(sigs))

        sample_base += B

    sigs_tensor = torch.stack(all_sigs).cpu()   # [N_patches, D]
    idx_tensor  = torch.tensor(sample_ids, dtype=torch.long)
    return sigs_tensor, all_patches, idx_tensor


# ── Signature standardization ─────────────────────────────────────────────────

class SignatureStandardizer:
    """Fit on train signatures, apply to any split."""

    def __init__(self):
        self.mean: Optional[torch.Tensor] = None
        self.std:  Optional[torch.Tensor] = None

    def fit(self, sigs: torch.Tensor):
        self.mean = sigs.mean(0)
        self.std  = sigs.std(0).clamp(min=1e-6)

    def transform(self, sigs: torch.Tensor) -> torch.Tensor:
        return (sigs - self.mean.to(sigs.device)) / self.std.to(sigs.device)

    def fit_transform(self, sigs: torch.Tensor) -> torch.Tensor:
        self.fit(sigs)
        return self.transform(sigs)

    def state_dict(self) -> Dict:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, d: Dict):
        self.mean = d["mean"]
        self.std  = d["std"]


 
