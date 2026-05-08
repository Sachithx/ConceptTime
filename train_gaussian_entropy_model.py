"""
train_gaussian_entropy.py
─────────────────────────
Trains GaussianGPT on a time-series dataset for entropy-guided patch segmentation.

Key differences vs the old discrete GPT (train_entropy_model.py):
  - No tokenizer: model receives raw normalised floats directly
  - Multivariate: model sees all C channels jointly at each timestep
  - Dataset-level normalisation: channel mean/std computed once on train set,
    preserving absolute amplitude differences across samples
  - Loss: Gaussian NLL + calibration regularization
  - Entropy: analytic 0.5*(1 + log(2pi*e*sigma^2)) averaged over channels

Checkpoint saved to output/<dataset>/gaussian_entropy_best.pt with keys:
  epoch, model_state_dict, channel_mixer_state_dict, val_loss,
  channel_mean, channel_std, n_channels, block_size
"""

import argparse
import glob
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import math
import numpy as np

from GaussianEntropyModel import GaussianGPT, GaussianGPTConfig

# ── Per-dataset configuration ─────────────────────────────────────────────────
DATASET_CONFIGS = {
    "HAR":                  {"root_path": "./dataset/HAR",                  "n_channels": 9,   "seq_len": 127,  "output_dir": "output/HAR",                  "save_path": "output/HAR/gaussian_entropy_best.pt"},
    "Epilepsy":             {"root_path": "./dataset/Epilepsy",             "n_channels": 1,   "seq_len": 177,  "output_dir": "output/Epilepsy",             "save_path": "output/Epilepsy/gaussian_entropy_best.pt"},
    "SLeep-EDF":            {"root_path": "./dataset/SLeep-EDF",            "n_channels": 1,   "seq_len": 2999, "output_dir": "output/SLeep-EDF",            "save_path": "output/SLeep-EDF/gaussian_entropy_best.pt"},
    "FD-A":                 {"root_path": "./dataset/FD-A",                 "n_channels": 1,   "seq_len": 5119, "output_dir": "output/FD-A",                 "save_path": "output/FD-A/gaussian_entropy_best.pt"},
    "FD-B":                 {"root_path": "./dataset/FD-B",                 "n_channels": 1,   "seq_len": 5119, "output_dir": "output/FD-B",                 "save_path": "output/FD-B/gaussian_entropy_best.pt"},
    "FD-C":                 {"root_path": "./dataset/FD-C",                 "n_channels": 1,   "seq_len": 5119, "output_dir": "output/FD-C",                 "save_path": "output/FD-C/gaussian_entropy_best.pt"},
    "FD-D":                 {"root_path": "./dataset/FD-D",                 "n_channels": 1,   "seq_len": 5119, "output_dir": "output/FD-D",                 "save_path": "output/FD-D/gaussian_entropy_best.pt"},
    # ── UEA/UCR archive datasets ──────────────────────────────────────────────
    "EthanolConcentration": {"root_path": "./dataset/EthanolConcentration", "n_channels": 3,   "seq_len": 1750, "output_dir": "output/EthanolConcentration", "save_path": "output/EthanolConcentration/gaussian_entropy_best.pt"},
    "FaceDetection":        {"root_path": "./dataset/FaceDetection",        "n_channels": 144, "seq_len": 61,   "output_dir": "output/FaceDetection",        "save_path": "output/FaceDetection/gaussian_entropy_best.pt"},
    "Handwriting":          {"root_path": "./dataset/Handwriting",          "n_channels": 3,   "seq_len": 151,  "output_dir": "output/Handwriting",          "save_path": "output/Handwriting/gaussian_entropy_best.pt"},
    "Heartbeat":            {"root_path": "./dataset/Heartbeat",            "n_channels": 61,  "seq_len": 404,  "output_dir": "output/Heartbeat",            "save_path": "output/Heartbeat/gaussian_entropy_best.pt"},
    "JapaneseVowels":       {"root_path": "./dataset/JapaneseVowels",       "n_channels": 12,  "seq_len": 28,   "output_dir": "output/JapaneseVowels",       "save_path": "output/JapaneseVowels/gaussian_entropy_best.pt"},
    "PEMS-SF":              {"root_path": "./dataset/PEMS-SF",              "n_channels": 963, "seq_len": 143,  "output_dir": "output/PEMS-SF",              "save_path": "output/PEMS-SF/gaussian_entropy_best.pt"},
    "SelfRegulationSCP1":   {"root_path": "./dataset/SelfRegulationSCP1",   "n_channels": 6,   "seq_len": 895,  "output_dir": "output/SelfRegulationSCP1",   "save_path": "output/SelfRegulationSCP1/gaussian_entropy_best.pt"},
    "SelfRegulationSCP2":   {"root_path": "./dataset/SelfRegulationSCP2",   "n_channels": 7,   "seq_len": 1151, "output_dir": "output/SelfRegulationSCP2",   "save_path": "output/SelfRegulationSCP2/gaussian_entropy_best.pt"},
    "SpokenArabicDigits":   {"root_path": "./dataset/SpokenArabicDigits",   "n_channels": 13,  "seq_len": 92,   "output_dir": "output/SpokenArabicDigits",   "save_path": "output/SpokenArabicDigits/gaussian_entropy_best.pt"},
    "UWaveGestureLibrary":  {"root_path": "./dataset/UWaveGestureLibrary",  "n_channels": 3,   "seq_len": 314,  "output_dir": "output/UWaveGestureLibrary",  "save_path": "output/UWaveGestureLibrary/gaussian_entropy_best.pt"},
}


def get_lr(iteration, max_iters, warmup_iters, learning_rate, min_lr,
           decay_lr=True):
    if iteration < warmup_iters:
        return learning_rate * (iteration / max(warmup_iters, 1))
    if not decay_lr:
        return learning_rate
    decay_ratio = (iteration - warmup_iters) / max(max_iters - warmup_iters, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ============================================================================
# CONFIG
# ============================================================================

class Config:
    # Hardware
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    dtype       = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") \
                  if torch.cuda.is_available() else "float32"

    # Model
    n_layer      = 4
    n_head       = 8       # transformer only
    n_embd       = 128
    dropout      = 0.05
    bias         = False
    logvar_min   = -10.0
    logvar_max   =   4.0
    calib_weight = 0.01
    # Backbone
    backbone     = "transformer"   # "transformer" | "mamba"
    d_state      = 16              # Mamba: SSM state dimension
    d_conv       = 4               # Mamba: depthwise conv kernel width
    d_expand     = 2               # Mamba: inner-dim expansion factor

    # Data (set from DATASET_CONFIGS in main)
    root_path   = ""
    n_channels  = 1
    seq_len     = 1
    batch_size  = 128
    num_workers = 4

    # Training
    epochs               = 50
    lr                   = 3e-4
    weight_decay         = 0.05
    beta1, beta2         = 0.9, 0.95
    grad_accumulation    = 1
    clip_grad            = 1.0
    warmup_steps         = 0
    min_lr_factor        = 0.05
    decay_lr             = True
    patience             = 10

    # Output (set from DATASET_CONFIGS in main)
    output_dir = ""
    save_path  = ""


# ============================================================================
# DATASET  (identical to caption_pipeline HARDataset)
# ============================================================================

class TimeSeriesDataset(Dataset):
    """Generic loader for all supported datasets.

    Handles three storage formats:
      [N, T]    (FD flat)         → unsqueeze to [N, T, 1]
      [N, C, T] (channel-first datasets) → permute to [N, T, C]
    FD splits are stored as multiple files (train_a.pt, …) and are concatenated.
    """
    def __init__(self, root_path, flag="train", seq_len=127):
        sub_files = sorted(glob.glob(f"{root_path}/{flag}_*.pt"))
        if sub_files:
            sigs, lbls = [], []
            for f in sub_files:
                s, lb = self._load(f)
                sigs.append(s); lbls.append(lb)
            self.samples = torch.cat(sigs, dim=0)
            self.labels  = torch.cat(lbls, dim=0)
        else:
            self.samples, self.labels = self._load(f"{root_path}/{flag}.pt")
        self.seq_len = seq_len
        N, T, C = self.samples.shape
        print(f"[{flag}] {root_path.split('/')[-1]}: {N} samples  T={T}  C={C}")

    @staticmethod
    def _load(path):
        data = torch.load(path, weights_only=False)
        s    = data["samples"].float()
        lb   = data["labels"]
        if s.dim() == 2:      # [N, T] → [N, T, 1]
            s = s.unsqueeze(-1)
        elif s.dim() == 3:    # [N, C, T] → [N, T, C]
            s = s.permute(0, 2, 1)
        lb = (torch.mode(lb.flatten(1), dim=1).values.long()
              if lb.dim() > 1 else lb.long())
        return s, lb

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]        # [T, C]
        T   = s.shape[0]
        # Random start for signals longer than the context window (data augmentation)
        max_start = max(0, T - self.seq_len - 1)
        start = int(torch.randint(0, max_start + 1, (1,)).item()) if max_start > 0 else 0
        x     = s[start : start + self.seq_len]
        y     = s[start + 1 : start + self.seq_len + 1]
        dummy = torch.zeros(self.seq_len, 4)
        return x, y, dummy, dummy


# Keep alias for any external code that references HARDataset
HARDataset = TimeSeriesDataset


# ============================================================================
# CHANNEL MIXER  (same architecture as caption_pipeline training)
# ============================================================================

class ChannelMixer(nn.Module):
    def __init__(self, n_channels=9, dropout=0.1):
        super().__init__()
        self.mix  = nn.Linear(n_channels, n_channels, bias=False)
        self.norm = nn.LayerNorm(n_channels)
        self.drop = nn.Dropout(dropout)
        nn.init.eye_(self.mix.weight)

    def forward(self, x):           # x: [B, C, T]
        x_t = x.permute(0, 2, 1)   # [B, T, C]
        return x + self.norm(self.drop(self.mix(x_t))).permute(0, 2, 1)


# ============================================================================
# DATASET-LEVEL NORMALISATION STATISTICS
# ============================================================================

def compute_channel_stats(dataset: TimeSeriesDataset, device: torch.device):
    """
    Compute channel-wise mean and std from the full training set.

    Uses dataset-level statistics rather than per-sample normalization so that
    absolute amplitude differences between activities are preserved.

    Returns: channel_mean [C], channel_std [C]  (on device, float32)
    """
    samples = dataset.samples.float()       # [N, T, C]
    flat    = samples.reshape(-1, samples.shape[-1])  # [N*T, C]
    mean    = flat.mean(0)                  # [C]
    std     = flat.std(0).clamp(min=1e-8)   # [C]
    print(f"Channel stats computed from {len(dataset)} samples")
    print(f"  mean: {mean.tolist()}")
    print(f"  std:  {std.tolist()}")
    return mean.to(device), std.to(device)


# ============================================================================
# PREPROCESSING  (channel mixer + dataset-level normalisation)
# ============================================================================

def preprocess(x, y, channel_mixer, channel_mean, channel_std, device):
    """
    x, y:          [B, T, C]
    channel_mean:  [C]  dataset-level mean (computed once on train set)
    channel_std:   [C]  dataset-level std  (computed once on train set)
    Returns x_norm, y_norm: [B, T, C] normalised floats ready for GaussianGPT

    Order: normalise FIRST (so input to mixer has unit std per channel),
    then apply ChannelMixer.  The mixer's LayerNorm residual adds ~unit-std
    noise, giving combined std ≈ sqrt(2) — stable for the density model.
    Reversing this order causes the LayerNorm residual (std≈1) to dominate
    the raw signal (std≈0.1–0.4), then dividing by raw std amplifies ≈5×.
    """
    x = x.to(device)   # [B, T, C]
    y = y.to(device)

    # Step 1: dataset-level normalisation → each channel has mean≈0, std≈1
    x = (x - channel_mean) / channel_std
    y = (y - channel_mean) / channel_std

    # Step 2: ChannelMixer (sees unit-std inputs, adds unit-std residual → std≈√2)
    x = channel_mixer(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T, C]
    y = channel_mixer(y.permute(0, 2, 1)).permute(0, 2, 1)

    return x, y   # [B, T, C]


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=10, save_path="best.pt"):
        self.patience   = patience
        self.counter    = 0
        self.best_score = None
        self.early_stop = False
        self.val_min    = float("inf")
        self.save_path  = save_path

    def __call__(self, val_loss, model, channel_mixer, epoch,
                 channel_mean, channel_std):
        score = -val_loss
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter    = 0
            self.val_min    = val_loss
            sd = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") \
                 else model.state_dict()
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({
                "epoch":                    epoch,
                "model_state_dict":         sd,
                "channel_mixer_state_dict": channel_mixer.state_dict(),
                "val_loss":                 val_loss,
                "n_channels":               channel_mixer.mix.in_features,
                "block_size":               raw_model.config.block_size,
                "channel_mean":             channel_mean.cpu(),
                "channel_std":              channel_std.cpu(),
                "model_config":             vars(raw_model.config),
            }, self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ============================================================================
# EVALUATE
# ============================================================================

@torch.no_grad()
def evaluate(model, channel_mixer, val_loader, config, channel_mean, channel_std):
    model.eval()
    channel_mixer.eval()
    total_loss    = 0.0
    n_channels    = config.n_channels
    channel_nlls  = torch.zeros(n_channels, device=config.device)
    amp_dtype     = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16 if config.dtype == "float16" else torch.float32

    for batch_x, batch_y, *_ in val_loader:
        x_norm, y_norm = preprocess(
            batch_x, batch_y, channel_mixer, channel_mean, channel_std, config.device
        )
        with torch.amp.autocast(config.device_type, enabled=(config.device_type == "cuda"), dtype=amp_dtype):
            _, _, loss = model(x_norm, y_norm)
            per_ch     = model.channel_nll(x_norm, y_norm)  # [C]
        total_loss   += loss.item()
        channel_nlls += per_ch

    n = len(val_loader)
    return total_loss / n, (channel_nlls / n).tolist()


# ============================================================================
# TRAIN ONE EPOCH
# ============================================================================

def train_epoch(model, channel_mixer, loader, optimizer, scaler,
                config, epoch, total_steps, es, channel_mean, channel_std):
    model.train()
    channel_mixer.train()
    num_batches = len(loader)
    epoch_loss  = 0.0
    current_lr  = config.lr
    t0          = time.time()
    amp_dtype   = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16

    # Accumulators for monitoring mean-head activation (Issue 9 diagnostic)
    residual_sq_acc = 0.0

    pbar = tqdm(enumerate(loader), total=num_batches,
                desc=f"Epoch {epoch+1}/{config.epochs}")

    optimizer.zero_grad(set_to_none=True)

    for i, (batch_x, batch_y, *_) in pbar:
        iteration = epoch * num_batches + i

        # LR schedule
        min_lr     = config.lr * config.min_lr_factor
        current_lr = get_lr(iteration, total_steps, config.warmup_steps,
                             config.lr, min_lr, config.decay_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        x_norm, y_norm = preprocess(
            batch_x, batch_y, channel_mixer, channel_mean, channel_std, config.device
        )

        with torch.amp.autocast(config.device_type, enabled=(config.device_type == "cuda"), dtype=amp_dtype):
            mu, log_var, loss = model(x_norm, y_norm)
            loss = loss / config.grad_accumulation

        scaler.scale(loss).backward()

        if (i + 1) % config.grad_accumulation == 0:
            scaler.unscale_(optimizer)
            all_params = list(model.parameters()) + list(channel_mixer.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, config.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_loss  = loss.item() * config.grad_accumulation
        epoch_loss += batch_loss
        avg_loss    = epoch_loss / (i + 1)

        # Track mean squared residual to detect mean-head collapse (Issue 9)
        with torch.no_grad():
            residual_sq_acc += (y_norm - mu.detach()).pow(2).mean().item()

        pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}",
                         pat=f"{es.counter}/{es.patience}")

    mean_residual_sq = residual_sq_acc / num_batches
    return epoch_loss / num_batches, time.time() - t0, current_lr, mean_residual_sq


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default="Epilepsy",
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to train on")
    parser.add_argument("--backbone",  default="transformer",
                        choices=["transformer", "mamba"],
                        help="Density-model backbone")
    parser.add_argument("--d_state",   type=int, default=16,
                        help="Mamba SSM state dimension")
    parser.add_argument("--d_conv",    type=int, default=4,
                        help="Mamba depthwise-conv kernel width")
    parser.add_argument("--d_expand",  type=int, default=2,
                        help="Mamba inner-dim expansion factor")
    args = parser.parse_args()

    cfg = Config()
    dc  = DATASET_CONFIGS[args.dataset]
    cfg.root_path      = dc["root_path"]
    cfg.n_channels     = dc["n_channels"]
    cfg.seq_len        = dc["seq_len"]
    cfg.output_dir     = dc["output_dir"]

    # Backbone config
    cfg.backbone  = args.backbone
    cfg.d_state   = args.d_state
    cfg.d_conv    = args.d_conv
    cfg.d_expand  = args.d_expand

    # Separate checkpoint per backbone so both can coexist in output/
    base = dc["save_path"]
    cfg.save_path = (base.replace("_best.pt", f"_{args.backbone}_best.pt")
                     if args.backbone != "transformer" else base)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("high")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = TimeSeriesDataset(cfg.root_path, "train", cfg.seq_len)
    val_ds   = TimeSeriesDataset(cfg.root_path, "val",   cfg.seq_len)
    train_bs = min(cfg.batch_size, len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=train_bs,
                              shuffle=True,  num_workers=cfg.num_workers,
                              drop_last=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, num_workers=cfg.num_workers,
                              drop_last=False, pin_memory=True)

    # ── Dataset-level normalisation stats (computed once on train set) ────────
    channel_mean, channel_std = compute_channel_stats(train_ds, cfg.device)

    # ── Model ─────────────────────────────────────────────────────────────────
    gpt_cfg = GaussianGPTConfig(
        block_size   = cfg.seq_len,
        n_channels   = cfg.n_channels,
        n_layer      = cfg.n_layer,
        n_head       = cfg.n_head,
        n_embd       = cfg.n_embd,
        dropout      = cfg.dropout,
        bias         = cfg.bias,
        logvar_min   = cfg.logvar_min,
        logvar_max   = cfg.logvar_max,
        calib_weight = cfg.calib_weight,
        backbone     = cfg.backbone,
        d_state      = cfg.d_state,
        d_conv       = cfg.d_conv,
        d_expand     = cfg.d_expand,
    )
    model         = GaussianGPT(gpt_cfg).to(cfg.device)
    channel_mixer = ChannelMixer(n_channels=cfg.n_channels, dropout=0.1).to(cfg.device)

    # ── Optimiser ────────────────────────────────────────────────────────────
    decay_p   = [p for n, p in model.named_parameters()
                 if p.requires_grad and p.dim() >= 2]
    nodecay_p = [p for n, p in model.named_parameters()
                 if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay_p,                      "weight_decay": cfg.weight_decay},
         {"params": nodecay_p,                    "weight_decay": 0.0},
         {"params": channel_mixer.parameters(),   "weight_decay": 0.0}],
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
    )
    scaler = torch.amp.GradScaler(enabled=(cfg.device_type == "cuda" and cfg.dtype == "float16"))

    total_steps = (len(train_loader) // cfg.grad_accumulation) * cfg.epochs
    es          = EarlyStopping(patience=cfg.patience, save_path=cfg.save_path)

    total_params = (sum(p.numel() for p in model.parameters()) +
                    sum(p.numel() for p in channel_mixer.parameters()))
    print(f"Total params: {total_params:,}  |  device: {cfg.device}  |  steps: {total_steps}")

    # ── Training loop ─────────────────────────────────────────────────────────
    t_start = time.time()
    for epoch in range(cfg.epochs):
        train_loss, t_train, lr, mean_residual_sq = train_epoch(
            model, channel_mixer, train_loader, optimizer,
            scaler, cfg, epoch, total_steps, es, channel_mean, channel_std
        )
        val_loss, val_channel_nlls = evaluate(
            model, channel_mixer, val_loader, cfg, channel_mean, channel_std
        )

        print(f"Epoch {epoch+1:3d}/{cfg.epochs} | "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={lr:.2e}  t={t_train:.1f}s  "
              f"residual²={mean_residual_sq:.4f}")

        es(val_loss, model, channel_mixer, epoch + 1, channel_mean, channel_std)
        if es.early_stop:
            print(f"Early stopping at epoch {epoch+1}  (best val={es.val_min:.4f})")
            break

    # ── Finish ────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed/60:.1f} min")
    print(f"Best val loss: {es.val_min:.4f}  →  {cfg.save_path}")

    return model, channel_mixer


if __name__ == "__main__":
    model, channel_mixer = main()
