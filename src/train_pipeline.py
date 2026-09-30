"""
train_pipeline.py — End-to-end CAFE-TS pipeline trainer.

Phases executed in sequence (each gated by go/no-go checks):
  1. Load frozen density model checkpoint (Phase 2 must be done first)
  2. Extract signatures + fit concept space (Phases 4-5)
  3. Train ConceptEncoder (Phase 6)
  4. Train classifier head (Phase 7)
  5. Save full pipeline artifact + report metrics

Usage:
  python train_pipeline.py --dataset HAR --M 32 --patcher entropy
  python train_pipeline.py --dataset HAR --M 32 --patcher static  # ablation
  python train_pipeline.py --dataset HAR --sweep_M                 # scan M
"""

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
except ImportError:  # W&B is optional; use --no_wandb to skip logging.
    class _WandbStub:
        run = None
        def init(self, *a, **k): pass
        def log(self, *a, **k): pass
        def finish(self, *a, **k): pass
    wandb = _WandbStub()

from GaussianEntropyModel import GaussianGPT, GaussianGPTConfig
from train_gaussian_entropy_model import TimeSeriesDataset, ChannelMixer, DATASET_CONFIGS
from patcher import EntropyPatcher, StaticPatcher, GreedyDualThresholdPatcher, normalize_batch, get_patch_config
from signatures import (compute_marginal, extract_signatures_for_dataset,
                         SignatureStandardizer, signature_dim)
from concept_space import ConceptSpace, sweep_M
from concept_encoder import (ConceptEncoder, PatchDataset, patch_collate_fn,
                              augment_batch_patches, align_loss, sparse_loss,
                              stable_loss, build_patch_dataset)
from classifier import (TransformerClassifier, SparseLinearClassifier,
                         BigramLinearClassifier, ConceptPipeline, RawPatchTransformer)


# ── Dataset configs (augment existing DATASET_CONFIGS) ────────────────────────

PIPELINE_CONFIGS = {
    "HAR":       {"n_classes": 6,  "sig_mode": "surprise_full", "temperature": 1.0,
                  "lambda_align": 1.0, "lambda_sparse": 0.7, "lambda_stable": 0.5},
    "Epilepsy":  {"n_classes": 2,  "sig_mode": "surprise_full", "temperature": 1.0,
                  "lambda_align": 1.0, "lambda_sparse": 0.6, "lambda_stable": 0.5},
    "SLeep-EDF": {"n_classes": 5,  "sig_mode": "surprise_full", "temperature": 1.0,
                  "lambda_align": 1.0, "lambda_sparse": 0.05, "lambda_stable": 0.1},
    "FD-A":        {"n_classes": 3,  "sig_mode": "surprise_full", "temperature": 1.0,
                  "lambda_align": 1.0, "lambda_sparse": 0.1, "lambda_stable": 0.1},
    "FD-B":        {"n_classes": 3,  "sig_mode": "surprise_full", "temperature": 1.0,
                  "lambda_align": 1.0, "lambda_sparse": 0.1, "lambda_stable": 0.1},
    "FD-C":        {"n_classes": 3,  "sig_mode": "surprise_full", "temperature": 1.0,
                  "lambda_align": 1.0, "lambda_sparse": 0.1, "lambda_stable": 0.1},
    "FD-D":        {"n_classes": 3,  "sig_mode": "surprise_full", "temperature": 1.0,
                  "lambda_align": 1.0, "lambda_sparse": 0.1, "lambda_stable": 0.1},
    # ── UEA/UCR archive datasets ───────────────────────────────────────────────
    "EthanolConcentration": {"n_classes": 4,  "sig_mode": "surprise_full", "temperature": 1.0,
                             "lambda_align": 1.0, "lambda_sparse": 0.5, "lambda_stable": 0.3},
    "FaceDetection":        {"n_classes": 2,  "sig_mode": "surprise_full", "temperature": 1.0,
                             "lambda_align": 1.0, "lambda_sparse": 0.5, "lambda_stable": 0.3},
    "Handwriting":          {"n_classes": 26, "sig_mode": "surprise_full", "temperature": 1.0,
                             "lambda_align": 1.0, "lambda_sparse": 0.5, "lambda_stable": 0.3},
    "Heartbeat":            {"n_classes": 2,  "sig_mode": "surprise_full", "temperature": 1.0,
                             "lambda_align": 1.0, "lambda_sparse": 0.5, "lambda_stable": 0.3},
    "JapaneseVowels":       {"n_classes": 9,  "sig_mode": "surprise_full", "temperature": 1.0,
                             "lambda_align": 1.0, "lambda_sparse": 0.5, "lambda_stable": 0.3},
    "PEMS-SF":              {"n_classes": 7,  "sig_mode": "surprise_full", "temperature": 1.0,
                             "lambda_align": 1.0, "lambda_sparse": 0.5, "lambda_stable": 0.3},
    "SelfRegulationSCP1":   {"n_classes": 2,  "sig_mode": "surprise_full", "temperature": 1.0,
                             "lambda_align": 1.0, "lambda_sparse": 0.5, "lambda_stable": 0.3},
    "SelfRegulationSCP2":   {"n_classes": 2,  "sig_mode": "surprise_full", "temperature": 1.0,
                             "lambda_align": 1.0, "lambda_sparse": 0.5, "lambda_stable": 0.3},
    "SpokenArabicDigits":   {"n_classes": 10, "sig_mode": "surprise_full", "temperature": 1.0,
                             "lambda_align": 1.0, "lambda_sparse": 0.5, "lambda_stable": 0.3},
    "UWaveGestureLibrary":  {"n_classes": 8,  "sig_mode": "surprise_full", "temperature": 1.0,
                             "lambda_align": 1.0, "lambda_sparse": 0.5, "lambda_stable": 0.3},
}


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_density_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd   = ckpt["model_state_dict"]

    if "model_config" in ckpt:
        cfg = GaussianGPTConfig(**ckpt["model_config"])
    else:
        n_embd, n_channels = sd["transformer.input_proj.weight"].shape
        block_size          = sd["transformer.wpe.weight"].shape[0]
        n_layer             = sum(1 for k in sd
                                  if k.startswith("transformer.h.")
                                  and k.endswith(".ln_1.weight"))
        cfg = GaussianGPTConfig(block_size=block_size, n_channels=n_channels,
                                n_layer=n_layer, n_embd=n_embd)

    model = GaussianGPT(cfg).to(device)
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_ch          = ckpt.get("n_channels", cfg.n_channels)
    channel_mixer = ChannelMixer(n_channels=n_ch).to(device)
    channel_mixer.load_state_dict(ckpt["channel_mixer_state_dict"])
    channel_mixer.eval()
    for p in channel_mixer.parameters():
        p.requires_grad_(False)

    channel_mean = ckpt["channel_mean"].to(device)
    channel_std  = ckpt["channel_std"].to(device)

    print(f"Loaded density model  val_loss={ckpt.get('val_loss', '?'):.4f}"
          f"  epoch={ckpt.get('epoch', '?')}")
    return model, channel_mixer, channel_mean, channel_std


# ── Go/no-go gate ─────────────────────────────────────────────────────────────

def gate(condition: bool, msg: str,
         bypass: bool = False, gate_failures: Optional[List[str]] = None):
    """
    Raise on failure in production; log and continue in ablation mode.

    bypass=True (--no_gate):  skipped gates are appended to gate_failures
    so they appear in the artifact and ablation CSV — the gate firing IS
    a result, not just an error.
    """
    if not condition:
        if bypass:
            print(f"  [gate-SKIPPED] {msg}")
            if gate_failures is not None:
                gate_failures.append(msg)
        else:
            raise RuntimeError(f"Go/no-go FAILED: {msg}")
    else:
        print(f"  [gate] PASS: {msg}")


# ── Encoder training ──────────────────────────────────────────────────────────

def train_encoder(encoder: ConceptEncoder,
                  train_dataset: PatchDataset,
                  val_dataset: PatchDataset,
                  args, device: torch.device,
                  pc: dict,
                  wandb_step_offset: int = 0) -> tuple:
    """Returns (best_val_align_loss, epochs_run)."""
    loader = DataLoader(train_dataset, batch_size=args.enc_batch,
                        shuffle=True, collate_fn=patch_collate_fn,
                        num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.enc_batch,
                            shuffle=False, collate_fn=patch_collate_fn,
                            num_workers=0, drop_last=False)

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.enc_lr,
                                   weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = len(loader) * args.enc_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.enc_lr * 0.05)

    la = pc["lambda_align"]
    ls = pc["lambda_sparse"]
    lt = pc["lambda_stable"]

    best_val, patience_count, best_sd = float("inf"), 0, None

    for epoch in range(args.enc_epochs):
        encoder.train()
        ep_align = ep_sparse = ep_stable = ep_total = 0.0

        for patches, mask, pi_target, _ in loader:
            patches   = patches.to(device)
            mask      = mask.to(device)
            pi_target = pi_target.to(device)

            q = encoder(patches, mask)
            with torch.no_grad():                                       # no graph needed — q_aug is detached anyway
                q_aug = encoder(augment_batch_patches(patches, mask), mask)

            l_align  = align_loss(q, pi_target)
            l_sparse = sparse_loss(q)
            l_stable = stable_loss(q, q_aug)
            loss     = la * l_align + ls * l_sparse + lt * l_stable

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            a, sp, st, tot = torch.stack([l_align, l_sparse, l_stable, loss]).tolist()  # one sync
            ep_align += a; ep_sparse += sp; ep_stable += st; ep_total += tot

        n = len(loader)
        encoder.eval()
        val_align = 0.0
        hard_match = 0
        n_val = 0
        with torch.no_grad():
            for patches, mask, pi_target, _ in val_loader:
                patches   = patches.to(device)
                mask      = mask.to(device)
                pi_target = pi_target.to(device)
                q = encoder(patches, mask)
                b = len(q)
                val_align  += align_loss(q, pi_target).item() * b   # accumulate sum, not mean
                hard_match += (q.argmax(-1) == pi_target.argmax(-1)).sum().item()
                n_val += b

        val_align /= max(n_val, 1)
        hard_acc   = hard_match / max(n_val, 1)

        print(f"  Enc epoch {epoch+1:3d}/{args.enc_epochs}"
              f"  align={ep_align/n:.4f}  sparse={ep_sparse/n:.4f}"
              f"  stable={ep_stable/n:.4f}"
              f"  val_align={val_align:.4f}  hard_acc={hard_acc:.3f}")

        if wandb.run is not None:
            wandb.log({"enc/train_align": ep_align / n,
                       "enc/val_align":   val_align,
                       "enc/hard_acc":    hard_acc},
                      step=wandb_step_offset + epoch)

        if val_align < best_val:
            best_val       = val_align
            patience_count = 0
            best_sd        = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= args.enc_patience:
                print(f"  Early stopping encoder at epoch {epoch+1}")
                break

    # gate: hard assignment accuracy > 80%
    gate(hard_acc > 0.5, f"Encoder hard-assignment accuracy {hard_acc:.2%} > 50%")

    if best_sd is not None:
        encoder.load_state_dict({k: v.to(device) for k, v in best_sd.items()})
    epochs_run = min(epoch + 1, args.enc_epochs)
    return best_val, epochs_run


# ── Classifier training ───────────────────────────────────────────────────────

def build_concept_sequences(encoder: ConceptEncoder,
                              patch_dataset: PatchDataset,
                              K: int, device: torch.device,
                              batch_size: int = 64):
    """
    Encode all patches → group back into per-sample concept sequences [K, M].
    Returns: concept_seqs [N_samples, K, M], labels [N_samples]
    """
    loader = DataLoader(patch_dataset, batch_size=batch_size,
                        shuffle=False, collate_fn=patch_collate_fn,
                        num_workers=0, drop_last=False)
    encoder.eval()
    all_q, all_labels = [], []
    with torch.no_grad():
        for patches, mask, pi, labels in loader:
            patches = patches.to(device)
            mask    = mask.to(device)
            q = encoder(patches, mask)      # [B_patches, M]
            all_q.append(q.cpu())
            all_labels.append(labels)

    all_q      = torch.cat(all_q, 0)        # [N_patches_total, M]
    all_labels = torch.cat(all_labels, 0)   # [N_patches_total]

    # Group patches back into signals (K patches per signal)
    N_total = len(all_q)
    assert N_total % K == 0, f"Total patches {N_total} not divisible by K={K}"
    N_samples = N_total // K

    concept_seqs = all_q.reshape(N_samples, K, -1)   # [N, K, M]
    # take label from first patch of each sample
    labels_per_sample = all_labels[::K]               # [N]
    return concept_seqs, labels_per_sample


def build_concept_sequences_direct(
    sigs_std: torch.Tensor,
    sample_ids: torch.Tensor,
    concept_space,
    K: int,
    dataset_labels: torch.Tensor,
    device: torch.device,
    temperature: float = 1.0,
):
    """
    Direct path (--direct_sig): bypass encoder, use soft_assign from signatures.
    sigs_std: [N_patches, D], sample_ids: [N_patches].
    Returns: concept_seqs [N_samples, K, M], labels [N_samples].
    """
    with torch.no_grad():
        soft_probs = concept_space.soft_assign(
            sigs_std.to(device), temperature=temperature
        )  # [N_patches, M]
    N_patches = len(sigs_std)
    assert N_patches % K == 0, f"Total patches {N_patches} not divisible by K={K}"
    N_samples    = N_patches // K
    concept_seqs = soft_probs.cpu().reshape(N_samples, K, -1)   # [N, K, M]
    labels       = dataset_labels[sample_ids[::K]].long()        # [N]
    return concept_seqs, labels


def stratified_subset(cseq: torch.Tensor, labels: torch.Tensor,
                       fraction: float, seed: int = 42):
    """Return a stratified subset keeping `fraction` of samples per class (min 1)."""
    if fraction >= 1.0:
        return cseq, labels
    rng = torch.Generator().manual_seed(seed)
    keep_idx = []
    for c in labels.unique():
        idx = (labels == c).nonzero(as_tuple=True)[0]
        n_keep = max(1, round(len(idx) * fraction))
        perm = torch.randperm(len(idx), generator=rng)
        keep_idx.append(idx[perm[:n_keep]])
    keep_idx = torch.cat(keep_idx)
    return cseq[keep_idx], labels[keep_idx]


def train_classifier(classifier: nn.Module,
                      concept_seqs_train: torch.Tensor,
                      labels_train: torch.Tensor,
                      concept_seqs_val: torch.Tensor,
                      labels_val: torch.Tensor,
                      args, device: torch.device,
                      wandb_step_offset: int = 0,
                      tag: str = ""):
    """Returns best val accuracy."""
    from torch.utils.data import TensorDataset

    train_ds = TensorDataset(concept_seqs_train, labels_train)
    val_ds   = TensorDataset(concept_seqs_val,   labels_val)
    train_ld = DataLoader(train_ds, batch_size=args.cls_batch, shuffle=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.cls_batch, shuffle=False)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.cls_lr,
                                   weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_ld) * args.cls_epochs, eta_min=args.cls_lr * 0.05)

    use_l1 = hasattr(classifier, 'l1_loss')
    best_val_acc, best_sd, best_epoch = 0.0, None, None

    for epoch in range(args.cls_epochs):
        classifier.train()
        ep_loss = 0.0

        for cseq, lbl in train_ld:
            cseq = cseq.to(device)
            lbl  = lbl.to(device)
            logits = classifier(cseq)
            loss   = F.cross_entropy(logits, lbl)
            if use_l1:
                loss = loss + classifier.l1_loss()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ep_loss += loss.item()

        classifier.eval()
        correct = total = 0
        with torch.no_grad():
            for cseq, lbl in val_ld:
                cseq = cseq.to(device)
                lbl  = lbl.to(device)
                preds = classifier(cseq).argmax(-1)
                correct += (preds == lbl).sum().item()
                total   += len(lbl)

        val_acc = correct / max(total, 1)
        # print(f"  Cls epoch {epoch+1:3d}/{args.cls_epochs}"
        #       f"  loss={ep_loss/len(train_ld):.4f}  val_acc={val_acc:.4f}")
        if wandb.run is not None:
            wandb.log({f"cls{tag}/val_acc": val_acc, f"cls{tag}/train_loss": ep_loss / len(train_ld)},
                      step=wandb_step_offset + epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_sd = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}
            best_epoch = epoch + 1

    if best_sd is not None:
        classifier.load_state_dict({k: v.to(device) for k, v in best_sd.items()})
    classifier.selected_epoch = best_epoch
    return best_val_acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="HAR",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--ckpt",       default=None,
                        help="Density model checkpoint (defaults to dataset save_path)")
    parser.add_argument("--patcher",    default="entropy",
                        choices=["entropy", "static", "greedy"],
                        help="entropy: DP segmentation  static: fixed-window  greedy: dual-threshold EntroPE-style")
    parser.add_argument("--patch_scale", default=None,
                        choices=["xs", "s", "m", "l", "xl"],
                        help="Patch scale override (xs=tiny … xl=large). "
                             "Default: per-dataset value in DATASET_PATCH_SCALE.")
    parser.add_argument("--patch_K", type=int, default=None,
                        help="Explicit patch count. Rebuttal runs use this instead of "
                             "the mutable repository policy.")
    parser.add_argument("--patch_L_min", type=int, default=None,
                        help="Explicit minimum patch length.")
    parser.add_argument("--patch_L_max", type=int, default=None,
                        help="Explicit maximum patch length.")
    parser.add_argument("--patch_burn_in", type=int, default=None,
                        help="Explicit unreliable-prefix length for the boundary score.")
    parser.add_argument("--boundary_mode", default="surprise",
                        choices=["entropy", "surprise", "kl_shift", "residual"])
    parser.add_argument("--M",          type=int, default=16,
                        help="Vocabulary size (# concept prototypes)")
    parser.add_argument("--sweep_M",    action="store_true",
                        help="Sweep M and report metrics before training")
    parser.add_argument("--cluster_algo", default="gmm",
                        choices=["kmeans", "gmm", "hdbscan"])
    parser.add_argument("--sig_mode",   default=None, nargs="+",
                        metavar="MODE",
                        help="Signature mode(s): full | entropy_only | moments_only | "
                             "distributional_only | morphological_only | "
                             "residual_morphology | surprise_trajectory | surprise_full. "
                             "Multiple modes are concatenated. (default: from PIPELINE_CONFIGS)")
    parser.add_argument("--classifier_type", default="transformer",
                        choices=["transformer", "linear", "bigram"])

    # Concept-encoder loss weights
    parser.add_argument("--lambda_align", type=float, default=None,
                        help="Weight for KL alignment loss. If None, use dataset default.")
    parser.add_argument("--lambda_sparse", type=float, default=None,
                        help="Weight for concept sparsity / entropy loss. If None, use dataset default.")
    parser.add_argument("--lambda_stable", type=float, default=None,
                        help="Weight for augmentation-stability loss. If None, use dataset default.")

    # Encoder hyperparams
    parser.add_argument("--enc_d_model",  type=int,   default=64)
    parser.add_argument("--enc_n_layers", type=int,   default=2)
    parser.add_argument("--enc_n_head",   type=int,   default=4)
    parser.add_argument("--enc_epochs",   type=int,   default=50)
    parser.add_argument("--enc_lr",       type=float, default=3e-3)
    parser.add_argument("--enc_batch",    type=int,   default=128)
    parser.add_argument("--enc_patience", type=int,   default=10)

    # Classifier hyperparams
    parser.add_argument("--cls_epochs",   type=int,   default=200)
    parser.add_argument("--cls_lr",       type=float, default=3e-3)
    parser.add_argument("--cls_batch",    type=int,   default=64)

    parser.add_argument("--device",     default=None)
    parser.add_argument("--no_wandb",   action="store_true")
    parser.add_argument("--no_gate",    action="store_true",
                        help="Bypass go/no-go gates (ablation mode). "
                             "Skipped gates are logged to the artifact as gate_failures.")
    parser.add_argument("--direct_sig", action="store_true",
                        help="Skip encoder training (Phase 6). Use soft_assign(signatures) "
                             "directly as concept sequences. Density model still runs for patching.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed for reproducibility.")
    parser.add_argument("--quick_test", type=int, default=None, metavar="N",
                        help="Subsample each split to N samples for fast debugging.")
    parser.add_argument("--no_sig_cache", action="store_true",
                        help="Disable signature caching (forces re-extraction).")
    parser.add_argument(
        "--sig_cache_dir",
        default=None,
        help="Optional cache directory. Use a density-checkpoint-specific directory "
             "when comparing configurations that share frozen signatures.",
    )
    parser.add_argument("--label_fraction", type=float, default=None,
                        help="Train only this classifier fraction instead of 1/5/100%%.")
    parser.add_argument("--split_file", default=None,
                        help="Immutable JSON labeled indices; required for controlled rebuttal runs.")
    parser.add_argument("--fit_scope", choices=["all", "labeled", "indices"], default="all",
                        help="Inputs used to fit marginal, standardizer, and GMM.")
    parser.add_argument(
        "--fit_indices_file",
        default=None,
        help="JSON with `indices` or `unlabeled_indices`; required when "
             "--fit_scope=indices. Classifier labels remain controlled by --split_file.",
    )
    parser.add_argument("--output_dir", default=None,
                        help="Unique pipeline output directory override.")
    parser.add_argument("--protocol", default=None,
                        help="Protocol label saved in the artifact.")
    args = parser.parse_args()

    # Fix all RNGs so results are reproducible across runs
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )

    dc  = DATASET_CONFIGS[args.dataset]
    pc  = dict(PIPELINE_CONFIGS[args.dataset])  # copy so CLI overrides do not modify global config
    ppc = get_patch_config(args.dataset, dc["seq_len"], scale=args.patch_scale)
    explicit_patch_values = {
        "K": args.patch_K,
        "L_min": args.patch_L_min,
        "L_max": args.patch_L_max,
        "burn_in": args.patch_burn_in,
    }
    for key, value in explicit_patch_values.items():
        if value is not None:
            ppc[key] = value
    if any(value is not None for value in explicit_patch_values.values()):
        missing = [
            key for key in ("K", "L_min", "L_max", "burn_in")
            if explicit_patch_values[key] is None
        ]
        if missing:
            raise ValueError(
                "Explicit patch configuration must set all of --patch_K, "
                "--patch_L_min, --patch_L_max, and --patch_burn_in; missing "
                + ", ".join(missing)
            )
    effective_patch_length = int(dc["seq_len"])
    if ppc["K"] * ppc["L_min"] > effective_patch_length:
        raise ValueError(
            "Infeasible patch configuration: "
            f"K*L_min={ppc['K'] * ppc['L_min']} exceeds "
            f"T={effective_patch_length}"
        )
    if ppc["L_max"] is not None and ppc["K"] * ppc["L_max"] < effective_patch_length:
        raise ValueError(
            "Infeasible patch configuration: "
            f"K*L_max={ppc['K'] * ppc['L_max']} is below "
            f"T={effective_patch_length}"
        )

    # Optional CLI overrides for concept-encoder loss weights
    if args.lambda_align is not None:
        pc["lambda_align"] = args.lambda_align
    if args.lambda_sparse is not None:
        pc["lambda_sparse"] = args.lambda_sparse
    if args.lambda_stable is not None:
        pc["lambda_stable"] = args.lambda_stable

    print(
        f"Loss weights: "
        f"lambda_align={pc['lambda_align']}  "
        f"lambda_sparse={pc['lambda_sparse']}  "
        f"lambda_stable={pc['lambda_stable']}"
    )

    sig_mode     = args.sig_mode if args.sig_mode is not None else pc["sig_mode"]
    if isinstance(sig_mode, list) and len(sig_mode) == 1:
        sig_mode = sig_mode[0]   # single mode → plain string (artifact backward compat)
    ckpt_path    = args.ckpt or dc["save_path"]
    out_dir      = (
        Path(args.output_dir)
        if args.output_dir
        else Path(dc["output_dir"]) / "pipeline" / args.patcher
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gate state: accumulates skipped-gate messages when --no_gate is set.
    # The list is saved to the artifact so the ablation table can show gate_status.
    gate_failures: List[str] = []
    bypass_gates  = args.no_gate

    if not args.no_wandb:
        wandb.init(
            project=f"CAFE-TS-{args.dataset}",
            tags=[args.dataset, args.patcher, f"M{args.M}", args.cluster_algo],
            config=vars(args),
        )

    # ── Phase 2: Load frozen density model ───────────────────────────────────
    print("\n── Phase 2: Loading density model ─────────────────────────")
    density_model, channel_mixer, channel_mean, channel_std = load_density_model(
        ckpt_path, device
    )

    # ── Phase 1: Load datasets ────────────────────────────────────────────────
    print("\n── Phase 1: Loading datasets ───────────────────────────────")
    train_ds = TimeSeriesDataset(dc["root_path"], "train", dc["seq_len"])
    val_ds   = TimeSeriesDataset(dc["root_path"], "val",   dc["seq_len"])
    test_ds  = TimeSeriesDataset(dc["root_path"], "test",  dc["seq_len"])

    labeled_indices = None
    split_payload = None
    if args.split_file:
        split_payload = json.loads(Path(args.split_file).read_text())
        if split_payload.get("dataset") != args.dataset:
            raise ValueError(
                f"Split dataset={split_payload.get('dataset')} does not match {args.dataset}"
            )
        if split_payload.get("seed") != args.seed:
            raise ValueError(
                f"Split seed={split_payload.get('seed')} does not match --seed={args.seed}"
            )
        if args.label_fraction is not None and not math.isclose(
            float(split_payload.get("fraction")), args.label_fraction
        ):
            raise ValueError("Split fraction does not match --label_fraction")
        saved_labeled_indices = torch.tensor(
            split_payload["labeled_indices"], dtype=torch.long
        )
        if split_payload.get("total_train_count") != len(train_ds):
            raise ValueError("Split total_train_count does not match loaded training data")
        labeled_indices = saved_labeled_indices
    elif args.label_fraction is not None or args.fit_scope in {"labeled", "indices"}:
        raise ValueError("Controlled --label_fraction/--fit_scope labeled runs require --split_file")

    if args.quick_test is not None:
        N = args.quick_test
        if labeled_indices is not None:
            # A prefix-only smoke subset can accidentally contain none of the
            # immutable labeled examples. Preserve every saved labeled example,
            # add a small deterministic unlabeled prefix, then remap indices.
            prefix = torch.arange(min(N, len(train_ds)), dtype=torch.long)
            keep = torch.unique(torch.cat([prefix, labeled_indices]), sorted=True)
            remap = {int(old): new for new, old in enumerate(keep.tolist())}
            labeled_indices = torch.tensor(
                [remap[int(old)] for old in saved_labeled_indices.tolist()],
                dtype=torch.long,
            )
            train_ds.samples = train_ds.samples[keep]
            train_ds.labels = train_ds.labels[keep]
            print(
                f"  [quick_test] Train uses {len(keep)} deterministic samples "
                f"including all {len(labeled_indices)} immutable labeled inputs."
            )
        else:
            train_ds.samples = train_ds.samples[:N]
            train_ds.labels = train_ds.labels[:N]
        for ds in (val_ds, test_ds):
            ds.samples = ds.samples[:N]
            ds.labels = ds.labels[:N]
        print(f"  [quick_test] Validation/test use the first {N} samples.")

    if args.split_file:
        print(f"  Loaded {len(labeled_indices)} immutable labeled indices from {args.split_file}")

    fit_train_ds = train_ds
    marginal_train_ds = train_ds
    fit_sample_indices = None
    classifier_indices = labeled_indices
    if args.fit_scope == "labeled":
        fit_train_ds = copy.copy(train_ds)
        fit_train_ds.samples = train_ds.samples[labeled_indices]
        fit_train_ds.labels = train_ds.labels[labeled_indices]
        # From this point on the strict protocol must not even transform the
        # held-out training inputs. Sample ids are therefore local to the
        # immutable subset, while the artifact still retains the original
        # split indices for provenance.
        classifier_indices = torch.arange(len(fit_train_ds), dtype=torch.long)
        marginal_train_ds = fit_train_ds
        print(f"  Fit scope: labeled subset only ({len(fit_train_ds)} inputs)")
    elif args.fit_scope == "indices":
        if not args.fit_indices_file:
            raise ValueError("--fit_scope=indices requires --fit_indices_file")
        fit_payload = json.loads(Path(args.fit_indices_file).read_text())
        raw_indices = fit_payload.get(
            "indices", fit_payload.get("unlabeled_indices")
        )
        if raw_indices is None:
            raise ValueError(
                "Fit-index JSON must contain `indices` or `unlabeled_indices`"
            )
        fit_sample_indices = torch.tensor(raw_indices, dtype=torch.long)
        if len(fit_sample_indices) == 0:
            raise ValueError("Fit-index subset is empty")
        if fit_sample_indices.min() < 0 or fit_sample_indices.max() >= len(train_ds):
            raise ValueError("Fit-index subset contains an out-of-range sample")
        if torch.unique(fit_sample_indices).numel() != len(fit_sample_indices):
            raise ValueError("Fit-index subset contains duplicates")
        marginal_train_ds = copy.copy(train_ds)
        marginal_train_ds.samples = train_ds.samples[fit_sample_indices]
        marginal_train_ds.labels = train_ds.labels[fit_sample_indices]
        print(
            f"  Fit scope: explicit nested subset "
            f"({len(fit_sample_indices)} of {len(train_ds)} inputs)"
        )
    else:
        print(f"  Fit scope: all training inputs ({len(fit_train_ds)} inputs)")

    # ── Patch config (always needed: K used in Phase 6 + artifact) ──────────────
    K      = ppc["K"]
    L_min  = ppc["L_min"]
    L_max  = ppc["L_max"]
    burn_in = ppc["burn_in"]
    print(
        "  Patch config: "
        f"K={K} L_min={L_min} L_max={L_max} burn_in={burn_in} "
        f"effective_T={effective_patch_length}"
    )

    # ── Signature cache key ───────────────────────────────────────────────────
    _sig_mode_key   = "-".join(sig_mode) if isinstance(sig_mode, list) else sig_mode
    _patch_scale_key = (
        f"K{K}-Lmin{L_min}-Lmax{L_max}-burn{burn_in}"
        if any(value is not None for value in explicit_patch_values.values())
        else args.patch_scale or "default"
    )
    _sample_scope_key = (
        f"quick-{args.quick_test}" if args.quick_test is not None else "full"
    )
    _sig_cache_key  = (f"{args.dataset}__{args.patcher}__{args.boundary_mode}"
                       f"__{_sig_mode_key}__{_patch_scale_key}__{_sample_scope_key}")
    _sig_cache_dir  = (
        Path(args.sig_cache_dir)
        if args.sig_cache_dir
        else Path(dc["output_dir"]) / "sig_cache"
    )
    _sig_cache_path = _sig_cache_dir / f"{_sig_cache_key}.pt"
    _use_sig_cache  = (
        not args.no_sig_cache
        and args.fit_scope == "all"
        and (
            args.sig_cache_dir is not None
            or (args.quick_test is None and args.split_file is None)
        )
    )

    C = dc["n_channels"]
    D = signature_dim(C, sig_mode)

    if _use_sig_cache and _sig_cache_path.exists():
        # ── Phase 3+4: Load cached signatures ────────────────────────────────
        print(f"\n── Phase 3+4: Loading cached signatures ────────────────────")
        print(f"  Key:   {_sig_cache_key}")
        print(f"  Cache: {_sig_cache_path}")
        _cache = torch.load(_sig_cache_path, map_location="cpu", weights_only=False)
        train_sigs         = _cache["train_sigs"].to(device)
        train_patch_ranges = _cache["train_patch_ranges"]
        train_sample_ids   = _cache["train_sample_ids"]
        val_sigs           = _cache["val_sigs"].to(device)
        val_patch_ranges   = _cache["val_patch_ranges"]
        val_sample_ids     = _cache["val_sample_ids"]
        test_sigs          = _cache["test_sigs"].to(device)
        test_patch_ranges  = _cache["test_patch_ranges"]
        test_sample_ids    = _cache["test_sample_ids"]
        if "mu_marg" in _cache:
            mu_marg     = _cache["mu_marg"].to(device)
            sigma2_marg = _cache["sigma2_marg"].to(device)
        else:
            # Old cache pre-dates mu_marg storage — compute on the fly
            print("  (old cache: recomputing mu_marg/sigma2_marg)")
            mu_marg, sigma2_marg = compute_marginal(
                density_model, channel_mixer, marginal_train_ds,
                channel_mean, channel_std, device
            )
        print(f"  Signature dim = {D}  (C={C} mode={sig_mode})")
        print(f"  Patches: train={len(train_sigs)}  val={len(val_sigs)}  test={len(test_sigs)}")
        # Rebuild patcher (cheap — no data pass) if encoder training will need it
        if not args.direct_sig:
            print(f"  Rebuilding patcher for encoder training…")
            if args.patcher == "entropy":
                patcher = EntropyPatcher(density_model, channel_mixer,
                                          K=K, L_min=L_min, L_max=L_max,
                                          burn_in=burn_in,
                                          mode=args.boundary_mode)
            elif args.patcher == "greedy":
                patcher = GreedyDualThresholdPatcher(density_model, channel_mixer,
                                                      K=K, L_min=L_min, burn_in=burn_in,
                                                      mode=args.boundary_mode)
            else:
                patcher = StaticPatcher(K=K)
    else:
        # ── Phase 4 (marginal) ────────────────────────────────────────────────
        print("\n── Phase 4: Computing training marginal distribution ───────")
        mu_marg, sigma2_marg = compute_marginal(
            density_model, channel_mixer, marginal_train_ds,
            channel_mean, channel_std, device
        )

        # ── Phase 3: Build patcher ────────────────────────────────────────────
        print(f"\n── Phase 3: Patcher = {args.patcher} ──────────────────────")
        if args.patcher == "entropy":
            patcher = EntropyPatcher(density_model, channel_mixer,
                                      K=K, L_min=L_min, L_max=L_max,
                                      burn_in=burn_in,
                                      mode=args.boundary_mode)
        elif args.patcher == "greedy":
            patcher = GreedyDualThresholdPatcher(density_model, channel_mixer,
                                                  K=K, L_min=L_min, burn_in=burn_in,
                                                  mode=args.boundary_mode)
        else:
            patcher = StaticPatcher(K=K)

        # ── Phase 4: Extract signatures ───────────────────────────────────────
        print("\n── Phase 4: Extracting signatures ──────────────────────────")
        print(f"  Signature dim = {D}  (C={C} mode={sig_mode})")

        print("  Train split…")
        train_sigs, train_patch_ranges, train_sample_ids = extract_signatures_for_dataset(
            density_model, channel_mixer, fit_train_ds, patcher,
            channel_mean, channel_std, mu_marg, sigma2_marg, device, mode=sig_mode
        )
        print("  Val split…")
        val_sigs, val_patch_ranges, val_sample_ids = extract_signatures_for_dataset(
            density_model, channel_mixer, val_ds, patcher,
            channel_mean, channel_std, mu_marg, sigma2_marg, device, mode=sig_mode
        )
        print("  Test split…")
        test_sigs, test_patch_ranges, test_sample_ids = extract_signatures_for_dataset(
            density_model, channel_mixer, test_ds, patcher,
            channel_mean, channel_std, mu_marg, sigma2_marg, device, mode=sig_mode
        )
        print(f"  Patches: train={len(train_sigs)}  val={len(val_sigs)}  test={len(test_sigs)}")

        if _use_sig_cache:
            _sig_cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "train_sigs":         train_sigs.cpu(),
                "train_patch_ranges": train_patch_ranges,
                "train_sample_ids":   train_sample_ids,
                "val_sigs":           val_sigs.cpu(),
                "val_patch_ranges":   val_patch_ranges,
                "val_sample_ids":     val_sample_ids,
                "test_sigs":          test_sigs.cpu(),
                "test_patch_ranges":  test_patch_ranges,
                "test_sample_ids":    test_sample_ids,
                "mu_marg":            mu_marg.cpu(),
                "sigma2_marg":        sigma2_marg.cpu(),
            }, _sig_cache_path)
            print(f"  Saved signature cache → {_sig_cache_path}")

    def patch_diagnostics(ranges):
        layouts = [
            tuple((int(start), int(end)) for start, end in sample_ranges)
            for sample_ranges in ranges
        ]
        lengths = [
            end - start
            for sample_ranges in layouts
            for start, end in sample_ranges
        ]
        return {
            "sample_count": len(layouts),
            "unique_layout_count": len(set(layouts)),
            "observed_patch_counts": sorted(
                {len(sample_ranges) for sample_ranges in layouts}
            ),
            "minimum_patch_length": min(lengths) if lengths else None,
            "maximum_patch_length": max(lengths) if lengths else None,
        }

    segmentation_diagnostics = {
        "train": patch_diagnostics(train_patch_ranges),
        "validation": patch_diagnostics(val_patch_ranges),
        "test": patch_diagnostics(test_patch_ranges),
    }
    print(f"  Segmentation diagnostics: {json.dumps(segmentation_diagnostics)}")
    for split_name, diagnostics in segmentation_diagnostics.items():
        if diagnostics["observed_patch_counts"] != [K]:
            raise RuntimeError(
                f"{split_name} produced patch counts "
                f"{diagnostics['observed_patch_counts']} instead of exactly K={K}"
            )

    # Standardize
    if fit_sample_indices is None:
        fit_patch_mask = torch.ones(len(train_sigs), dtype=torch.bool)
    else:
        fit_patch_mask = torch.isin(
            train_sample_ids.long(), fit_sample_indices.long()
        )
    if fit_patch_mask.sum() < args.M:
        raise ValueError(
            f"Only {int(fit_patch_mask.sum())} fit patches for M={args.M}"
        )
    fit_sigs = train_sigs[fit_patch_mask]
    standardizer = SignatureStandardizer()
    standardizer.fit(fit_sigs)
    train_sigs_std = standardizer.transform(train_sigs)
    val_sigs_std   = standardizer.transform(val_sigs)
    test_sigs_std  = standardizer.transform(test_sigs)
    fit_sigs_std   = train_sigs_std[fit_patch_mask]

    # ── Phase 4 gate ──────────────────────────────────────────────────────────
    # Signatures should have non-trivial PCA variance
    from torch.linalg import svd
    _svd_input = train_sigs_std[:min(2000, len(train_sigs_std))]
    _finite_mask = _svd_input.isfinite().all(dim=1)
    if not _finite_mask.all():
        n_bad = (~_finite_mask).sum().item()
        print(f"  WARNING: {n_bad} train signatures contain NaN/Inf — dropping before SVD gate")
        _svd_input = _svd_input[_finite_mask]
    _, s, _ = svd(_svd_input, full_matrices=False)
    pca_top3_ratio = (s[:3].pow(2).sum() / s.pow(2).sum()).item()
    gate(pca_top3_ratio < 0.99,
         f"PCA top-3 explains {pca_top3_ratio:.1%} (< 99% → non-trivial structure)",
         bypass=bypass_gates, gate_failures=gate_failures)

    # ── Phase 5: Concept space ────────────────────────────────────────────────
    print(f"\n── Phase 5: Concept space  M={args.M}  algo={args.cluster_algo}")

    if args.sweep_M:
        print("  Sweeping M…")
        val_labels_per_patch = torch.zeros(len(val_sigs_std), dtype=torch.long)
        for i, sid in enumerate(val_sample_ids.tolist()):
            val_labels_per_patch[i] = val_ds.labels[sid]
        sweep_M(train_sigs_std, val_sigs_std, val_labels_per_patch,
                M_values=[8, 16, 32, 64, 128], algorithm=args.cluster_algo)

    concept_space = ConceptSpace(args.M, D)
    gmm_started = time.time()
    concept_space.fit(fit_sigs_std, algorithm=args.cluster_algo, seed=args.seed)
    gmm_training_seconds = time.time() - gmm_started
    print(concept_space.cluster_balance_report())
    fit_hard_assignments = concept_space.hard_assign(fit_sigs_std)
    fit_occupancy_counts = torch.bincount(
        fit_hard_assignments.cpu(), minlength=args.M
    )
    fit_occupancy_fraction = (
        fit_occupancy_counts.float() / max(len(fit_hard_assignments), 1)
    )

    # Gather val labels per patch for NMI
    val_labels_patch = val_ds.labels[val_sample_ids]
    nmi = concept_space.nmi_with_labels(val_sigs_std, val_labels_patch)
    sil = concept_space.silhouette(val_sigs_std)
    print(f"  Val NMI={nmi:.3f}  Silhouette={sil:.3f}")

    gate(nmi > 0.05, f"Concept NMI {nmi:.3f} > 0.05 (concepts carry task-relevant info)",
         bypass=bypass_gates, gate_failures=gate_failures)

    min_pct = (100 * concept_space.cluster_sizes.float().min() /
               concept_space.cluster_sizes.float().sum()).item()
    # gate(min_pct > 0.1, f"Smallest cluster {min_pct:.2f}% > 0.1%")

    if not args.no_wandb:
        wandb.log({"concept/nmi": nmi, "concept/silhouette": sil,
                   "concept/min_cluster_pct": min_pct})

    # ── Phase 6: ConceptEncoder ───────────────────────────────────────────────
    print("\n── Phase 6: ConceptEncoder ─────────────────────────────────")

    max_patch_len = max(dc["seq_len"] // max(K - 1, 1) * 4, dc["seq_len"], 64)
    enc_epochs_run = 0

    if args.direct_sig:
        print("  --direct_sig: skipping encoder training. "
              "Concept sequences = soft_assign(signatures).")
        encoder = None
    else:
        encoder = ConceptEncoder(
            n_channels=C, M=args.M,
            d_model=args.enc_d_model,
            n_layers=args.enc_n_layers,
            n_head=args.enc_n_head,
            max_patch_len=max_patch_len,
        ).to(device)

        print("  Building patch datasets…")
        enc_train_ds = build_patch_dataset(
            density_model, channel_mixer, fit_train_ds, patcher, concept_space,
            channel_mean, channel_std, device, temperature=pc["temperature"],
            sig_mode=sig_mode, mu_marg=mu_marg, sigma2_marg=sigma2_marg,
        )
        enc_val_ds = build_patch_dataset(
            density_model, channel_mixer, val_ds, patcher, concept_space,
            channel_mean, channel_std, device, temperature=pc["temperature"],
            sig_mode=sig_mode, mu_marg=mu_marg, sigma2_marg=sigma2_marg,
        )

        best_enc_val, enc_epochs_run = train_encoder(
            encoder, enc_train_ds, enc_val_ds, args, device, pc)
        print(f"  Encoder best val align loss: {best_enc_val:.4f}")

    # ── Phase 7: Classifier (1%, 5%, 100% labels) ────────────────────────────
    print("\n── Phase 7: Training classifier (few-shot: 1%, 5%, 100%) ──")

    from torch.utils.data import TensorDataset
    from sklearn.metrics import f1_score as _f1_score

    n_classes = pc["n_classes"]

    print("  Building concept sequences for all splits…")
    if args.direct_sig:
        train_cseq, train_lbls = build_concept_sequences_direct(
            train_sigs_std, train_sample_ids, concept_space, K,
            fit_train_ds.labels, device, temperature=pc["temperature"])
        val_cseq, val_lbls = build_concept_sequences_direct(
            val_sigs_std, val_sample_ids, concept_space, K,
            val_ds.labels, device, temperature=pc["temperature"])
        test_cseq, test_lbls = build_concept_sequences_direct(
            test_sigs_std, test_sample_ids, concept_space, K,
            test_ds.labels, device, temperature=pc["temperature"])
    else:
        train_cseq, train_lbls = build_concept_sequences(encoder, enc_train_ds, K, device)
        val_cseq,   val_lbls   = build_concept_sequences(encoder, enc_val_ds,   K, device)

        enc_test_ds = build_patch_dataset(
            density_model, channel_mixer, test_ds, patcher, concept_space,
            channel_mean, channel_std, device, temperature=pc["temperature"],
            sig_mode=sig_mode, mu_marg=mu_marg, sigma2_marg=sigma2_marg,
        )
        test_cseq, test_lbls = build_concept_sequences(encoder, enc_test_ds, K, device)

    if args.label_fraction is None:
        LABEL_FRACTIONS = [(0.01, "1pct"), (0.05, "5pct"), (1.0, "100pct")]
    else:
        pct = int(round(args.label_fraction * 100))
        LABEL_FRACTIONS = [(args.label_fraction, f"{pct}pct")]
    classifiers_trained = {}
    results_by_frac     = {}

    for frac, frac_label in LABEL_FRACTIONS:
        print(f"\n  -- Label fraction: {frac*100:.0f}% --")
        if classifier_indices is not None:
            sub_cseq = train_cseq[classifier_indices]
            sub_lbls = train_lbls[classifier_indices]
        else:
            sub_cseq, sub_lbls = stratified_subset(
                train_cseq, train_lbls, frac, seed=args.seed
            )
        print(f"     Training samples: {len(sub_lbls)} / {len(train_lbls)}")

        if args.classifier_type == "transformer":
            cls = TransformerClassifier(
                M=args.M, K=K, n_classes=n_classes,
                d_model=max(args.enc_d_model, 64), n_layers=2, n_head=4,
            ).to(device)
        elif args.classifier_type == "bigram":
            cls = BigramLinearClassifier(
                M=args.M, K=K, n_classes=n_classes, l1_lambda=1e-3,
            ).to(device)
        else:
            cls = SparseLinearClassifier(
                M=args.M, K=K, n_classes=n_classes, l1_lambda=1e-3,
            ).to(device)

        classifier_started = time.time()
        val_acc = train_classifier(
            cls, sub_cseq, sub_lbls, val_cseq, val_lbls, args, device,
            wandb_step_offset=enc_epochs_run, tag=f"_{frac_label}",
        )
        classifier_training_seconds = time.time() - classifier_started
        print(f"  Classifier ({frac*100:.0f}%) best val accuracy: {val_acc:.4f}")

        # ── Phase 8: Test evaluation ──────────────────────────────────────────
        cls.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            test_ld = DataLoader(TensorDataset(test_cseq, test_lbls),
                                 batch_size=args.cls_batch, shuffle=False)
            for cseq, lbl in test_ld:
                preds_b = cls(cseq.to(device)).argmax(-1)
                all_preds.extend(preds_b.cpu().tolist())
                all_true.extend(lbl.tolist())

        test_acc = sum(p == t for p, t in zip(all_preds, all_true)) / max(len(all_true), 1)
        try:
            test_f1 = _f1_score(all_true, all_preds, average="macro")
        except Exception:
            test_f1 = float("nan")

        pct_str = f"{frac*100:3.0f}%"
        print(f"  Test accuracy ({pct_str}): {test_acc:.4f}")
        print(f"  Test macro-F1 ({pct_str}): {test_f1:.4f}")

        if not args.no_wandb:
            wandb.log({f"test_{frac_label}/accuracy": test_acc,
                       f"test_{frac_label}/f1": test_f1,
                       f"test_{frac_label}/val_acc": val_acc})

        classifiers_trained[frac_label] = cls
        results_by_frac[frac_label] = {
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "val_accuracy": val_acc,
            "selected_epoch": getattr(cls, "selected_epoch", None),
            "classifier_training_seconds": classifier_training_seconds,
        }

    # convenience aliases
    canonical_fraction = "100pct" if "100pct" in classifiers_trained else next(
        iter(classifiers_trained)
    )
    classifier = classifiers_trained[canonical_fraction]
    test_acc   = results_by_frac[canonical_fraction]["test_accuracy"]
    test_f1    = results_by_frac[canonical_fraction]["test_f1"]
    val_acc    = results_by_frac[canonical_fraction]["val_accuracy"]

    # ── Save pipeline artifact ────────────────────────────────────────────────
    gate_status = ("passed" if not gate_failures
                   else "gate_bypassed: " + " | ".join(gate_failures))

    sig_tag = sig_mode if isinstance(sig_mode, str) else "_".join(sig_mode)

    artifact_name = (
        f"pipeline_M{args.M}"
        f"_{args.boundary_mode}"
        f"_{sig_tag}"
        f"_{args.cluster_algo}"
        f"_{args.classifier_type}"
        + ("_directsig" if args.direct_sig else "")
        + f"_la{pc['lambda_align']}"
        f"_ls{pc['lambda_sparse']}"
        f"_lt{pc['lambda_stable']}.pt"
    )

    artifact_path = out_dir / artifact_name
    if artifact_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {artifact_path}")
    torch.save({
        "dataset":          args.dataset,
        "density_checkpoint": str(Path(ckpt_path).resolve()),
        "patcher":          args.patcher,
        "boundary_mode":    args.boundary_mode,
        "M":                args.M,
        "K":                K,
        "patch_config": {
            "K": K,
            "L_min": L_min,
            "L_max": L_max,
            "burn_in": burn_in,
            "effective_length": effective_patch_length,
            "source": (
                "explicit_cli"
                if any(value is not None for value in explicit_patch_values.values())
                else "get_patch_config"
            ),
        },
        "segmentation_diagnostics": segmentation_diagnostics,
        "test_patch_ranges": test_patch_ranges,
        "n_classes":        n_classes,
        "sig_mode":         sig_mode,
        "cluster_algo":     args.cluster_algo,
        "seed":             args.seed,
        "protocol":         args.protocol,
        "label_fraction":   args.label_fraction,
        "split_file":       args.split_file,
        "labeled_indices":  (None if labeled_indices is None else labeled_indices),
        "fit_scope":        args.fit_scope,
        "fit_input_count":  len(marginal_train_ds),
        "fit_patch_count":  int(fit_patch_mask.sum()),
        "fit_indices_file": args.fit_indices_file,
        "concept_occupancy_counts": fit_occupancy_counts,
        "concept_occupancy_fraction": fit_occupancy_fraction,
        "empty_component_count": int((fit_occupancy_counts == 0).sum()),
        "near_empty_component_count": int(
            (fit_occupancy_fraction < 0.01).sum()
        ),
        "gmm_training_seconds": gmm_training_seconds,
        "loss_weights": {
            "lambda_align": pc["lambda_align"],
            "lambda_sparse": pc["lambda_sparse"],
            "lambda_stable": pc["lambda_stable"],
        },
        # normalisation
        "channel_mean":     channel_mean.cpu(),
        "channel_std":      channel_std.cpu(),
        "mu_marg":          mu_marg.cpu(),
        "sigma2_marg":      sigma2_marg.cpu(),
        # standardizer
        "sig_standardizer": standardizer.state_dict(),
        # concept space
        "concept_space":    concept_space.state_dict(),
        # encoder (None when --direct_sig)
        "direct_sig":       args.direct_sig,
        "encoder_config":   (None if args.direct_sig else {
            "n_channels": C, "M": args.M,
            "d_model": args.enc_d_model,
            "n_layers": args.enc_n_layers,
            "n_head": args.enc_n_head,
            "max_patch_len": max_patch_len,
        }),
        "encoder_state_dict": (None if args.direct_sig
                                else {k: v.cpu() for k, v in encoder.state_dict().items()}),
        # classifiers (one per label fraction)
        "classifier_type":  args.classifier_type,
        "classifier_state_dict": {k: v.cpu()
                                  for k, v in classifier.state_dict().items()},
        **{f"classifier_{fl}_state_dict": {k: v.cpu()
                                           for k, v in cls.state_dict().items()}
           for fl, cls in classifiers_trained.items()},
        # metrics (100% is the canonical; all fractions stored separately)
        "test_accuracy":    test_acc,
        "test_f1":          test_f1,
        "val_accuracy":     val_acc,
        **{f"test_accuracy_{fl}": r["test_accuracy"] for fl, r in results_by_frac.items()},
        **{f"test_f1_{fl}":       r["test_f1"]       for fl, r in results_by_frac.items()},
        "results_by_fraction": results_by_frac,
        "concept_nmi":      nmi,
        "concept_sil":      sil,
        # gate audit
        "gate_failures":    gate_failures,
        "gate_status":      gate_status,
    }, artifact_path)

    print(f"\n  gate_status: {gate_status}")
    print(f"Artifact saved to {artifact_path}")

    if not args.no_wandb:
        wandb.finish()

    return test_acc


if __name__ == "__main__":
    main()
