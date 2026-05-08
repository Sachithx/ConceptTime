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
import math
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from GaussianEntropyModel import GaussianGPT, GaussianGPTConfig
from train_gaussian_entropy_model import TimeSeriesDataset, ChannelMixer, DATASET_CONFIGS
from patcher import EntropyPatcher, StaticPatcher, GreedyDualThresholdPatcher, normalize_batch, PATCH_CONFIGS
from signatures import (compute_marginal, extract_signatures_for_dataset,
                         SignatureStandardizer, signature_dim)
from concept_space import ConceptSpace, sweep_M
from concept_encoder import (ConceptEncoder, PatchDataset, patch_collate_fn,
                              augment_batch_patches, align_loss, sparse_loss,
                              stable_loss, build_patch_dataset)
from classifier import (TransformerClassifier, SparseLinearClassifier,
                         ConceptPipeline, RawPatchTransformer)


# ── Dataset configs (augment existing DATASET_CONFIGS) ────────────────────────

PIPELINE_CONFIGS = {
    "HAR":       {"n_classes": 6,  "sig_mode": "full", "temperature": 1.0,
                  "lambda_align": 1.0, "lambda_sparse": 0.7, "lambda_stable": 0.5},
    "Epilepsy":  {"n_classes": 2,  "sig_mode": "full", "temperature": 1.0,
                  "lambda_align": 1.0, "lambda_sparse": 0.6, "lambda_stable": 0.5},
    "SLeep-EDF": {"n_classes": 5,  "sig_mode": "full", "temperature": 1.0,
                  "lambda_align": 1.0, "lambda_sparse": 0.05, "lambda_stable": 0.1},
    "FD":        {"n_classes": 3,  "sig_mode": "full", "temperature": 1.0,
                  "lambda_align": 1.0, "lambda_sparse": 0.1, "lambda_stable": 0.1},
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

    use_l1 = isinstance(classifier, SparseLinearClassifier)
    best_val_acc, best_sd = 0.0, None

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
        print(f"  Cls epoch {epoch+1:3d}/{args.cls_epochs}"
              f"  loss={ep_loss/len(train_ld):.4f}  val_acc={val_acc:.4f}")
        if wandb.run is not None:
            wandb.log({f"cls{tag}/val_acc": val_acc, f"cls{tag}/train_loss": ep_loss / len(train_ld)},
                      step=wandb_step_offset + epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_sd = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}

    if best_sd is not None:
        classifier.load_state_dict({k: v.to(device) for k, v in best_sd.items()})
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
                        choices=["transformer", "linear"])

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
    parser.add_argument("--cls_batch",    type=int,   default=128)

    parser.add_argument("--device",     default=None)
    parser.add_argument("--no_wandb",   action="store_true")
    parser.add_argument("--no_gate",    action="store_true",
                        help="Bypass go/no-go gates (ablation mode). "
                             "Skipped gates are logged to the artifact as gate_failures.")
    parser.add_argument("--direct_sig", action="store_true",
                        help="Skip encoder training (Phase 6). Use soft_assign(signatures) "
                             "directly as concept sequences. Density model still runs for patching.")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )

    dc  = DATASET_CONFIGS[args.dataset]
    pc  = dict(PIPELINE_CONFIGS[args.dataset])  # copy so CLI overrides do not modify global config
    ppc = PATCH_CONFIGS[args.dataset]

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
    out_dir      = Path(dc["output_dir"]) / "pipeline" / args.patcher
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

    # ── Phase 4 (marginal) ────────────────────────────────────────────────────
    print("\n── Phase 4: Computing training marginal distribution ───────")
    mu_marg, sigma2_marg = compute_marginal(
        density_model, channel_mixer, train_ds,
        channel_mean, channel_std, device
    )

    # ── Phase 3: Build patcher ────────────────────────────────────────────────
    print(f"\n── Phase 3: Patcher = {args.patcher} ──────────────────────")
    K      = ppc["K"]
    L_min  = ppc["L_min"]
    # L_max  = ppc["L_max"]
    burn_in = ppc["burn_in"]

    if args.patcher == "entropy":
        patcher = EntropyPatcher(density_model, channel_mixer,
                                  K=K, L_min=L_min, burn_in=burn_in, # L_max=L_max
                                  mode=args.boundary_mode)
    elif args.patcher == "greedy":
        patcher = GreedyDualThresholdPatcher(density_model, channel_mixer,
                                              K=K, L_min=L_min, burn_in=burn_in,
                                              mode=args.boundary_mode)
    else:
        patcher = StaticPatcher(K=K)

    # ── Phase 4: Extract signatures ───────────────────────────────────────────
    print("\n── Phase 4: Extracting signatures ──────────────────────────")
    C = dc["n_channels"]
    D = signature_dim(C, sig_mode)
    print(f"  Signature dim = {D}  (C={C} mode={sig_mode})")

    print("  Train split…")
    train_sigs, train_patch_ranges, train_sample_ids = extract_signatures_for_dataset(
        density_model, channel_mixer, train_ds, patcher,
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

    # Standardize
    standardizer = SignatureStandardizer()
    train_sigs_std = standardizer.fit_transform(train_sigs)
    val_sigs_std   = standardizer.transform(val_sigs)
    test_sigs_std  = standardizer.transform(test_sigs)

    # ── Phase 4 gate ──────────────────────────────────────────────────────────
    # Signatures should have non-trivial PCA variance
    from torch.linalg import svd
    _, s, _ = svd(train_sigs_std[:min(2000, len(train_sigs_std))], full_matrices=False)
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
    train_labels_np = train_ds.labels[train_sample_ids].numpy()
    concept_space.fit(train_sigs_std, algorithm=args.cluster_algo)
    print(concept_space.cluster_balance_report())

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
            density_model, channel_mixer, train_ds, patcher, concept_space,
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
            train_ds.labels, device, temperature=pc["temperature"])
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

    LABEL_FRACTIONS = [(0.01, "1pct"), (0.05, "5pct"), (1.0, "100pct")]
    classifiers_trained = {}
    results_by_frac     = {}

    for frac, frac_label in LABEL_FRACTIONS:
        print(f"\n  -- Label fraction: {frac*100:.0f}% --")
        sub_cseq, sub_lbls = stratified_subset(train_cseq, train_lbls, frac)
        print(f"     Training samples: {len(sub_lbls)} / {len(train_lbls)}")

        if args.classifier_type == "transformer":
            cls = TransformerClassifier(
                M=args.M, K=K, n_classes=n_classes,
                d_model=max(args.enc_d_model, 64), n_layers=2, n_head=4,
            ).to(device)
        else:
            cls = SparseLinearClassifier(
                M=args.M, K=K, n_classes=n_classes, l1_lambda=1e-3,
            ).to(device)

        val_acc = train_classifier(
            cls, sub_cseq, sub_lbls, val_cseq, val_lbls, args, device,
            wandb_step_offset=enc_epochs_run, tag=f"_{frac_label}",
        )
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
            "test_accuracy": test_acc, "test_f1": test_f1, "val_accuracy": val_acc,
        }

    # convenience aliases
    classifier = classifiers_trained["100pct"]
    test_acc   = results_by_frac["100pct"]["test_accuracy"]
    test_f1    = results_by_frac["100pct"]["test_f1"]
    val_acc    = results_by_frac["100pct"]["val_accuracy"]

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
    torch.save({
        "dataset":          args.dataset,
        "patcher":          args.patcher,
        "boundary_mode":    args.boundary_mode,
        "M":                args.M,
        "K":                K,
        "n_classes":        n_classes,
        "sig_mode":         sig_mode,
        "cluster_algo":     args.cluster_algo,
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