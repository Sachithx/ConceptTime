"""
train_pipeline.py — End-to-end ConceptTime pipeline trainer.

Phases executed in sequence:
  1. Load frozen density model checkpoint
  2. Extract signatures + fit concept space
  3. Train classifier head (few-shot: 1%, 5%, 100%)
  4. Save pipeline artifact + report metrics

Usage:
  python train_pipeline.py --dataset Epilepsy
  python train_pipeline.py --dataset Epilepsy --M 8 --sig_mode surprise_full
"""

import argparse
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

from GaussianEntropyModel import GaussianGPT, GaussianGPTConfig
from train_gaussian_entropy_model import TimeSeriesDataset, ChannelMixer, DATASET_CONFIGS
from patcher import EntropyPatcher, StaticPatcher, GreedyDualThresholdPatcher, get_patch_config
from signatures import (compute_marginal, extract_signatures_for_dataset,
                         SignatureStandardizer, signature_dim)
from concept_space import ConceptSpace, sweep_M
from classifier import (TransformerClassifier, SparseLinearClassifier, BigramLinearClassifier)


# ── Dataset configs ────────────────────────────────────────────────────────────

PIPELINE_CONFIGS = {
    "HAR":                   {"n_classes": 6},
    "Epilepsy":              {"n_classes": 2},
    "SLeep-EDF":             {"n_classes": 5},
    "FD-A":                  {"n_classes": 3},
    "FD-B":                  {"n_classes": 3},
    "FD-C":                  {"n_classes": 3},
    "FD-D":                  {"n_classes": 3},
    "EthanolConcentration":  {"n_classes": 4},
    "FaceDetection":         {"n_classes": 2},
    "Handwriting":           {"n_classes": 26},
    "Heartbeat":             {"n_classes": 2},
    "JapaneseVowels":        {"n_classes": 9},
    "PEMS-SF":               {"n_classes": 7},
    "SelfRegulationSCP1":    {"n_classes": 2},
    "SelfRegulationSCP2":    {"n_classes": 2},
    "SpokenArabicDigits":    {"n_classes": 10},
    "UWaveGestureLibrary":   {"n_classes": 8},
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
    if not condition:
        if bypass:
            print(f"  [gate-SKIPPED] {msg}")
            if gate_failures is not None:
                gate_failures.append(msg)
        else:
            raise RuntimeError(f"Go/no-go FAILED: {msg}")
    else:
        print(f"  [gate] PASS: {msg}")


# ── Direct concept sequence building ─────────────────────────────────────────

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
    Bypass encoder: use soft_assign(signatures) directly as concept sequences.
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
    concept_seqs = soft_probs.cpu().reshape(N_samples, K, -1)
    labels       = dataset_labels[sample_ids[::K]].long()
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


# ── Classifier training ───────────────────────────────────────────────────────

def train_classifier(classifier: nn.Module,
                      concept_seqs_train: torch.Tensor,
                      labels_train: torch.Tensor,
                      concept_seqs_val: torch.Tensor,
                      labels_val: torch.Tensor,
                      args, device: torch.device,
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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_sd = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}

    if best_sd is not None:
        classifier.load_state_dict({k: v.to(device) for k, v in best_sd.items()})
    return best_val_acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="Epilepsy",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--ckpt",       default=None,
                        help="Density model checkpoint (defaults to dataset save_path)")
    parser.add_argument("--patcher",    default="entropy",
                        choices=["entropy", "static", "greedy"])
    parser.add_argument("--patch_scale", default=None,
                        choices=["xs", "s", "m", "l", "xl"])
    parser.add_argument("--boundary_mode", default="surprise",
                        choices=["entropy", "surprise", "kl_shift", "residual"])
    parser.add_argument("--M",          type=int, default=8,
                        help="Vocabulary size (# concept prototypes)")
    parser.add_argument("--sweep_M",    action="store_true",
                        help="Sweep M and report metrics before training")
    parser.add_argument("--cluster_algo", default="gmm",
                        choices=["kmeans", "gmm", "hdbscan"])
    parser.add_argument("--sig_mode",   default="surprise_full",
                        help="Signature mode: full | entropy_only | moments_only | "
                             "distributional_only | morphological_only | "
                             "residual_morphology | surprise_trajectory | surprise_full")
    parser.add_argument("--classifier_type", default="transformer",
                        choices=["transformer", "linear", "bigram"])

    # Classifier hyperparams
    parser.add_argument("--cls_epochs",   type=int,   default=200)
    parser.add_argument("--cls_lr",       type=float, default=3e-3)
    parser.add_argument("--cls_batch",    type=int,   default=64)

    parser.add_argument("--device",     default=None)
    parser.add_argument("--no_gate",    action="store_true",
                        help="Bypass go/no-go gates (ablation mode).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick_test", type=int, default=None, metavar="N",
                        help="Subsample each split to N samples for fast debugging.")
    parser.add_argument("--no_sig_cache", action="store_true",
                        help="Disable signature caching (forces re-extraction).")
    args = parser.parse_args()

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
    pc  = PIPELINE_CONFIGS[args.dataset]
    ppc = get_patch_config(args.dataset, dc["seq_len"], scale=args.patch_scale)

    sig_mode  = args.sig_mode
    ckpt_path = args.ckpt or dc["save_path"]
    out_dir   = Path(dc["output_dir"]) / "pipeline" / args.patcher
    out_dir.mkdir(parents=True, exist_ok=True)

    gate_failures: List[str] = []
    bypass_gates  = args.no_gate

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

    if args.quick_test is not None:
        N = args.quick_test
        for ds in (train_ds, val_ds, test_ds):
            ds.samples = ds.samples[:N]
            ds.labels  = ds.labels[:N]
        print(f"  [quick_test] Using {N} samples per split.")

    K      = ppc["K"]
    L_min  = ppc["L_min"]
    burn_in = ppc["burn_in"]

    # ── Signature cache key ───────────────────────────────────────────────────
    _patch_scale_key = args.patch_scale or "default"
    _sig_cache_key  = (f"{args.dataset}__{args.patcher}__{args.boundary_mode}"
                       f"__{sig_mode}__{_patch_scale_key}")
    _sig_cache_dir  = Path(dc["output_dir"]) / "sig_cache"
    _sig_cache_path = _sig_cache_dir / f"{_sig_cache_key}.pt"
    _use_sig_cache  = not args.no_sig_cache and args.quick_test is None

    C = dc["n_channels"]
    D = signature_dim(C, sig_mode)

    if _use_sig_cache and _sig_cache_path.exists():
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
            print("  (old cache: recomputing mu_marg/sigma2_marg)")
            mu_marg, sigma2_marg = compute_marginal(
                density_model, channel_mixer, train_ds,
                channel_mean, channel_std, device
            )
        print(f"  Signature dim = {D}  (C={C} mode={sig_mode})")
        print(f"  Patches: train={len(train_sigs)}  val={len(val_sigs)}  test={len(test_sigs)}")
    else:
        # ── Phase 4 (marginal) ────────────────────────────────────────────────
        print("\n── Phase 4: Computing training marginal distribution ───────")
        mu_marg, sigma2_marg = compute_marginal(
            density_model, channel_mixer, train_ds,
            channel_mean, channel_std, device
        )

        # ── Phase 3: Build patcher ────────────────────────────────────────────
        print(f"\n── Phase 3: Patcher = {args.patcher} ──────────────────────")
        if args.patcher == "entropy":
            patcher = EntropyPatcher(density_model, channel_mixer,
                                      K=K, L_min=L_min, burn_in=burn_in,
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

    # Standardize
    standardizer = SignatureStandardizer()
    train_sigs_std = standardizer.fit_transform(train_sigs)
    val_sigs_std   = standardizer.transform(val_sigs)
    test_sigs_std  = standardizer.transform(test_sigs)

    # ── Phase 4 gate ──────────────────────────────────────────────────────────
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
    concept_space.fit(train_sigs_std, algorithm=args.cluster_algo)
    print(concept_space.cluster_balance_report())

    val_labels_patch = val_ds.labels[val_sample_ids]
    nmi = concept_space.nmi_with_labels(val_sigs_std, val_labels_patch)
    sil = concept_space.silhouette(val_sigs_std)
    print(f"  Val NMI={nmi:.3f}  Silhouette={sil:.3f}")

    gate(nmi > 0.05, f"Concept NMI {nmi:.3f} > 0.05 (concepts carry task-relevant info)",
         bypass=bypass_gates, gate_failures=gate_failures)

    # ── Phase 6: Build concept sequences (direct from signatures) ────────────
    print("\n── Phase 6: Building concept sequences from signatures ─────")
    train_cseq, train_lbls = build_concept_sequences_direct(
        train_sigs_std, train_sample_ids, concept_space, K,
        train_ds.labels, device)
    val_cseq, val_lbls = build_concept_sequences_direct(
        val_sigs_std, val_sample_ids, concept_space, K,
        val_ds.labels, device)
    test_cseq, test_lbls = build_concept_sequences_direct(
        test_sigs_std, test_sample_ids, concept_space, K,
        test_ds.labels, device)

    # ── Phase 7: Classifier (1%, 5%, 100% labels) ────────────────────────────
    print("\n── Phase 7: Training classifier (few-shot: 1%, 5%, 100%) ──")

    from torch.utils.data import TensorDataset
    from sklearn.metrics import f1_score as _f1_score

    n_classes = pc["n_classes"]

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
                d_model=64, n_layers=2, n_head=4,
            ).to(device)
        elif args.classifier_type == "bigram":
            cls = BigramLinearClassifier(
                M=args.M, K=K, n_classes=n_classes, l1_lambda=1e-3,
            ).to(device)
        else:
            cls = SparseLinearClassifier(
                M=args.M, K=K, n_classes=n_classes, l1_lambda=1e-3,
            ).to(device)

        val_acc = train_classifier(
            cls, sub_cseq, sub_lbls, val_cseq, val_lbls, args, device,
            tag=f"_{frac_label}",
        )
        print(f"  Classifier ({frac*100:.0f}%) best val accuracy: {val_acc:.4f}")

        # ── Evaluate on test ──────────────────────────────────────────────────
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

        classifiers_trained[frac_label] = cls
        results_by_frac[frac_label] = {
            "test_accuracy": test_acc, "test_f1": test_f1, "val_accuracy": val_acc,
        }

    classifier = classifiers_trained["100pct"]
    test_acc   = results_by_frac["100pct"]["test_accuracy"]
    test_f1    = results_by_frac["100pct"]["test_f1"]
    val_acc    = results_by_frac["100pct"]["val_accuracy"]

    # ── Save pipeline artifact ────────────────────────────────────────────────
    gate_status = ("passed" if not gate_failures
                   else "gate_bypassed: " + " | ".join(gate_failures))

    artifact_name = (
        f"pipeline_M{args.M}"
        f"_{args.boundary_mode}"
        f"_{sig_mode}"
        f"_{args.cluster_algo}"
        f"_{args.classifier_type}.pt"
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
        # normalisation
        "channel_mean":     channel_mean.cpu(),
        "channel_std":      channel_std.cpu(),
        "mu_marg":          mu_marg.cpu(),
        "sigma2_marg":      sigma2_marg.cpu(),
        # standardizer
        "sig_standardizer": standardizer.state_dict(),
        # concept space
        "concept_space":    concept_space.state_dict(),
        # classifiers (one per label fraction)
        "classifier_type":  args.classifier_type,
        "classifier_state_dict": {k: v.cpu()
                                  for k, v in classifier.state_dict().items()},
        **{f"classifier_{fl}_state_dict": {k: v.cpu()
                                           for k, v in cls.state_dict().items()}
           for fl, cls in classifiers_trained.items()},
        # metrics
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

    return test_acc


if __name__ == "__main__":
    main()
