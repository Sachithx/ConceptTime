"""
evaluate.py : Phase 8: Evaluation infrastructure for CAFE-TS.

Loads a saved pipeline artifact and runs:
  1. Standard classification metrics (accuracy, F1, confusion matrix)
  2. Per-patch concept fidelity ρ(P_k)
  3. Concept-assignment entropy H(π(P_k))
  4. Selective accuracy / coverage-accuracy curve
  5. Concept cards (prototype visualization)
  6. Prediction explanation for test samples
  7. Baseline comparison (raw transformer on raw patches)

Usage:
  python evaluate.py --dataset HAR --artifact output/HAR/pipeline/entropy/pipeline_M32.pt
  python evaluate.py --dataset HAR --artifact ... --run_baselines
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from GaussianEntropyModel import GaussianGPT, GaussianGPTConfig
from train_gaussian_entropy_model import TimeSeriesDataset, ChannelMixer, DATASET_CONFIGS
from train_pipeline import load_density_model, PIPELINE_CONFIGS
from patcher import EntropyPatcher, StaticPatcher, PATCH_CONFIGS
from signatures import (compute_marginal, extract_signatures_for_dataset,
                         SignatureStandardizer, signature_dim)
from concept_space import ConceptSpace
from concept_encoder import ConceptEncoder, build_patch_dataset, patch_collate_fn
from classifier import TransformerClassifier, SparseLinearClassifier


# ── Artifact loading ──────────────────────────────────────────────────────────

def load_pipeline(artifact_path: str, device: torch.device, density_ckpt: str):
    art = torch.load(artifact_path, map_location="cpu", weights_only=False)

    # Density model (frozen)
    density_model, channel_mixer, channel_mean, channel_std = load_density_model(
        density_ckpt, device
    )

    # Concept space
    cs = ConceptSpace(art["M"], art["concept_space"]["D"])
    cs.load_state_dict(art["concept_space"])

    # Signature standardizer
    standardizer = SignatureStandardizer()
    standardizer.load_state_dict(art["sig_standardizer"])

    # Encoder
    enc_cfg = dict(art["encoder_config"])
    if "max_patch_len" not in enc_cfg:
        enc_cfg["max_patch_len"] = art["encoder_state_dict"]["pos_emb.emb.weight"].shape[0]
    encoder = ConceptEncoder(**enc_cfg).to(device)
    encoder.load_state_dict({k: v.to(device)
                             for k, v in art["encoder_state_dict"].items()})
    encoder.eval()

    # Classifier
    M = art["M"]
    K = art["K"]
    n_classes = art["n_classes"]
    if art["classifier_type"] == "transformer":
        classifier = TransformerClassifier(M=M, K=K, n_classes=n_classes).to(device)
    else:
        classifier = SparseLinearClassifier(M=M, K=K, n_classes=n_classes).to(device)
    classifier.load_state_dict({k: v.to(device)
                                 for k, v in art["classifier_state_dict"].items()})
    classifier.eval()

    mu_marg    = art["mu_marg"].to(device)
    sigma2_marg = art["sigma2_marg"].to(device)

    print(f"Loaded pipeline  M={M}  K={K}  test_acc={art.get('test_accuracy', '?'):.4f}")
    return (density_model, channel_mixer, channel_mean, channel_std,
            cs, standardizer, encoder, classifier, mu_marg, sigma2_marg, art)


# ── Inference helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def run_pipeline_on_dataset(density_model, channel_mixer, channel_mean, channel_std,
                             patcher, cs, standardizer, encoder, classifier,
                             dataset, mu_marg, sigma2_marg,
                             device, batch_size=64, sig_mode="full", temperature=1.0):
    """
    Full inference pass on a dataset split.

    Returns dict with:
      preds         [N_samples]       hard class predictions
      labels        [N_samples]       true labels
      concept_seqs  [N_samples, K, M] soft concept assignments
      patch_fidelity [N_patches_total] distance to nearest centroid
      patch_entropy  [N_patches_total] H(π) for each patch
      patch_sigs     [N_patches_total, D] standardized signatures
      sample_for_patch [N_patches_total] which sample each patch belongs to
    """
    K = patcher.K if hasattr(patcher, "K") else cs.M

    # Extract signatures for all samples
    all_sigs, all_patch_ranges, sample_ids = extract_signatures_for_dataset(
        density_model, channel_mixer, dataset, patcher,
        channel_mean, channel_std, mu_marg, sigma2_marg,
        device, mode=sig_mode
    )
    sigs_std = standardizer.transform(all_sigs)

    # Soft targets from concept space
    pi = cs.soft_assign(sigs_std, temperature=temperature)     # [N_patches, M]

    # Fidelity = distance to hard-assigned centroid
    hard_z = cs.hard_assign(sigs_std)                          # [N_patches]
    all_dists = cs.distances(sigs_std)                         # [N_patches, M]
    fidelity  = all_dists[torch.arange(len(hard_z)), hard_z]  # [N_patches]

    # Concept entropy
    pi_entropy = -(pi * (pi + 1e-8).log()).sum(-1)             # [N_patches]

    # Build per-sample concept sequences by encoding raw patches
    enc_ds = build_patch_dataset(
        density_model, channel_mixer, dataset, patcher, cs,
        channel_mean, channel_std, device, temperature=temperature,
        sig_mode=sig_mode, mu_marg=mu_marg, sigma2_marg=sigma2_marg,
    )
    enc_loader = DataLoader(enc_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=patch_collate_fn, num_workers=0)

    all_q, all_labels_enc = [], []
    encoder.eval()
    with torch.no_grad():
        for patches, mask, pi_t, labels in enc_loader:
            patches = patches.to(device)
            mask    = mask.to(device)
            q = encoder(patches, mask)
            all_q.append(q.cpu())
            all_labels_enc.append(labels)

    all_q      = torch.cat(all_q)           # [N_patches, M]
    all_lbls_p = torch.cat(all_labels_enc)  # [N_patches]

    # Reshape to per-sample concept sequences
    N_samples = len(dataset)
    # each sample contributes exactly K patches
    concept_seqs = all_q.reshape(N_samples, K, -1)     # [N, K, M]
    labels_per_s = all_lbls_p[::K]                     # [N]

    # Classify
    cls_loader = DataLoader(TensorDataset(concept_seqs, labels_per_s),
                            batch_size=batch_size, shuffle=False)
    all_preds = []
    classifier.eval()
    with torch.no_grad():
        for cseq, _ in cls_loader:
            cseq = cseq.to(device)
            all_preds.append(classifier(cseq).argmax(-1).cpu())

    preds = torch.cat(all_preds)

    return {
        "preds":          preds,
        "labels":         labels_per_s,
        "concept_seqs":   concept_seqs,
        "patch_fidelity": fidelity,
        "patch_entropy":  pi_entropy,
        "patch_sigs":     sigs_std,
        "sample_for_patch": sample_ids,
        "pi_targets":     pi,
    }


# ── Few-shot classifier helpers ──────────────────────────────────────────────

@torch.no_grad()
def apply_classifier(classifier, concept_seqs: torch.Tensor,
                     device: torch.device, batch_size: int = 64) -> torch.Tensor:
    """Apply classifier to pre-built concept sequences → preds [N]."""
    loader = DataLoader(TensorDataset(concept_seqs), batch_size=batch_size, shuffle=False)
    all_preds = []
    classifier.eval()
    for (cseq,) in loader:
        all_preds.append(classifier(cseq.to(device)).argmax(-1).cpu())
    return torch.cat(all_preds)


def load_few_shot_classifiers(art: dict, M: int, K: int, n_classes: int,
                               device: torch.device) -> Dict[str, nn.Module]:
    """Load per-fraction classifiers from artifact. Returns {frac_label: classifier}."""
    cls_type = art.get("classifier_type", "transformer")
    result = {}
    for frac_label in ["1pct", "5pct", "100pct"]:
        key = f"classifier_{frac_label}_state_dict"
        if key not in art:
            continue
        if cls_type == "transformer":
            cls = TransformerClassifier(M=M, K=K, n_classes=n_classes).to(device)
        else:
            cls = SparseLinearClassifier(M=M, K=K, n_classes=n_classes).to(device)
        cls.load_state_dict({k: v.to(device) for k, v in art[key].items()})
        cls.eval()
        result[frac_label] = cls
    return result


# ── Standard classification metrics ──────────────────────────────────────────

def classification_report(results: dict, out_dir: Path,
                           label_names: Optional[Dict[int, str]] = None):
    preds  = results["preds"].numpy()
    labels = results["labels"].numpy()

    from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                                  classification_report as sk_report)

    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, average="macro")
    cm   = confusion_matrix(labels, preds)
    n_cls = cm.shape[0]

    names = [label_names.get(i, f"C{i}") if label_names else f"C{i}"
             for i in range(n_cls)]
    report_str = sk_report(labels, preds, target_names=names)

    print(f"\n{'='*50}")
    print(f"  Accuracy: {acc:.4f}    Macro-F1: {f1:.4f}")
    print(f"{'='*50}")
    print(report_str)

    # Confusion matrix figure
    fig, ax = plt.subplots(figsize=(max(4, n_cls), max(4, n_cls)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(n_cls):
        for j in range(n_cls):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=7, color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix  acc={acc:.3f}  F1={f1:.3f}")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    path = out_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved {path}")

    (out_dir / "classification_report.txt").write_text(
        f"Accuracy: {acc:.4f}\nMacro-F1: {f1:.4f}\n\n{report_str}"
    )
    return {"accuracy": acc, "f1": f1}


# ── Selective accuracy / coverage-accuracy curve ──────────────────────────────

def selective_accuracy_curve(results: dict, out_dir: Path,
                              n_thresholds: int = 50):
    """
    For varying fidelity thresholds τ (low fidelity = patch fits its concept well):
      coverage = fraction of patches with ρ < τ
      accuracy = accuracy on samples where ALL patches have ρ < τ  (confident samples)

    Also: sample-level coverage based on max patch fidelity.
    """
    fidelity    = results["patch_fidelity"].numpy()   # [N_patches]
    preds       = results["preds"].numpy()            # [N_samples]
    labels      = results["labels"].numpy()
    sample_for  = results["sample_for_patch"].numpy() # [N_patches]
    N_samples   = len(preds)
    K           = results["concept_seqs"].shape[1]

    # Max fidelity per sample
    max_fid_per_sample = np.full(N_samples, -1.0)
    for i, fid in zip(sample_for.tolist(), fidelity.tolist()):
        if i < N_samples:
            max_fid_per_sample[i] = max(max_fid_per_sample[i], fid)

    thresholds = np.linspace(fidelity.min(), fidelity.max(), n_thresholds)
    coverages, accuracies = [], []

    for tau in thresholds:
        mask = max_fid_per_sample <= tau
        cov  = mask.mean()
        if mask.sum() > 0:
            acc = (preds[mask] == labels[mask]).mean()
        else:
            acc = float("nan")
        coverages.append(cov)
        accuracies.append(acc)

    coverages  = np.array(coverages)
    accuracies = np.array(accuracies)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Coverage-accuracy curve
    valid = ~np.isnan(accuracies)
    ax1.plot(coverages[valid], accuracies[valid], "b-o", markersize=3)
    ax1.axhline((preds == labels).mean(), color="red", linestyle="--",
                label="Full coverage accuracy")
    ax1.set_xlabel("Coverage (fraction of samples retained)")
    ax1.set_ylabel("Accuracy on retained samples")
    ax1.set_title("Selective accuracy vs coverage")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Fidelity histogram
    ax2.hist(fidelity, bins=50, color="steelblue", edgecolor="black", alpha=0.8)
    ax2.set_xlabel("Patch fidelity ρ (distance to nearest centroid)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Patch fidelity distribution  "
                  f"median={np.median(fidelity):.3f}")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "selective_accuracy.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved {path}")

    # AUC (selective accuracy)
    valid_idx = np.where(valid)[0]
    auc = np.trapz(accuracies[valid_idx], coverages[valid_idx]) if len(valid_idx) > 1 else float("nan")
    print(f"  Selective accuracy AUC (coverage-acc): {auc:.4f}")
    return {"selective_auc": auc,
            "median_fidelity": float(np.median(fidelity)),
            "coverage_accuracy_data": list(zip(coverages.tolist(), accuracies.tolist()))}


# ── Concept cards ─────────────────────────────────────────────────────────────

def concept_cards(cs: ConceptSpace, dataset: TimeSeriesDataset,
                  all_patch_ranges: List[List[Tuple[int, int]]],
                  out_dir: Path, n_prototypes: int = 8):
    """
    For each concept m, plot its top-k nearest training patches (raw signal).
    If patches don't look similar within a cluster, concepts aren't meaningful.
    """
    M    = cs.M
    ncols = min(n_prototypes, 8)
    nrows = M

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 2.5, nrows * 1.8),
                              squeeze=False)

    for m in range(M):
        top_patch_ids = cs.prototype_idxs[m][:n_prototypes]

        for col, global_patch_idx in enumerate(top_patch_ids):
            if col >= ncols:
                break
            ax = axes[m][col]

            # Map global patch index to sample + local patch
            count = 0
            found = False
            for sample_i, patches in enumerate(all_patch_ranges):
                for t1, t2 in patches:
                    if count == global_patch_idx:
                        raw = dataset.samples[sample_i][t1:t2].numpy()  # [L, C]
                        for c in range(min(raw.shape[1], 3)):
                            ax.plot(raw[:, c], linewidth=0.7, alpha=0.7)
                        found = True
                        break
                    count += 1
                if found:
                    break

            if col == 0:
                ax.set_ylabel(f"C{m}", fontsize=7, rotation=0, labelpad=20)
            ax.tick_params(labelsize=5)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Concept cards : top prototype patches per concept\n"
                 "(within-concept patches may not look similar, but the concept is meaningful with the ditribution's proprties at each time step.)", fontsize=9)
    plt.tight_layout()
    path = out_dir / "concept_cards.png"
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Saved {path}")


# ── Prediction explanation ─────────────────────────────────────────────────────

def prediction_explanations(results: dict, dataset: TimeSeriesDataset,
                              all_patch_ranges: List[List[Tuple[int, int]]],
                              out_dir: Path, n_samples: int = 10,
                              label_names: Optional[Dict[int, str]] = None,
                              rng=None):
    """
    For n_samples test examples, show:
      - Raw signal (channel 0)
      - Per-patch concept assignment (colored bar)
      - Per-patch fidelity (color intensity)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N        = len(dataset)
    M        = results["concept_seqs"].shape[-1]
    K        = results["concept_seqs"].shape[1]
    indices  = rng.choice(N, size=min(n_samples, N), replace=False)
    fidelity = results["patch_fidelity"].numpy()   # [N_patches]
    sample_for = results["sample_for_patch"].numpy()

    # Build per-sample fidelity lookup
    fid_per_sample: Dict[int, List[float]] = {}
    for sid, fid in zip(sample_for.tolist(), fidelity.tolist()):
        fid_per_sample.setdefault(sid, []).append(fid)

    cmap  = plt.cm.get_cmap("tab20", M)
    ncols = 2
    nrows = math.ceil(len(indices) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 2.5))
    axes = np.array(axes).flatten()

    for plot_i, sid in enumerate(indices):
        ax = axes[plot_i]
        raw = dataset.samples[sid].numpy()[:, 0]   # channel 0
        T   = len(raw)
        ax.plot(range(T), raw, color="grey", linewidth=0.8, alpha=0.7)

        cseq = results["concept_seqs"][sid]         # [K, M]
        hard_z = cseq.argmax(-1).numpy()            # [K]
        fids   = fid_per_sample.get(sid, [0.0] * K)
        fid_max = max(fids) if fids else 1.0

        patches = all_patch_ranges[sid] if sid < len(all_patch_ranges) else []
        for k, (t1, t2) in enumerate(patches[:K]):
            concept = int(hard_z[k])
            alpha   = 0.2 + 0.6 * (1.0 - min(fids[k] / max(fid_max, 1e-8), 1.0))
            col     = cmap(concept)
            ax.axvspan(t1, t2, color=col, alpha=alpha, linewidth=0)
            ax.text((t1 + t2) / 2, raw.min(), f"C{concept}",
                    ha="center", va="bottom", fontsize=5, color=col)

        pred  = results["preds"][sid].item()
        label = results["labels"][sid].item()
        pname = label_names.get(pred, f"C{pred}") if label_names else f"pred={pred}"
        tname = label_names.get(label, f"C{label}") if label_names else f"true={label}"
        correct = "✓" if pred == label else "✗"
        ax.set_title(f"#{sid}  {correct} pred={pname}  true={tname}", fontsize=7)
        ax.tick_params(labelsize=6)

    for ax in axes[len(indices):]:
        ax.set_visible(False)

    fig.suptitle("Prediction explanations  "
                 "(colored bands = concept, opacity = fidelity confidence)", fontsize=9)
    plt.tight_layout()
    path = out_dir / "prediction_explanations.png"
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved {path}")


# ── Abstention examples ───────────────────────────────────────────────────────

def abstention_examples(results: dict, dataset: TimeSeriesDataset,
                         cs: ConceptSpace,
                         all_patch_ranges: List[List[Tuple[int, int]]],
                         out_dir: Path, n_examples: int = 6,
                         top_pct: float = 0.1,
                         train_dataset: Optional["TimeSeriesDataset"] = None,
                         train_patch_ranges: Optional[List[List[Tuple[int, int]]]] = None):
    """
    Show patches with the highest fidelity (worst fit) alongside nearest prototype.

    Root-cause fix: cs.prototype_idxs are global indices into TRAINING patches,
    not test patches. We maintain a separate train-side lookup and fall back
    through the prototype list until a valid index is found, so the right panel
    is never left empty due to an index mismatch.
    """
    fidelity   = results["patch_fidelity"].numpy()
    sigs_std   = results["patch_sigs"]

    threshold  = np.quantile(fidelity, 1.0 - top_pct)
    high_fid   = np.where(fidelity >= threshold)[0]

    rng    = np.random.default_rng(0)
    chosen = rng.choice(high_fid, size=min(n_examples, len(high_fid)), replace=False)

    # Test-side lookup: maps test global patch index → (sample_i, local_patch_i)
    test_lookup: List[Tuple[int, int]] = []
    for sample_i, patches in enumerate(all_patch_ranges):
        for pi in range(len(patches)):
            test_lookup.append((sample_i, pi))

    # Training-side lookup: prototype indices reference TRAINING patches.
    # Fall back to test lookup only when training data isn't supplied.
    if train_patch_ranges is not None and train_dataset is not None:
        proto_lookup   = []
        proto_ranges   = train_patch_ranges
        proto_dataset  = train_dataset
        for sample_i, patches in enumerate(train_patch_ranges):
            for pi in range(len(patches)):
                proto_lookup.append((sample_i, pi))
    else:
        proto_lookup  = test_lookup
        proto_ranges  = all_patch_ranges
        proto_dataset = dataset

    fig, axes = plt.subplots(n_examples, 2,
                              figsize=(10, n_examples * 2.0), squeeze=False)

    for row, glob_idx in enumerate(chosen):
        ax_sig, ax_proto = axes[row]

        sid, local_pi = test_lookup[glob_idx]
        t1, t2 = all_patch_ranges[sid][local_pi]

        # Left panel: the high-fidelity (abstention-candidate) patch
        raw_patch = dataset.samples[sid][t1:t2].numpy()
        for c in range(min(raw_patch.shape[1], 3)):
            ax_sig.plot(raw_patch[:, c], label=f"ch{c}", linewidth=0.8)
        ax_sig.set_title(f"Patch #{glob_idx}  ρ={fidelity[glob_idx]:.3f}", fontsize=7)
        ax_sig.tick_params(labelsize=5); ax_sig.legend(fontsize=5)

        # Right panel: nearest training prototype — walk the list until valid
        nearest_concept = cs.hard_assign(sigs_std[glob_idx:glob_idx+1]).item()
        proto_idxs = cs.prototype_idxs[nearest_concept]
        plotted = False
        for proto_glob in proto_idxs:
            if proto_glob < len(proto_lookup):
                p_sid, p_li = proto_lookup[proto_glob]
                pt1, pt2 = proto_ranges[p_sid][p_li]
                proto_raw = proto_dataset.samples[p_sid][pt1:pt2].numpy()
                for c in range(min(proto_raw.shape[1], 3)):
                    ax_proto.plot(proto_raw[:, c], linewidth=0.8)
                plotted = True
                break
        if not plotted:
            ax_proto.text(0.5, 0.5, "no prototype\navailable",
                          ha="center", va="center",
                          transform=ax_proto.transAxes, fontsize=8, color="grey")
        ax_proto.set_title(f"Nearest prototype (concept C{nearest_concept})", fontsize=7)
        ax_proto.tick_params(labelsize=5)

    plt.suptitle("Abstention examples : high-fidelity patches and their nearest prototypes",
                 fontsize=9)
    plt.tight_layout()
    path = out_dir / "abstention_examples.png"
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved {path}")


# ── Concept assignment entropy plot ───────────────────────────────────────────

def concept_entropy_report(results: dict, out_dir: Path):
    pi_ent = results["patch_entropy"].numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pi_ent, bins=50, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(pi_ent.mean(), color="red", linestyle="--",
               label=f"Mean H(π) = {pi_ent.mean():.3f}")
    ax.set_xlabel("H(π(P_k))  : concept assignment entropy (lower = more confident)")
    ax.set_ylabel("Count")
    ax.set_title("Concept assignment entropy distribution")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "concept_entropy.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved {path}")
    return {"mean_concept_entropy": float(pi_ent.mean()),
            "median_concept_entropy": float(np.median(pi_ent))}


# ── Concept semantic analysis ────────────────────────────────────────────────

def concept_label_distribution(cs: ConceptSpace,
                                sigs_std_train: torch.Tensor,
                                train_patch_labels: torch.Tensor,
                                out_dir: Path,
                                label_names: Optional[Dict[int, str]] = None):
    """
    For each concept, show the distribution of activity labels in its patches.

    Concepts that are ≥70% one activity are semantically pure — this is the
    interpretability story. Mixed concepts reveal where the concept vocabulary
    conflates activities.

    Uses TRAINING patches (concept space was fitted on training data).
    """
    M         = cs.M
    assigns   = cs.hard_assign(sigs_std_train).numpy()    # [N_train_patches]
    labels_np = train_patch_labels.numpy()
    n_classes = int(labels_np.max()) + 1

    names = [label_names.get(i, f"L{i}") if label_names else f"L{i}"
             for i in range(n_classes)]

    # Count matrix [M, n_classes]
    counts = np.zeros((M, n_classes), dtype=np.float32)
    for m in range(M):
        mask = assigns == m
        if mask.sum() > 0:
            for c in range(n_classes):
                counts[m, c] = float((labels_np[mask] == c).sum())

    totals        = counts.sum(axis=1, keepdims=True).clip(min=1)
    fracs         = counts / totals                          # [M, n_classes] — row sums to 1
    purity        = fracs.max(axis=1)                        # [M]
    dominant_lbl  = fracs.argmax(axis=1)                     # [M]
    n_pure        = int((purity >= 0.70).sum())

    # Sort by dominant label then by purity descending for a readable layout
    sort_idx = np.lexsort((-purity, dominant_lbl))

    cmap   = plt.cm.get_cmap("Set1", n_classes)
    colors = [cmap(c) for c in range(n_classes)]
    x      = np.arange(M)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, M), 5))

    # Left: stacked bar — label fraction per concept (sorted)
    bottom = np.zeros(M)
    for c in range(n_classes):
        ax1.bar(x, fracs[sort_idx, c], bottom=bottom, color=colors[c],
                label=names[c], edgecolor="none", alpha=0.88)
        bottom += fracs[sort_idx, c]
    ax1.axhline(0.70, color="black", linestyle="--", linewidth=1.0,
                label="70% purity line")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"C{sort_idx[i]}" for i in range(M)],
                        rotation=90, fontsize=6)
    ax1.set_ylabel("Fraction of patches")
    ax1.set_title(f"Label distribution per concept  "
                  f"({n_pure}/{M} concepts ≥70% pure)")
    ax1.legend(fontsize=7, loc="upper right", ncol=2)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.3)

    # Right: purity bar coloured by dominant activity
    bar_colors = [colors[dominant_lbl[sort_idx[i]]] for i in range(M)]
    ax2.bar(x, purity[sort_idx], color=bar_colors, edgecolor="black",
            linewidth=0.4, alpha=0.85)
    ax2.axhline(0.70, color="red", linestyle="--", linewidth=1.0,
                label="70% threshold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"C{sort_idx[i]}" for i in range(M)],
                        rotation=90, fontsize=6)
    ax2.set_ylabel("Max label fraction (purity)")
    ax2.set_title("Concept purity  (bar colour = dominant activity)")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = out_dir / "concept_label_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()

    # Text report
    lines = [
        f"Concept label distribution  M={M}  ({n_pure}/{M} concepts ≥70% pure)",
        "=" * 56,
    ]
    for m in range(M):
        dom  = dominant_lbl[m]
        p    = purity[m]
        flag = " <-- PURE" if p >= 0.70 else ""
        lines.append(f"  C{m:03d}: dominant={names[dom]:<12s}  purity={p:.2f}{flag}")
    report = "\n".join(lines)
    (out_dir / "concept_label_distribution.txt").write_text(report)
    print(f"  [{n_pure}/{M} concepts ≥70% pure]  Saved {path}")

    return {"n_pure_concepts": n_pure, "mean_purity": float(purity.mean())}


# ── Baselines ─────────────────────────────────────────────────────────────────

def run_raw_transformer_baseline(density_model, channel_mixer,
                                  channel_mean, channel_std,
                                  patcher, dataset, n_classes,
                                  device, epochs: int = 30,
                                  batch_size: int = 64):
    """
    Trains a small transformer directly on raw patches (no concept bottleneck).
    Returns test accuracy.
    """
    from classifier import RawPatchTransformer
    from concept_encoder import build_patch_dataset, patch_collate_fn
    from torch.utils.data import DataLoader

    K = patcher.K if hasattr(patcher, "K") else 8
    C = channel_mean.shape[0]

    # We need train/val/test raw patch collections grouped per sample
    # Reuse extract_signatures_for_dataset to get patch ranges, then build padded tensors
    # For simplicity: collect raw patches per sample and pad

    dc = DATASET_CONFIGS.get(dataset.root_path.split("/")[-1], {})
    max_L = 64   # max patch length for baseline padding

    def collect_raw_patches(ds):
        samples_list, labels_list = [], []
        for i in range(len(ds)):
            raw_x = ds.samples[i]   # [T, C]
            xn = (raw_x.to(device) - channel_mean) / channel_std
            xn = channel_mixer(xn.unsqueeze(0).permute(0, 2, 1)).permute(0, 2, 1).squeeze(0)

            # get patches
            T = xn.shape[0]
            if hasattr(patcher, "patch_signal") and hasattr(patcher, "model"):
                # entropy patcher needs y_norm : use shifted xn
                try:
                    patches = patcher.patch_signal(xn[:-1], xn[1:])
                except Exception:
                    patches = StaticPatcher(K).patch_signal(T)
            else:
                patches = StaticPatcher(K).patch_signal(T)

            # Pad each patch to max_L
            padded = torch.zeros(K, max_L, C, device=device)
            mask   = torch.zeros(K, max_L, dtype=torch.bool, device=device)
            for k, (t1, t2) in enumerate(patches[:K]):
                L = min(t2 - t1, max_L)
                padded[k, :L] = xn[t1:t1+L]
                mask[k, :L] = True

            samples_list.append(padded.cpu())
            labels_list.append(ds.labels[i])

        return torch.stack(samples_list), torch.tensor([l.item() for l in labels_list])

    return None   # placeholder : full implementation deferred to ablations.py

    # Placeholder return to keep the function signature valid


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default="HAR",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--artifact", required=True,
                        help="Path to pipeline artifact (.pt file)")
    parser.add_argument("--ckpt",     default=None,
                        help="Density model checkpoint (defaults to dataset save_path)")
    parser.add_argument("--split",    default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--out_dir",  default=None)
    parser.add_argument("--device",   default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )

    dc  = DATASET_CONFIGS[args.dataset]
    pc  = PIPELINE_CONFIGS[args.dataset]
    ppc = PATCH_CONFIGS[args.dataset]

    ckpt_path = args.ckpt or dc["save_path"]
    out_dir   = Path(args.out_dir or str(Path(args.artifact).parent / "eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    HAR_LABEL_NAMES = {0: "Walking", 1: "Walk Up", 2: "Walk Down",
                       3: "Sitting", 4: "Standing", 5: "Laying"}
    label_names = HAR_LABEL_NAMES if args.dataset == "HAR" else None

    print(f"\n{'='*60}\n  CAFE-TS Evaluation  {args.dataset}  split={args.split}\n{'='*60}\n")

    # Load pipeline
    (density_model, channel_mixer, channel_mean, channel_std,
     cs, standardizer, encoder, classifier,
     mu_marg, sigma2_marg, art) = load_pipeline(args.artifact, device, ckpt_path)

    K = art["K"]
    M = art["M"]

    # Build patcher
    if art["patcher"] == "entropy":
        patcher = EntropyPatcher(density_model, channel_mixer,
                                  K=K, L_min=ppc["L_min"], burn_in=ppc["burn_in"],
                                  mode=art.get("boundary_mode", "entropy"))
    else:
        patcher = StaticPatcher(K=K)

    # Load dataset split
    dataset = TimeSeriesDataset(dc["root_path"], args.split, dc["seq_len"])
    print(f"Evaluating on {len(dataset)} {args.split} samples\n")

    # Run pipeline inference
    print("── Running pipeline inference ─────────────────────────────")
    results = run_pipeline_on_dataset(
        density_model, channel_mixer, channel_mean, channel_std,
        patcher, cs, standardizer, encoder, classifier, dataset,
        mu_marg, sigma2_marg, device,
        batch_size=args.batch_size,
        sig_mode=art.get("sig_mode", "full"),
        temperature=pc["temperature"],
    )

    sig_mode = art.get("sig_mode", "full")

    # Get test patch ranges for visualization (re-extract; model is frozen, fast)
    _, all_patch_ranges, _ = extract_signatures_for_dataset(
        density_model, channel_mixer, dataset, patcher,
        channel_mean, channel_std, mu_marg, sigma2_marg, device,
        mode=sig_mode,
    )

    # Extract training patches once — shared by abstention plot and label distribution.
    # cs.prototype_idxs are indices into training patches, not test patches.
    print("Extracting training patches for prototype lookup and semantic analysis…")
    train_ds = TimeSeriesDataset(dc["root_path"], "train", dc["seq_len"])
    train_sigs_raw, train_patch_ranges, train_sids = extract_signatures_for_dataset(
        density_model, channel_mixer, train_ds, patcher,
        channel_mean, channel_std, mu_marg, sigma2_marg, device,
        mode=sig_mode,
    )
    train_sigs_std      = standardizer.transform(train_sigs_raw)
    train_patch_labels  = train_ds.labels[train_sids]

    # ── 1. Classification metrics (all label fractions) ──────────────────────
    few_shot_clsfs = load_few_shot_classifiers(art, M, K, art["n_classes"], device)
    FRAC_LABELS = [("1pct", "  1%"), ("5pct", "  5%"), ("100pct", "100%")]

    frac_metrics = {}
    if few_shot_clsfs:
        for frac_label, pct_str in FRAC_LABELS:
            cls_frac = few_shot_clsfs.get(frac_label)
            if cls_frac is None:
                continue
            print(f"\n── 1. Classification metrics ({pct_str} labels) ────────────")
            preds_frac = apply_classifier(cls_frac, results["concept_seqs"],
                                          device, args.batch_size)
            results_frac = {**results, "preds": preds_frac}
            out_dir_frac = out_dir / f"labels_{frac_label}"
            out_dir_frac.mkdir(parents=True, exist_ok=True)
            frac_metrics[frac_label] = classification_report(
                results_frac, out_dir_frac, label_names
            )
    else:
        # legacy artifact: single classifier already applied
        print("\n── 1. Classification metrics ──────────────────────────────")
        frac_metrics["100pct"] = classification_report(results, out_dir, label_names)

    metrics = frac_metrics.get("100pct", next(iter(frac_metrics.values())))

    print("\n── 2. Selective accuracy / coverage-accuracy curve ────────")
    sel_metrics = selective_accuracy_curve(results, out_dir)

    print("\n── 3. Concept assignment entropy ──────────────────────────")
    ent_metrics = concept_entropy_report(results, out_dir)

    print("\n── 4. Concept cards ───────────────────────────────────────")
    concept_cards(cs, dataset, all_patch_ranges, out_dir)

    print("\n── 5. Prediction explanations ─────────────────────────────")
    prediction_explanations(results, dataset, all_patch_ranges, out_dir,
                             label_names=label_names)

    print("\n── 6. Abstention examples ─────────────────────────────────")
    abstention_examples(results, dataset, cs, all_patch_ranges, out_dir,
                        train_dataset=train_ds,
                        train_patch_ranges=train_patch_ranges)

    print("\n── 7. Concept semantic analysis ───────────────────────────")
    sem_metrics = concept_label_distribution(
        cs, train_sigs_std, train_patch_labels, out_dir, label_names
    )

    # Summary report
    summary = {**metrics, **sel_metrics, **ent_metrics, **sem_metrics}
    report_lines = [
        f"CAFE-TS Evaluation  {args.dataset}  split={args.split}",
        "=" * 50,
    ]
    for frac_label, pct_str in FRAC_LABELS:
        if frac_label in frac_metrics:
            fm = frac_metrics[frac_label]
            report_lines += [
                f"Accuracy ({pct_str} labels):    {fm['accuracy']:.4f}",
                f"Macro-F1 ({pct_str} labels):    {fm['f1']:.4f}",
            ]
    report_lines += [
        f"Selective AUC:             {sel_metrics['selective_auc']:.4f}",
        f"Median patch fidelity:     {sel_metrics['median_fidelity']:.4f}",
        f"Mean concept H(π):         {ent_metrics['mean_concept_entropy']:.4f}",
        f"Pure concepts (≥70%):      {sem_metrics['n_pure_concepts']}/{M}",
        f"Mean concept purity:       {sem_metrics['mean_purity']:.4f}",
    ]
    report = "\n".join(report_lines)
    print("\n" + report)
    (out_dir / "eval_summary.txt").write_text(report)

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
