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
  python evaluate.py --dataset Epilepsy --artifact output/Epilepsy/pipeline/entropy/pipeline_M8.pt
  python evaluate.py --dataset Epilepsy --artifact ... --run_baselines
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
from train_pipeline import load_density_model
from patcher import EntropyPatcher, StaticPatcher, get_patch_config
from signatures import (compute_marginal, extract_signatures_for_dataset,
                         SignatureStandardizer, signature_dim)
from concept_space import ConceptSpace
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

    encoder = None

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
                             device, batch_size=64, sig_mode="full", temperature=1.0,
                             precomputed_sigs=None, precomputed_patch_ranges=None,
                             precomputed_sample_ids=None):
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
      patch_ranges   list of (start, end) per patch
    """
    K = patcher.K if hasattr(patcher, "K") else cs.M

    if precomputed_sigs is not None:
        all_sigs         = precomputed_sigs
        all_patch_ranges = precomputed_patch_ranges
        sample_ids       = precomputed_sample_ids
    else:
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

    # Build per-sample concept sequences
    N_samples = len(dataset)
    concept_seqs = pi.cpu().reshape(N_samples, K, -1)   # [N, K, M]
    labels_per_s = dataset.labels[:N_samples].long()

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
        "patch_fidelity": fidelity.cpu(),
        "patch_entropy":  pi_entropy.cpu(),
        "patch_sigs":     sigs_std.cpu(),
        "sample_for_patch": sample_ids.cpu() if isinstance(sample_ids, torch.Tensor) else torch.tensor(sample_ids),
        "pi_targets":     pi.cpu(),
        "patch_ranges":   all_patch_ranges,
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


# ── Pipeline trace figures ────────────────────────────────────────────────────

def _sig_feature_names(n_channels: int, mode="full") -> List[str]:
    """Short human-readable names for each signature dimension (matches signatures.py)."""
    if isinstance(mode, (list, tuple)):
        names = []
        for m in mode:
            names += _sig_feature_names(n_channels, m)
        return names
    if mode == "entropy_only":
        return ["mean_H", "var_H", "max_H", "pk_pos", "log_L"]
    if mode == "moments_only":
        return ([f"μ_c{c}" for c in range(n_channels)] +
                [f"σ²_c{c}" for c in range(n_channels)] + ["log_L"])
    if mode == "distributional_only":
        return ["drift", "mean_NLL", "max_NLL", "KL"]
    if mode == "morphological_only":
        return ["fft_pk", "ac1", "ac2", "ac3", "skew", "kurt", "ZCR", "p2p"]
    if mode == "residual_morphology":
        return ["r_fft", "r_ac1", "r_ac2", "r_ac3", "r_skew", "r_kurt", "r_ZCR", "r_p2p"]
    if mode == "surprise_trajectory":
        return ["s_pos", "s_sprd", "c_surp", "r_en"]
    if mode == "surprise_full":
        return ([f"μ_c{c}" for c in range(n_channels)] +
                [f"σ²_c{c}" for c in range(n_channels)] + ["log_L"] +
                ["r_fft", "r_ac1", "r_ac2", "r_ac3", "r_skew", "r_kurt", "r_ZCR", "r_p2p"] +
                ["s_pos", "s_sprd", "c_surp", "r_en"])
    # full: entropy(5) + mean_mu(C) + mean_sigma2(C) + distrib(4) + morph(8) = 17+2C
    return (["mean_H", "var_H", "max_H", "pk_pos", "log_L"] +
            [f"μ_c{c}" for c in range(n_channels)] +
            [f"σ²_c{c}" for c in range(n_channels)] +
            ["drift", "mean_NLL", "max_NLL", "KL"] +
            ["fft_pk", "ac1", "ac2", "ac3", "skew", "kurt", "ZCR", "p2p"])


def pipeline_trace_figure(
    results: dict,
    dataset,
    all_patch_ranges: List[List[Tuple[int, int]]],
    cs: ConceptSpace,
    out_dir: Path,
    n_samples: int = 5,
    label_names: Optional[Dict[int, str]] = None,
    sig_mode: str = "full",
    n_channels: int = 1,
    rng=None,
):
    """
    For n_samples test examples render the full pipeline story:

      Row 0 — raw signal (ch0) with one colored band per patch, labeled by
              dominant concept (C0..CM-1)
      Row 1 — K individual patch waveforms in separate panels, colored by concept
      Row 2 — K concept-distribution bars π[M] (winning concept highlighted)
      Row 3 — signature fingerprint: most-confident patch vs its concept centroid
              (grouped bar chart in standardized / z-score units)

    A mix of correctly and incorrectly classified samples is chosen.
    One PNG per sample is saved to out_dir.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N          = len(dataset)
    M          = cs.M
    K          = results["concept_seqs"].shape[1]
    preds      = results["preds"]
    labels_    = results["labels"]
    cseqs      = results["concept_seqs"]      # [N, K, M]
    patch_sigs = results["patch_sigs"]        # [N_patches, D]
    sample_for = results["sample_for_patch"]  # [N_patches]
    fidelity   = results["patch_fidelity"]    # [N_patches]

    # Build per-sample → list-of-global-patch-indices lookup
    sample_to_pidx: Dict[int, List[int]] = {}
    for pi, sid in enumerate(sample_for.tolist()):
        sample_to_pidx.setdefault(int(sid), []).append(pi)

    # Select a mix of correct + incorrect samples
    correct_idx   = (preds == labels_).nonzero(as_tuple=True)[0].tolist()
    incorrect_idx = (preds != labels_).nonzero(as_tuple=True)[0].tolist()
    n_wrong = min(max(1, n_samples // 3), len(incorrect_idx))
    n_right = n_samples - n_wrong
    chosen: List[int] = []
    if incorrect_idx:
        chosen += rng.choice(incorrect_idx, size=n_wrong, replace=False).tolist()
    chosen += rng.choice(correct_idx,
                         size=min(n_right, len(correct_idx)),
                         replace=False).tolist()
    chosen = [int(x) for x in chosen[:n_samples]]

    cmap_c     = plt.cm.get_cmap("tab20", M)
    feat_names = _sig_feature_names(n_channels, sig_mode)
    D_feat     = len(feat_names)
    centroids  = cs.centroids   # [M, D] — already in standardized space

    out_dir.mkdir(parents=True, exist_ok=True)

    for sid in chosen:
        raw   = dataset.samples[sid].numpy()   # [T, C_raw]
        T     = raw.shape[0]
        ptchs = all_patch_ranges[sid] if sid < len(all_patch_ranges) else []
        n_p   = min(len(ptchs), K)
        if n_p == 0:
            continue

        pred_i = int(preds[sid].item())
        true_i = int(labels_[sid].item())
        ok_str = "✓" if pred_i == true_i else "✗"
        pname  = label_names.get(pred_i, str(pred_i)) if label_names else str(pred_i)
        tname  = label_names.get(true_i, str(true_i)) if label_names else str(true_i)

        pidxs   = sample_to_pidx.get(sid, [])
        sigs_s  = patch_sigs[pidxs][:n_p]   # [n_p, D]
        fid_s   = fidelity[pidxs][:n_p]     # [n_p]
        cseq_s  = cseqs[sid]                 # [K, M]
        hard_z  = cseq_s.argmax(-1)          # [K]

        # Most-confident patch = lowest concept-assignment entropy
        pi_ent = -(cseq_s * (cseq_s + 1e-8).log()).sum(-1)   # [K]
        best_k = int(pi_ent.argmin().item())

        fig = plt.figure(figsize=(max(14, n_p * 2.2), 10))
        gs  = gridspec.GridSpec(
            4, n_p,
            height_ratios=[2.5, 1.6, 1.6, 2.5],
            hspace=0.55, wspace=0.25,
        )

        # ── Row 0: full signal with colored concept bands ─────────────────
        ax0 = fig.add_subplot(gs[0, :])
        ax0.plot(np.arange(T), raw[:, 0], color="#222222", lw=0.9, alpha=0.9)
        y_lo = float(raw[:, 0].min())
        y_hi = float(raw[:, 0].max())

        for k, (t1, t2) in enumerate(ptchs[:n_p]):
            cidx = int(hard_z[k])
            col  = cmap_c(cidx)
            ax0.axvspan(t1, t2, color=col, alpha=0.25, linewidth=0)
            ax0.axvline(t1, color="gray", lw=0.4, alpha=0.45)
            ax0.text((t1 + t2) / 2, y_lo, f"C{cidx}",
                     ha="center", va="bottom", fontsize=6,
                     color=col, fontweight="bold")
            ax0.text((t1 + t2) / 2, y_hi, f"P{k+1}",
                     ha="center", va="top", fontsize=5, color="gray")

        ax0.set_title(
            f"Sample #{sid}   {ok_str}   pred = {pname}   true = {tname}"
            f"   │   colored bands = concept per patch",
            fontsize=8.5, pad=4,
        )
        ax0.set_ylabel("Signal (ch0)", fontsize=7)
        ax0.tick_params(labelsize=6)
        ax0.grid(True, alpha=0.18)

        # ── Row 1: individual patch waveforms ─────────────────────────────
        for k, (t1, t2) in enumerate(ptchs[:n_p]):
            ax1  = fig.add_subplot(gs[1, k])
            cidx = int(hard_z[k])
            col  = cmap_c(cidx)
            seg  = raw[t1:t2]   # [L, C_raw]
            ax1.plot(seg[:, 0], color=col, lw=0.9, alpha=0.9)
            fid_val = float(fid_s[k].item()) if k < len(fid_s) else 0.0
            ax1.set_facecolor((*col[:3], 0.10))
            ax1.set_title(f"P{k+1} | C{cidx}\nρ={fid_val:.2f}", fontsize=5.5, pad=2)
            ax1.set_xticks([]); ax1.set_yticks([])
            for spine in ax1.spines.values():
                spine.set_edgecolor(col)
                spine.set_linewidth(1.2)

        # ── Row 2: concept distribution π[M] per patch ───────────────────
        for k in range(n_p):
            ax2  = fig.add_subplot(gs[2, k])
            pi_k = cseq_s[k].numpy() if k < len(cseq_s) else np.zeros(M)
            cidx = int(hard_z[k]) if k < len(hard_z) else 0
            bar_colors = [cmap_c(m) if m == cidx else "#cccccc" for m in range(M)]
            ax2.bar(np.arange(M), pi_k, color=bar_colors, linewidth=0, width=1.0)
            ax2.set_xlim(-0.5, M - 0.5)
            ax2.set_ylim(0, 1.0)
            ax2.set_xticks([]); ax2.set_yticks([])
            ax2.set_xlabel(f"concepts (M={M})", fontsize=4)
            if k == 0:
                ax2.set_ylabel("π(C|patch)", fontsize=5)

        # ── Row 3: signature fingerprint — patch vs concept centroid ──────
        ax3 = fig.add_subplot(gs[3, :])
        if best_k < len(sigs_s) and D_feat > 0 and centroids is not None:
            sig_v  = sigs_s[best_k].numpy()[:D_feat]
            c_idx  = int(hard_z[best_k])
            cent_v = centroids[c_idx].numpy()[:D_feat]
            col    = cmap_c(c_idx)

            x     = np.arange(D_feat)
            width = 0.38
            ax3.bar(x - width / 2, sig_v,  width,
                    label=f"Patch P{best_k + 1} (most confident)",
                    color=col, alpha=0.82)
            ax3.bar(x + width / 2, cent_v, width,
                    label=f"Concept C{c_idx} centroid",
                    color="gray", alpha=0.60)
            ax3.set_xticks(x)
            ax3.set_xticklabels(feat_names, rotation=55, ha="right", fontsize=4.5)
            ax3.axhline(0, color="black", lw=0.5)
            ax3.set_ylabel("z-score", fontsize=6)
            ax3.set_title(
                f"Signature fingerprint — patch P{best_k + 1} vs concept C{c_idx} centroid"
                f"   │   bars near 0 = near the training mean for that feature",
                fontsize=7,
            )
            ax3.legend(fontsize=6, loc="upper right")
            ax3.grid(axis="y", alpha=0.25)
            ax3.tick_params(axis="y", labelsize=5)
        else:
            ax3.axis("off")
            ax3.text(0.5, 0.5, "signature data unavailable",
                     ha="center", va="center", transform=ax3.transAxes,
                     fontsize=9, color="gray")

        fig.suptitle(
            "Pipeline trace  ·  signal  →  entropy-patches  "
            "→  concept assignments  →  prediction",
            fontsize=10, fontweight="bold",
        )
        fpath = out_dir / f"trace_{sid:04d}.png"
        plt.savefig(fpath, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fpath}")


# ── Concept label derivation (prediction-vs-observation vocabulary) ───────────

def _concept_tag(c: np.ndarray,
                 med: np.ndarray,
                 pct_lo: np.ndarray,
                 pct_hi: np.ndarray) -> str:
    """
    Map a centroid to a short human label describing the prediction–observation
    relationship (not raw waveform shape).

    surprise_full dims (C channels, but here we use channel-0 proxies):
      0=mean_mu  1=mean_sigma2  2=log_len
      3=res_fft  4=res_ac1  …  8=res_kurt  9=res_zcr  10=res_p2p
      11=argmax_pos  12=spread  13=conf_surp  14=res_energy
    For multi-channel artifacts D > 15; we index relative to the last 13 dims
    (residual morph + surprise traj) and use dim 0 / dim C for mu / sigma2.
    """
    D = len(c)
    # For surprise_full: moments = first 2C+1 dims, then res_morph(8), surp_traj(4)
    # We always use dim 0 (mean_mu ch0) and the last 13 dims for morph+traj.
    # This is exact for C=1 (D=15) and a good proxy for multi-channel.
    mu_d      = 0
    sigma2_d  = 1
    res_ac1_d = D - 13 + 1   # residual morph starts at D-13; ac1 is offset 1
    res_kurt_d= D - 13 + 5   # kurtosis offset 5
    res_zcr_d = D - 13 + 6
    argpos_d  = D - 4         # surprise traj starts at D-4
    spread_d  = D - 3
    conf_d    = D - 2
    energy_d  = D - 1

    def ab_med(v, d): return v > med[d]
    def hi(v, d):     return v > pct_hi[d]
    def lo(v, d):     return v < pct_lo[d]

    violated = ab_med(c[energy_d], energy_d) or ab_med(c[conf_d], conf_d)

    if violated:
        if hi(c[sigma2_d], sigma2_d):
            return "unpredictable region"
        strong    = hi(c[energy_d], energy_d)
        conf      = ab_med(c[conf_d], conf_d)
        kurt_h    = hi(c[res_kurt_d], res_kurt_d)
        ac1_h     = hi(c[res_ac1_d], res_ac1_d)
        zcr_h     = hi(c[res_zcr_d], res_zcr_d)
        argp_hi   = hi(c[argpos_d], argpos_d)
        argp_lo   = lo(c[argpos_d], argpos_d)
        quiet_ctx = lo(c[mu_d], mu_d)
        actv_ctx  = hi(c[mu_d], mu_d)

        if strong and conf:
            if kurt_h:
                if quiet_ctx: return "spike vs quiet prediction"
                if actv_ctx:  return "spike vs active prediction"
                return "impulsive surprise"
            if ac1_h:
                if quiet_ctx: return "drift from quiet prediction"
                if actv_ctx:  return "drift from active prediction"
                return "sustained drift"
            if zcr_h:          return "erratic violation"
            if argp_hi:
                if quiet_ctx:  return "quiet→surprise at tail"
                return "transition at tail"
            if argp_lo:
                if quiet_ctx:  return "surprise at onset"
                return "onset shift"
            if quiet_ctx:      return "violated quiet expectation"
            if actv_ctx:       return "violated active expectation"
            return "confident surprise"

        if argp_hi:            return "trailing shift"
        if argp_lo:            return "onset deviation"
        if ac1_h:              return "smooth rhythmic deviation"
        if kurt_h:             return "brief spike"
        if zcr_h:              return "rapid fluctuation"
        if quiet_ctx:          return "mild excitation (quiet)"
        if actv_ctx:           return "mild suppression (active)"
        return "moderate deviation"

    # Confirmed
    if hi(c[sigma2_d], sigma2_d): return "uncertain but matched"
    quiet_ctx = lo(c[mu_d], mu_d)
    actv_ctx  = hi(c[mu_d], mu_d)
    ac1_h     = hi(c[res_ac1_d], res_ac1_d)
    if quiet_ctx: return "smooth quiet (as predicted)" if ac1_h else "quiet as predicted"
    if actv_ctx:  return "smooth active (as predicted)" if ac1_h else "active as predicted"
    if ac1_h:     return "rhythmic match"
    return "prediction confirmed"


def build_concept_labels(centroids: np.ndarray) -> Dict[int, str]:
    """Derive a human-readable label for every concept centroid."""
    med    = np.percentile(centroids, 50, axis=0)
    pct_lo = np.percentile(centroids, 25, axis=0)
    pct_hi = np.percentile(centroids, 75, axis=0)
    return {m: _concept_tag(centroids[m], med, pct_lo, pct_hi)
            for m in range(len(centroids))}


# ── Concept trajectory through cluster space ──────────────────────────────────

def _get_2d_embedding(X: np.ndarray):
    """UMAP → t-SNE → PCA fallback for 2-D embedding."""
    try:
        import umap as umap_lib
        reducer = umap_lib.UMAP(n_components=2, random_state=42,
                                n_neighbors=15, min_dist=0.1,
                                metric="euclidean", verbose=False)
        return reducer.fit_transform(X), "UMAP"
    except ImportError:
        pass
    try:
        from sklearn.manifold import TSNE
        perp = min(30, max(5, len(X) // 10))
        return (TSNE(n_components=2, random_state=42, perplexity=perp,
                     n_iter=1000, verbose=0).fit_transform(X), "t-SNE")
    except Exception:
        pass
    Xc = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T, "PCA"


def test_concept_trajectory(
    results: dict,
    train_sigs_std: torch.Tensor,
    dataset,
    all_patch_ranges: List[List[Tuple[int, int]]],
    cs: ConceptSpace,
    out_dir: Path,
    n_samples: int = 4,
    max_bg_patches: int = 2000,
    label_names: Optional[Dict[int, str]] = None,
    concept_labels: Optional[Dict[int, str]] = None,
    rng=None,
):
    """
    For n_samples test examples show how their patches move through the learned
    concept space, one PNG per sample:

      Row 0 — embedding (UMAP/t-SNE/PCA) of training patches as background cloud,
               with this sample's K test patches overlaid as numbered stars and
               connected in temporal order P1 → P2 → … → PK by arrows.
               Concept centroid labels mark where each cluster lives.
      Row 1 — K patch waveforms, one panel per patch (top of each panel connected
               by a colored arrow to its star in the embedding above).
      Row 2 — K concept-distribution bars π[M].

    All selected test patches are included in the single embedding fit so they
    share coordinates with the training cloud — no separate transform needed.
    """
    from matplotlib.patches import ConnectionPatch

    if rng is None:
        rng = np.random.default_rng(42)

    M          = cs.M
    K          = results["concept_seqs"].shape[1]
    preds      = results["preds"]
    labels_    = results["labels"]
    cseqs      = results["concept_seqs"]      # [N_test, K, M]
    test_sigs  = results["patch_sigs"]        # [N_test_patches, D]
    sample_for = results["sample_for_patch"]  # [N_test_patches]
    fidelity   = results["patch_fidelity"]    # [N_test_patches]

    # Build per-sample → ordered patch-index list
    sample_to_pidx: Dict[int, List[int]] = {}
    for pi, sid in enumerate(sample_for.tolist()):
        sample_to_pidx.setdefault(int(sid), []).append(pi)

    # Select a mix of correct + incorrect samples
    correct_idx   = (preds == labels_).nonzero(as_tuple=True)[0].tolist()
    incorrect_idx = (preds != labels_).nonzero(as_tuple=True)[0].tolist()
    n_wrong = min(max(1, n_samples // 3), len(incorrect_idx))
    n_right = n_samples - n_wrong
    chosen: List[int] = []
    if incorrect_idx:
        chosen += rng.choice(incorrect_idx, size=n_wrong, replace=False).tolist()
    chosen += rng.choice(correct_idx,
                         size=min(n_right, len(correct_idx)), replace=False).tolist()
    chosen = [int(x) for x in chosen[:n_samples]]

    cmap_c = plt.cm.get_cmap("tab20", M)

    # Build concept labels from centroids if not supplied
    if concept_labels is None:
        concept_labels = build_concept_labels(cs.centroids.cpu().numpy())

    # Training background — subsampled
    N_train = len(train_sigs_std)
    bg_idxs = list(range(N_train))
    if N_train > max_bg_patches:
        bg_idxs = rng.choice(N_train, size=max_bg_patches, replace=False).tolist()
    bg_idxs = [int(x) for x in bg_idxs]
    X_bg        = train_sigs_std[bg_idxs].float().cpu().numpy()
    assigns_bg  = cs.hard_assign(train_sigs_std[bg_idxs]).cpu().numpy()

    # Collect test patches for all chosen samples into one block so we embed once
    chosen_pidxs: List[List[int]] = []   # per chosen sample
    all_test_flat: List[int] = []
    for sid in chosen:
        pidxs = sample_to_pidx.get(sid, [])[:K]
        chosen_pidxs.append(pidxs)
        all_test_flat.extend(pidxs)

    if not all_test_flat:
        print("  [trajectory] No test patches found for chosen samples.")
        return

    X_test_block = test_sigs[all_test_flat].float().numpy()   # [sum_K, D]
    X_combined   = np.vstack([X_bg, X_test_block])

    print(f"  [trajectory] Embedding {len(X_combined)} patches "
          f"({len(X_bg)} bg + {len(X_test_block)} test)…", flush=True)
    Z_combined, embed_name = _get_2d_embedding(X_combined)

    Z_bg_2d   = Z_combined[:len(X_bg)]
    Z_test_2d = Z_combined[len(X_bg):]   # rows match all_test_flat order

    # Compute approximate centroid positions in 2D from background cloud
    cent_z = np.zeros((M, 2))
    cent_n = np.zeros(M)
    for z, c in zip(Z_bg_2d, assigns_bg):
        cent_z[c] += z
        cent_n[c] += 1
    for m in range(M):
        if cent_n[m] > 0:
            cent_z[m] /= cent_n[m]

    # Slice Z_test_2d back into per-sample blocks
    offset = 0
    sample_z: Dict[int, np.ndarray] = {}
    for sid, pidxs in zip(chosen, chosen_pidxs):
        n_p = len(pidxs)
        sample_z[sid] = Z_test_2d[offset: offset + n_p]
        offset += n_p

    out_dir.mkdir(parents=True, exist_ok=True)

    for sid, pidxs in zip(chosen, chosen_pidxs):
        n_p    = len(pidxs)
        if n_p == 0:
            continue
        Z_s    = sample_z[sid]              # [n_p, 2]
        cseq_s = cseqs[sid]                 # [K, M]
        hard_z = cseq_s.argmax(-1)          # [K]
        fid_s  = fidelity[pidxs][:n_p]     # [n_p]
        ptchs  = all_patch_ranges[sid] if sid < len(all_patch_ranges) else []
        raw    = dataset.samples[sid].numpy()

        pred_i = int(preds[sid].item())
        true_i = int(labels_[sid].item())
        ok_str = "✓" if pred_i == true_i else "✗"
        pname  = label_names.get(pred_i, str(pred_i)) if label_names else str(pred_i)
        tname  = label_names.get(true_i, str(true_i)) if label_names else str(true_i)

        fig = plt.figure(figsize=(max(14, n_p * 2.0), 12))
        gs  = gridspec.GridSpec(
            3, n_p,
            height_ratios=[4.0, 1.6, 1.6],
            hspace=0.45, wspace=0.25,
            left=0.05, right=0.97, top=0.91, bottom=0.06,
        )

        # ── Row 0: embedding with trajectory ─────────────────────────────
        ax_u = fig.add_subplot(gs[0, :])

        # Background cloud
        ax_u.scatter(Z_bg_2d[:, 0], Z_bg_2d[:, 1],
                     c=assigns_bg, cmap="tab20", vmin=-0.5, vmax=M - 0.5,
                     s=4, alpha=0.18, linewidths=0, zorder=1)

        # Concept centroid labels
        for m in range(M):
            if cent_n[m] > 0:
                clbl  = concept_labels.get(m, "")
                # Show "C{id}: short-label" — truncate long labels
                short = clbl[:22] + "…" if len(clbl) > 22 else clbl
                ax_u.text(cent_z[m, 0], cent_z[m, 1],
                          f"C{m}\n{short}" if short else f"C{m}",
                          fontsize=6.5, ha='center', va='center',
                          color=cmap_c(m), fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                    alpha=0.70, lw=0))

        # Temporal arrows P_k → P_{k+1}
        for k in range(n_p - 1):
            x0, y0 = Z_s[k]
            x1, y1 = Z_s[k + 1]
            ax_u.annotate("", xy=(x1, y1), xytext=(x0, y0),
                          arrowprops=dict(arrowstyle='->', color='#333333',
                                          lw=1.1, alpha=0.55),
                          zorder=3)

        # Star markers for each patch
        for k in range(n_p):
            cidx = int(hard_z[k])
            col  = cmap_c(cidx)
            x, y = Z_s[k]
            ax_u.scatter(x, y, marker='*', color=col,
                         s=230, zorder=5, edgecolors='black', linewidths=0.6)
            ax_u.annotate(f"P{k + 1}", xy=(x, y), xytext=(x, y + 0.25),
                          fontsize=12, color=col, fontweight='bold',
                          ha='center', zorder=6)

        ax_u.set_title(
            f"Concept space trajectory — sample #{sid}   {ok_str}   "
            f"pred={pname}   true={tname}\n"
            f"★ = test patches  │  arrows = temporal order P1→PK  │  "
            f"cloud = training patches ({embed_name})",
            fontsize=8.5,
        )
        ax_u.set_xlabel(f"{embed_name}-1", fontsize=12)
        ax_u.set_ylabel(f"{embed_name}-2", fontsize=12)
        ax_u.tick_params(labelsize=10)
        ax_u.grid(True, alpha=0.12)

        # ── Row 1: patch waveforms ────────────────────────────────────────
        ax_waves: List = []
        for k in range(n_p):
            ax_w = fig.add_subplot(gs[1, k])
            cidx = int(hard_z[k])
            col  = cmap_c(cidx)
            if k < len(ptchs):
                t1, t2 = ptchs[k]
                ax_w.plot(raw[t1:t2, 0], color=col, lw=0.9)
            fid_val  = float(fid_s[k].item()) if k < len(fid_s) else 0.0
            clbl     = concept_labels.get(cidx, "")
            short    = clbl[:20] + "…" if len(clbl) > 20 else clbl
            ax_w.set_facecolor((*col[:3], 0.09))
            ax_w.set_title(
                f"P{k + 1} │ C{cidx}  ρ={fid_val:.2f}\n{short}",
                fontsize=7.5, pad=3,
            )
            ax_w.set_xticks([]); ax_w.set_yticks([])
            for spine in ax_w.spines.values():
                spine.set_edgecolor(col)
                spine.set_linewidth(1.0)
            ax_waves.append(ax_w)

        # ── Row 2: concept distribution bars ─────────────────────────────
        for k in range(n_p):
            ax_c  = fig.add_subplot(gs[2, k])
            pi_k  = cseq_s[k].numpy()
            cidx  = int(hard_z[k])
            bar_c = [cmap_c(m) if m == cidx else "#cccccc" for m in range(M)]
            ax_c.bar(np.arange(M), pi_k, color=bar_c, linewidth=0, width=1.0)
            ax_c.set_xlim(-0.5, M - 0.5)
            ax_c.set_ylim(0, 1.0)
            ax_c.set_xticks([]); ax_c.set_yticks([])
            ax_c.set_xlabel(f"M={M}", fontsize=10)
            if k == 0:
                ax_c.set_ylabel("π(C|patch)", fontsize=10)

        # Arrows: top of waveform panel → UMAP star
        for k, ax_w in enumerate(ax_waves):
            x, y = Z_s[k]
            col  = cmap_c(int(hard_z[k]))
            con  = ConnectionPatch(
                xyA=(0.5, 1.0), coordsA='axes fraction', axesA=ax_w,
                xyB=(x, y),     coordsB='data',           axesB=ax_u,
                arrowstyle='->', color=col, lw=0.8, alpha=0.55,
                shrinkA=3, shrinkB=6,
            )
            fig.add_artist(con)

        fig.suptitle(
            "Test sample concept trajectory  ·  "
            "how the signal's patches move through learned concept space",
            fontsize=12, fontweight='bold',
        )
        fpath = out_dir / f"trajectory_{sid:04d}.png"
        plt.savefig(fpath, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fpath}")


# ── Paired concept trajectory (concept space in centre) ──────────────────────

def _sig_region_strip(ax, sig_vec: np.ndarray, n_channels: int, sig_mode: str = "full"):
    """
    Draw a compact 1-row signature heatmap on *ax*, divided into colour-coded
    regions.  Each region's normalised values are shown as a heatmap strip;
    region boundaries and labels are annotated below.

    Regions (full mode):
      Entropy      — mean_H, var_H, max_H, pk_pos, log_L              (5)
      Moments      — μ_c0…cC, σ²_c0…cC                              (2C)
      Distributional — drift, mean_NLL, max_NLL, KL                   (4)
      Morphological  — fft_pk, ac1–ac3, skew, kurt, ZCR, p2p          (8)
    """
    REGION_COLOURS = ['#4e79a7', '#59a14f', '#f28e2b', '#e15759']
    REGION_LABELS  = ['Entropy', 'Moments', 'Distrib.', 'Morphol.']

    if sig_mode == "entropy_only":
        boundaries = [(0, len(sig_vec), 'Entropy', REGION_COLOURS[0])]
    elif sig_mode == "moments_only":
        boundaries = [(0, len(sig_vec), 'Moments', REGION_COLOURS[1])]
    elif sig_mode == "distributional_only":
        boundaries = [(0, len(sig_vec), 'Distrib.', REGION_COLOURS[2])]
    elif sig_mode == "morphological_only":
        boundaries = [(0, len(sig_vec), 'Morphol.', REGION_COLOURS[3])]
    elif sig_mode == "residual_morphology":
        boundaries = [(0, len(sig_vec), 'Res.Morph', REGION_COLOURS[3])]
    elif sig_mode == "surprise_trajectory":
        boundaries = [(0, len(sig_vec), 'Surprise', '#9467bd')]
    elif sig_mode == "surprise_full":
        # Moments: μ_c0..cC, σ²_c0..cC, log_L  → 2C+1 features
        # Residual morphology: r_fft..r_p2p      → 8 features
        # Surprise trajectory: s_pos..r_en       → 4 features
        m_end  = 2 * n_channels + 1
        rm_end = m_end + 8
        st_end = min(rm_end + 4, len(sig_vec))
        boundaries = [
            (0,     m_end,  'Moments',   REGION_COLOURS[1]),
            (m_end, rm_end, 'Res.Morph', REGION_COLOURS[3]),
            (rm_end, st_end,'Surprise',  '#9467bd'),
        ]
        boundaries = [(s, e, l, c) for s, e, l, c in boundaries if s < e]
    else:
        # full — entropy(5) + moments(2C) + distributional(4) + morphological(8)
        e_end  = 5
        m_end  = e_end + 2 * n_channels
        d_end  = m_end + 4
        mo_end = min(d_end + 8, len(sig_vec))
        boundaries = [
            (0,     e_end,  REGION_LABELS[0], REGION_COLOURS[0]),
            (e_end, m_end,  REGION_LABELS[1], REGION_COLOURS[1]),
            (m_end, d_end,  REGION_LABELS[2], REGION_COLOURS[2]),
            (d_end, mo_end, REGION_LABELS[3], REGION_COLOURS[3]),
        ]
        boundaries = [(s, e, l, c) for s, e, l, c in boundaries if s < e]

    ax.set_xlim(0, len(sig_vec))
    ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for s, e, lbl, col in boundaries:
        region = sig_vec[s:e]
        if len(region) == 0:
            continue
        # Normalise to [0,1] for display
        r_min, r_max = region.min(), region.max()
        norm = (region - r_min) / (r_max - r_min + 1e-8)
        # Draw the heatmap strip as thin coloured blocks
        for fi, val in enumerate(norm):
            ax.barh(0.5, 1, left=s + fi, height=0.9,
                    color=col, alpha=0.25 + 0.65 * float(val),
                    linewidth=0)
        # Region label
        mid = (s + e) / 2
        ax.text(mid, 0.50, lbl, ha='center', va='center',
                fontsize=12, color=col, fontweight='bold', clip_on=True)
        # Boundary line
        if s > 0:
            ax.axvline(s, color='white', lw=0.6, alpha=0.9)


def paired_concept_trajectory(
    results: dict,
    train_sigs_std: torch.Tensor,
    dataset,
    all_patch_ranges: List[List[Tuple[int, int]]],
    cs: ConceptSpace,
    out_dir: Path,
    n_pairs: int = 20,
    max_bg_patches: int = 2000,
    label_names: Optional[Dict[int, str]] = None,
    sig_mode: str = "full",
    n_channels: int = 1,
    rng=None,
):
    """
    For n_pairs pairs of test samples render one PNG each:

      Top row    — K patch waveforms for sample A (blue ★ in concept space)
      Middle     — shared concept space embedding (UMAP/t-SNE/PCA) with both
                   samples' patches overlaid; ★ = sample A, ● = sample B,
                   each connected in temporal order by coloured arrows
      Bottom row — K patch waveforms for sample B (red ● in concept space)

    ConnectionPatch arrows link each waveform panel to its star/circle in the
    embedding.  Pairs are drawn as a mix of same-label and different-label.
    No bar charts are produced.
    """
    from matplotlib.patches import ConnectionPatch
    from collections import defaultdict

    if rng is None:
        rng = np.random.default_rng(42)

    M          = cs.M
    K          = results["concept_seqs"].shape[1]
    preds      = results["preds"]
    labels_    = results["labels"]
    cseqs      = results["concept_seqs"]      # [N_test, K, M]
    test_sigs  = results["patch_sigs"]        # [N_test_patches, D]
    sample_for = results["sample_for_patch"]  # [N_test_patches]
    fidelity   = results["patch_fidelity"]    # [N_test_patches]
    N          = len(dataset)

    # Per-sample ordered patch index list
    sample_to_pidx: Dict[int, List[int]] = {}
    for pi, sid in enumerate(sample_for.tolist()):
        sample_to_pidx.setdefault(int(sid), []).append(pi)

    # Group samples by label
    label_to_sids: Dict[int, List[int]] = defaultdict(list)
    for sid in range(N):
        lbl = int(labels_[sid].item())
        label_to_sids[lbl].append(sid)
    label_list = sorted(label_to_sids.keys())

    # Build pairs: half same-label, half different-label
    n_same = n_pairs // 2
    n_diff = n_pairs - n_same
    pairs: List[Tuple[int, int]] = []

    seen: set = set()
    for _ in range(n_same * 10):
        lbl = int(rng.choice(label_list))
        sids = label_to_sids[lbl]
        if len(sids) < 2:
            continue
        a, b = (int(x) for x in rng.choice(sids, size=2, replace=False))
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((a, b))
        if len(pairs) >= n_same:
            break

    for _ in range(n_diff * 10):
        if len(label_list) < 2:
            break
        l1, l2 = (int(x) for x in rng.choice(label_list, size=2, replace=False))
        a = int(rng.choice(label_to_sids[l1]))
        b = int(rng.choice(label_to_sids[l2]))
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((a, b))
        if len(pairs) >= n_pairs:
            break

    pairs = pairs[:n_pairs]

    cmap_c = plt.cm.get_cmap("tab20", M)

    # Training background — subsampled
    N_train = len(train_sigs_std)
    bg_idxs = list(range(N_train))
    if N_train > max_bg_patches:
        bg_idxs = rng.choice(N_train, size=max_bg_patches, replace=False).tolist()
    bg_idxs    = [int(x) for x in bg_idxs]
    X_bg       = train_sigs_std[bg_idxs].float().cpu().numpy()
    assigns_bg = cs.hard_assign(train_sigs_std[bg_idxs]).cpu().numpy()

    # Collect all unique sample patches; embed once so all samples share axes
    unique_sids: List[int] = list(dict.fromkeys(s for pair in pairs for s in pair))
    chosen_pidxs: Dict[int, List[int]] = {}
    all_test_flat: List[int] = []
    for sid in unique_sids:
        pidxs = sample_to_pidx.get(sid, [])[:K]
        chosen_pidxs[sid] = pidxs
        all_test_flat.extend(pidxs)

    if not all_test_flat:
        print("  [paired_trajectory] No test patches found.")
        return

    X_test_block = test_sigs[all_test_flat].float().numpy()
    X_combined   = np.vstack([X_bg, X_test_block])

    print(f"  [paired_trajectory] Embedding {len(X_combined)} patches "
          f"({len(X_bg)} bg + {len(X_test_block)} test)…", flush=True)
    Z_combined, embed_name = _get_2d_embedding(X_combined)

    Z_bg_2d   = Z_combined[:len(X_bg)]
    Z_test_2d = Z_combined[len(X_bg):]

    # Slice embedding back per sample
    offset = 0
    sid_to_z: Dict[int, np.ndarray] = {}
    for sid in unique_sids:
        n_p = len(chosen_pidxs[sid])
        sid_to_z[sid] = Z_test_2d[offset: offset + n_p]
        offset += n_p

    # Approximate centroid positions from background cloud
    cent_z = np.zeros((M, 2))
    cent_n = np.zeros(M)
    for z, c in zip(Z_bg_2d, assigns_bg):
        cent_z[c] += z
        cent_n[c] += 1
    for m in range(M):
        if cent_n[m] > 0:
            cent_z[m] /= cent_n[m]

    out_dir.mkdir(parents=True, exist_ok=True)

    for pair_i, (sid_a, sid_b) in enumerate(pairs):
        pidxs_a = chosen_pidxs.get(sid_a, [])
        pidxs_b = chosen_pidxs.get(sid_b, [])
        n_pa = len(pidxs_a)
        n_pb = len(pidxs_b)
        if n_pa == 0 or n_pb == 0:
            continue

        Z_a = sid_to_z[sid_a]       # [n_pa, 2]
        Z_b = sid_to_z[sid_b]       # [n_pb, 2]
        cseq_a = cseqs[sid_a]       # [K, M]
        cseq_b = cseqs[sid_b]
        hard_a = cseq_a.argmax(-1)  # [K]
        hard_b = cseq_b.argmax(-1)
        fid_a  = fidelity[pidxs_a][:n_pa]
        fid_b  = fidelity[pidxs_b][:n_pb]
        ptchs_a = all_patch_ranges[sid_a] if sid_a < len(all_patch_ranges) else []
        ptchs_b = all_patch_ranges[sid_b] if sid_b < len(all_patch_ranges) else []
        raw_a   = dataset.samples[sid_a].numpy()
        raw_b   = dataset.samples[sid_b].numpy()

        lbl_a  = int(labels_[sid_a].item())
        lbl_b  = int(labels_[sid_b].item())
        name_a = label_names.get(lbl_a, str(lbl_a)) if label_names else str(lbl_a)
        name_b = label_names.get(lbl_b, str(lbl_b)) if label_names else str(lbl_b)
        same_str = "same label" if lbl_a == lbl_b else "diff label"

        n_cols = max(n_pa, n_pb)

        fig = plt.figure(figsize=(max(14, n_cols * 2.2), 12))
        # 5 rows: top waveforms | sig strip A | concept space | sig strip B | bot waveforms
        gs  = gridspec.GridSpec(
            5, n_cols,
            height_ratios=[1.6, 0.38, 5.0, 0.38, 1.6],
            hspace=0.22, wspace=0.22,
            left=0.05, right=0.97, top=0.92, bottom=0.05,
        )

        # ── Row 2: concept space ──────────────────────────────────────────
        ax_u = fig.add_subplot(gs[2, :])

        ax_u.scatter(Z_bg_2d[:, 0], Z_bg_2d[:, 1],
                     c=assigns_bg, cmap="tab20", vmin=-0.5, vmax=M - 0.5,
                     s=4, alpha=0.18, linewidths=0, zorder=1)

        for m in range(M):
            if cent_n[m] > 0:
                ax_u.text(cent_z[m, 0], cent_z[m, 1], f"C{m}",
                          fontsize=6, ha='center', va='center',
                          color=cmap_c(m), fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.12', fc='white',
                                    alpha=0.60, lw=0))

        # Sample A trajectory (★, blue outlines)
        for k in range(n_pa - 1):
            x0, y0 = Z_a[k]; x1, y1 = Z_a[k + 1]
            ax_u.annotate("", xy=(x1, y1), xytext=(x0, y0),
                          arrowprops=dict(arrowstyle='->', color='#2255bb',
                                          lw=1.1, alpha=0.55), zorder=3)
        for k in range(n_pa):
            cidx = int(hard_a[k])
            col  = cmap_c(cidx)
            x, y = Z_a[k]
            ax_u.scatter(x, y, marker='*', color=col, s=260,
                         zorder=5, edgecolors='#2255bb', linewidths=0.9)
            ax_u.annotate(f"A{k+1}", xy=(x, y), xytext=(x, y + 0.28),
                          fontsize=6.5, color='#2255bb', fontweight='bold',
                          ha='center', zorder=6)

        # Sample B trajectory (●, red outlines)
        for k in range(n_pb - 1):
            x0, y0 = Z_b[k]; x1, y1 = Z_b[k + 1]
            ax_u.annotate("", xy=(x1, y1), xytext=(x0, y0),
                          arrowprops=dict(arrowstyle='->', color='#bb3311',
                                          lw=1.1, alpha=0.55), zorder=3)
        for k in range(n_pb):
            cidx = int(hard_b[k])
            col  = cmap_c(cidx)
            x, y = Z_b[k]
            ax_u.scatter(x, y, marker='o', color=col, s=130,
                         zorder=5, edgecolors='#bb3311', linewidths=0.9)
            ax_u.annotate(f"B{k+1}", xy=(x, y), xytext=(x, y - 0.30),
                          fontsize=6.5, color='#bb3311', fontweight='bold',
                          ha='center', zorder=6)

        ax_u.set_title(
            f"★ Sample A  #{sid_a}  label={name_a}    "
            f"●  Sample B  #{sid_b}  label={name_b}    [{same_str}]\n"
            f"arrows = temporal order  │  cloud = training patches ({embed_name})",
            fontsize=8.5,
        )
        ax_u.set_xlabel(f"{embed_name}-1", fontsize=8)
        ax_u.set_ylabel(f"{embed_name}-2", fontsize=8)
        ax_u.tick_params(labelsize=7)
        ax_u.grid(True, alpha=0.12)

        patch_sigs = results["patch_sigs"]   # [N_test_patches, D]

        # ── Row 0: sample A waveforms ─────────────────────────────────────
        ax_waves_a: List = []
        for k in range(n_pa):
            ax_w = fig.add_subplot(gs[0, k])
            cidx = int(hard_a[k])
            col  = cmap_c(cidx)
            if k < len(ptchs_a):
                t1, t2 = ptchs_a[k]
                ax_w.plot(raw_a[t1:t2, 0], color=col, lw=0.9)
            ax_w.set_facecolor((*col[:3], 0.09))
            ax_w.set_title(f"A{k+1} | C{cidx}", fontsize=5.5, pad=2)
            ax_w.set_xticks([]); ax_w.set_yticks([])
            for spine in ax_w.spines.values():
                spine.set_edgecolor('#2255bb')
                spine.set_linewidth(1.0)
            ax_waves_a.append(ax_w)
        for k in range(n_pa, n_cols):
            fig.add_subplot(gs[0, k]).set_visible(False)

        # ── Row 1: signature region strips for A ──────────────────────────
        for k in range(n_pa):
            ax_s = fig.add_subplot(gs[1, k])
            if k < len(pidxs_a):
                sig_v = patch_sigs[pidxs_a[k]].numpy()
                _sig_region_strip(ax_s, sig_v, n_channels, sig_mode)
        for k in range(n_pa, n_cols):
            fig.add_subplot(gs[1, k]).set_visible(False)

        # ── Row 3: signature region strips for B ──────────────────────────
        for k in range(n_pb):
            ax_s = fig.add_subplot(gs[3, k])
            if k < len(pidxs_b):
                sig_v = patch_sigs[pidxs_b[k]].numpy()
                _sig_region_strip(ax_s, sig_v, n_channels, sig_mode)
        for k in range(n_pb, n_cols):
            fig.add_subplot(gs[3, k]).set_visible(False)

        # ── Row 4: sample B waveforms ─────────────────────────────────────
        ax_waves_b: List = []
        for k in range(n_pb):
            ax_w = fig.add_subplot(gs[4, k])
            cidx = int(hard_b[k])
            col  = cmap_c(cidx)
            if k < len(ptchs_b):
                t1, t2 = ptchs_b[k]
                ax_w.plot(raw_b[t1:t2, 0], color=col, lw=0.9)
            ax_w.set_facecolor((*col[:3], 0.09))
            ax_w.set_title(f"B{k+1} | C{cidx}", fontsize=5.5, pad=2)
            ax_w.set_xticks([]); ax_w.set_yticks([])
            for spine in ax_w.spines.values():
                spine.set_edgecolor('#bb3311')
                spine.set_linewidth(1.0)
            ax_waves_b.append(ax_w)
        for k in range(n_pb, n_cols):
            fig.add_subplot(gs[4, k]).set_visible(False)

        # Arrows: bottom of top waveform → star in concept space (A)
        for k, ax_w in enumerate(ax_waves_a):
            x, y = Z_a[k]
            con = ConnectionPatch(
                xyA=(0.5, 0.0), coordsA='axes fraction', axesA=ax_w,
                xyB=(x, y),     coordsB='data',           axesB=ax_u,
                arrowstyle='->', color='#2255bb', lw=0.8, alpha=0.50,
                shrinkA=3, shrinkB=6,
            )
            fig.add_artist(con)

        # Arrows: circle in concept space → top of bottom waveform (B)
        for k, ax_w in enumerate(ax_waves_b):
            x, y = Z_b[k]
            con = ConnectionPatch(
                xyA=(x, y),     coordsA='data',           axesA=ax_u,
                xyB=(0.5, 1.0), coordsB='axes fraction', axesB=ax_w,
                arrowstyle='->', color='#bb3311', lw=0.8, alpha=0.50,
                shrinkA=6, shrinkB=3,
            )
            fig.add_artist(con)

        fig.suptitle(
            f"Paired concept trajectories  ·  "
            f"A (top, blue ★)  vs  B (bottom, red ●)  [{same_str}]",
            fontsize=10, fontweight='bold',
        )
        fpath = out_dir / f"pair_{pair_i:02d}_A{sid_a:04d}_B{sid_b:04d}.png"
        plt.savefig(fpath, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fpath}")


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
    assigns   = cs.hard_assign(sigs_std_train).cpu().numpy()    # [N_train_patches]
    labels_np = train_patch_labels.cpu().numpy()
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
    parser.add_argument("--patch_scale", default=None,
                        help="Patch scale key used during training (for sig cache lookup)")
    parser.add_argument("--no_sig_cache", action="store_true",
                        help="Skip signature cache and always re-extract")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )

    dc  = DATASET_CONFIGS[args.dataset]
    ppc = get_patch_config(args.dataset, dc["seq_len"])

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

    sig_mode = art.get("sig_mode", "full")

    # Build patcher
    if art["patcher"] == "entropy":
        patcher = EntropyPatcher(density_model, channel_mixer,
                                  K=K, L_min=ppc["L_min"], burn_in=ppc["burn_in"],
                                  mode=art.get("boundary_mode", "entropy"))
    else:
        patcher = StaticPatcher(K=K)

    # ── Signature cache ───────────────────────────────────────────────────────
    _sig_mode_key    = "-".join(sig_mode) if isinstance(sig_mode, list) else sig_mode
    _patch_scale_key = args.patch_scale or "default"
    _boundary_mode   = art.get("boundary_mode", "entropy")
    _sig_cache_key   = (f"{args.dataset}__{art['patcher']}__{_boundary_mode}"
                        f"__{_sig_mode_key}__{_patch_scale_key}")
    _sig_cache_dir   = Path(dc["output_dir"]) / "sig_cache"
    _sig_cache_path  = _sig_cache_dir / f"{_sig_cache_key}.pt"
    _use_sig_cache   = not args.no_sig_cache

    eval_sigs = eval_patch_ranges = eval_sample_ids = None
    train_sigs_raw = train_patch_ranges = train_sids = None

    if _use_sig_cache and _sig_cache_path.exists():
        print(f"\n── Loading cached signatures ───────────────────────────────")
        print(f"  Key:   {_sig_cache_key}")
        print(f"  Cache: {_sig_cache_path}")
        _cache = torch.load(_sig_cache_path, map_location="cpu", weights_only=False)
        split_key = args.split
        eval_sigs         = _cache[f"{split_key}_sigs"].to(device)
        eval_patch_ranges = _cache[f"{split_key}_patch_ranges"]
        eval_sample_ids   = _cache[f"{split_key}_sample_ids"]
        train_sigs_raw    = _cache["train_sigs"].to(device)
        train_patch_ranges = _cache["train_patch_ranges"]
        train_sids        = _cache["train_sample_ids"]
        if "mu_marg" in _cache:
            mu_marg     = _cache["mu_marg"].to(device)
            sigma2_marg = _cache["sigma2_marg"].to(device)
        print(f"  Patches (eval={args.split}): {len(eval_sigs)}  train: {len(train_sigs_raw)}")

    # Load dataset split
    dataset = TimeSeriesDataset(dc["root_path"], args.split, dc["seq_len"])
    print(f"Evaluating on {len(dataset)} {args.split} samples\n")

    # Extract eval-split sigs if not loaded from cache
    if eval_sigs is None:
        print(f"Extracting {args.split} signatures…")
        eval_sigs, eval_patch_ranges, eval_sample_ids = extract_signatures_for_dataset(
            density_model, channel_mixer, dataset, patcher,
            channel_mean, channel_std, mu_marg, sigma2_marg, device, mode=sig_mode,
        )

    # Extract training sigs if not loaded from cache
    train_ds = TimeSeriesDataset(dc["root_path"], "train", dc["seq_len"])
    if train_sigs_raw is None:
        print("Extracting training signatures for prototype lookup and semantic analysis…")
        train_sigs_raw, train_patch_ranges, train_sids = extract_signatures_for_dataset(
            density_model, channel_mixer, train_ds, patcher,
            channel_mean, channel_std, mu_marg, sigma2_marg, device, mode=sig_mode,
        )

    # Save to cache if freshly extracted
    if _use_sig_cache and not _sig_cache_path.exists():
        _sig_cache_dir.mkdir(parents=True, exist_ok=True)
        split_key = args.split
        torch.save({
            f"{split_key}_sigs":         eval_sigs.cpu(),
            f"{split_key}_patch_ranges": eval_patch_ranges,
            f"{split_key}_sample_ids":   eval_sample_ids,
            "train_sigs":         train_sigs_raw.cpu(),
            "train_patch_ranges": train_patch_ranges,
            "train_sample_ids":   train_sids,
            "mu_marg":            mu_marg.cpu(),
            "sigma2_marg":        sigma2_marg.cpu(),
        }, _sig_cache_path)
        print(f"  Saved signature cache → {_sig_cache_path}")

    train_sigs_std     = standardizer.transform(train_sigs_raw)
    train_patch_labels = train_ds.labels[train_sids]

    # Run pipeline inference
    print("── Running pipeline inference ─────────────────────────────")
    results = run_pipeline_on_dataset(
        density_model, channel_mixer, channel_mean, channel_std,
        patcher, cs, standardizer, encoder, classifier, dataset,
        mu_marg, sigma2_marg, device,
        batch_size=args.batch_size,
        sig_mode=sig_mode,
        temperature=1.0,
        precomputed_sigs=eval_sigs,
        precomputed_patch_ranges=eval_patch_ranges,
        precomputed_sample_ids=eval_sample_ids,
    )

    all_patch_ranges = results["patch_ranges"]

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

    print("\n── 8. Pipeline trace figures ──────────────────────────────")
    n_channels = dc.get("n_channels", 1)
    pipeline_trace_figure(
        results, dataset, all_patch_ranges, cs,
        out_dir=out_dir / "traces",
        n_samples=5,
        label_names=label_names,
        sig_mode=sig_mode,
        n_channels=n_channels,
    )

    print("\n── 9. Concept trajectory figures ──────────────────────────")
    _centroids_np = art["concept_space"]["centroids"].numpy()
    _concept_labels = build_concept_labels(_centroids_np)
    test_concept_trajectory(
        results, train_sigs_std, dataset, all_patch_ranges, cs,
        out_dir=out_dir / "trajectories",
        n_samples=4,
        label_names=label_names,
        concept_labels=_concept_labels,
    )

    print("\n── 10. Paired concept trajectory figures ──────────────────")
    paired_concept_trajectory(
        results, train_sigs_std, dataset, all_patch_ranges, cs,
        out_dir=out_dir / "paired_trajectories",
        n_pairs=20,
        label_names=label_names,
        sig_mode=sig_mode,
        n_channels=n_channels,
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
