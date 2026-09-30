"""
evaluate.py — minimal evaluation of a trained ConceptTime pipeline.

Loads a trained pipeline artifact together with its frozen density model,
runs the full concept pipeline on a data split, and reports classification
accuracy and macro-F1. Reproduces the numbers stored in the checkpoint.

Example
-------
    python src/evaluate.py \
        --dataset Epilepsy \
        --artifact checkpoints/Epilepsy/pipeline_best.pt \
        --density  checkpoints/Epilepsy/gaussian_entropy_best.pt \
        --split    test

What a checkpoint contains / needs
----------------------------------
The pipeline artifact (`pipeline_best.pt`) is self-contained for the concept
space, the signature standardizer, the (optional) patch encoder, and the
classifier(s). To run inference you additionally need:
  1. the per-dataset density model  (`gaussian_entropy_best.pt`), used by the
     entropy patcher and to compute predictive-surprise signatures; and
  2. the data split you want to score (e.g. `dataset/<Dataset>/test.pt`).
Signatures are recomputed on the fly (no signature cache required).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

from train_gaussian_entropy_model import TimeSeriesDataset, DATASET_CONFIGS
from train_pipeline import load_density_model
from patcher import EntropyPatcher, StaticPatcher, get_patch_config
from signatures import extract_signatures_for_dataset, SignatureStandardizer
from concept_space import ConceptSpace
from concept_encoder import ConceptEncoder, build_patch_dataset, patch_collate_fn
from classifier import TransformerClassifier, SparseLinearClassifier


def build_classifier(cls_type, M, K, n_classes, state_dict, device):
    cls = (TransformerClassifier(M=M, K=K, n_classes=n_classes)
           if cls_type == "transformer"
           else SparseLinearClassifier(M=M, K=K, n_classes=n_classes)).to(device)
    cls.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
    cls.eval()
    return cls


@torch.no_grad()
def concept_sequences(art, density_model, channel_mixer, channel_mean, channel_std,
                      cs, standardizer, encoder, patcher, dataset,
                      mu_marg, sigma2_marg, device, batch_size, sig_mode):
    """Return per-sample concept sequences [N, K, M] and labels [N]."""
    K = art["K"]

    if encoder is None:
        # Direct-signature path: soft-assign standardized signatures to prototypes.
        sigs, _, _ = extract_signatures_for_dataset(
            density_model, channel_mixer, dataset, patcher,
            channel_mean, channel_std, mu_marg, sigma2_marg,
            device, mode=sig_mode,
        )
        sigs_std = standardizer.transform(sigs)
        pi = cs.soft_assign(sigs_std, temperature=1.0)          # [N_patches, M]
        N = len(dataset)
        seqs = pi.cpu().reshape(N, K, -1)                       # [N, K, M]
        labels = dataset.labels[:N].long()
    else:
        # Learned-encoder path (e.g. moments_only artifacts).
        enc_ds = build_patch_dataset(
            density_model, channel_mixer, dataset, patcher, cs,
            channel_mean, channel_std, device, temperature=1.0,
            sig_mode=sig_mode, mu_marg=mu_marg, sigma2_marg=sigma2_marg,
        )
        loader = DataLoader(enc_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=patch_collate_fn, num_workers=0)
        qs, lbls = [], []
        for patches, mask, _pi, labels in loader:
            qs.append(encoder(patches.to(device), mask.to(device)).cpu())
            lbls.append(labels)
        q = torch.cat(qs)
        seqs = q.reshape(len(dataset), K, -1)
        labels = torch.cat(lbls)[::K]
    return seqs, labels


@torch.no_grad()
def classify(classifier, seqs, labels, device, batch_size):
    loader = DataLoader(TensorDataset(seqs, labels), batch_size=batch_size, shuffle=False)
    preds = []
    for cseq, _ in loader:
        preds.append(classifier(cseq.to(device)).argmax(-1).cpu())
    preds = torch.cat(preds).numpy()
    y = labels.numpy()
    return accuracy_score(y, preds), f1_score(y, preds, average="macro")


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained ConceptTime pipeline.")
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--artifact", required=True, help="Path to pipeline_best.pt")
    p.add_argument("--density", required=True, help="Path to gaussian_entropy_best.pt")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--data_root", default="./dataset",
                   help="Folder containing <data_root>/<Dataset>/<split>.pt")
    p.add_argument("--device", default=None, help="cuda / cpu (auto if unset)")
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    dc = DATASET_CONFIGS[args.dataset]
    seq_len = dc["seq_len"]

    print(f"\n{'='*60}\n  ConceptTime evaluation  |  {args.dataset}  |  split={args.split}\n{'='*60}")

    # ── Load pipeline artifact ────────────────────────────────────────────────
    art = torch.load(args.artifact, map_location="cpu", weights_only=False)
    M, K, n_classes = art["M"], art["K"], art["n_classes"]
    sig_mode = art.get("sig_mode", "surprise_full")
    cls_type = art.get("classifier_type", "transformer")

    # Frozen density model (used by patcher + signature extraction).
    density_model, channel_mixer, channel_mean, channel_std = load_density_model(
        args.density, device)

    # Concept space + signature standardizer (stored inside the artifact).
    cs = ConceptSpace(M, art["concept_space"]["D"])
    cs.load_state_dict(art["concept_space"])
    standardizer = SignatureStandardizer()
    standardizer.load_state_dict(art["sig_standardizer"])

    # Optional patch encoder (None for direct-signature artifacts).
    if art.get("direct_sig", False):
        encoder = None
    else:
        enc_cfg = dict(art["encoder_config"])
        enc_cfg.setdefault("max_patch_len",
                           art["encoder_state_dict"]["pos_emb.emb.weight"].shape[0])
        encoder = ConceptEncoder(**enc_cfg).to(device)
        encoder.load_state_dict({k: v.to(device)
                                 for k, v in art["encoder_state_dict"].items()})
        encoder.eval()

    mu_marg = art["mu_marg"].to(device)
    sigma2_marg = art["sigma2_marg"].to(device)

    # ── Patcher ───────────────────────────────────────────────────────────────
    ppc = get_patch_config(args.dataset, seq_len)
    if art["patcher"] == "entropy":
        patcher = EntropyPatcher(density_model, channel_mixer, K=K,
                                 L_min=ppc["L_min"], burn_in=ppc["burn_in"],
                                 mode=art.get("boundary_mode", "entropy"))
    else:
        patcher = StaticPatcher(K=K)

    print(f"  M={M}  K={K}  patcher={art['patcher']}  boundary={art.get('boundary_mode')}"
          f"  sig_mode={sig_mode}  encoder={'yes' if encoder else 'direct'}")

    # ── Data ──────────────────────────────────────────────────────────────────
    root = f"{args.data_root}/{args.dataset}"
    dataset = TimeSeriesDataset(root, args.split, seq_len)

    seqs, labels = concept_sequences(
        art, density_model, channel_mixer, channel_mean, channel_std,
        cs, standardizer, encoder, patcher, dataset,
        mu_marg, sigma2_marg, device, args.batch_size, sig_mode)

    # ── Score each label-fraction classifier stored in the artifact ───────────
    print(f"\n  {'Labels':<10}{'Accuracy':>12}{'Macro-F1':>12}")
    print("  " + "-" * 34)
    fractions = [("1%", "classifier_1pct_state_dict"),
                 ("5%", "classifier_5pct_state_dict"),
                 ("Full", "classifier_100pct_state_dict")]
    have_fracs = any(k in art for _, k in fractions)
    if not have_fracs:
        fractions = [("Full", "classifier_state_dict")]
    for name, key in fractions:
        if key not in art:
            continue
        clf = build_classifier(cls_type, M, K, n_classes, art[key], device)
        acc, f1 = classify(clf, seqs, labels, device, args.batch_size)
        print(f"  {name:<10}{acc*100:>11.2f}%{f1*100:>11.2f}%")
    print()


if __name__ == "__main__":
    main()
