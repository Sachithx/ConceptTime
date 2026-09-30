# (NeurIPS'26) ConceptTime: Predictive Surprise as Self-Grounding Concept Bottleneck for Interpretable Time Series

**Predictive Surprise as a Self-Grounding Concept Bottleneck for Interpretable Time Series**

ConceptTime is a concept-bottleneck framework for interpretable time-series
classification. It defines patch-level concepts from the *predictive
distribution* of a probabilistic world model: a causal density model predicts
each next step as a Gaussian, and the **surprise** (negative log-likelihood)
of the observed signal under that model both (i) segments the series into
variable-length patches and (ii) produces a fixed-length *signature* per patch.
Signatures are clustered into a discrete concept vocabulary, and a small
transformer classifies the resulting concept sequence.

This repository contains the **minimal code and pretrained checkpoints** needed
to reproduce the Epilepsy pipeline end-to-end (training → evaluation) and to
evaluate pretrained checkpoints on **Epilepsy**, **UCI-HAR**, and **Sleep-EDF**
without any training.

---

## Repository structure

```
ConceptTime_NeurIPS/
├── README.md
├── LICENSE
├── requirements.txt
├── run.sh                         # optional libstdc++ preload wrapper (see below)
├── download_data.py               # fetch HAR / Sleep-EDF test splits from HuggingFace
├── scripts/
│   └── upload_data_to_hf.sh       # (maintainers) publish the large splits to the Hub
├── src/
│   ├── GaussianEntropyModel.py        # causal Gaussian density (world) model
│   ├── train_gaussian_entropy_model.py# Stage 1: pretrain the density model
│   ├── patcher.py                     # surprise/entropy-guided patch segmentation
│   ├── signatures.py                  # per-patch fixed-length feature extraction
│   ├── concept_space.py               # cluster signatures into M concept prototypes
│   ├── concept_encoder.py             # (optional) learned patch encoder
│   ├── classifier.py                  # transformer / sparse-linear concept classifier
│   ├── train_pipeline.py              # Stage 2: train the full pipeline
│   └── evaluate.py                    # load a checkpoint and report accuracy / macro-F1
├── dataset/
│   └── Epilepsy/{train,val,test}.pt   # in this repo (train + evaluate)
│       # HAR/ and SLeep-EDF/ test splits are fetched from HuggingFace
└── checkpoints/
    └── Epilepsy/{gaussian_entropy_best.pt, pipeline_best.pt}   # in this repo
        # HAR/ and SLeep-EDF/ checkpoints are fetched from HuggingFace
```

**What lives where.** Only **Epilepsy** (data + checkpoints) ships in this
GitHub repo, so `git clone` alone reproduces the full Epilepsy train→eval
pipeline. For **HAR** and **Sleep-EDF**, both the test split and the trained
checkpoints are hosted on the **HuggingFace Hub**
([`sachithabey/ConceptTime`](https://huggingface.co/datasets/sachithabey/ConceptTime))
and pulled with `python download_data.py` (see [Datasets](#datasets)). That Hub
repo also archives *all* datasets (including FD) and *all* checkpoints under
`archive/`.

Each dataset has **two** checkpoints:

| File | What it is |
|------|------------|
| `gaussian_entropy_best.pt` | the frozen causal density model (Stage 1) |
| `pipeline_best.pt`         | the full pipeline: concept space, signature standardizer, (optional) encoder, and the trained classifiers (Stage 2) |

Both are required for evaluation: the density model drives the patcher and the
predictive-surprise signatures; the pipeline artifact holds the concept
vocabulary and the classifier.

---

## Installation

```bash
conda create -n concepttime python=3.10 -y
conda activate concepttime
pip install -r requirements.txt
```

Requirements: PyTorch ≥ 2.0, NumPy, scikit-learn, tqdm. A GPU is recommended
(especially for Sleep-EDF, whose sequences are long) but not required. Weights &
Biases are optional — the training scripts run fine without them; pass `--no_wandb`.

**libstdc++ note.** Some conda + recent-PyTorch setups raise
`GLIBCXX_3.4.xx not found` on `import torch`. If that happens, prefix any
command with the provided wrapper, which preloads the conda env's libstdc++:

```bash
./run.sh python src/evaluate.py ...
```

Otherwise, you can ignore `run.sh` and call `python` directly.

---

## Datasets

All splits are PyTorch `.pt` files, each a dict:

```python
{
  "samples": torch.Tensor,   # float32, shape [N, C, T]  (channel-first; auto-permuted on load)
  "labels":  torch.Tensor,   # long,    shape [N]
}
```

| Dataset | C | T | Classes | Needed to evaluate | Hosted on |
|---------|---|---|---------|--------------------|-----------|
| Epilepsy   | 1 | 178  | 2 | train + val + test | **this GitHub repo** |
| UCI-HAR    | 9 | 128  | 6 | test split | HuggingFace Hub |
| Sleep-EDF  | 1 | 2999 | 5 | test split | HuggingFace Hub |

**Where things live.** Only **Epilepsy** (its data *and* checkpoints) ships in
this GitHub repo, so a bare clone reproduces the full Epilepsy pipeline. For
**HAR** and **Sleep-EDF**, both the test split and the trained checkpoints are on
the **HuggingFace Hub** (the data is large — Sleep-EDF's test.pt is ~103 MB,
above GitHub's 100 MB per-file limit).

**Fetch HAR / Sleep-EDF (test data + checkpoints) before evaluating them:**

```bash
pip install huggingface_hub
python download_data.py                       # both HAR + Sleep-EDF
# or: python download_data.py --datasets HAR  # just one
```

This pulls from [`sachithabey/ConceptTime`](https://huggingface.co/datasets/sachithabey/ConceptTime)
into `dataset/<D>/test.pt` and `checkpoints/<D>/{gaussian_entropy_best,pipeline_best}.pt`
— the exact paths the eval command expects.

**Full archive.** The same Hub repo also holds **every** dataset (all 17,
including FD-A/B/C/D with full train/val/test) and **every** trained checkpoint
under `archive/datasets/` and `archive/checkpoints/`, for anyone who wants to
retrain or explore beyond the three headline datasets.

To use your own data, drop `train.pt` / `val.pt` / `test.pt` into
`dataset/<Name>/` in the format above and add an entry to `DATASET_CONFIGS` in
[`src/train_gaussian_entropy_model.py`](src/train_gaussian_entropy_model.py).

---

## Quick start — evaluate pretrained checkpoints (no training)

Run all commands from the repository root. Epilepsy data ships with the repo;
for **HAR** and **Sleep-EDF** run `python download_data.py` first (see Datasets).

**Epilepsy**
```bash
python src/evaluate.py \
    --dataset  Epilepsy \
    --artifact checkpoints/Epilepsy/pipeline_best.pt \
    --density  checkpoints/Epilepsy/gaussian_entropy_best.pt \
    --split    test
```

**UCI-HAR**
```bash
python src/evaluate.py \
    --dataset  HAR \
    --artifact checkpoints/HAR/pipeline_best.pt \
    --density  checkpoints/HAR/gaussian_entropy_best.pt \
    --split    test
```

**Sleep-EDF** (run `python download_data.py --datasets SLeep-EDF` first — see Datasets)
```bash
python src/evaluate.py \
    --dataset  SLeep-EDF \
    --artifact checkpoints/SLeep-EDF/pipeline_best.pt \
    --density  checkpoints/SLeep-EDF/gaussian_entropy_best.pt \
    --split    test
```

`evaluate.py` prints test **accuracy** and **macro-F1** for each label-budget
classifier stored in the checkpoint (1 %, 5 %, and full supervision).
Signatures are recomputed on the fly, so no signature cache is needed.

---

## Reproduce the Epilepsy pipeline from scratch (training → evaluation)

Two stages. Run from the repository root.

**Stage 1 — pretrain the density (world) model**
```bash
python src/train_gaussian_entropy_model.py --dataset Epilepsy --no_wandb
# → output/Epilepsy/gaussian_entropy_best.pt
```

**Stage 2 — train the full concept pipeline**
```bash
python src/train_pipeline.py \
    --dataset      Epilepsy \
    --M            32 \
    --boundary_mode surprise \
    --sig_mode     surprise_full \
    --cluster_algo gmm \
    --no_wandb
# → output/Epilepsy/pipeline/.../pipeline_*.pt
```

**Evaluate what you just trained**
```bash
python src/evaluate.py \
    --dataset  Epilepsy \
    --artifact output/Epilepsy/pipeline/<the file written above>.pt \
    --density  output/Epilepsy/gaussian_entropy_best.pt \
    --split    test
```

> Newly trained artifacts are written under `output/`, so they never overwrite
> the released `checkpoints/`. To retrain HAR or Sleep-EDF, add their full
> `train.pt` / `val.pt` splits under `dataset/<Name>/` and rerun the two stages
> with `--dataset HAR` / `--dataset SLeep-EDF`.

### Key `train_pipeline.py` arguments

| Argument | Default | Meaning |
|----------|---------|---------|
| `--dataset` | `HAR` | dataset name (must be in `DATASET_CONFIGS`) |
| `--M` | per-dataset | concept vocabulary size (# prototypes) |
| `--boundary_mode` | `surprise` | patch-boundary signal: `surprise`, `entropy`, `kl_shift`, `residual` |
| `--sig_mode` | `surprise_full` | patch signature features (see below) |
| `--cluster_algo` | `gmm` | `gmm` or `kmeans` |
| `--seed` | `42` | random seed |
| `--no_wandb` | off | disable Weights & Biases logging |
| `--no_gate` | off | skip the training-time quality gates |

Signature modes (`--sig_mode`): `surprise_full` (default), `full`,
`entropy_only`, `moments_only`, `distributional_only`, `morphological_only`,
`residual_morphology`, `surprise_trajectory`.

---

## Method at a glance

1. **Density model** (`GaussianEntropyModel.py`) — a causal transformer trained
   to predict each next timestep as a Gaussian `(μ, σ²)`.
2. **Patcher** (`patcher.py`) — segments each series into variable-length
   patches at points of high predictive surprise / entropy shift.
3. **Signatures** (`signatures.py`) — a fixed-length descriptor per patch built
   from predictive surprise, residual morphology, and moment statistics.
4. **Concept space** (`concept_space.py`) — signatures are clustered (GMM/k-means)
   into `M` interpretable prototypes; each patch gets a soft concept assignment.
5. **Classifier** (`classifier.py`) — a small transformer over the length-`K`
   concept sequence predicts the label.

---

## Citation

Coming soon!

Released under the Apache License 2.0 (see [LICENSE](LICENSE)).
