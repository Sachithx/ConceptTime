# ConceptTime

This repository contains the code to reproduce the Epilepsy classification results reported in the paper.

---

## Requirements

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended. The pipeline also runs on CPU but is slower for signature extraction.

---

## Dataset

The Epilepsy dataset should be placed at `dataset/Epilepsy/` with three files:

```
dataset/Epilepsy/
  train.pt
  val.pt
  test.pt
```

Each `.pt` file is a dict with keys `"samples"` (tensor `[N, C, T]` or `[N, T, C]`) and `"labels"` (tensor `[N]`).  
Epilepsy: C=1, T=177, 2 classes (seizure vs. non-seizure).

---

## Reproducing results in three steps

### Step 1 — Train the density model

```bash
python train_gaussian_entropy_model.py --dataset Epilepsy
```

Trains GaussianGPT on the Epilepsy training split and saves the checkpoint to:

```
output/Epilepsy/gaussian_entropy_best.pt
```

Training runs for up to 50 epochs with early stopping (patience=10). Typical runtime: ~5 minutes on a single GPU.

---

### Step 2 — Train the pipeline

```bash
python train_pipeline.py --dataset Epilepsy
```

This executes the full ConceptTime pipeline in sequence:

1. Loads the frozen density model from Step 1
2. Computes the training marginal distribution
3. Segments each signal into K=8 entropy-guided patches
4. Extracts `surprise_full` signatures per patch (moments + residual morphology + surprise trajectory)
5. Fits a GMM concept space with M=8 prototypes
6. Builds soft concept sequences for all splits
7. Trains a transformer classifier at 1%, 5%, and 100% label fractions
8. Reports test accuracy and macro-F1 at each fraction

Signatures are cached in `output/Epilepsy/sig_cache/` after the first run, so re-running with different classifiers is fast.

The pipeline artifact is saved to:

```
output/Epilepsy/pipeline/entropy/pipeline_M8_surprise_surprise_full_gmm_transformer.pt
```

Key arguments (all have sensible defaults):

| Argument | Default | Description |
|---|---|---|
| `--M` | `8` | Number of concept prototypes |
| `--boundary_mode` | `surprise` | Patch boundary signal |
| `--sig_mode` | `surprise_full` | Signature feature set |
| `--cluster_algo` | `gmm` | Concept clustering algorithm |
| `--classifier_type` | `transformer` | Classifier head (`transformer` / `linear` / `bigram`) |
| `--cls_epochs` | `200` | Classifier training epochs |
| `--seed` | `42` | Random seed |

---

### Step 3 — Evaluate

```bash
python evaluate.py \
  --dataset Epilepsy \
  --artifact output/Epilepsy/pipeline/entropy/pipeline_M8_surprise_surprise_full_gmm_transformer.pt
```

Produces accuracy, macro-F1, confusion matrix, concept-assignment entropy, and per-concept visualizations.

---

## Expected results (Epilepsy)

| Label fraction | Test accuracy  
|---|---| 
| 1% | 98% |  
| 5% | 98% |  
| 100% | 98% |  

*(Fill in after running to confirm reproducibility.)*

---

## File overview

| File | Role |
|---|---|
| `train_gaussian_entropy_model.py` | Step 1: train GaussianGPT density model |
| `GaussianEntropyModel.py` | GaussianGPT architecture (Transformer / Mamba backbone) |
| `train_pipeline.py` | Step 2: full ConceptTime pipeline |
| `patcher.py` | Entropy-guided patch segmentation |
| `signatures.py` | Per-patch signature extraction |
| `concept_space.py` | GMM concept space and soft assignment |
| `classifier.py` | Classifier heads (transformer, sparse linear, bigram) |
| `evaluate.py` | Step 3: evaluation and visualization |
