# (NeurIPS'26) ConceptTime: Predictive Surprise as Self-Grounding Concept Bottleneck for Interpretable Time Series

Interpretable time-series classification with a predictive-surprise concept
bottleneck. Code and checkpoints to train the Epilepsy pipeline end-to-end and
to evaluate pretrained checkpoints on Epilepsy, UCI-HAR, and Sleep-EDF.

## Install

```bash
conda create -n concepttime python=3.10 -y && conda activate concepttime
pip install -r requirements.txt
```

If `import torch` raises `GLIBCXX_3.4.xx not found`, prefix commands with
`./run.sh` (preloads the conda env's libstdc++); otherwise ignore `run.sh`.

## Data & checkpoints

- **Epilepsy** — data (`dataset/Epilepsy/`) and checkpoints
  (`checkpoints/Epilepsy/`) ship in this repo.
- **UCI-HAR, Sleep-EDF** — download the test split + checkpoints from the
  HuggingFace Hub:

  ```bash
  python download_data.py                    # HAR + Sleep-EDF
  python download_data.py --datasets HAR     # one only
  ```

  This writes `dataset/<D>/test.pt` and
  `checkpoints/<D>/{gaussian_entropy_best,pipeline_best}.pt`.

Each dataset uses **two** checkpoints: `gaussian_entropy_best.pt` (the density
model) and `pipeline_best.pt` (concept space + classifier) — both are required
to evaluate.

Data format: every `.pt` is a dict
`{"samples": FloatTensor[N, C, T], "labels": LongTensor[N]}`.

## Evaluate pretrained checkpoints

Run from the repo root:

```bash
python src/evaluate.py --dataset Epilepsy \
    --artifact checkpoints/Epilepsy/pipeline_best.pt \
    --density  checkpoints/Epilepsy/gaussian_entropy_best.pt --split test

python src/evaluate.py --dataset HAR \
    --artifact checkpoints/HAR/pipeline_best.pt \
    --density  checkpoints/HAR/gaussian_entropy_best.pt --split test

python src/evaluate.py --dataset SLeep-EDF \
    --artifact checkpoints/SLeep-EDF/pipeline_best.pt \
    --density  checkpoints/SLeep-EDF/gaussian_entropy_best.pt --split test
```

Each run prints accuracy and macro-F1 at 1% / 5% / full label budgets.

## Train Epilepsy from scratch

Two stages, from the repo root:

```bash
# 1) density model
python src/train_gaussian_entropy_model.py --dataset Epilepsy --no_wandb
#    -> output/Epilepsy/gaussian_entropy_best.pt

# 2) concept pipeline
python src/train_pipeline.py --dataset Epilepsy --M 32 \
    --boundary_mode surprise --sig_mode surprise_full --cluster_algo gmm --no_wandb
#    -> output/Epilepsy/pipeline/.../pipeline_*.pt
```

Then evaluate what you trained:

```bash
python src/evaluate.py --dataset Epilepsy \
    --artifact output/Epilepsy/pipeline/<file written above>.pt \
    --density  output/Epilepsy/gaussian_entropy_best.pt --split test
```

Training writes under `output/`, so the released `checkpoints/` are left
untouched. To train HAR or Sleep-EDF, place their `train.pt` / `val.pt` under
`dataset/<D>/` and rerun both stages with `--dataset HAR` / `--dataset SLeep-EDF`.

## Citation

Coming soon!

Released under the Apache License 2.0 (see [LICENSE](LICENSE)).
