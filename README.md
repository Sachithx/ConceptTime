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

Data and checkpoints for HAR / Sleep-EDF are hosted on the HuggingFace Hub:
**https://huggingface.co/datasets/sachithabey/ConceptTime**

- **Epilepsy** — data (`dataset/Epilepsy/`) and checkpoints
  (`checkpoints/Epilepsy/`) ship in this repo; nothing to download.
- **UCI-HAR, Sleep-EDF** — download the test split + checkpoints from the Hub
  (needs `huggingface_hub`, already in `requirements.txt`):

  ```bash
  python download_data.py                    # HAR + Sleep-EDF
  python download_data.py --datasets HAR     # one only
  ```

  This pulls from [`sachithabey/ConceptTime`](https://huggingface.co/datasets/sachithabey/ConceptTime)
  and writes `dataset/<D>/test.pt` and
  `checkpoints/<D>/{density_model_best,pipeline_best}.pt` — the exact paths
  the eval commands below expect.

Each dataset uses **two** checkpoints: `density_model_best.pt` (the density
model) and `pipeline_best.pt` (concept space + classifier) — both are required
to evaluate.

Data format: every `.pt` is a dict
`{"samples": FloatTensor[N, C, T], "labels": LongTensor[N]}`.

> The same Hub repo also archives **all** datasets (including FD-A/B/C/D, with
> full train/val/test) and **all** trained checkpoints under `archive/`, for
> retraining or exploration beyond the three headline datasets.

## Evaluate pretrained checkpoints

Run from the repo root:

```bash
python src/evaluate.py --dataset Epilepsy \
    --artifact checkpoints/Epilepsy/pipeline_best.pt \
    --density  checkpoints/Epilepsy/density_model_best.pt --split test

python src/evaluate.py --dataset HAR \
    --artifact checkpoints/HAR/pipeline_best.pt \
    --density  checkpoints/HAR/density_model_best.pt --split test

python src/evaluate.py --dataset SLeep-EDF \
    --artifact checkpoints/SLeep-EDF/pipeline_best.pt \
    --density  checkpoints/SLeep-EDF/density_model_best.pt --split test
```

Each run prints accuracy and macro-F1 at 1% / 5% / full label budgets.

## Train Epilepsy from scratch

Two stages, from the repo root:

```bash
# 1) density model
python src/train_density_model.py --dataset Epilepsy --no_wandb
#    -> output/Epilepsy/density_model_best.pt

# 2) concept pipeline
python src/train_pipeline.py --dataset Epilepsy --M 32 \
    --boundary_mode surprise --sig_mode surprise_full --cluster_algo gmm --no_wandb
#    -> output/Epilepsy/pipeline/.../pipeline_*.pt
```

Then evaluate what you trained:

```bash
python src/evaluate.py --dataset Epilepsy \
    --artifact output/Epilepsy/pipeline/<file written above>.pt \
    --density  output/Epilepsy/density_model_best.pt --split test
```

Training writes under `output/`, so the released `checkpoints/` are left
untouched. To train HAR or Sleep-EDF, place their `train.pt` / `val.pt` under
`dataset/<D>/` and rerun both stages with `--dataset HAR` / `--dataset SLeep-EDF`.

## Citation

If you found this work useful for you, please consider citing it.

```bibtex
@inproceedings{sachith_concepttime_26,
  title={Predictive Surprise as Self-Grounding Concept Bottleneck for Interpretable Time Series},
  author={Abeywickrama, Sachith and Eldele, Emadeldeen and Wu, Min and Li, Xiaoli and Yuen, Chau},
  booktitle = {Advances in Neural Information Processing Systems},
  year={2026}
}
```

Released under the Apache License 2.0 (see [LICENSE](LICENSE)).
