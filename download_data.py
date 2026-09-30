"""
download_data.py — fetch the HAR / Sleep-EDF checkpoints and test splits from
the HuggingFace Hub into this repo.

Epilepsy is self-contained in the GitHub repo (data + checkpoints ship with it).
Everything for the other datasets — the test split *and* the trained
checkpoints (density model + pipeline) — lives on the Hub, because the data is
large (Sleep-EDF's test.pt alone is >100 MB, above GitHub's per-file limit).

Run this once before evaluating HAR or Sleep-EDF:

    pip install huggingface_hub
    python download_data.py                    # HAR + Sleep-EDF
    python download_data.py --datasets HAR     # just one

Files are written to the same paths the eval command expects
(dataset/<D>/test.pt and checkpoints/<D>/*.pt).

The full archive of *all* datasets (incl. FD) and *all* checkpoints is also on
the Hub under archive/ — see the repo card at
https://huggingface.co/datasets/sachithabey/ConceptTime
"""

import argparse
import os

# ── HuggingFace dataset repo ──────────────────────────────────────────────────
HF_REPO = "sachithabey/ConceptTime"
HF_REPO_TYPE = "dataset"

# For each dataset: the files to pull, at the same relative path used here.
FILES = {
    "HAR": [
        "dataset/HAR/test.pt",
        "checkpoints/HAR/gaussian_entropy_best.pt",
        "checkpoints/HAR/pipeline_best.pt",
    ],
    "SLeep-EDF": [
        "dataset/SLeep-EDF/test.pt",
        "checkpoints/SLeep-EDF/gaussian_entropy_best.pt",
        "checkpoints/SLeep-EDF/pipeline_best.pt",
    ],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=HF_REPO,
                    help="HuggingFace dataset repo id (default: %(default)s)")
    ap.add_argument("--datasets", nargs="+", default=list(FILES), choices=list(FILES),
                    help="which datasets to fetch (default: all non-Epilepsy)")
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("Please `pip install huggingface_hub` first.")

    for name in args.datasets:
        print(f"[{name}]")
        for rel in FILES[name]:
            os.makedirs(os.path.dirname(rel), exist_ok=True)
            hf_hub_download(repo_id=args.repo, repo_type=HF_REPO_TYPE,
                            filename=rel, local_dir=".")
            print(f"    -> {rel}")
    print("Done. You can now run src/evaluate.py on these datasets.")


if __name__ == "__main__":
    main()
