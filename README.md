# PRECEPT-

PRECEPT: Predictive Distribution Concepts for Interpretable Time Series. A concept bottleneck framework that defines patch-level concepts from predictive distributions of a probabilistic world model.

'''
train_gaussian_entropy_model.py --dataset HAR   # Phase 2
entropy_diagnostics.py --dataset HAR            # Phase 2 gates
train_pipeline.py --dataset HAR --M 32          # Phases 3–7
pipeline_diagnostics.py --artifact ...          # Phases 5–6 gates
evaluate.py --artifact ...                      # Phase 8
ablations.py --dataset HAR                      # Phase 9
'''