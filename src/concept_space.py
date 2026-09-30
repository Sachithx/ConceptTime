"""
concept_space.py — Phase 5: Cluster standardized signatures into M concept prototypes.

Supports: KMeans (default), GMM, HDBSCAN.
Produces soft concept assignments π_m(P_k) via temperature-scaled distances.
"""

import math
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


# ── Clustering algorithms ─────────────────────────────────────────────────────

def fit_kmeans(sigs: np.ndarray, M: int, n_init: int = 10, max_iter: int = 300,
               seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Returns centroids [M, D], labels [N]."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=M, n_init=n_init, max_iter=max_iter,
                random_state=seed)
    labels = km.fit_predict(sigs)
    return km.cluster_centers_.astype(np.float32), labels


def fit_gmm(sigs: np.ndarray, M: int, seed: int = 42,
            covariance_type: str = "diag",
            reg_covar: float = 1e-4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns means [M,D], covariances [M,D] (diag), labels [N].

    Retries with increasing reg_covar if components collapse (common when data
    is near-degenerate or M is large relative to the intrinsic dimensionality).
    """
    from sklearn.mixture import GaussianMixture
    for reg in [reg_covar, 1e-3, 1e-2, 0.1]:
        try:
            gm = GaussianMixture(n_components=M, covariance_type=covariance_type,
                                 random_state=seed, max_iter=300, n_init=3,
                                 reg_covar=reg)
            gm.fit(sigs)
            labels = gm.predict(sigs)
            if reg > reg_covar:
                print(f"  [GMM] converged with reg_covar={reg:.0e}")
            return (gm.means_.astype(np.float32),
                    gm.covariances_.astype(np.float32),
                    labels)
        except ValueError:
            print(f"  [GMM] collapsed with reg_covar={reg:.0e}, retrying …")
    raise RuntimeError(
        f"GMM fitting failed for M={M} even with reg_covar=0.1. "
        "Consider reducing M or switching to --cluster_algo kmeans."
    )


# ── Concept prototype store ───────────────────────────────────────────────────

class ConceptSpace:
    """
    Stores M concept prototypes and supports soft assignment computation.

    centroids:       [M, D]  cluster centroids in standardized signature space
    covariances:     [M, D]  per-cluster diagonal covariance (optional, for GMM)
    prototype_idxs:  list of lists — top-50 training patch indices per cluster
    cluster_sizes:   [M]
    """

    def __init__(self, M: int, D: int):
        self.M = M
        self.D = D
        self.centroids:      Optional[torch.Tensor] = None
        self.covariances:    Optional[torch.Tensor] = None
        self.prototype_idxs: Optional[List[List[int]]] = None
        self.cluster_sizes:  Optional[torch.Tensor] = None
        self.algorithm: str = "kmeans"

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, sigs: torch.Tensor, algorithm: str = "kmeans",
            n_prototypes_per_cluster: int = 50, seed: int = 42, **kwargs):
        """
        sigs: [N, D] standardized training signatures.
        Fits clusters and stores prototypes.
        """
        self.algorithm = algorithm
        sigs_np = sigs.cpu().numpy().astype(np.float64)

        if algorithm == "kmeans":
            centroids_np, labels = fit_kmeans(sigs_np, self.M, seed=seed, **kwargs)
            self.centroids  = torch.from_numpy(centroids_np)
            self.covariances = None

        elif algorithm == "gmm":
            means_np, covs_np, labels = fit_gmm(sigs_np, self.M, seed=seed, **kwargs)
            self.centroids   = torch.from_numpy(means_np)
            self.covariances = torch.from_numpy(covs_np)

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        labels = np.array(labels)
        self.cluster_sizes = torch.zeros(self.M, dtype=torch.long)
        self.prototype_idxs = []

        for m in range(self.M):
            mask = np.where(labels == m)[0]
            self.cluster_sizes[m] = len(mask)
            if len(mask) == 0:
                self.prototype_idxs.append([])
                continue
            # find top-k nearest to centroid
            dists = np.linalg.norm(sigs_np[mask] - self.centroids[m].cpu().numpy(), axis=1)
            top_k = mask[np.argsort(dists)[:n_prototypes_per_cluster]]
            self.prototype_idxs.append(top_k.tolist())

        return labels

    # ── Distance computation ──────────────────────────────────────────────────

    def distances(self, sigs: torch.Tensor,
                  distance_type: str = "euclidean") -> torch.Tensor:
        """
        sigs: [N, D]
        Returns: [N, M] distances to each centroid.
        """
        centroids = self.centroids.to(sigs.device)  # [M, D]
        if distance_type == "euclidean":
            # ||sig - c||^2 for each (sig, centroid) pair
            diff = sigs.unsqueeze(1) - centroids.unsqueeze(0)  # [N, M, D]
            return diff.pow(2).sum(-1).sqrt()                   # [N, M]

        elif distance_type == "mahalanobis":
            if self.covariances is None:
                return self.distances(sigs, "euclidean")
            # per-cluster diagonal Mahalanobis
            diff = sigs.unsqueeze(1) - centroids.unsqueeze(0)   # [N, M, D]
            cov  = self.covariances.to(sigs.device).unsqueeze(0) + 1e-8  # [1, M, D]
            return (diff.pow(2) / cov).sum(-1).sqrt()            # [N, M]

        elif distance_type == "cosine":
            sigs_n = torch.nn.functional.normalize(sigs, dim=-1)
            cent_n = torch.nn.functional.normalize(centroids, dim=-1)
            return 1.0 - sigs_n @ cent_n.T                       # [N, M]

        else:
            raise ValueError(f"Unknown distance: {distance_type}")

    # ── Soft assignment ───────────────────────────────────────────────────────

    def soft_assign(self, sigs: torch.Tensor, temperature: float = 1.0,
                    distance_type: str = "euclidean") -> torch.Tensor:
        """
        Temperature-scaled soft concept distribution.
        π_m(P) = exp(-d(s,c_m)/τ) / Σ exp(-d(s,c_m')/τ)

        sigs: [N, D]
        Returns: [N, M] soft distributions (sum to 1 over M).
        """
        d = self.distances(sigs, distance_type)    # [N, M]
        logits = -d / max(temperature, 1e-6)
        return torch.softmax(logits, dim=-1)

    def hard_assign(self, sigs: torch.Tensor,
                    distance_type: str = "euclidean") -> torch.Tensor:
        """Returns [N] integer cluster assignments."""
        d = self.distances(sigs, distance_type)
        return d.argmin(dim=-1)

    # ── State ─────────────────────────────────────────────────────────────────

    def state_dict(self) -> Dict:
        return {
            "M":              self.M,
            "D":              self.D,
            "algorithm":      self.algorithm,
            "centroids":      self.centroids,
            "covariances":    self.covariances,
            "prototype_idxs": self.prototype_idxs,
            "cluster_sizes":  self.cluster_sizes,
        }

    def load_state_dict(self, d: Dict):
        self.M              = d["M"]
        self.D              = d["D"]
        self.algorithm      = d["algorithm"]
        self.centroids      = d["centroids"]
        self.covariances    = d["covariances"]
        self.prototype_idxs = d["prototype_idxs"]
        self.cluster_sizes  = d["cluster_sizes"]

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def cluster_balance_report(self) -> str:
        sizes = self.cluster_sizes.float()
        total = sizes.sum().item()
        lines = ["Cluster sizes:"]
        for m in range(self.M):
            pct = 100 * sizes[m].item() / max(total, 1)
            flag = " <-- small" if pct < 1.0 else ""
            lines.append(f"  C{m:03d}: {int(sizes[m]):5d}  ({pct:5.1f}%){flag}")
        return "\n".join(lines)

    def nmi_with_labels(self, sigs: torch.Tensor,
                         labels: torch.Tensor) -> float:
        from sklearn.metrics import normalized_mutual_info_score
        assignments = self.hard_assign(sigs).cpu().numpy()
        return float(normalized_mutual_info_score(labels.cpu().numpy(), assignments,
                                                   average_method="arithmetic"))

    def silhouette(self, sigs: torch.Tensor,
                   max_samples: int = 5000) -> float:
        from sklearn.metrics import silhouette_score
        assignments = self.hard_assign(sigs).cpu().numpy()
        n = min(len(sigs), max_samples)
        idx = np.random.choice(len(sigs), n, replace=False) if len(sigs) > n else np.arange(len(sigs))
        try:
            return float(silhouette_score(sigs[idx].cpu().numpy(), assignments[idx]))
        except Exception:
            return float("nan")


# ── M-sweep utility ───────────────────────────────────────────────────────────

def sweep_M(sigs_train: torch.Tensor,
            sigs_val: torch.Tensor,
            labels_val: torch.Tensor,
            M_values: List[int] = [8, 16, 32, 64, 128],
            algorithm: str = "kmeans") -> Dict:
    """
    Sweep over vocabulary sizes M. Returns dict with metrics per M.
    Use this to pick the best M before committing to Phase 6.
    """
    results = {}
    for M in M_values:
        cs = ConceptSpace(M, sigs_train.shape[1])
        cs.fit(sigs_train, algorithm=algorithm)
        nmi = cs.nmi_with_labels(sigs_val, labels_val)
        sil = cs.silhouette(sigs_val)
        sizes = cs.cluster_sizes.float()
        min_pct = (100 * sizes.min() / sizes.sum()).item()
        results[M] = {"nmi": nmi, "silhouette": sil, "min_cluster_pct": min_pct}
        print(f"  M={M:4d}  NMI={nmi:.3f}  Sil={sil:.3f}  min_cluster={min_pct:.1f}%")
    return results
