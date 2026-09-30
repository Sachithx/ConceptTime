"""
concept_encoder.py — Phase 6: Concept encoder q_psi.

Maps raw patches [L, C] → concept distribution [M].
Trained to align with signature-derived targets (distillation),
be sparse (peaked assignments), and be stable under perturbation.

Training loss:
  L_total = L_task + λ_align * L_align + λ_sparse * L_sparse + λ_stable * L_stable

  L_align  = KL(q_ψ(P) || π(P))          distil from signature target
  L_sparse = H(q_ψ(P))                    entropy penalty → peaked assignments
  L_stable = KL(q_ψ(P) || q_ψ(aug(P)))  perturbation invariance
  L_task   = cross-entropy on labels      only this term uses labels
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# ── Positional embedding ──────────────────────────────────────────────────────

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        pos = torch.arange(L, device=x.device)
        return x + self.emb(pos)


# ── Attention pooling ─────────────────────────────────────────────────────────

class AttentionPool(nn.Module):
    """Pools variable-length sequence → single vector via learned query."""

    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model) * 0.02)
        self.scale = d_model ** -0.5

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:    [B, L, D]
        mask: [B, L]  True = valid position
        Returns: [B, D]
        """
        q = self.q.unsqueeze(0).unsqueeze(0)    # [1, 1, D]
        scores = (x * q).sum(-1) * self.scale   # [B, L]
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # [B, L]
        return (weights.unsqueeze(-1) * x).sum(1)  # [B, D]


# ── Transformer block (shared with density model but standalone) ──────────────

class ConceptBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout,
                                           batch_first=True)
        self.mlp  = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # non-causal self-attention (patch encoder sees full patch)
        xn = self.ln1(x)
        attn_out, _ = self.attn(xn, xn, xn, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


# ── Concept Encoder ───────────────────────────────────────────────────────────

class ConceptEncoder(nn.Module):
    """
    q_psi: [L, C] patch → [M] concept distribution.

    Architecture:
      Linear(C, d_model) → positional emb → N transformer blocks (non-causal)
      → attention pool → Linear(d_model, M) → softmax
    """

    def __init__(self, n_channels: int, M: int,
                 d_model: int = 64, n_layers: int = 2, n_head: int = 4,
                 dropout: float = 0.1, max_patch_len: int = 256):
        super().__init__()
        self.M = M
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_emb    = LearnedPositionalEmbedding(max_patch_len, d_model)
        self.blocks     = nn.ModuleList([
            ConceptBlock(d_model, n_head, dropout) for _ in range(n_layers)
        ])
        self.ln_f       = nn.LayerNorm(d_model)
        self.pool       = AttentionPool(d_model)
        self.head       = nn.Linear(d_model, M)

        total = sum(p.numel() for p in self.parameters())
        print(f"ConceptEncoder: {total:,} params  "
              f"(C={n_channels} M={M} d={d_model} L={n_layers})")

    def forward(self, patches: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        patches: [B, L, C]  (padded to max L in batch)
        mask:    [B, L]     True = valid position
        Returns: [B, M]  soft concept distribution (sums to 1)
        """
        x = self.input_proj(patches)    # [B, L, d]
        x = self.pos_emb(x)

        kpm = ~mask if mask is not None else None   # nn.MHA expects True = ignore
        for block in self.blocks:
            x = block(x, key_padding_mask=kpm)

        x = self.ln_f(x)
        x = self.pool(x, mask)    # [B, d]
        logits = self.head(x)     # [B, M]
        return F.softmax(logits, dim=-1)


# ── Data augmentations for stability loss ─────────────────────────────────────

def augment_patch(patch: torch.Tensor, noise_std: float = 0.05,
                  shift_max: int = 2, scale_range: Tuple = (0.95, 1.05)) -> torch.Tensor:
    """
    patch: [L, C] single patch.
    Returns augmented version (in-place safe, returns new tensor).
    """
    L, C = patch.shape
    sig_std = patch.std().clamp(min=1e-6)

    # Additive Gaussian noise
    noise = torch.randn_like(patch) * noise_std * sig_std
    aug   = patch + noise

    # Amplitude scale
    scale = torch.empty(1).uniform_(*scale_range).item()
    aug   = aug * scale

    # Time shift (circular within patch)
    shift = torch.randint(-shift_max, shift_max + 1, (1,)).item()
    if shift != 0:
        aug = torch.roll(aug, shift, dims=0)

    return aug


def augment_batch_patches(patches: torch.Tensor, mask: torch.Tensor,
                           **aug_kwargs) -> torch.Tensor:
    """patches: [B, L, C] padded batch. Augments each valid subsequence."""
    aug = patches.clone()
    B = patches.shape[0]
    for i in range(B):
        valid_len = mask[i].sum().item() if mask is not None else patches.shape[1]
        aug[i, :valid_len] = augment_patch(patches[i, :valid_len], **aug_kwargs)
    return aug


# ── Loss functions ────────────────────────────────────────────────────────────

def align_loss(q: torch.Tensor, pi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL(q || pi) — encoder distribution vs signature target."""
    return (q * (q.clamp(min=eps).log() - pi.clamp(min=eps).log())).sum(-1).mean()


def sparse_loss(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Entropy of q — penalize flat distributions."""
    return -(q * q.clamp(min=eps).log()).sum(-1).mean()


def stable_loss(q: torch.Tensor, q_aug: torch.Tensor,
                eps: float = 1e-8) -> torch.Tensor:
    """KL(q || q_aug) — encourage augmentation invariance."""
    return (q * (q.clamp(min=eps).log() - q_aug.clamp(min=eps).log())).sum(-1).mean()


def task_loss(concept_seq: torch.Tensor, labels: torch.Tensor,
              classifier) -> torch.Tensor:
    """cross-entropy via classifier head."""
    logits = classifier(concept_seq)   # [B, n_classes]
    return F.cross_entropy(logits, labels)


# ── Patch collation helper ────────────────────────────────────────────────────

def collate_patches(patch_list: List[torch.Tensor],
                    max_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    patch_list: list of [L_i, C] tensors with variable L.
    Returns:
      padded: [N, L_max, C]
      mask:   [N, L_max]  True = valid
    """
    max_L = max(p.shape[0] for p in patch_list) if max_len is None else max_len
    C     = patch_list[0].shape[1]
    N     = len(patch_list)

    padded = torch.zeros(N, max_L, C)
    mask   = torch.zeros(N, max_L, dtype=torch.bool)
    for i, p in enumerate(patch_list):
        L = min(p.shape[0], max_L)
        padded[i, :L] = p[:L]
        mask[i,   :L] = True

    return padded, mask


# ── Training dataset for encoder ─────────────────────────────────────────────

class PatchDataset(torch.utils.data.Dataset):
    """
    Holds pre-extracted patches + signature-derived soft targets.
    Each item: (patch_tensor [L, C], pi_target [M], label int)
    """

    def __init__(self, patches: List[torch.Tensor],
                 pi_targets: torch.Tensor,
                 labels: torch.Tensor):
        assert len(patches) == len(pi_targets) == len(labels)
        self.patches    = patches
        self.pi_targets = pi_targets
        self.labels     = labels

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], self.pi_targets[idx], self.labels[idx]


def patch_collate_fn(batch):
    patches, pi_targets, labels = zip(*batch)
    padded, mask = collate_patches(list(patches))
    pi   = torch.stack(pi_targets)
    lbls = torch.stack(labels)
    return padded, mask, pi, lbls


# ── Build patch dataset from pre-extracted signatures ─────────────────────────

@torch.no_grad()
def build_patch_dataset(density_model, channel_mixer,
                         source_dataset, patch_fn,
                         concept_space,
                         channel_mean, channel_std,
                         device, temperature: float = 1.0,
                         sig_mode: str = "full",
                         mu_marg=None, sigma2_marg=None):
    """
    Iterates the source_dataset, extracts patches, computes signature-derived
    soft targets, and returns a PatchDataset ready for ConceptEncoder training.
    """
    from signatures import extract_signatures_for_dataset

    all_sigs, all_patch_ranges, sample_ids = extract_signatures_for_dataset(
        density_model, channel_mixer, source_dataset, patch_fn,
        channel_mean, channel_std, mu_marg, sigma2_marg, device, mode=sig_mode,
    )

    # soft concept targets from concept space
    pi_targets = concept_space.soft_assign(all_sigs, temperature=temperature)

    # extract raw patch tensors + labels — batched by sample length to minimise
    # device transfers and channel_mixer kernel launches
    patches, labels_list = [], []

    from itertools import groupby as _groupby
    lengths = [source_dataset.samples[i].shape[0] for i in range(len(source_dataset.samples))]
    order   = sorted(range(len(lengths)), key=lambda i: lengths[i])

    for _, grp in _groupby(order, key=lambda i: lengths[i]):
        indices = list(grp)
        # single H2D transfer + single channel_mixer call per unique length
        raw_batch = torch.stack([source_dataset.samples[i] for i in indices]).to(device)  # [B, T, C]
        xn_batch  = (raw_batch - channel_mean) / channel_std
        xn_batch  = channel_mixer(xn_batch.permute(0, 2, 1)).permute(0, 2, 1)            # [B, T, C']

        for b, sample_i in enumerate(indices):
            xn = xn_batch[b]
            label = source_dataset.labels[sample_i]
            for (t1, t2) in all_patch_ranges[sample_i]:
                patches.append(xn[t1:t2].cpu())
                labels_list.append(label)

    labels_tensor = torch.stack([
        l if isinstance(l, torch.Tensor) else torch.tensor(l) for l in labels_list
    ]).long()
    return PatchDataset(patches, pi_targets.cpu(), labels_tensor)
