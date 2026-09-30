"""
classifier.py — temporal classifier heads over concept sequences.

Two heads (select via --classifier_type):
  TransformerClassifier  — small transformer over K concept activations
  SparseLinearClassifier — fully interpretable L1-regularized linear head

Both take: concept sequence [B, K, M] → logits [B, n_classes]
"""

import torch
import torch.nn as nn


# ── Shared positional embedding ────────────────────────────────────────────────

class PatchPositionalEmbedding(nn.Module):
    def __init__(self, K: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(K, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, _ = x.shape
        pos = torch.arange(K, device=x.device)
        return x + self.emb(pos)


# ── Transformer classifier ────────────────────────────────────────────────────

class TransformerClassifier(nn.Module):
    """
    Input:  concept sequence [B, K, M]
    Output: class logits     [B, n_classes]

    Architecture: project M→d, add positional emb, N transformer blocks,
    mean pool over K, linear to n_classes.
    """

    def __init__(self, M: int, K: int, n_classes: int,
                 d_model: int = 64, n_layers: int = 2, n_head: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.proj    = nn.Linear(M, d_model)
        self.pos_emb = PatchPositionalEmbedding(K, d_model)
        self.blocks  = nn.ModuleList([
            _ClassifierBlock(d_model, n_head, dropout) for _ in range(n_layers)
        ])
        self.ln_f    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, n_classes)

        total = sum(p.numel() for p in self.parameters())
        print(f"TransformerClassifier: {total:,} params  "
              f"(M={M} K={K} d={d_model} L={n_layers} cls={n_classes})")

    def forward(self, concept_seq: torch.Tensor) -> torch.Tensor:
        """concept_seq: [B, K, M]"""
        x = self.proj(concept_seq)    # [B, K, d]
        x = self.pos_emb(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x).mean(dim=1) # [B, d]  mean pool over patches
        return self.head(x)           # [B, n_classes]


class _ClassifierBlock(nn.Module):
    def __init__(self, d: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1  = nn.LayerNorm(d)
        self.ln2  = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_head, dropout=dropout, batch_first=True)
        self.mlp  = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(),
            nn.Linear(4 * d, d), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.ln1(x)
        attn_out, _ = self.attn(xn, xn, xn)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


# ── Sparse linear classifier ──────────────────────────────────────────────────

class SparseLinearClassifier(nn.Module):
    """
    Flatten concept activations [B, K, M] → [B, K*M],
    then a single linear layer with L1 regularization on weights.
    Fully interpretable: weight[k*M + m] = contribution of concept m at patch k.
    """

    def __init__(self, M: int, K: int, n_classes: int, l1_lambda: float = 1e-3):
        super().__init__()
        self.M          = M
        self.K          = K
        self.n_classes  = n_classes
        self.l1_lambda  = l1_lambda
        self.linear     = nn.Linear(K * M, n_classes)

        total = sum(p.numel() for p in self.parameters())
        print(f"SparseLinearClassifier: {total:,} params  "
              f"(K*M={K*M} → cls={n_classes}  λ_L1={l1_lambda})")

    def forward(self, concept_seq: torch.Tensor) -> torch.Tensor:
        """concept_seq: [B, K, M]"""
        flat = concept_seq.reshape(concept_seq.shape[0], -1)   # [B, K*M]
        return self.linear(flat)

    def l1_loss(self) -> torch.Tensor:
        return self.l1_lambda * self.linear.weight.abs().sum()

    def concept_importance(self) -> torch.Tensor:
        """
        Returns [K, M] matrix of concept importance across classes.
        Entry (k, m) = mean |weight| connecting concept m at patch k to all classes.
        """
        W = self.linear.weight.detach()   # [n_classes, K*M]
        W = W.abs().mean(0)               # [K*M]
        return W.reshape(self.K, self.M)
