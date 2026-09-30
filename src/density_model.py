"""
GaussianDensityModel — a causal Gaussian density (world) model for time series.

At each timestep it predicts the parameters (mean μ and log-variance) of a
Gaussian over the next continuous value, per channel:

    p(x_t | x_{<t}) = ∏_c N(μ^(c)_t, σ²^(c)_t),   H(t) = 0.5·(1 + log 2πσ²_t)

The predictive entropy / surprise of this model is what drives ConceptTime's
patch segmentation and per-patch signatures.

The transformer blocks (LayerNorm / CausalSelfAttention / MLP / Block) are
adapted from nanoGPT (https://github.com/karpathy/nanoGPT).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ============================================================================
# GaussianDensityModel — continuous predictive density, no tokenizer required
# ============================================================================
#
# Architecture vs a discrete GPT LM:
#   - token embedding      -> input_proj:  Linear(n_channels, n_embd)
#   - vocab-logit lm_head  -> mean_head:   Linear(n_embd, n_channels)   (μ)
#                             logvar_head: Linear(n_embd, n_channels)   (log σ²)
#   - Loss: Gaussian NLL = 0.5·(log σ² + (y − μ)²/σ²) + calibration term.
#   - Attention / MLP / LayerNorm / positional embedding: identical to a GPT.

@dataclass
class GaussianDensityConfig:
    block_size:   int   = 127
    n_channels:   int   = 1       # number of input/output sensor channels
    n_layer:      int   = 4
    n_head:       int   = 8
    n_embd:       int   = 128
    dropout:      float = 0.05
    bias:         bool  = False
    logvar_min:   float = -10.0
    logvar_max:   float =   4.0
    calib_weight: float = 0.01


class GaussianDensityModel(nn.Module):
    """
    Causal GPT that jointly predicts μ_{t+1} and log σ²_{t+1} for all C channels.

    Input:  [B, T, C]  dataset-normalised continuous values
    Output: (mu [B, T, C], log_var [B, T, C], loss scalar or None)

    The predictive distribution is a factorised diagonal Gaussian:
        p(x_t | x_{<t}) = ∏_c N(μ^(c)_t, σ²^(c)_t)
    Cross-channel temporal dependencies are captured through joint attention over
    the C-dim input embedding at each timestep.

    Normalization contract: callers must supply dataset-level channel stats
    (computed once on the training set) rather than per-sample normalization,
    so that absolute amplitude information is preserved across samples.
    """

    def __init__(self, config: GaussianDensityConfig):
        super().__init__()
        self.config = config
        C = config.n_channels

        # Input embedding + positional embedding + causal Block stack + final LN.
        self.transformer = nn.ModuleDict(dict(
            input_proj = nn.Linear(C, config.n_embd, bias=config.bias),
            drop       = nn.Dropout(config.dropout),
            wpe        = nn.Embedding(config.block_size, config.n_embd),
            h          = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f       = LayerNorm(config.n_embd, bias=config.bias),
        ))

        # ── Output heads ──────────────────────────────────────────────────────
        self.mean_head   = nn.Linear(config.n_embd, C, bias=True)
        self.logvar_head = nn.Linear(config.n_embd, C, bias=True)

        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        # Init logvar near 0.5: with normalise-then-mix, input std ≈ √2, so
        # log(σ²) ≈ log(2) ≈ 0.69 — starting at 0.5 keeps early training stable.
        nn.init.constant_(self.logvar_head.bias, 0.5)

        total = sum(p.numel() for p in self.parameters())
        print(f"GaussianDensityModel[transformer(n_head={config.n_head})]: {total:,} params  "
              f"(n_layer={config.n_layer} n_embd={config.n_embd} n_channels={C})")

    def _run_transformer(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C] → hidden states [B, T, n_embd]"""
        B, T, C = x.shape
        assert T <= self.config.block_size, \
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        h = self.transformer.input_proj(x)                  # [B, T, n_embd]
        h = h + self.transformer.wpe(torch.arange(T, device=x.device))
        h = self.transformer.drop(h)
        for block in self.transformer.h:
            h = block(h)
        return self.transformer.ln_f(h)                     # [B, T, n_embd]

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        """
        x:       [B, T, C]  dataset-normalised input
        targets: [B, T, C]  dataset-normalised targets (x shifted by 1 step)
        Returns: (mu [B,T,C], log_var [B,T,C], loss or None)

        Loss = Gaussian NLL + calibration regularization.
        Calibration term forces E[(y-μ)²/σ²] → 1 so that σ² is empirically
        meaningful and entropy values are trustworthy for downstream patching.
        """
        h       = self._run_transformer(x)
        mu      = self.mean_head(h)                                 # [B, T, C]
        log_var = self.logvar_head(h).clamp(
            self.config.logvar_min, self.config.logvar_max
        )                                                           # [B, T, C]

        loss = None
        if targets is not None:
            inv_var  = torch.exp(-log_var)
            residual = targets - mu
            nll      = 0.5 * (log_var + residual.pow(2) * inv_var) # [B, T, C]

            # Calibration: normalized squared residuals should have mean ≈ 1
            calib = (residual.detach().pow(2) * inv_var).mean() - 1.0
            loss  = nll.mean() + self.config.calib_weight * calib.pow(2)

        return mu, log_var, loss

    def forward_with_hidden(self, x: torch.Tensor):
        """
        Like forward() but also returns the final transformer hidden states.

        x:       [B, T, C]  dataset-normalised input
        Returns: (mu [B,T,C], log_var [B,T,C], h [B,T,n_embd])

        Differentiable w.r.t. x and all model parameters (unlike entropy()).
        """
        h       = self._run_transformer(x)
        mu      = self.mean_head(h)
        log_var = self.logvar_head(h).clamp(
            self.config.logvar_min, self.config.logvar_max
        )
        return mu, log_var, h

    @torch.no_grad()
    def entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-timestep predictive entropy, averaged across channels.

        x:       [T, C]  or  [B, T, C]  dataset-normalised values
        Returns: [T]     or  [B, T]     channel-averaged Gaussian entropy
                 H(t) = (1/C) Σ_c 0.5*(log(2πe) + log σ²_{t,c})

        High entropy → high predicted variance → uncertain / transitioning region.
        Note: entropy at the first ~10 timesteps is unreliable due to short context.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)              # [1, T, C]
        _, log_var, _ = self.forward(x)     # [B, T, C]
        LOG2PIE = math.log(2 * math.pi * math.e)
        H = 0.5 * (LOG2PIE + log_var)      # [B, T, C]
        return H.mean(dim=-1).squeeze(0)   # [B, T] or [T]

    @torch.no_grad()
    def channel_nll(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Per-channel mean NLL for diagnostics and logging.

        x, targets: [B, T, C]
        Returns:    [C]  per-channel mean Gaussian NLL
        """
        mu, log_var, _ = self.forward(x, targets)
        inv_var = torch.exp(-log_var)
        nll = 0.5 * (log_var + (targets - mu).pow(2) * inv_var)
        return nll.mean(dim=(0, 1))         # [C]
