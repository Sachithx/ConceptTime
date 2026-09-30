"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
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

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        # print("number of parameters: %.2fk" % (self.get_num_params()/1e3,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x) # note: using list [-1] to preserve the time dim >>>>  self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ============================================================================
# Mamba (S6) backbone — pure PyTorch, no mamba_ssm dependency
# ============================================================================

def _parallel_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Parallel prefix scan for h_t = a_t * h_{t-1} + b_t.

    Hillis-Steele algorithm: O(T log T) work, log(T) iterations — fully
    out-of-place so autograd can differentiate through it safely.
    a, b : [B, T, d_inner, N]
    """
    T = a.shape[1]
    level = 1
    while level < T:
        new_b = a[:, level:] * b[:, :T - level] + b[:, level:]
        new_a = a[:, level:] * a[:, :T - level]
        b = torch.cat([b[:, :level], new_b], dim=1)
        a = torch.cat([a[:, :level], new_a], dim=1)
        level *= 2
    return b


class MambaBlock(nn.Module):
    """
    Selective state-space (S6 / Mamba) block in pure PyTorch.

    Implements the Mamba mixer described in Gu & Dao, arXiv:2312.00752,
    without the custom CUDA selective-scan kernels.  The scan is replaced
    by _parallel_scan, which is O(T log T) and fully vectorised.

    Same pre-norm residual contract as TransformerBackbone.Block:
        h_out = h_in + MambaBlock(LayerNorm(h_in))

    Args:
        d_model  : model dimension (= n_embd)
        d_state  : SSM state dimension N
        d_conv   : depthwise-conv kernel width (causal padding applied)
        expand   : inner-dim expansion factor  (d_inner = expand * d_model)
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        di = int(expand * d_model)
        self.d_inner = di

        # Input: split into SSM branch and gate branch
        self.in_proj  = nn.Linear(d_model, 2 * di, bias=False)
        # Causal depthwise conv for local context (padding trimmed in forward)
        self.conv1d   = nn.Conv1d(di, di, kernel_size=d_conv,
                                  padding=d_conv - 1, groups=di, bias=True)
        # Input-dependent SSM parameters: B (N), C (N), dt (1)
        self.x_proj   = nn.Linear(di, 2 * d_state + 1, bias=False)
        # dt: scalar per timestep → expand to d_inner
        self.dt_proj  = nn.Linear(1, di, bias=True)
        # A: stable negative-real diagonal, log-parameterised
        A_init = torch.arange(1, d_state + 1, dtype=torch.float
                              ).unsqueeze(0).expand(di, -1)   # [di, N]
        self.A_log    = nn.Parameter(torch.log(A_init))
        # D: skip connection (direct term)
        self.D        = nn.Parameter(torch.ones(di))
        # Output projection
        self.out_proj = nn.Linear(di, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, T, d_model]  →  [B, T, d_model]"""
        B, T, _ = x.shape
        N = self.d_state

        # ── Input split ──────────────────────────────────────────────────────
        xz        = self.in_proj(x)                    # [B, T, 2*d_inner]
        x_in, z   = xz.chunk(2, dim=-1)                # each [B, T, d_inner]

        # ── Causal depthwise conv ────────────────────────────────────────────
        x_conv = (self.conv1d(x_in.transpose(1, 2))    # [B, d_inner, T+pad]
                  [:, :, :T]                            # trim look-ahead
                  .transpose(1, 2))                    # [B, T, d_inner]
        x_conv = F.silu(x_conv)

        # ── Input-dependent SSM parameters ───────────────────────────────────
        params  = self.x_proj(x_conv)                  # [B, T, 2N+1]
        B_ssm   = params[..., :N]                      # [B, T, N]
        C_ssm   = params[..., N:2 * N]                 # [B, T, N]
        dt_raw  = params[..., -1:]                     # [B, T, 1]
        dt      = F.softplus(self.dt_proj(dt_raw))     # [B, T, d_inner]

        # ── Discretize (zero-order hold) ─────────────────────────────────────
        A  = -torch.exp(self.A_log.float())            # [d_inner, N]  (< 0 → stable)
        # dA[b,t,d,n] = exp(dt[b,t,d] * A[d,n])
        dA = torch.exp(
            dt.unsqueeze(-1) *                         # [B, T, d_inner, 1]
            A.unsqueeze(0).unsqueeze(0)                # [1,  1, d_inner, N]
        )                                              # [B, T, d_inner, N]
        # dBx[b,t,d,n] = dt[b,t,d] * B_ssm[b,t,n] * x_conv[b,t,d]
        dBx = (dt.unsqueeze(-1) *                      # [B, T, d_inner, 1]
               B_ssm.unsqueeze(2) *                    # [B, T,       1, N]
               x_conv.unsqueeze(-1))                   # [B, T, d_inner, 1]
        # → [B, T, d_inner, N]

        # ── Parallel scan: h[t] = dA[t]*h[t-1] + dBx[t] ───────────────────
        h = _parallel_scan(dA, dBx)                    # [B, T, d_inner, N]

        # ── Output: y_t = C_t·h_t + D*x_t  (then gate) ─────────────────────
        y = (h * C_ssm.unsqueeze(2)).sum(-1)           # [B, T, d_inner]
        y = y + self.D * x_conv                        # skip connection
        y = y * F.silu(z)                              # input-dependent gate
        return self.out_proj(y)                        # [B, T, d_model]


class MambaBackbone(nn.Module):
    """
    Stack of MambaBlocks with pre-norm residual connections.

    Drop-in replacement for TransformerBackbone: same input/output contract
    ([B, T, n_embd] → [B, T, n_embd]), no positional embeddings needed.
    """

    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model  = config.n_embd,
                       d_state  = config.d_state,
                       d_conv   = config.d_conv,
                       expand   = config.d_expand)
            for _ in range(config.n_layer)
        ])
        self.norms  = nn.ModuleList([
            nn.LayerNorm(config.n_embd) for _ in range(config.n_layer)
        ])
        self.ln_f   = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for layer, norm in zip(self.layers, self.norms):
            h = h + layer(norm(h))
        return self.ln_f(h)


class TransformerBackbone(nn.Module):
    """
    Stack of causal transformer Blocks with a final LayerNorm.

    Extracted from the original GaussianGPT so the backbone is swappable.
    Same input/output contract as MambaBackbone.
    """

    def __init__(self, config):
        super().__init__()
        self.h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for block in self.h:
            h = block(h)
        return self.ln_f(h)


# ============================================================================
# GaussianGPT — continuous predictive entropy, no tokenizer required
# ============================================================================
#
# Motivation (vs discrete GPT + tokenizer):
#   The discrete model computes H = -Σ p_k log p_k over 252 vocabulary bins.
#   This entropy depends on bin width and spacing — it's partly a binning
#   artefact.  GaussianGPT predicts a Gaussian distribution over the NEXT
#   CONTINUOUS VALUE, so entropy is the *provably correct* information-theoretic
#   uncertainty of a well-defined probabilistic model:
#
#       H(t) = 0.5 * (1 + log(2πσ²_t))
#
#   High σ² → model is uncertain → signal is in a complex/transitioning region.
#   Low  σ² → model is confident → stable, predictable regime.
#   No binning, no vocab size sensitivity.
#
# Architecture changes vs GPT:
#   - wte (token embed) replaced by input_proj: Linear(1, n_embd)
#   - lm_head (vocab_size logits) replaced by:
#       mean_head:    Linear(n_embd, 1)   — predicted next value
#       logvar_head:  Linear(n_embd, 1)   — log predicted variance (clamped)
#   - Loss: Gaussian NLL = 0.5 * (log_var + (y - mu)² / exp(log_var))
#   - Everything else (attention, MLP, LayerNorm, positional embed) identical.
#
# Training loop changes (minimal):
#   - Skip tokenizer.context_input_transform / label_input_transform
#   - Pass raw normalised float tensor instead of token_ids
#   - Use model.entropy_from_logvar() to get entropy signal during inference
#
# ============================================================================

@dataclass
class GaussianGPTConfig:
    block_size:   int   = 127
    n_channels:   int   = 1       # number of input/output sensor channels
    n_layer:      int   = 4
    n_head:       int   = 8       # transformer only (ignored for mamba)
    n_embd:       int   = 128
    dropout:      float = 0.05
    bias:         bool  = False
    logvar_min:   float = -10.0
    logvar_max:   float =   4.0
    calib_weight: float = 0.01
    # Backbone selection — old checkpoints default to "transformer"
    backbone:     str   = "transformer"   # "transformer" | "mamba"
    d_state:      int   = 16              # Mamba: SSM state dimension
    d_conv:       int   = 4               # Mamba: depthwise conv kernel width
    d_expand:     int   = 2               # Mamba: inner-dim expansion factor


class GaussianGPT(nn.Module):
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

    def __init__(self, config: GaussianGPTConfig):
        super().__init__()
        self.config = config
        C = config.n_channels

        if config.backbone not in ("transformer", "mamba"):
            raise ValueError(
                f"Unknown backbone '{config.backbone}'. Choose 'transformer' or 'mamba'."
            )

        # ── Shared input embedding ────────────────────────────────────────────
        # For transformer: wpe + Block stack + ln_f live inside self.transformer
        # so that old checkpoint keys (transformer.h.*, transformer.ln_f.*,
        # transformer.wpe.*) are preserved and load without any migration.
        # For mamba: only input_proj and drop live in self.transformer;
        # the SSM stack lives in self.mamba_backbone.
        td: dict = dict(
            input_proj = nn.Linear(C, config.n_embd, bias=config.bias),
            drop       = nn.Dropout(config.dropout),
        )
        if config.backbone == "transformer":
            td["wpe"]  = nn.Embedding(config.block_size, config.n_embd)
            td["h"]    = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
            td["ln_f"] = LayerNorm(config.n_embd, bias=config.bias)
        self.transformer = nn.ModuleDict(td)

        if config.backbone == "mamba":
            self.mamba_backbone = MambaBackbone(config)

        # ── Output heads (identical for both backbones) ───────────────────────
        self.mean_head   = nn.Linear(config.n_embd, C, bias=True)
        self.logvar_head = nn.Linear(config.n_embd, C, bias=True)

        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        # Init logvar near 0.5: with normalise-then-mix, input std ≈ √2, so
        # log(σ²) ≈ log(2) ≈ 0.69 — starting at 0.5 keeps early training stable.
        nn.init.constant_(self.logvar_head.bias, 0.5)

        total = sum(p.numel() for p in self.parameters())
        if config.backbone == "mamba":
            bkb = (f"mamba(d_state={config.d_state}, d_conv={config.d_conv},"
                   f" expand={config.d_expand})")
        else:
            bkb = f"transformer(n_head={config.n_head})"
        print(f"GaussianGPT[{bkb}]: {total:,} params  "
              f"(n_layer={config.n_layer} n_embd={config.n_embd} n_channels={C})")

    def _run_transformer(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C] → hidden states [B, T, n_embd] (works for both backbones)"""
        B, T, C = x.shape
        assert T <= self.config.block_size, \
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        h = self.transformer.input_proj(x)                  # [B, T, n_embd]
        if self.config.backbone == "transformer":
            # Positional embeddings are redundant for Mamba (its recurrence
            # already encodes position implicitly).
            h = h + self.transformer.wpe(torch.arange(T, device=x.device))
        h = self.transformer.drop(h)
        if self.config.backbone == "transformer":
            for block in self.transformer.h:
                h = block(h)
            return self.transformer.ln_f(h)                 # [B, T, n_embd]
        return self.mamba_backbone(h)                       # [B, T, n_embd]

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

        Use this when you need per-timestep representations for downstream
        tasks (e.g. the TrainableBoundaryDetector's channel aggregator).
        Unlike entropy(), this method is differentiable w.r.t. x and all
        model parameters.
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
        Per-channel mean NLL for diagnostics and wandb logging.

        x, targets: [B, T, C]
        Returns:    [C]  per-channel mean Gaussian NLL
        """
        mu, log_var, _ = self.forward(x, targets)
        inv_var = torch.exp(-log_var)
        nll = 0.5 * (log_var + (targets - mu).pow(2) * inv_var)
        return nll.mean(dim=(0, 1))         # [C]