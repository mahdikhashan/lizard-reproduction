import math

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def hedgehog(x, weight):
    xw = x @ weight.T
    return torch.cat([xw.softmax(-1), (-xw).softmax(-1)], dim=-1)


def window_mask(length, window, device=None):
    i = torch.arange(length, device=device)[:, None]
    t = torch.arange(length, device=device)[None, :]
    return (t <= i) & (i - t < window)


def gate_products(gamma):
    causal = window_mask(gamma.shape[-1], gamma.shape[-1], gamma.device)
    g = torch.where(causal, gamma[..., None, :], 1.0)
    products = g.flip(-1).cumprod(-1).flip(-1)
    ones = torch.ones_like(products[..., :1])
    return torch.cat([products[..., 1:], ones], dim=-1) * causal


def gla(fq, fk, v, gamma):
    w = (fq @ fk.transpose(-1, -2)) * gate_products(gamma)[:, None]
    tiny = torch.finfo(w.dtype).tiny
    return (w @ v) / w.sum(-1, keepdim=True).clamp_min(tiny)


def sink_softmax(scores, meta):
    m = torch.maximum(scores.amax(-1, keepdim=True), meta.max()).detach()
    e = (scores - m).exp()
    sinks = (meta - m).exp().sum(-1, keepdim=True)
    return e / (sinks + e.sum(-1, keepdim=True))


def awa(q, k, v, meta, window):
    mask = window_mask(q.shape[-2], window, q.device)
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    return sink_softmax(scores.masked_fill(~mask, float("-inf")), meta) @ v


class LizardAttention(nn.Module):
    def __init__(self, config, layer_idx, window=128, num_meta=4,
                 feature_dim=128):
        super().__init__()
        hidden = config.hidden_size
        self.layer_idx = layer_idx
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = hidden // self.heads
        self.window = window
        kv_dim = self.kv_heads * self.head_dim
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.phi_q = nn.Linear(self.head_dim, feature_dim, bias=False)
        self.phi_k = nn.Linear(self.head_dim, feature_dim, bias=False)
        self.W_gamma = nn.Linear(hidden, 1, bias=False)
        self.meta_tokens = nn.Parameter(torch.zeros(num_meta))
        self.alpha_blend = nn.Parameter(torch.ones(()))
        nn.init.normal_(self.phi_q.weight, std=0.02)
        nn.init.normal_(self.phi_k.weight, std=0.02)
        nn.init.zeros_(self.W_gamma.weight)

    def split(self, x, heads):
        b, n, _ = x.shape
        x = x.view(b, n, heads, self.head_dim).transpose(1, 2)
        return x.repeat_interleave(self.heads // heads, dim=1)

    def forward(self, hidden_states, position_embeddings, **kwargs):
        x = hidden_states
        dtype = torch.promote_types(x.dtype, torch.float32)
        q = self.split(self.q_proj(x), self.heads).to(dtype)
        k = self.split(self.k_proj(x), self.kv_heads).to(dtype)
        v = self.split(self.v_proj(x), self.kv_heads).to(dtype)
        gamma = torch.sigmoid(self.W_gamma(x).to(dtype)).squeeze(-1)
        fq = hedgehog(q, self.phi_q.weight.to(dtype))
        fk = hedgehog(k, self.phi_k.weight.to(dtype))
        cos, sin = (t.to(dtype) for t in position_embeddings)
        qr, kr = apply_rotary_pos_emb(q, k, cos, sin)
        y = gla(fq, fk, v, gamma) + self.alpha_blend.to(dtype) * awa(
            qr, kr, v, self.meta_tokens.to(dtype), self.window)
        return self.o_proj(y.to(x.dtype).transpose(1, 2).flatten(2)), None


def from_llama(attention, config, **options):
    weight = attention.q_proj.weight
    layer = LizardAttention(config, attention.layer_idx, **options)
    layer = layer.to(weight.device, weight.dtype)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        getattr(layer, name).load_state_dict(
            getattr(attention, name).state_dict())
    return layer
