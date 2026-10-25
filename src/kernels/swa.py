import math

import torch
import triton
import triton.language as tl

from .dot import INTERPRET, block_size, dot

FWD_BLOCKS = (64, 64)
BWD_BLOCKS = (64, 64)


@triton.jit
def _window(rm, rn, L, WINDOW):
    return ((rm[:, None] >= rn[None, :]) & (rm[:, None] - rn[None, :] < WINDOW)
            & (rm[:, None] < L) & (rn[None, :] < L))


@triton.jit
def _swa_fwd(Q, K, V, META, Y, LSE, L, ND, H, NMETA, WINDOW, scale,
             s_b, s_h, s_l,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
             BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr,
             F32: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    base = (pid_bh // H) * s_b + (pid_bh % H) * s_h
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = tl.arange(0, BLOCK_D)
    rs = tl.arange(0, BLOCK_S)
    row_m = rm[:, None] < L
    q = tl.load(Q + base + rm[:, None] * s_l + rd[None, :],
                mask=row_m & (rd[None, :] < ND), other=0.0)
    meta = tl.load(META + rs, mask=rs < NMETA, other=float("-inf"))
    m_s = tl.maximum(tl.max(meta, 0), -1e30)
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) + m_s
    l_s = tl.sum(tl.exp(meta - m_s), 0)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32) + l_s
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    lo = tl.maximum(pid_m * BLOCK_M - WINDOW + 1, 0) // BLOCK_N * BLOCK_N
    for start in range(lo, (pid_m + 1) * BLOCK_M, BLOCK_N):
        rn = start + tl.arange(0, BLOCK_N)
        row_n = rn[:, None] < L
        k = tl.load(K + base + rn[:, None] * s_l + rd[None, :],
                    mask=row_n & (rd[None, :] < ND), other=0.0)
        v = tl.load(V + base + rn[:, None] * s_l + rd[None, :],
                    mask=row_n & (rd[None, :] < ND), other=0.0)
        s = tl.where(_window(rm, rn, L, WINDOW),
                     dot(q, tl.trans(k), F32) * scale, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + dot(p.to(tl.bfloat16), v, F32)
        m_i = m_new
    y = acc / l_i[:, None]
    tl.store(Y + base + rm[:, None] * s_l + rd[None, :], y.to(tl.bfloat16),
             mask=row_m & (rd[None, :] < ND))
    tl.store(LSE + pid_bh * L + rm, m_i + tl.log(l_i), mask=rm < L)


@triton.jit
def _swa_bwd_q(Q, K, V, DY, LSE, DELTA, DQ, L, ND, H, WINDOW, scale,
               s_b, s_h, s_l,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
               BLOCK_D: tl.constexpr, F32: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    base = (pid_bh // H) * s_b + (pid_bh % H) * s_h
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = tl.arange(0, BLOCK_D)
    row_m = rm[:, None] < L
    q = tl.load(Q + base + rm[:, None] * s_l + rd[None, :],
                mask=row_m & (rd[None, :] < ND), other=0.0)
    dy = tl.load(DY + base + rm[:, None] * s_l + rd[None, :],
                 mask=row_m & (rd[None, :] < ND), other=0.0)
    lse = tl.load(LSE + pid_bh * L + rm, mask=rm < L, other=0.0)
    delta = tl.load(DELTA + pid_bh * L + rm, mask=rm < L, other=0.0)
    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    lo = tl.maximum(pid_m * BLOCK_M - WINDOW + 1, 0) // BLOCK_N * BLOCK_N
    for start in range(lo, (pid_m + 1) * BLOCK_M, BLOCK_N):
        rn = start + tl.arange(0, BLOCK_N)
        row_n = rn[:, None] < L
        k = tl.load(K + base + rn[:, None] * s_l + rd[None, :],
                    mask=row_n & (rd[None, :] < ND), other=0.0)
        v = tl.load(V + base + rn[:, None] * s_l + rd[None, :],
                    mask=row_n & (rd[None, :] < ND), other=0.0)
        s = tl.where(_window(rm, rn, L, WINDOW),
                     dot(q, tl.trans(k), F32) * scale, float("-inf"))
        p = tl.exp(s - lse[:, None])
        ds = p * (dot(dy, tl.trans(v), F32) - delta[:, None])
        dq += dot(ds.to(tl.bfloat16), k, F32)
    tl.store(DQ + base + rm[:, None] * s_l + rd[None, :], dq * scale,
             mask=row_m & (rd[None, :] < ND))


@triton.jit
def _swa_bwd_kv(Q, K, V, DY, LSE, DELTA, DK, DV, L, ND, H, WINDOW, scale,
                s_b, s_h, s_l,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                BLOCK_D: tl.constexpr, F32: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    base = (pid_bh // H) * s_b + (pid_bh % H) * s_h
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rd = tl.arange(0, BLOCK_D)
    row_n = rn[:, None] < L
    k = tl.load(K + base + rn[:, None] * s_l + rd[None, :],
                mask=row_n & (rd[None, :] < ND), other=0.0)
    v = tl.load(V + base + rn[:, None] * s_l + rd[None, :],
                mask=row_n & (rd[None, :] < ND), other=0.0)
    dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    hi = tl.minimum((pid_n + 1) * BLOCK_N + WINDOW - 1, L)
    for start in range(pid_n * BLOCK_N // BLOCK_M * BLOCK_M, hi, BLOCK_M):
        rm = start + tl.arange(0, BLOCK_M)
        row_m = rm[:, None] < L
        q = tl.load(Q + base + rm[:, None] * s_l + rd[None, :],
                    mask=row_m & (rd[None, :] < ND), other=0.0)
        dy = tl.load(DY + base + rm[:, None] * s_l + rd[None, :],
                     mask=row_m & (rd[None, :] < ND), other=0.0)
        lse = tl.load(LSE + pid_bh * L + rm, mask=rm < L, other=0.0)
        delta = tl.load(DELTA + pid_bh * L + rm, mask=rm < L, other=0.0)
        s = tl.where(_window(rm, rn, L, WINDOW),
                     dot(q, tl.trans(k), F32) * scale, float("-inf"))
        p = tl.exp(s - lse[:, None])
        dv += dot(tl.trans(p).to(tl.bfloat16), dy, F32)
        ds = p * (dot(dy, tl.trans(v), F32) - delta[:, None])
        dk += dot(tl.trans(ds).to(tl.bfloat16), q, F32)
    tl.store(DK + base + rn[:, None] * s_l + rd[None, :], dk * scale,
             mask=row_n & (rd[None, :] < ND))
    tl.store(DV + base + rn[:, None] * s_l + rd[None, :], dv,
             mask=row_n & (rd[None, :] < ND))


class _SWA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, meta, window, fwd_blocks, bwd_blocks):
        ctx.dtypes = (q.dtype, k.dtype, v.dtype, meta.dtype)
        q, k, v = (t.to(torch.bfloat16).contiguous() for t in (q, k, v))
        meta = meta.float().contiguous()
        batch, heads, length, nd = q.shape
        scale = 1 / math.sqrt(nd)
        y = torch.empty_like(v)
        lse = torch.empty(batch, heads, length, device=v.device,
                          dtype=torch.float32)
        grid = (triton.cdiv(length, fwd_blocks[0]), batch * heads)
        _swa_fwd[grid](q, k, v, meta, y, lse, length, nd, heads,
                       meta.numel(), window, scale, *q.stride()[:3],
                       BLOCK_M=fwd_blocks[0], BLOCK_N=fwd_blocks[1],
                       BLOCK_D=block_size(nd),
                       BLOCK_S=max(2, triton.next_power_of_2(meta.numel())),
                       F32=INTERPRET)
        ctx.save_for_backward(q, k, v, meta, y, lse)
        ctx.window, ctx.scale, ctx.bwd_blocks = window, scale, bwd_blocks
        return y

    @staticmethod
    def backward(ctx, dy):
        q, k, v, meta, y, lse = ctx.saved_tensors
        blocks = ctx.bwd_blocks
        batch, heads, length, nd = q.shape
        dy = dy.to(torch.bfloat16).contiguous()
        delta = (dy.float() * y.float()).sum(-1).contiguous()
        dq = torch.empty(q.shape, device=q.device, dtype=torch.float32)
        dk = torch.empty_like(dq)
        dv = torch.empty_like(dq)
        args = (length, nd, heads, ctx.window, ctx.scale, *q.stride()[:3])
        meta_kw = dict(BLOCK_M=blocks[0], BLOCK_N=blocks[1],
                       BLOCK_D=block_size(nd), F32=INTERPRET)
        _swa_bwd_q[(triton.cdiv(length, blocks[0]), batch * heads)](
            q, k, v, dy, lse, delta, dq, *args, **meta_kw)
        _swa_bwd_kv[(triton.cdiv(length, blocks[1]), batch * heads)](
            q, k, v, dy, lse, delta, dk, dv, *args, **meta_kw)
        sink = torch.exp(meta[:, None, None, None] - lse[None])
        dmeta = -(sink * delta[None]).sum((1, 2, 3))
        grads = (dq, dk, dv, dmeta)
        dq, dk, dv, dmeta = (t.to(d) for t, d in zip(grads, ctx.dtypes))
        return dq, dk, dv, dmeta, None, None, None


def swa(q, k, v, meta, window, fwd_blocks=FWD_BLOCKS, bwd_blocks=BWD_BLOCKS):
    return _SWA.apply(q, k, v, meta, window, fwd_blocks, bwd_blocks)
