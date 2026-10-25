import torch
import triton
import triton.language as tl

from .dot import INTERPRET, block_size, dot

TINY = tl.constexpr(1.1754943508222875e-38)
FWD_BLOCKS = (64, 32)
BWD_BLOCKS = (32, 32)


@triton.jit
def _gla_fwd(FQ, FK, V, G, Y, DEN, L, NF, ND, H,
             s_fb, s_fh, s_fl, s_vb, s_vh, s_vl, s_gb,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
             BLOCK_F: tl.constexpr, BLOCK_D: tl.constexpr,
             F32: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rf = tl.arange(0, BLOCK_F)
    rd = tl.arange(0, BLOCK_D)
    f_base = b * s_fb + h * s_fh
    v_base = b * s_vb + h * s_vh
    row_m = rm[:, None] < L
    fq = tl.load(FQ + f_base + rm[:, None] * s_fl + rf[None, :],
                 mask=row_m & (rf[None, :] < NF), other=0.0)
    g_i = tl.load(G + b * s_gb + rm, mask=rm < L, other=0.0)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    den = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start in range(0, (pid_m + 1) * BLOCK_M, BLOCK_N):
        rn = start + tl.arange(0, BLOCK_N)
        row_n = rn[:, None] < L
        fk = tl.load(FK + f_base + rn[:, None] * s_fl + rf[None, :],
                     mask=row_n & (rf[None, :] < NF), other=0.0)
        v = tl.load(V + v_base + rn[:, None] * s_vl + rd[None, :],
                    mask=row_n & (rd[None, :] < ND), other=0.0)
        g_t = tl.load(G + b * s_gb + rn, mask=rn < L, other=0.0)
        causal = (rm[:, None] >= rn[None, :]) & row_m & (rn[None, :] < L)
        decay = tl.exp(tl.where(causal, g_i[:, None] - g_t[None, :],
                                float("-inf")))
        a = (dot(fq, tl.trans(fk), F32) * decay).to(tl.bfloat16)
        acc += dot(a, v, F32)
        den += tl.sum(a.to(tl.float32), 1)
    y = acc / tl.maximum(den, TINY)[:, None]
    tl.store(Y + v_base + rm[:, None] * s_vl + rd[None, :],
             y.to(tl.bfloat16), mask=row_m & (rd[None, :] < ND))
    tl.store(DEN + pid_bh * L + rm, den, mask=rm < L)


@triton.jit
def _gla_bwd_q(FQ, FK, V, G, DY, DEN, DELTA, DFQ, RROW, L, NF, ND, H,
               s_fb, s_fh, s_fl, s_vb, s_vh, s_vl, s_gb,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
               BLOCK_F: tl.constexpr, BLOCK_D: tl.constexpr,
               F32: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rf = tl.arange(0, BLOCK_F)
    rd = tl.arange(0, BLOCK_D)
    f_base = b * s_fb + h * s_fh
    v_base = b * s_vb + h * s_vh
    row_m = rm[:, None] < L
    fq = tl.load(FQ + f_base + rm[:, None] * s_fl + rf[None, :],
                 mask=row_m & (rf[None, :] < NF), other=0.0)
    dy = tl.load(DY + v_base + rm[:, None] * s_vl + rd[None, :],
                 mask=row_m & (rd[None, :] < ND), other=0.0)
    g_i = tl.load(G + b * s_gb + rm, mask=rm < L, other=0.0)
    den = tl.maximum(tl.load(DEN + pid_bh * L + rm, mask=rm < L, other=1.0),
                     TINY)
    delta = tl.load(DELTA + pid_bh * L + rm, mask=rm < L, other=0.0)
    dfq = tl.zeros((BLOCK_M, BLOCK_F), dtype=tl.float32)
    rrow = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start in range(0, (pid_m + 1) * BLOCK_M, BLOCK_N):
        rn = start + tl.arange(0, BLOCK_N)
        row_n = rn[:, None] < L
        fk = tl.load(FK + f_base + rn[:, None] * s_fl + rf[None, :],
                     mask=row_n & (rf[None, :] < NF), other=0.0)
        v = tl.load(V + v_base + rn[:, None] * s_vl + rd[None, :],
                    mask=row_n & (rd[None, :] < ND), other=0.0)
        g_t = tl.load(G + b * s_gb + rn, mask=rn < L, other=0.0)
        causal = (rm[:, None] >= rn[None, :]) & row_m & (rn[None, :] < L)
        decay = tl.exp(tl.where(causal, g_i[:, None] - g_t[None, :],
                                float("-inf")))
        a = (dot(fq, tl.trans(fk), F32) * decay).to(tl.bfloat16)
        dp = dot(dy, tl.trans(v), F32)
        da = tl.where(causal, (dp - delta[:, None]) / den[:, None], 0.0)
        dfq += dot((da * decay).to(tl.bfloat16), fk, F32)
        rrow += tl.sum(da * a.to(tl.float32), 1)
    tl.store(DFQ + f_base + rm[:, None] * s_fl + rf[None, :], dfq,
             mask=row_m & (rf[None, :] < NF))
    tl.store(RROW + pid_bh * L + rm, rrow, mask=rm < L)


@triton.jit
def _gla_bwd_kv(FQ, FK, V, G, DY, DEN, DELTA, DFK, DV, RCOL, L, NF, ND, H,
                s_fb, s_fh, s_fl, s_vb, s_vh, s_vl, s_gb,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                BLOCK_F: tl.constexpr, BLOCK_D: tl.constexpr,
                F32: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rf = tl.arange(0, BLOCK_F)
    rd = tl.arange(0, BLOCK_D)
    f_base = b * s_fb + h * s_fh
    v_base = b * s_vb + h * s_vh
    row_n = rn[:, None] < L
    fk = tl.load(FK + f_base + rn[:, None] * s_fl + rf[None, :],
                 mask=row_n & (rf[None, :] < NF), other=0.0)
    v = tl.load(V + v_base + rn[:, None] * s_vl + rd[None, :],
                mask=row_n & (rd[None, :] < ND), other=0.0)
    g_t = tl.load(G + b * s_gb + rn, mask=rn < L, other=0.0)
    dfk = tl.zeros((BLOCK_N, BLOCK_F), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    rcol = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for start in range(pid_n * BLOCK_N // BLOCK_M * BLOCK_M, L, BLOCK_M):
        rm = start + tl.arange(0, BLOCK_M)
        row_m = rm[:, None] < L
        fq = tl.load(FQ + f_base + rm[:, None] * s_fl + rf[None, :],
                     mask=row_m & (rf[None, :] < NF), other=0.0)
        dy = tl.load(DY + v_base + rm[:, None] * s_vl + rd[None, :],
                     mask=row_m & (rd[None, :] < ND), other=0.0)
        g_i = tl.load(G + b * s_gb + rm, mask=rm < L, other=0.0)
        den = tl.maximum(tl.load(DEN + pid_bh * L + rm, mask=rm < L,
                                 other=1.0), TINY)
        delta = tl.load(DELTA + pid_bh * L + rm, mask=rm < L, other=0.0)
        causal = (rm[:, None] >= rn[None, :]) & row_m & (rn[None, :] < L)
        decay = tl.exp(tl.where(causal, g_i[:, None] - g_t[None, :],
                                float("-inf")))
        a = (dot(fq, tl.trans(fk), F32) * decay).to(tl.bfloat16)
        w = a.to(tl.float32) / den[:, None]
        dv += dot(tl.trans(w).to(tl.bfloat16), dy, F32)
        dp = dot(dy, tl.trans(v), F32)
        da = tl.where(causal, (dp - delta[:, None]) / den[:, None], 0.0)
        dfk += dot(tl.trans(da * decay).to(tl.bfloat16), fq, F32)
        rcol += tl.sum(da * a.to(tl.float32), 0)
    tl.store(DFK + f_base + rn[:, None] * s_fl + rf[None, :], dfk,
             mask=row_n & (rf[None, :] < NF))
    tl.store(DV + v_base + rn[:, None] * s_vl + rd[None, :], dv,
             mask=row_n & (rd[None, :] < ND))
    tl.store(RCOL + pid_bh * L + rn, rcol, mask=rn < L)


def _args(fq, v, g):
    _, heads, length, nf = fq.shape
    return (length, nf, v.shape[-1], heads, *fq.stride()[:3],
            *v.stride()[:3], g.stride(0))


def _meta(fq, v, blocks):
    return dict(BLOCK_M=blocks[0], BLOCK_N=blocks[1],
                BLOCK_F=block_size(fq.shape[-1]),
                BLOCK_D=block_size(v.shape[-1]), F32=INTERPRET,
                num_warps=4, num_stages=2)


class _GLA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fq, fk, v, g, fwd_blocks, bwd_blocks):
        ctx.dtypes = (fq.dtype, fk.dtype, v.dtype)
        fq, fk, v = (t.to(torch.bfloat16).contiguous() for t in (fq, fk, v))
        g = g.float().contiguous()
        batch, heads, length, _ = fq.shape
        y = torch.empty_like(v)
        den = torch.empty(batch, heads, length, device=v.device,
                          dtype=torch.float32)
        grid = (triton.cdiv(length, fwd_blocks[0]), batch * heads)
        _gla_fwd[grid](fq, fk, v, g, y, den, *_args(fq, v, g),
                       **_meta(fq, v, fwd_blocks))
        ctx.save_for_backward(fq, fk, v, g, y, den)
        ctx.bwd_blocks = bwd_blocks
        return y

    @staticmethod
    def backward(ctx, dy):
        fq, fk, v, g, y, den = ctx.saved_tensors
        blocks = ctx.bwd_blocks
        batch, heads, length, _ = fq.shape
        dy = dy.to(torch.bfloat16).contiguous()
        delta = (dy.float() * y.float()).sum(-1).contiguous()
        dfq = torch.empty(fq.shape, device=fq.device, dtype=torch.float32)
        dfk = torch.empty_like(dfq)
        dv = torch.empty(v.shape, device=v.device, dtype=torch.float32)
        rrow = torch.empty_like(delta)
        rcol = torch.empty_like(delta)
        args, meta = _args(fq, v, g), _meta(fq, v, blocks)
        _gla_bwd_q[(triton.cdiv(length, blocks[0]), batch * heads)](
            fq, fk, v, g, dy, den, delta, dfq, rrow, *args, **meta)
        _gla_bwd_kv[(triton.cdiv(length, blocks[1]), batch * heads)](
            fq, fk, v, g, dy, den, delta, dfk, dv, rcol, *args, **meta)
        dg = (rrow - rcol).sum(1)
        dfq, dfk, dv = (t.to(d) for t, d in zip((dfq, dfk, dv), ctx.dtypes))
        return dfq, dfk, dv, dg, None, None


def gla(fq, fk, v, log_gamma, fwd_blocks=FWD_BLOCKS, bwd_blocks=BWD_BLOCKS):
    g = torch.cumsum(log_gamma.float(), dim=-1)
    return _GLA.apply(fq, fk, v, g, fwd_blocks, bwd_blocks)
