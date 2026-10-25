import torch
import triton
import triton.language as tl

from .dot import INTERPRET, block_size, dot

TINY = tl.constexpr(1.1754943508222875e-38)
CHUNK = 64
BLOCK = 64


def _configs():
    if INTERPRET:
        return [triton.Config({}, num_warps=4, num_stages=1)]
    return [triton.Config({}, num_warps=w, num_stages=s)
            for w in (4, 8) for s in (1, 2, 3)]


@triton.jit
def _gates(G, n, L, BT: tl.constexpr):
    rt = n * BT + tl.arange(0, BT)
    g = tl.load(G + rt, mask=rt < L, other=0.0)
    start = tl.where(n > 0, tl.load(G + tl.maximum(n * BT - 1, 0)), 0.0)
    end = tl.load(G + tl.minimum(n * BT + BT, L) - 1)
    return rt, g, start, end


@triton.jit
def _rescale(rt, g, ref, L):
    return tl.exp(tl.where(rt < L, g - ref, float("-inf")))


@triton.autotune(configs=_configs(), key=["NF", "ND"])
@triton.jit
def _gla_fwd_h(FK, V, G, S, Z, L, NF, ND, NT,
               BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
               F32: tl.constexpr):
    pid_k, pid_v, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rk = pid_k * BK + tl.arange(0, BK)
    rv = pid_v * BV + tl.arange(0, BV)
    s_mask = (rk[:, None] < NF) & (rv[None, :] < ND)
    h = tl.zeros((BK, BV), dtype=tl.float32)
    z = tl.zeros((BK,), dtype=tl.float32)
    for n in range(NT):
        chunk = bh * NT + n
        tl.store(S + (chunk * NF + rk[:, None]) * ND + rv[None, :],
                 h.to(tl.bfloat16), mask=s_mask)
        tl.store(Z + chunk * NF + rk, z, mask=(rk < NF) & (pid_v == 0))
        rt, g, start, end = _gates(G + bh * L, n, L, BT)
        row = rt[:, None] < L
        fk = tl.load(FK + (bh * L + rt[:, None]) * NF + rk[None, :],
                     mask=row & (rk[None, :] < NF), other=0.0)
        v = tl.load(V + (bh * L + rt[:, None]) * ND + rv[None, :],
                    mask=row & (rv[None, :] < ND), other=0.0)
        kt = (fk * _rescale(rt, end, g, L)[:, None]).to(tl.bfloat16)
        h = h * tl.exp(end - start) + dot(tl.trans(kt), v, F32)
        z = z * tl.exp(end - start) + tl.sum(kt.to(tl.float32), 0)


@triton.autotune(configs=_configs(), key=["NF", "ND"])
@triton.jit
def _gla_fwd_o(FQ, FK, V, G, S, Z, Y, Y32, DEN, L, NF, ND, NT,
               BT: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr,
               SAVE: tl.constexpr, F32: tl.constexpr):
    n, bh = tl.program_id(0), tl.program_id(1)
    chunk = bh * NT + n
    rt, g, start, end = _gates(G + bh * L, n, L, BT)
    rd = tl.arange(0, BD)
    row = rt[:, None] < L
    qs = _rescale(rt, g, start, L)
    s = tl.zeros((BT, BT), dtype=tl.float32)
    o = tl.zeros((BT, BD), dtype=tl.float32)
    den = tl.zeros((BT,), dtype=tl.float32)
    for k in range(0, NF, BK):
        rk = k + tl.arange(0, BK)
        f_mask = row & (rk[None, :] < NF)
        fq = tl.load(FQ + (bh * L + rt[:, None]) * NF + rk[None, :],
                     mask=f_mask, other=0.0)
        fk = tl.load(FK + (bh * L + rt[:, None]) * NF + rk[None, :],
                     mask=f_mask, other=0.0)
        h = tl.load(S + (chunk * NF + rk[:, None]) * ND + rd[None, :],
                    mask=(rk[:, None] < NF) & (rd[None, :] < ND), other=0.0)
        z = tl.load(Z + chunk * NF + rk, mask=rk < NF, other=0.0)
        qt = (fq * qs[:, None]).to(tl.bfloat16)
        s += dot(fq, tl.trans(fk), F32)
        o += dot(qt, h, F32)
        den += tl.sum(qt.to(tl.float32) * z[None, :], 1)
    causal = (rt[:, None] >= rt[None, :]) & row
    decay = tl.exp(tl.where(causal, g[:, None] - g[None, :], float("-inf")))
    a = (s * decay).to(tl.bfloat16)
    v_mask = row & (rd[None, :] < ND)
    v = tl.load(V + (bh * L + rt[:, None]) * ND + rd[None, :], mask=v_mask,
                other=0.0)
    o += dot(a, v, F32)
    den += tl.sum(a.to(tl.float32), 1)
    y = o / tl.maximum(den, TINY)[:, None]
    out = (bh * L + rt[:, None]) * ND + rd[None, :]
    tl.store(Y + out, y.to(tl.bfloat16), mask=v_mask)
    if SAVE:
        tl.store(Y32 + out, y, mask=v_mask)
        tl.store(DEN + bh * L + rt, den, mask=rt < L)


@triton.autotune(configs=_configs(), key=["NF", "ND"])
@triton.jit
def _gla_bwd_dh(FQ, DY, G, DEN, DELTA, DS, DZ, L, NF, ND, NT,
                BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                F32: tl.constexpr):
    pid_k, pid_v, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rk = pid_k * BK + tl.arange(0, BK)
    rv = pid_v * BV + tl.arange(0, BV)
    s_mask = (rk[:, None] < NF) & (rv[None, :] < ND)
    dh = tl.zeros((BK, BV), dtype=tl.float32)
    dz = tl.zeros((BK,), dtype=tl.float32)
    for i in range(NT):
        n = NT - 1 - i
        chunk = bh * NT + n
        tl.store(DS + (chunk * NF + rk[:, None]) * ND + rv[None, :],
                 dh.to(tl.bfloat16), mask=s_mask)
        tl.store(DZ + chunk * NF + rk, dz, mask=(rk < NF) & (pid_v == 0))
        rt, g, start, end = _gates(G + bh * L, n, L, BT)
        row = rt[:, None] < L
        fq = tl.load(FQ + (bh * L + rt[:, None]) * NF + rk[None, :],
                     mask=row & (rk[None, :] < NF), other=0.0)
        dy = tl.load(DY + (bh * L + rt[:, None]) * ND + rv[None, :],
                     mask=row & (rv[None, :] < ND), other=0.0)
        den = tl.load(DEN + bh * L + rt, mask=rt < L, other=1.0)
        delta = tl.load(DELTA + bh * L + rt, mask=rt < L, other=0.0)
        qt = (fq * (_rescale(rt, g, start, L) / den)[:, None]).to(tl.bfloat16)
        dh = dh * tl.exp(end - start) + dot(tl.trans(qt), dy, F32)
        dz = dz * tl.exp(end - start) + tl.sum(
            qt.to(tl.float32) * delta[:, None], 0)


@triton.autotune(configs=_configs(), key=["NF", "ND"])
@triton.jit
def _gla_bwd_dqkv(FQ, FK, V, G, DEN, DELTA, DY, S, Z, DS, DZ, DFQ, DFK, DV,
                  L, NF, ND, NT,
                  BT: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr,
                  F32: tl.constexpr):
    n, bh = tl.program_id(0), tl.program_id(1)
    chunk = bh * NT + n
    rt, g, start, end = _gates(G + bh * L, n, L, BT)
    rd = tl.arange(0, BD)
    row = rt[:, None] < L
    v_mask = row & (rd[None, :] < ND)
    den = tl.load(DEN + bh * L + rt, mask=rt < L, other=1.0)
    delta = tl.load(DELTA + bh * L + rt, mask=rt < L, other=0.0)
    dy = tl.load(DY + (bh * L + rt[:, None]) * ND + rd[None, :], mask=v_mask,
                 other=0.0)
    v = tl.load(V + (bh * L + rt[:, None]) * ND + rd[None, :], mask=v_mask,
                other=0.0)
    s = tl.zeros((BT, BT), dtype=tl.float32)
    for k in range(0, NF, BK):
        rk = k + tl.arange(0, BK)
        f_mask = row & (rk[None, :] < NF)
        fq = tl.load(FQ + (bh * L + rt[:, None]) * NF + rk[None, :],
                     mask=f_mask, other=0.0)
        fk = tl.load(FK + (bh * L + rt[:, None]) * NF + rk[None, :],
                     mask=f_mask, other=0.0)
        s += dot(fq, tl.trans(fk), F32)
    causal = (rt[:, None] >= rt[None, :]) & row
    decay = tl.exp(tl.where(causal, g[:, None] - g[None, :], float("-inf")))
    a = (s * decay).to(tl.bfloat16)
    da = tl.where(causal, (dot(dy, tl.trans(v), F32) - delta[:, None])
                  / den[:, None], 0.0)
    dad = da * decay
    dv = dot(tl.trans(a.to(tl.float32) / den[:, None]), dy, F32)
    qs = _rescale(rt, g, start, L) / den
    ks = _rescale(rt, end, g, L)
    for k in range(0, NF, BK):
        rk = k + tl.arange(0, BK)
        f_mask = row & (rk[None, :] < NF)
        s_mask = (rk[:, None] < NF) & (rd[None, :] < ND)
        fq = tl.load(FQ + (bh * L + rt[:, None]) * NF + rk[None, :],
                     mask=f_mask, other=0.0)
        fk = tl.load(FK + (bh * L + rt[:, None]) * NF + rk[None, :],
                     mask=f_mask, other=0.0)
        h = tl.load(S + (chunk * NF + rk[:, None]) * ND + rd[None, :],
                    mask=s_mask, other=0.0)
        dh = tl.load(DS + (chunk * NF + rk[:, None]) * ND + rd[None, :],
                     mask=s_mask, other=0.0)
        z = tl.load(Z + chunk * NF + rk, mask=rk < NF, other=0.0)
        dz = tl.load(DZ + chunk * NF + rk, mask=rk < NF, other=0.0)
        dfq = dot(dad, fk, F32) + qs[:, None] * (
            dot(dy, tl.trans(h), F32) - delta[:, None] * z[None, :])
        dfk = dot(tl.trans(dad), fq, F32) + ks[:, None] * (
            dot(v, tl.trans(dh), F32) - dz[None, :])
        dv += dot(fk * ks[:, None], dh, F32)
        tl.store(DFQ + (bh * L + rt[:, None]) * NF + rk[None, :], dfq,
                 mask=f_mask)
        tl.store(DFK + (bh * L + rt[:, None]) * NF + rk[None, :], dfk,
                 mask=f_mask)
    tl.store(DV + (bh * L + rt[:, None]) * ND + rd[None, :], dv, mask=v_mask)


def _sizes(fq, v, chunk):
    batch, heads, length, nf = fq.shape
    nd = v.shape[-1]
    return batch * heads, length, nf, nd, triton.cdiv(length, chunk)


def _state_grid(bh, nf, nd):
    bk, bv = min(BLOCK, block_size(nf)), min(BLOCK, block_size(nd))
    return (triton.cdiv(nf, bk), triton.cdiv(nd, bv), bh), dict(BK=bk, BV=bv)


def _chunk_meta(nf, nd, chunk):
    return dict(BT=chunk, BK=min(BLOCK, block_size(nf)),
                BD=block_size(nd), F32=INTERPRET)


class _GLA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fq, fk, v, g, chunk):
        ctx.dtypes = (fq.dtype, fk.dtype, v.dtype)
        fq, fk, v = (t.to(torch.bfloat16).contiguous() for t in (fq, fk, v))
        g = g.float().contiguous()
        bh, length, nf, nd, nt = _sizes(fq, v, chunk)
        save = any(ctx.needs_input_grad[:4])
        s = fq.new_empty(bh, nt, nf, nd)
        z = g.new_empty(bh, nt, nf)
        y = torch.empty_like(v)
        y32 = torch.empty_like(v, dtype=torch.float32) if save else y
        den = torch.empty_like(g) if save else g
        grid, blocks = _state_grid(bh, nf, nd)
        _gla_fwd_h[grid](fk, v, g, s, z, length, nf, nd, nt, BT=chunk,
                         F32=INTERPRET, **blocks)
        _gla_fwd_o[(nt, bh)](fq, fk, v, g, s, z, y, y32, den, length, nf, nd,
                             nt, SAVE=save, **_chunk_meta(nf, nd, chunk))
        if save:
            ctx.save_for_backward(fq, fk, v, g, y32, den, s, z)
        ctx.chunk = chunk
        return y

    @staticmethod
    def backward(ctx, dy):
        fq, fk, v, g, y, den, s, z = ctx.saved_tensors
        chunk = ctx.chunk
        bh, length, nf, nd, nt = _sizes(fq, v, chunk)
        dy = dy.to(torch.bfloat16).contiguous()
        delta = (dy.float() * y).sum(-1).contiguous()
        ds = torch.empty_like(s)
        dz = torch.empty_like(z)
        grid, blocks = _state_grid(bh, nf, nd)
        _gla_bwd_dh[grid](fq, dy, g, den, delta, ds, dz, length, nf, nd, nt,
                          BT=chunk, F32=INTERPRET, **blocks)
        dfq = torch.empty(fq.shape, device=fq.device, dtype=torch.float32)
        dfk = torch.empty_like(dfq)
        dv = torch.empty(v.shape, device=v.device, dtype=torch.float32)
        _gla_bwd_dqkv[(nt, bh)](fq, fk, v, g, den, delta, dy, s, z, ds, dz,
                                dfq, dfk, dv, length, nf, nd, nt,
                                **_chunk_meta(nf, nd, chunk))
        dg = (fq.float() * dfq - fk.float() * dfk).sum(-1)
        dfq, dfk, dv = (t.to(d) for t, d in zip((dfq, dfk, dv), ctx.dtypes))
        return dfq, dfk, dv, dg, None


def gla(fq, fk, v, log_gamma, chunk=CHUNK):
    g = torch.cumsum(log_gamma.float(), dim=-1)
    if g.dim() == 2:
        g = g[:, None].expand(-1, fq.shape[1], -1)
    return _GLA.apply(fq, fk, v, g, chunk)
