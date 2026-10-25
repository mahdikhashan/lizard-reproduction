import argparse

import torch
import triton

import kernels

SHAPES = ["16x2048", "16x4096", "16x8192", "32x8192"]
PAPER = {(16, 2048): 32, (16, 4096): 35, (16, 8192): 36, (32, 8192): 36}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Latency of the Lizard GLA kernel and of the GLA kernels "
                    "of flash-linear-attention, as in Table 3 of the paper")
    parser.add_argument("--shapes", nargs="+", default=SHAPES,
                        metavar="BxL", help="batch size x sequence length")
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--dim", type=int, default=128,
                        help="head dimension of q, k and v")
    parser.add_argument("--backward", action="store_true",
                        help="time the forward and the backward pass")
    parser.add_argument("--plot", metavar="FILE",
                        help="save a bar chart of the first shape")
    return parser.parse_args(argv)


def features(*shape):
    return torch.rand(shape, device="cuda", dtype=torch.bfloat16)


def log_gates(*shape):
    x = torch.randn(shape, device="cuda")
    return x.neg_().exp_().log1p_().neg_()


def lizard(batch, length, heads, dim):
    shape = (batch, heads, length, dim)
    inputs = (features(*shape), features(*shape), features(*shape),
              log_gates(batch, heads, length))
    return kernels.gla, inputs


def fla(elementwise):
    def make(batch, length, heads, dim):
        from fla.ops.gla import chunk_gla
        from fla.ops.simple_gla import chunk_simple_gla
        kernel = chunk_gla if elementwise else chunk_simple_gla
        shape = (batch, length, heads, dim)
        gates = shape if elementwise else shape[:3]
        inputs = (features(*shape), features(*shape), features(*shape),
                  log_gates(*gates))
        return lambda *x: kernel(*x, scale=1.0)[0], inputs
    return make


def measure(make, shape, heads, dim, backward):
    fn, inputs = make(*shape, heads, dim)
    if not backward:
        with torch.no_grad():
            return triton.testing.do_bench(lambda: fn(*inputs),
                                           return_mode="median")
    inputs = [x.requires_grad_() for x in inputs]
    grad = torch.randn_like(fn(*inputs))
    return triton.testing.do_bench(lambda: fn(*inputs).backward(grad),
                                   grad_to_none=inputs, return_mode="median")


def methods():
    try:
        from fla.ops.gla import chunk_gla  # noqa: F401
        from fla.ops.simple_gla import chunk_simple_gla  # noqa: F401
    except ImportError as error:
        print(f"[skip] flash-linear-attention: {error} "
              "(pip install fla-core)")
        return {"Lizard": lizard}
    return {"GLA": fla(True), "Simple GLA": fla(False), "Lizard": lizard}


def cell(ms):
    return f"{ms:.2f} ms" if ms is not None else "OOM"


def speedup(times):
    gla, ours = times.get("GLA"), times.get("Lizard")
    return f"{gla / ours - 1:+.0%}" if gla and ours else "-"


def plot(path, shape, times):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [name for name, ms in times.items() if ms is not None]
    values = [times[name] for name in names]
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    bars = ax.bar(names, values, width=0.45, color="#2a78d6", zorder=2)
    ax.bar_label(bars, labels=[f"{ms:.3f} ms" for ms in values], padding=4,
                 color="#0b0b0b")
    ax.set_ylabel("Median time (ms)", color="#52514e")
    ax.set_title(f"B = {shape[0]}, L = {shape[1]}", color="#0b0b0b")
    ax.grid(axis="y", color="#e4e3df", linewidth=1, zorder=0)
    ax.tick_params(colors="#52514e")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"saved {path}")


def main(shapes, heads, dim, backward, plot_path=None):
    assert torch.cuda.is_available(), "the benchmark needs a CUDA GPU"
    makers = methods()
    passes = "forward and backward" if backward else "forward"
    print(f"{torch.cuda.get_device_name()}, bfloat16, {heads} heads, "
          f"head dimension {dim}, {passes} pass")
    print(f"{'Configuration':<18}" + "".join(f"{n:>12}" for n in makers)
          + f"{'Speedup':>10}{'Paper':>8}")
    results = {}
    for text in shapes:
        shape = tuple(int(x) for x in text.lower().split("x"))
        times = {}
        for name, make in makers.items():
            try:
                times[name] = measure(make, shape, heads, dim, backward)
            except torch.cuda.OutOfMemoryError:
                times[name] = None
            torch.cuda.empty_cache()
        paper = PAPER.get(shape) if (heads, dim, backward) == (32, 128,
                                                               False) else None
        print(f"{f'B = {shape[0]}, L = {shape[1]}':<18}"
              + "".join(f"{cell(times[n]):>12}" for n in makers)
              + f"{speedup(times):>10}"
              + f"{f'+{paper}%' if paper else '-':>8}")
        results[shape] = times
    if plot_path:
        first = next(iter(results))
        plot(plot_path, first, results[first])
    return results


if __name__ == "__main__":
    args = parse_args()
    main(args.shapes, args.heads, args.dim, args.backward, args.plot)
