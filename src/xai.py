import argparse
import json
import math
from pathlib import Path

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

import config
import lizard_attention as lz

SPANS = [(0, 128), (128, 512), (512, 1024), (1024, 2048)]
HEAD_GROUP = 8
MAP_SIZE = 256
METRICS = ("tv", "teacher_outside", "lizard_outside", "window_share")
EDGES = (0, 128, 512)
CANDIDATES = {"G": "gated branch",
              "W": "window branch of the checkpoint, without alpha",
              "WR": "oracle: softmax with RoPE over the window, without sinks"}
SETS = {"G": [1], "W": [2], "WR": [3], "G+W": [1, 2], "G+WR": [1, 3],
        "W+WR": [2, 3], "G+W+WR": [1, 2, 3]}
COLORS = {"PyTorch": "#2a78d6", "Triton": "#eb6834", "Teacher": "#1baf7a",
          "Paper": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TASKS = ["piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande",
         "avg", "mmlu"]
TASK_NAMES = ["PIQA", "ARC-e", "ARC-c", "HellaSwag", "WinoGrande", "Avg.",
              "MMLU"]


def span_names(length):
    names = [f"{a}-{b - 1}" for a, b in SPANS if a < length]
    return names + ["all"]


def span_slices(length):
    slices = [slice(a, min(b, length)) for a, b in SPANS if a < length]
    return slices + [slice(0, length)]


def held_out(tokenizer, cfg, sequences):
    from datasets import load_dataset

    from train import format_example, pack
    data = load_dataset(cfg.dataset_name, split="train").shuffle(seed=cfg.seed)
    data = data.select(range(cfg.num_examples, len(data)))
    tokens = tokenizer([format_example(e) for e in data])["input_ids"]
    chunks = pack([t + [tokenizer.eos_token_id] for t in tokens], cfg.seq_len)
    assert len(chunks) >= sequences, f"only {len(chunks)} held-out sequences"
    return chunks[:sequences]


def capture(teacher, ids):
    found = {}
    hooks = []
    for i, block in enumerate(teacher.model.layers):
        def hook(module, args, kwargs, output, i=i):
            x = args[0] if args else kwargs["hidden_states"]
            found[i] = (x, kwargs["position_embeddings"], output[0])
        hooks.append(block.self_attn.register_forward_hook(hook,
                                                           with_kwargs=True))
    with torch.no_grad():
        teacher.model(input_ids=ids, use_cache=False)
    for hook in hooks:
        hook.remove()
    return found


def heads(x, proj, n, head_dim, repeat):
    b, length, _ = x.shape
    y = proj(x).view(b, length, n, head_dim).transpose(1, 2).float()
    return y.repeat_interleave(repeat, dim=1)


def teacher_qk(attn, x, rope, cfg):
    h, kv = cfg.num_attention_heads, cfg.num_key_value_heads
    d = cfg.hidden_size // h
    q = heads(x, attn.q_proj, h, d, 1)
    k = heads(x, attn.k_proj, kv, d, h // kv)
    cos, sin = (t.float() for t in rope)
    return apply_rotary_pos_emb(q, k, cos, sin)


def lizard_parts(attn, x, rope):
    d = attn.head_dim
    q = heads(x, attn.q_proj, attn.heads, d, 1)
    k = heads(x, attn.k_proj, attn.kv_heads, d, attn.heads // attn.kv_heads)
    v = heads(x, attn.v_proj, attn.kv_heads, d, attn.heads // attn.kv_heads)
    z = attn.W_gamma(x).float().squeeze(-1)
    fq = lz.hedgehog(q, attn.phi_q.weight.float())
    fk = lz.hedgehog(k, attn.phi_k.weight.float())
    cos, sin = (t.float() for t in rope)
    qr, kr = apply_rotary_pos_emb(q, k, cos, sin)
    return dict(fq=fq, fk=fk, v=v, qr=qr, kr=kr, gamma=torch.sigmoid(z),
                alpha=attn.alpha_blend.float(), meta=attn.meta_tokens.float(),
                window=attn.window)


def lizard_weights(p, h):
    w = (p["fq"][:, h] @ p["fk"][:, h].transpose(-1, -2))
    w = w * lz.gate_products(p["gamma"])[:, None]
    gated = w / w.sum(-1, keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)
    length = w.shape[-1]
    mask = lz.window_mask(length, p["window"], w.device)
    scores = p["qr"][:, h] @ p["kr"][:, h].transpose(-1, -2)
    scores = scores / math.sqrt(p["qr"].shape[-1])
    window = lz.sink_softmax(scores.masked_fill(~mask, float("-inf")),
                             p["meta"])
    return gated, window


def teacher_weights(q, k, h):
    s = q[:, h] @ k[:, h].transpose(-1, -2) / math.sqrt(q.shape[-1])
    causal = lz.window_mask(s.shape[-1], s.shape[-1], s.device)
    return s.masked_fill(~causal, float("-inf")).softmax(-1)


def weight_metrics(teacher, gated, window, alpha, size):
    total = gated + alpha * window
    rows = total.sum(-1)
    lizard = total / rows[..., None]
    length = teacher.shape[-1]
    i = torch.arange(length, device=teacher.device)[:, None]
    t = torch.arange(length, device=teacher.device)[None, :]
    outside = (t <= i - size).float()
    return {"tv": 0.5 * (teacher - lizard).abs().sum(-1),
            "teacher_outside": (teacher * outside).sum(-1),
            "lizard_outside": (lizard * outside).sum(-1),
            "window_share": alpha * window.sum(-1) / rows}, lizard


def rounded(t):
    return [[float(f"{x:.4g}") for x in row] for row in t.tolist()]


def bounds(length):
    edges = sorted({min(e, length) for e in EDGES} | {length})
    return list(zip(edges[:-1], edges[1:]))


def bucket_names(length):
    return [f"{lo}\u2013{hi - 1}" for lo, hi in bounds(length)] + ["all"]


def window_rope(p, h):
    s = p["qr"][:, h] @ p["kr"][:, h].transpose(-1, -2)
    s = s / math.sqrt(p["qr"].shape[-1])
    mask = lz.window_mask(s.shape[-1], p["window"], s.device)
    return s.masked_fill(~mask, float("-inf")).softmax(-1) @ p["v"][:, h]


def before_o_proj(teacher_v, y_true, p, output, length):
    gated = lz.gla(p["fq"], p["fk"], p["v"], p["gamma"])
    window = lz.awa(p["qr"], p["kr"], p["v"], p["meta"], p["window"])
    oracle = torch.cat([window_rope(p, slice(i, i + HEAD_GROUP))
                        for i in range(0, p["v"].shape[1], HEAD_GROUP)], 1)
    stack = torch.stack([y_true, gated, window, oracle, output])[:, 0]
    stack = stack.double()
    error = (output - y_true)[0].double().pow(2)
    target = y_true[0].double().pow(2)
    grams, err, tsq = [], [], []
    for lo, hi in bounds(length):
        part = stack[:, :, lo:hi]
        grams.append(torch.einsum("shld,thld->hst", part, part).cpu())
        err.append(error[:, lo:hi].sum((1, 2)).cpu())
        tsq.append(target[:, lo:hi].sum((1, 2)).cpu())
    return torch.stack(grams), torch.stack(err), torch.stack(tsq)


def layer_stats(attn, teacher_attn, x, rope, target, model_cfg, maps, layer):
    out = attn(x, rope)[0].float()
    err = (out - target.float()).pow(2).mean(-1)[0]
    norm = target.float().pow(2).mean(-1)[0]
    p = lizard_parts(attn, x, rope)
    q, k = teacher_qk(teacher_attn, x, rope, model_cfg)
    n_heads = q.shape[1]
    kv = model_cfg.num_key_value_heads
    teacher_v = heads(x, teacher_attn.v_proj, kv, q.shape[-1], n_heads // kv)
    sums = {m: [] for m in METRICS}
    crops = {}
    output = torch.zeros_like(p["v"])
    y_true = torch.zeros_like(p["v"])
    for start in range(0, n_heads, HEAD_GROUP):
        h = slice(start, start + HEAD_GROUP)
        gated, window = lizard_weights(p, h)
        output[:, h] = (gated + p["alpha"] * window) @ p["v"][:, h]
        weights = teacher_weights(q, k, h)
        y_true[:, h] = weights @ teacher_v[:, h]
        values, lizard = weight_metrics(weights, gated,
                                        window, p["alpha"], p["window"])
        for m in METRICS:
            sums[m].append(values[m][0])
        for head in range(start, min(start + HEAD_GROUP, n_heads)):
            if (layer, head) in maps:
                j = head - start
                crops[f"{layer}:{head}"] = {
                    "teacher": rounded(weights[0, j, :MAP_SIZE, :MAP_SIZE]),
                    "lizard": rounded(lizard[0, j, :MAP_SIZE, :MAP_SIZE])}
    reference = lz.gla(p["fq"], p["fk"], p["v"], p["gamma"])
    reference = reference + p["alpha"] * lz.awa(p["qr"], p["kr"], p["v"],
                                                p["meta"], p["window"])
    check = ((output - reference).norm() / reference.norm()).item()
    per_head = {m: torch.cat(v) for m, v in sums.items()}
    gamma = p["gamma"][0]
    grams, head_err, head_tsq = before_o_proj(teacher_v, y_true, p, output,
                                              x.shape[1])
    return dict(err=err, norm=norm, per_head=per_head, crops=crops,
                grams=grams, head_err=head_err, head_tsq=head_tsq,
                head_dim=q.shape[-1],
                check=check, gamma_mean=gamma.mean().item(),
                memory=(1 / (1 - gamma).clamp_min(1e-6)).median().item(),
                alpha=p["alpha"].item(), meta=p["meta"].mean().item())


def compute(cfg, checkpoint, label, stage, out, sequences, maps):
    from transformers import AutoTokenizer

    from evaluate import load
    from train import load_llama
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    teacher = load_llama(cfg).eval()
    model = load(cfg, "lizard", checkpoint)
    model_cfg = teacher.config
    chunks = held_out(tokenizer, cfg, sequences)
    length = chunks.shape[1]
    names = span_names(length)
    n_layers = len(teacher.model.layers)
    acc = [dict(err=[0.0] * len(names), norm=[0.0] * len(names),
                per_head=None, check=0.0, gamma_mean=0.0, memory=0.0,
                grams=0, head_err=0, head_tsq=0)
           for _ in range(n_layers)]
    crops = {}
    for s, ids in enumerate(chunks):
        found = capture(teacher, ids[None].to(cfg.device))
        with torch.no_grad():
            for i in range(n_layers):
                x, rope, target = found[i]
                stats = layer_stats(model.model.layers[i].self_attn,
                                    teacher.model.layers[i].self_attn, x,
                                    rope, target, model_cfg,
                                    maps if s == 0 else set(), i)
                a = acc[i]
                for j, sl in enumerate(span_slices(length)):
                    a["err"][j] += stats["err"][sl].sum().item()
                    a["norm"][j] += stats["norm"][sl].sum().item()
                spans = {m: [v[:, sl].mean(-1) for sl in span_slices(length)]
                         for m, v in stats["per_head"].items()}
                if a["per_head"] is None:
                    a["per_head"] = spans
                else:
                    for m in METRICS:
                        a["per_head"][m] = [x + y for x, y in
                                            zip(a["per_head"][m], spans[m])]
                for key in ("check", "gamma_mean", "memory", "grams",
                            "head_err", "head_tsq"):
                    a[key] += stats[key]
                a["alpha"], a["meta"] = stats["alpha"], stats["meta"]
                a["head_dim"] = stats["head_dim"]
                crops.update(stats["crops"])
        print(f"sequence {s + 1}/{len(chunks)} done")
    layers = []
    for i, a in enumerate(acc):
        layers.append({
            "layer": i,
            "relative_mse": dict(zip(names, (e / n for e, n in
                                             zip(a["err"], a["norm"])))),
            **{m: {name: [x / len(chunks) for x in v.tolist()]
                   for name, v in zip(names, a["per_head"][m])}
               for m in METRICS},
            "gate_mean": a["gamma_mean"] / len(chunks),
            "gate_memory": a["memory"] / len(chunks),
            "alpha": a["alpha"], "meta_mean": a["meta"],
            "weights_check": a["check"] / len(chunks)})
    result = {"label": label, "stage": stage, "config": cfg.name,
              "backend": cfg.lizard.backend, "checkpoint": str(checkpoint),
              "sequences": len(chunks), "seq_len": length,
              "spans": names, "layers": layers, "maps": crops}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result))
    for kind, data in thesis_results(acc, cfg, label, stage, checkpoint,
                                     len(chunks), length).items():
        path = out.parent / "thesis" / kind / out.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1))
        print(f"wrote {path}")
    worst = max(layer["weights_check"] for layer in layers)
    print(f"wrote {out}; largest weights check error {worst:.1e}")
    return result


def solve(gram):
    yy = gram[:, 0, 0]
    fits = {}
    for name, idx in SETS.items():
        a = gram[:, idx][:, :, idx]
        r = gram[:, idx, 0]
        c = (torch.linalg.pinv(a, hermitian=True) @ r[..., None])[..., 0]
        fits[name] = {"error": (yy - (c * r).sum(-1)).clamp_min(0),
                      "weights": c}
    trained = yy - 2 * gram[:, 0, 4] + gram[:, 4, 4]
    fits["trained"] = {"error": trained.clamp_min(0)}
    return fits, yy


def short(x):
    return float(f"{x:.6g}")


def thesis_results(acc, cfg, label, stage, checkpoint, sequences, length):
    names = bucket_names(length)
    spans = bounds(length)
    head = {"label": f"{label}, {stage.replace('stage', 'stage ')}",
            "stage": stage, "config": cfg.name,
            "backend": cfg.lizard.backend, "torch_dtype": cfg.dtype,
            "checkpoint": {"path": str(checkpoint)},
            "data": {"split": "held-out", "batches": sequences,
                     "tokens": sequences * length, "sequence_length": length,
                     "all_batches": True}}
    layer_mse, position, fits = [], [], []
    for i, a in enumerate(acc):
        err, tsq = a["head_err"], a["head_tsq"]
        heads_n = err.shape[1]
        size = torch.tensor([hi - lo for lo, hi in spans], dtype=torch.float64)
        dim = a["head_dim"]
        count = size * dim * sequences
        buckets = []
        for b, (lo, hi) in enumerate(spans):
            mse = err[b].sum().item() / (count[b].item() * heads_n)
            tms = tsq[b].sum().item() / (count[b].item() * heads_n)
            buckets.append({"start": lo, "end": hi, "mse": mse,
                            "target_mean_square": tms,
                            "relative_mse": mse / tms,
                            "mse_per_head": (err[b] / count[b]).tolist()})
        total = count.sum().item()
        mse = err.sum().item() / (total * heads_n)
        tms = tsq.sum().item() / (total * heads_n)
        layer = {"layer": i, "mse": mse, "target_mean_square": tms,
                 "relative_mse": mse / tms,
                 "mse_per_head": (err.sum(0) / total).tolist()}
        layer_mse.append(layer)
        position.append({**layer, "buckets": buckets})
        grams = torch.cat([a["grams"], a["grams"].sum(0, keepdim=True)])
        layer_fits = {}
        for name, gram in zip(names, grams):
            solved, yy = solve(gram)
            layer_fits[name] = {
                s: {"relative_error": (f["error"].sum() / yy.sum()).item(),
                    "per_head": [short(e) for e in (f["error"] / yy).tolist()],
                    **({"weights": [[short(c) for c in row]
                                    for row in f["weights"].tolist()]}
                       if "weights" in f else {})}
                for s, f in solved.items()}
        fits.append({"layer": i, "alpha": [short(a["alpha"])] * heads_n,
                     "check_relative_error": a["check"] / sequences,
                     "fits": layer_fits})
    factor = cfg.mse_factor
    loss = factor * sum(l["mse"] for l in layer_mse) / len(layer_mse)
    summary = [{"start": lo, "end": hi,
                "loss": factor * sum(p["buckets"][b]["mse"] for p in position)
                / len(position),
                "relative_mse": sum(p["buckets"][b]["relative_mse"]
                                    for p in position) / len(position)}
               for b, (lo, hi) in enumerate(spans)]
    columns = list(SETS) + ["trained"]
    return {
        "xai-layer-wise-mse": {**head, "mse_factor": factor, "loss": loss,
                               "stored_loss": None, "layers": layer_mse},
        "xai-position-mse": {**head, "mse_factor": factor, "loss": loss,
                             "stored_loss": None, "buckets": summary,
                             "layers": position},
        "xai-branch-fit": {
            **head, "candidates": CANDIDATES, "buckets": names,
            "summary": {n: {c: sum(f["fits"][n][c]["relative_error"]
                                   for f in fits) / len(fits)
                            for c in columns} for n in names},
            "max_check_relative_error": max(f["check_relative_error"]
                                            for f in fits),
            "layers": fits}}


def curves(runs, out, project):
    import wandb
    api = wandb.Api()
    found = {}
    for spec in runs:
        label, stage, name = spec.split(":", 2)
        path = f"{api.default_entity}/{project}"
        [run] = api.runs(path, filters={"display_name": name})
        key = f"{stage}/loss"
        rows = [r for r in run.scan_history(keys=["_step", key])
                if r.get(key) is not None]
        found.setdefault(stage, {})[label] = {
            "run": name, "runtime": run.summary.get("_runtime"),
            "loss": [r[key] for r in rows]}
        print(f"{label} {stage}: {len(rows)} steps from {name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(found))
    print(f"wrote {out}")


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)


def by_stage(results):
    stages = {}
    for r in results:
        stages.setdefault(r["stage"], []).append(r)
    return dict(sorted(stages.items()))


def figure(plt, panels, width=6.0, height=4.0):
    fig, axes = plt.subplots(1, panels, figsize=(width * panels, height),
                             squeeze=False, facecolor="#fcfcfb")
    for ax in axes[0]:
        ax.set_facecolor("#fcfcfb")
        style(ax)
    return fig, axes[0]


def save(fig, out, name):
    fig.tight_layout()
    path = out / f"{name}.png"
    fig.savefig(path, dpi=200)
    print(f"wrote {path}")


def mean(xs):
    return sum(xs) / len(xs)


def line_per_layer(plt, results, value, ylabel, title, out, name):
    stages = by_stage(results)
    fig, axes = figure(plt, len(stages))
    for ax, (stage, rs) in zip(axes, stages.items()):
        for r in rs:
            ys = [value(layer) for layer in r["layers"]]
            ax.plot(range(len(ys)), ys, marker="o", markersize=4,
                    linewidth=2, color=COLORS[r["label"]], label=r["label"])
        ax.set_title(f"{title}, {stage.replace('stage', 'stage ')}",
                     color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Layer", color=MUTED)
        ax.set_ylabel(ylabel, color=MUTED)
        ax.set_xticks(range(len(rs[0]["layers"])))
        ax.legend(frameon=False, fontsize=9)
    save(fig, out, name)


def bars_per_span(plt, results, value, ylabel, title, out, name):
    stages = by_stage(results)
    fig, axes = figure(plt, len(stages))
    for ax, (stage, rs) in zip(axes, stages.items()):
        names = rs[0]["spans"]
        width = 0.8 / len(rs)
        for j, r in enumerate(rs):
            ys = [value(r, s) for s in names]
            xs = [i + (j - (len(rs) - 1) / 2) * width for i in
                  range(len(names))]
            bars = ax.bar(xs, ys, width * 0.92, color=COLORS[r["label"]],
                          label=r["label"], zorder=2)
            ax.bar_label(bars, labels=[f"{y:.3g}" for y in ys], fontsize=7,
                         padding=2, color=INK)
        ax.set_xticks(range(len(names)), names)
        ax.set_xlabel("Query positions", color=MUTED)
        ax.set_ylabel(ylabel, color=MUTED)
        ax.set_title(f"{title}, {stage.replace('stage', 'stage ')}",
                     color=INK, fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=9)
        ax.margins(y=0.15)
    save(fig, out, name)


def plot_maps(plt, results, out):
    keys = sorted({k for r in results for k in r["maps"]},
                  key=lambda k: tuple(map(int, k.split(":"))))
    if not keys:
        return
    from matplotlib.colors import PowerNorm
    rows = [("Teacher", results[0], "teacher")]
    rows += [(f"{r['label']}, {r['stage'].replace('stage', 'stage ')}", r,
              "lizard") for r in results]
    fig, axes = plt.subplots(len(rows), len(keys),
                             figsize=(3.2 * len(keys), 3.0 * len(rows)),
                             squeeze=False, facecolor="#fcfcfb")
    for c, key in enumerate(keys):
        for r_i, (name, r, part) in enumerate(rows):
            ax = axes[r_i][c]
            ax.imshow(r["maps"][key][part], cmap="Blues",
                      norm=PowerNorm(0.4, vmin=0, vmax=1))
            ax.set_xticks([])
            ax.set_yticks([])
            if r_i == 0:
                layer, head = key.split(":")
                ax.set_title(f"Layer {layer}, head {head}", color=INK,
                             fontsize=10)
            if c == 0:
                ax.set_ylabel(name, color=INK, fontsize=9)
    fig.suptitle(f"Attention weights, first {MAP_SIZE} tokens", color=INK,
                 fontsize=11)
    save(fig, out, "attention_maps")


def plot_branch_fit(plt, paths, out):
    from matplotlib.colors import LogNorm
    found = [Path(p).parent / "thesis" / "xai-branch-fit" / Path(p).name
             for p in paths]
    found = [json.loads(f.read_text()) for f in found if f.exists()]
    if not found:
        return
    sets = ["G", "W", "WR", "G+W", "G+WR", "trained"]
    bucket = found[0]["buckets"][0]
    norm = LogNorm(vmin=1e-3, vmax=1.0, clip=True)
    fig, axes = plt.subplots(1, len(found), figsize=(3.4 * len(found), 5.2),
                             squeeze=False, facecolor="#fcfcfb")
    for ax, d in zip(axes[0], found):
        m = [[layer["fits"][bucket][s]["relative_error"] for s in sets]
             for layer in d["layers"]]
        ax.imshow(m, cmap="Blues", norm=norm, aspect="auto")
        for i, row in enumerate(m):
            for j, v in enumerate(row):
                ax.text(j, i, f"{v:.2g}" if v >= 1e-3 else "<.001",
                        ha="center", va="center", fontsize=6,
                        color="white" if norm(v) > 0.55 else INK)
        ax.set_xticks(range(len(sets)), sets, fontsize=8)
        ax.set_yticks(range(len(m)))
        ax.tick_params(length=0, labelsize=7, colors=MUTED)
        ax.set_title(d["label"], color=INK, fontsize=10)
    axes[0][0].set_ylabel("Layer", color=MUTED)
    fig.suptitle(f"Branch fit at query positions {bucket}: remaining error "
                 "/ ||y||²", color=INK, fontsize=11)
    save(fig, out, "branch_fit")


def plot_eval(plt, evals, out):
    from evaluate import PAPER
    series = [("Teacher", evals.get("Teacher"))]
    series += [(label, scores) for label, scores in evals.items()
               if label != "Teacher"]
    series.append(("Paper", PAPER["lizard"]))
    series = [(label, s) for label, s in series if s]
    fig, axes = figure(plt, 1, width=10.0, height=4.4)
    ax = axes[0]
    width = 0.8 / len(series)
    for j, (label, scores) in enumerate(series):
        xs = [i + (j - (len(series) - 1) / 2) * width for i in
              range(len(TASKS))]
        ys = [scores[t] for t in TASKS]
        color = COLORS.get(label.split(" ")[0], MUTED)
        bars = ax.bar(xs, ys, width * 0.92, color=color, zorder=2,
                      label="Lizard, paper" if label == "Paper" else label)
        ax.bar_label(bars, labels=[f"{y:.1f}" for y in ys], fontsize=6.5,
                     padding=2, color=INK, rotation=90)
    ax.set_xticks(range(len(TASKS)), TASK_NAMES)
    ax.set_ylabel("Accuracy (%)", color=MUTED)
    ax.set_ylim(0, 90)
    ax.set_title("Table 9 of the paper: teacher, PyTorch and Triton models "
                 "after stage 2", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=len(series), loc="upper left")
    save(fig, out, "evaluation")


def plot_curves(plt, data, out):
    fig, axes = figure(plt, len(data))
    for ax, (stage, runs) in zip(axes, sorted(data.items())):
        for label, run in runs.items():
            ax.plot(range(1, len(run["loss"]) + 1), run["loss"], linewidth=1.5,
                    color=COLORS[label], label=label)
        ax.set_yscale("log")
        ax.set_xlabel("Optimizer step", color=MUTED)
        ax.set_ylabel("Training loss", color=MUTED)
        ax.set_title(f"Training loss, {stage.replace('stage', 'stage ')}",
                     color=INK, fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=9)
    save(fig, out, "training_loss")


def table(results, evals, curve_data):
    lines = ["# XAI results", ""]
    lines += ["| Result | Stage | Checkpoint | Sequences "
              "| Relative MSE (all) |"
              " TV distance (all) | Window share (all) | Weights check |",
              "|---|---|---|---|---|---|---|---|"]
    for r in results:
        ls = r["layers"]
        lines.append(
            f"| {r['label']} | {r['stage']} | `{Path(r['checkpoint']).name}` "
            f"| {r['sequences']} "
            f"| {mean([l['relative_mse']['all'] for l in ls]):.4f} "
            f"| {mean([mean(l['tv']['all']) for l in ls]):.4f} "
            f"| {mean([mean(l['window_share']['all']) for l in ls]):.4f} "
            f"| {max(l['weights_check'] for l in ls):.1e} |")
    for r in results:
        lines += ["", f"## {r['label']}, {r['stage']}", "",
                  "| Layer | Relative MSE | TV distance | Teacher outside "
                  "the window | Lizard outside the window | Window share | "
                  "Mean gate | Gate memory (tokens) | α | Mean sink |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for l in r["layers"]:
            lines.append(
                f"| {l['layer']} | {l['relative_mse']['all']:.4f} "
                f"| {mean(l['tv']['all']):.4f} "
                f"| {mean(l['teacher_outside']['all']):.4f} "
                f"| {mean(l['lizard_outside']['all']):.4f} "
                f"| {mean(l['window_share']['all']):.4f} "
                f"| {l['gate_mean']:.4f} | {l['gate_memory']:.2f} "
                f"| {l['alpha']:.4f} | {l['meta_mean']:.4f} |")
    if evals:
        lines += ["", "## Table 9", "", "| Model | " + " | ".join(TASK_NAMES)
                  + " |", "|---" * (len(TASKS) + 1) + "|"]
        for label, scores in evals.items():
            lines.append(f"| {label} | " + " | ".join(
                f"{scores[t]:.1f}" for t in TASKS) + " |")
    if curve_data:
        lines += ["", "## Training runs", "",
                  "| Stage | Result | Run | Steps | Last loss "
                  "| Run time (s) |",
                  "|---|---|---|---|---|---|"]
        for stage, runs in sorted(curve_data.items()):
            for label, run in runs.items():
                lines.append(f"| {stage} | {label} | `{run['run']}` "
                             f"| {len(run['loss'])} | {run['loss'][-1]:.4f} "
                             f"| {run['runtime']} |")
    return "\n".join(lines) + "\n"


def plot(paths, evals, curve_path, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    results = [json.loads(Path(p).read_text()) for p in paths]
    results.sort(key=lambda r: (r["stage"], list(COLORS).index(r["label"])))
    line_per_layer(plt, results, lambda l: l["relative_mse"]["all"],
                   "MSE / mean square of the teacher output",
                   "Relative MSE of each layer", out, "layer_mse")
    bars_per_span(plt, results,
                  lambda r, s: mean([l["relative_mse"][s]
                                     for l in r["layers"]]),
                  "Relative MSE, mean over layers",
                  "Relative MSE by query position", out, "position_mse")
    line_per_layer(plt, results, lambda l: mean(l["tv"]["all"]),
                   "TV distance, mean over heads",
                   "Attention weights against the teacher", out,
                   "attention_tv")
    bars_per_span(plt, results,
                  lambda r, s: mean([mean(l["tv"][s]) for l in r["layers"]]),
                  "TV distance, mean over layers and heads",
                  "TV distance by query position", out, "position_tv")
    line_per_layer(plt, results, lambda l: mean(l["window_share"]["all"]),
                   "Share of the window branch", "Window branch share", out,
                   "window_share")
    line_per_layer(plt, results, lambda l: l["gate_memory"],
                   "Median 1 / (1 − γ) (tokens)", "Gate memory", out,
                   "gate_memory")
    plot_maps(plt, [r for r in results if r["stage"] == results[-1]["stage"]],
              out)
    plot_branch_fit(plt, sorted(paths), out)
    scores = {}
    for spec in evals:
        label, path = spec.split("=", 1)
        if not Path(path).exists():
            print(f"skipped {label}: {path} does not exist")
            continue
        scores[label] = json.loads(Path(path).read_text())["scores"]
    if scores:
        plot_eval(plt, scores, out)
    curve_data = json.loads(Path(curve_path).read_text()) if curve_path \
        else None
    if curve_data:
        plot_curves(plt, curve_data, out)
    (out / "xai.md").write_text(table(results, scores, curve_data))
    print(f"wrote {out / 'xai.md'}")


def parse_maps(text):
    return {tuple(int(x) for x in item.split(":")) for item in text}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAI of Lizard checkpoints "
                                     "against the teacher")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compute", help="metrics of one checkpoint")
    c.add_argument("config")
    c.add_argument("--checkpoint", required=True, type=Path)
    c.add_argument("--label", required=True, choices=["PyTorch", "Triton"])
    c.add_argument("--stage", required=True, choices=["stage1", "stage2"])
    c.add_argument("--out", required=True, type=Path)
    c.add_argument("--sequences", type=int, default=16)
    c.add_argument("--maps", nargs="*", default=["2:0", "8:5", "15:14"],
                   help="layer:head pairs for the attention maps")
    w = sub.add_parser("curves", help="training losses from wandb")
    w.add_argument("out", type=Path)
    w.add_argument("--runs", nargs="+", required=True,
                   help="LABEL:STAGE:RUN_NAME, e.g. PyTorch:stage1:<run>")
    w.add_argument("--project", default="lizard-1b")
    p = sub.add_parser("plot", help="figures and a table of the results")
    p.add_argument("results", nargs="+")
    p.add_argument("--eval", nargs="*", default=[],
                   help="LABEL=PATH of evaluate.py JSON files")
    p.add_argument("--curves", help="JSON written by the curves command")
    p.add_argument("--out", type=Path, default=Path("xai"))
    args = parser.parse_args()
    if args.command == "compute":
        compute(config.load(args.config), args.checkpoint, args.label,
                args.stage, args.out, args.sequences, parse_maps(args.maps))
    elif args.command == "curves":
        curves(args.runs, args.out, args.project)
    else:
        plot(args.results, args.eval, args.curves, args.out)
