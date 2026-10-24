# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Thesis codebase implementing "Lizard" linearization (arXiv:2507.09025) applied to Llama-3.2-1B: replacing softmax attention with gated linear attention (GLA) plus a sliding-window softmax branch with sink tokens and RoPE (AWA), trained via two-stage distillation from the original model. No package/build system — standalone scripts in `src/` run inside the conda env in `envs/`. Setup and run steps for humans are in [README.md](README.md).

## Commands

```bash
make env                 # conda env create -f envs/distill_lizard_llama_3_2_1B.yaml
conda activate distill_lizard_llama_3_2_1B

make test                # PYTHONPATH=src pytest tests; CPU-only, tiny models
PYTHONPATH=src pytest tests/test_lizard.py::test_gla_matches_formula_loops
LIZARD_TEST_MODEL=meta-llama/Llama-3.2-1B PYTHONPATH=src pytest tests/test_lizard.py::test_whole_model_logits_unchanged_after_swap

make train               # stage 1 + stage 2, needs CUDA; WANDB_MODE=disabled to skip wandb
make train STAGE=stage1   # or stage2 (needs C.STAGE1_CKPT) or both (default); = python src/train.py <stage>
make evaluate            # lm-eval on the trained Lizard checkpoint
make evaluate-llama      # same tasks on the unmodified teacher
make push                # upload src/ to the Hugging Face repo in config
```

Run everything from the repo root: `checkpoints/` is relative to the working directory. All hyperparameters and paths live in [src/config.py](src/config.py); scripts take no flags except one positional choice (`train.py`: `stage1`/`stage2`/`both`; `evaluate.py`: `lizard`/`llama`). Code must read config as `C.NAME` at call time, not `from config import NAME`, so tests can `monkeypatch.setattr(C, ...)`.

## Architecture

- **`src/lizard_attention.py`** — pure-PyTorch, dense (materializes L×L) implementation. `hedgehog`, `gla`, `awa`, `sink_softmax` are free functions so tests can call them directly; `LizardAttention` is a drop-in for `LlamaAttention` (`forward(hidden_states, position_embeddings)` returns `(out, None)`, RoPE on the window branch only, no KV cache so generation needs `use_cache=False`). Weights stay in `C.DTYPE` (bfloat16); the Lizard math runs in at least float32. `from_llama` builds one from a Llama attention layer by copying q/k/v/o weights.
- **`src/train.py`**:
  - Stage 1 builds standalone student `LizardAttention` layers (not inserted into the model), captures each teacher attention layer's (input, RoPE cos/sin, output) with forward hooks, and trains only Lizard params with a per-layer summed-squared-error loss, backpropagating layer by layer. Returns a state dict keyed like `model.layers.{i}.self_attn.*`.
  - Stage 2 swaps Lizard into a fresh model (`to_lizard`), loads the stage 1 params, wraps with LoRA on q/k/v, re-enables Lizard params (PEFT freezes them), trains with causal-LM loss, then `merge_and_unload()` so the saved checkpoint loads with `strict=True`.
  - `main` names the run `run_name()` (Lizard config, dtype, start time; also the wandb name), and `save` writes `C.OUT_DIR/<run>-stage1.pt` and `<run>-stage2.pt` and uploads each with `push_to_hub.upload` to `C.HF_REPO` right after its stage. `main(stage)` runs `stage1`, `stage2` or `both`. `stage2` alone loads its stage 1 params from `C.STAGE1_CKPT` (default `None`, asserted to be set).
  - `is_lizard` matches parameter names by exact dotted component against `C.LIZARD_PARAMS` — the single source of truth for which params are Lizard-owned.
- **`src/evaluate.py`** — wraps the model in lm-eval's `HFLM` (`add_bos_token=False`, as the lolcats harness) and loads the newest `C.OUT_DIR/*-stage2.pt` with `strict=True`. Runs the paper's Table 9: PIQA/ARC-e/WinoGrande `acc`, ARC-c/HellaSwag `acc_norm` zero-shot, MMLU 5-shot. `summarize` scores MMLU as the unweighted mean over the 57 leaf subjects (lolcats' definition, not lm-eval's size-weighted group score) and Avg. over the five zero-shot tasks only, as the paper does. `PAPER` holds the paper's teacher and Lizard values. `C.EVAL_LIMIT` limits questions per task. Right padding in lm-eval's loglikelihood batches keeps Lizard (which ignores the padding mask) correct for batch sizes above 1.
- **`tests/test_evaluate.py`** — tests of `summarize` and of the `PAPER` averages.
- **`tests/test_lizard.py`** — slow loop/recurrent versions of every formula (`gla_loop`, `gla_recurrent`, `awa_loop`, `rope_loop`, `lizard_loop`) plus tiny-config helpers, and the tests. The loops are the ground truth; tests assert `rel_err < 1e-10` in float64 against them.
