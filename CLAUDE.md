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

make check               # Hugging Face and wandb access; make train depends on it
make train               # stage 1 + stage 2, needs CUDA; WANDB_MODE=disabled to skip wandb
make train STAGE=stage1   # or stage2 or both (default)
make train STAGE=stage2 STAGE1_CKPT=<path or name>   # stage 2 from a stage 1 checkpoint
make train CONFIG=configs/<name>.yaml   # any config; default configs/lizard-1b.yaml
make evaluate            # lm-eval on the trained Lizard checkpoint
make evaluate-llama      # same tasks on the unmodified teacher
make mmlu-fast           # MMLU 5-shot, 5 questions per subject (285); MODEL=llama for the teacher
make push                # upload src/ and the config to the Hugging Face repo in config
```

Run everything from the repo root: `out_dir` (default `checkpoints/`) is relative to the working directory. All hyperparameters and paths live in a YAML file in `configs/`, one file per experiment. Every script parses its arguments with `argparse`: the config path is positional, choices are options with `choices=` (so argparse rejects bad values): `python src/train.py <config> [--stage stage1|stage2|both] [--stage1-ckpt PATH]`, `python src/evaluate.py <config> [--model lizard|llama] [--tasks all|mmlu-fast]`, `python src/push_to_hub.py <config> [files ...]`, `python src/check.py <config>`. `main` functions take plain values, so tests call them directly.

## Config

[src/config.py](src/config.py) defines frozen dataclasses `Config` (every key of the YAML, plus `name` = file stem and `path`) and `Lizard` (`window`, `num_meta`, `feature_dim`, passed to `LizardAttention` with `asdict`). `load(path)` reads the YAML, casts the float fields (PyYAML reads `1e-3` as a string), and builds the dataclasses, so missing or unknown keys raise `TypeError`. `cfg.torch_dtype` maps `dtype` to the torch dtype. `LIZARD_PARAMS` stays a module constant, not a config value. Code gets the `Config` object as a `cfg` argument; there is no global config.

## Architecture

- **`src/lizard_attention.py`** — pure-PyTorch, dense (materializes L×L) implementation. `hedgehog`, `gla`, `awa`, `sink_softmax` are free functions so tests can call them directly; `LizardAttention` is a drop-in for `LlamaAttention` (`forward(hidden_states, position_embeddings)` returns `(out, None)`, RoPE on the window branch only, no KV cache so generation needs `use_cache=False`). Weights stay in `cfg.dtype` (bfloat16); the Lizard math runs in at least float32. `from_llama` builds one from a Llama attention layer by copying q/k/v/o weights.
- **`src/check.py`** — preflight before training: `CHECKS` is a list of `check_*(api, cfg)` functions (HF login, `auth_check` read on `model_name` and `dataset_name`, `create_repo` + `auth_check(write=True)` on `hf_repo`, wandb `Api().viewer`, skipped when `WANDB_MODE` is `disabled`/`offline`). `main` prints `[ok]`/`[fail]` for each and returns the number of failures; the exit code is 1 if any failed, so `make train: check` stops.
- **`src/train.py`**:
  - Stage 1 builds standalone student `LizardAttention` layers (not inserted into the model), captures each teacher attention layer's (input, RoPE cos/sin, output) with forward hooks, and trains only Lizard params with a per-layer summed-squared-error loss, backpropagating layer by layer. Returns a state dict keyed like `model.layers.{i}.self_attn.*`.
  - Stage 2 swaps Lizard into a fresh model (`to_lizard`), loads the stage 1 params, wraps with LoRA on q/k/v, re-enables Lizard params (PEFT freezes them), trains with causal-LM loss, then `merge_and_unload()` so the saved checkpoint loads with `strict=True`.
  - `main(cfg, stage)` names the run `<cfg.name>-<start time>` (also the wandb name), copies the YAML to `cfg.out_dir/<run>.yaml`, and `save` writes `<run>-stage1.pt` and `<run>-stage2.pt` right after each stage. Each of the three files goes to `cfg.hf_repo` with `push_to_hub.upload`. `stage` is `stage1`, `stage2` or `both`. `stage2` alone loads its stage 1 params with `load_stage1(cfg)` from `cfg.stage1_ckpt` (asserted to be set). `--stage1-ckpt` overrides the YAML value with `dataclasses.replace`. If the file does not exist locally, `load_stage1` downloads `cfg.hf_repo/<file name>` into `cfg.out_dir` with `hf_hub_download`.
  - `train` prints the step count first (`len(loader) // grad_accum` steps per epoch × `epochs`) and shows a tqdm bar over optimizer steps, refreshed every second on a terminal and every 30 s otherwise (log files). `log` returns the bar postfix (loss, lr, grad norm, mean `alpha_blend`, mean `meta_tokens`) and sends the same values to wandb.
  - `is_lizard` matches parameter names by exact dotted component against `config.LIZARD_PARAMS` — the single source of truth for which params are Lizard-owned.
- **`src/evaluate.py`** — wraps the model in lm-eval's `HFLM` (`add_bos_token=False`, as the lolcats harness) and loads the newest `cfg.out_dir/<cfg.name>-*-stage2.pt` with `strict=True`. Runs the paper's Table 9: PIQA/ARC-e/WinoGrande `acc`, ARC-c/HellaSwag `acc_norm` zero-shot, MMLU 5-shot. `summarize` scores MMLU as the unweighted mean over the 57 leaf subjects (lolcats' definition, not lm-eval's size-weighted group score) and Avg. over the five zero-shot tasks only, as the paper does. `PAPER` holds the paper's teacher and Lizard values. `cfg.eval_limit` limits questions per task. The `mmlu-fast` mode runs only `run_mmlu(lm, FAST_MMLU_LIMIT=5)`: lm-eval applies `limit` to each of the 57 subject tasks, so it scores 285 questions, as lolcats' `eval.sh --limit 5` (lm-eval 0.4 takes the first 5 test questions; lolcats' old harness took the first 5 after a seed-42 shuffle). Results go to `<cfg.name>-eval-<model>-<all|mmlu-fast>.json`. Right padding in lm-eval's loglikelihood batches keeps Lizard (which ignores the padding mask) correct for batch sizes above 1.
- **`tests/test_evaluate.py`** — tests of `summarize` and of the `PAPER` averages.
- **`tests/test_check.py`** — the checks with fake HF and wandb clients: all pass, and failures are counted.
- **`tests/test_config.py`** — the default YAML loads with the right types, and an unknown key fails.
- **`tests/test_lizard.py`** — slow loop/recurrent versions of every formula (`gla_loop`, `gla_recurrent`, `awa_loop`, `rope_loop`, `lizard_loop`) plus tiny-config helpers, and the tests. The loops are the ground truth; tests assert `rel_err < 1e-10` in float64 against them.
