Lizard Reproduction
---

Turns Llama-3.2-1B into a [Lizard](https://arxiv.org/abs/2507.09025) model: gated linear attention plus a sliding-window softmax attention with sink tokens and RoPE in the window. Training is a two-stage distillation from the original model. The weights stay in bfloat16.

### Setup

Create and activate the conda environment:

```bash
make env
conda activate distill_lizard_llama_3_2_1B
```

Llama-3.2-1B needs access on Hugging Face (`hf auth login`). Training logs to wandb (`wandb login`). To run without wandb, set `WANDB_MODE=disabled`.

### Usage

```bash
make test            # tests on CPU with a tiny model
make test-kernels    # only the checks of the Triton kernels against PyTorch
make check           # check Hugging Face and wandb access (make train runs it first)
make train           # stage 1 and stage 2, needs a GPU, saves and uploads each stage
make train STAGE=stage1  # only stage 1 (STAGE=stage2 for only stage 2)
make train-kernel    # the same training with the Triton kernels (KERNEL_CONFIG)
make benchmark       # latency of the GLA kernel against flash-linear-attention, needs a GPU
make evaluate        # paper metrics (Table 9) for the trained Lizard model
make evaluate-llama  # the same metrics for the original model
make mmlu-fast       # quick MMLU 5-shot, 5 questions per subject (MODEL=llama for the original model)
make mmlu-fast CKPT=checkpoints/<run>-stage1.pt  # the same on a given checkpoint
make push            # upload the code and the config to Hugging Face
```

Every command uses `CONFIG=configs/lizard-1b.yaml` by default. Pass another config with `make train CONFIG=configs/<name>.yaml`.

#### Configs

Each experiment is one YAML file in `configs/`. To add one, copy `configs/lizard-1b.yaml`, give it a new name and change the values. `src/config.py` loads a file into a typed `Config` dataclass, with the attention shape in a nested `Lizard` dataclass. A missing or unknown key is an error. Without make: `python src/train.py configs/<name>.yaml --stage stage1`. Each script prints its arguments with `--help`.

#### Triton kernels

`src/kernels/` has custom Triton kernels for the two branches of Lizard attention, with a forward and a backward pass in bfloat16. They load bf16 tensors, compute the dot products on tensor cores, accumulate in float32 and return bf16 outputs. They never build the L x L matrices.

- `kernels.gla(fq, fk, v, log_gamma)`: the gated branch, with the hardware-aware algorithm of Section 4 of the paper. The paper absorbs the cumulative log gate `log C` into the features, `Q~ = phi(Q) exp(log C)` and `K~ = phi(K) exp(-log C)`, so that the gated attention becomes a plain GEMM in bf16. Over a whole sequence, `exp(-log C_t)` overflows: after 2048 tokens with a gate of 0.5, `-log C_t` is about 1400, and float32 and bf16 overflow above 88. Thus the kernel splits the sequence into chunks of 64 tokens and measures `log C` from the chunk boundary. Each absorbed factor is then at most 1:
  - between chunks: `Q~_i = phi(q_i) exp(G_i - G_start)` and `K~_t = phi(k_t) exp(G_end - G_t)`. The kernel computes a state `K~^T V` for each chunk and multiplies it with `Q~`. Both are bf16 GEMMs on tensor cores, with no float32 fallback.
  - inside a chunk: `(phi(Q) phi(K)^T) * exp(G_i - G_t)` with the causal mask. Here one factor of the absorbed product would grow with the gates of the chunk, so the kernel applies the decay to the 64 x 64 scores after the GEMM.

  The work and the memory grow linearly with the sequence length. The gates can be one per token (`[B, L]`, as in the model) or one per head (`[B, H, L]`).
- `kernels.swa(q, k, v, meta, window)`: the window branch with sinks. It keeps a running maximum and sum over the window, and starts them with the sinks.

A config selects the attention with `lizard.backend`: `torch` (the PyTorch code of `src/lizard_attention.py`) or `triton`. The `-triton` configs are copies of the other configs with `backend: triton`:

```bash
make train-kernel STAGE=stage1                                    # configs/lizard-1b-mse1000-triton.yaml
make train-kernel KERNEL_CONFIG=configs/lizard-1b-triton.yaml
make mmlu-fast CONFIG=configs/lizard-1b-mse1000-triton.yaml       # evaluation with the kernels
```

The Triton backend needs a CUDA GPU. `make check` and `make train-kernel` check that Triton imports and that a GPU is there. `make test-kernels` compares the forward and backward pass of each kernel, and of a whole layer, with the PyTorch implementation. Without a GPU, the tests run the kernels in the Triton interpreter (`TRITON_INTERPRET=1`, set by `tests/conftest.py`), which computes the dot products in float32 because numpy has no bf16.

#### Kernel benchmark

`src/benchmark.py` measures the latency of the GLA kernel, as Table 3 and Figure 5 of the paper. It uses random inputs, so it needs no model and no training. It needs a CUDA GPU and the kernels of [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (`fla-core` has only the kernels; the full `flash-linear-attention` package also registers its models with transformers). `--plot` needs matplotlib.

```bash
pip install fla-core matplotlib
make benchmark                                                # Table 3: forward pass, 32 heads, head dimension 128
python src/benchmark.py --shapes 16x2048 --plot figure5.png   # Figure 5
python src/benchmark.py --backward                            # forward and backward pass
python src/benchmark.py --shapes 1x2048 --heads 32 --dim 64   # the shape of the training
```

The columns are:

- `GLA`: `chunk_gla` of flash-linear-attention, the baseline of the paper. It has a gate for each key dimension.
- `Simple GLA`: `chunk_simple_gla` of flash-linear-attention. Like Lizard, it has one gate for each head.
- `Lizard`: `kernels.gla`. It also computes the row sums that normalize the output.
- `Speedup`: the GLA time divided by the Lizard time, minus 1, as in the paper.
- `Paper`: the speedup in Table 3 of the paper, on an A100-80GB GPU.

The times are medians of `triton.testing.do_bench`. A shape that does not fit in the GPU memory shows `OOM`.

#### Checks before training

`make train` first runs `make check` (`src/check.py`), and stops if one check fails. The checks are:

- a Hugging Face login;
- read access to `model_name` (Llama is gated) and to `dataset_name`;
- write access to `hf_repo`, which is created as a private repo if it does not exist;
- a wandb login. This check is skipped when `WANDB_MODE` is `disabled` or `offline`;
- Triton and a CUDA GPU, only when the config has `backend: triton`.

#### Checkpoints

Each run gets the name of its config file and the start time, for example `lizard-1b-20261024-203343`. At the start, `src/train.py` saves a copy of the config as `<run>.yaml`. After each stage, it saves `<run>-stage1.pt` and `<run>-stage2.pt`. It uploads all three files to the private Hugging Face repo `hf_repo` and keeps them in `out_dir`. The wandb run has the same name. `STAGE=stage2` trains only stage 2 from the stage 1 checkpoint in `stage1_ckpt`. `make evaluate` loads the newest `<config name>-*-stage2.pt` in `out_dir`.

#### Start stage 2 from a checkpoint

Pass a stage 1 checkpoint and train only stage 2:

```bash
make train STAGE=stage2 STAGE1_CKPT=checkpoints/lizard-1b-20261024-203343-stage1.pt
```

Without make: `python src/train.py configs/lizard-1b.yaml --stage stage2 --stage1-ckpt <path>`. You can also set `stage1_ckpt` in the config, and `STAGE1_CKPT` overrides it. If the file is not on this machine, it is downloaded by its name from `hf_repo` into `out_dir`. Thus the name of a checkpoint of an earlier run, for example `lizard-1b-20261024-203343-stage1.pt`, is enough. Stage 2 gets a new run name, and its checkpoint is saved and uploaded as usual.

#### Evaluation

`src/evaluate.py` runs LM Evaluation Harness with the metrics of Table 9 of the paper: PIQA (acc), ARC-Easy (acc), ARC-Challenge (acc_norm), HellaSwag (acc_norm) and WinoGrande (acc) zero-shot, and MMLU 5-shot. As in lolcats, MMLU is the unweighted mean over its 57 subjects, the prompts have no BOS token, and Avg. is the mean of the five zero-shot tasks. It prints the scores next to the paper values and saves them to `<out_dir>/<config name>-eval-<model>-all.json`. Set `eval_limit` in the config for a quick run on a few questions per task. `make mmlu-fast` runs only MMLU 5-shot on the first 5 test questions of each of the 57 subjects (285 questions), as `./eval.sh --limit 5` in lolcats, and saves `<config name>-eval-<model>-mmlu-fast.json`.

#### Evaluate a given checkpoint

`CKPT` evaluates one checkpoint, of stage 1 or of stage 2, in place of the newest stage 2 checkpoint:

```bash
make mmlu-fast CONFIG=configs/lizard-1b-mse1000.yaml CKPT=checkpoints/lizard-1b-mse1000-20261024-232317-stage1.pt
make evaluate CKPT=checkpoints/<run>-stage2.pt
```

A stage 1 checkpoint has only the Lizard parameters. They go into the original model, with the q, k, v and o weights of the teacher and no LoRA. Use the config of the run, because it sets the window, the sinks and the feature dimension. If the file is not on this machine, it is downloaded by its name from `hf_repo`. The results are saved as `<checkpoint name>-eval-lizard-<tasks>.json`. Without make: `python src/evaluate.py <config> --tasks mmlu-fast --checkpoint <path>`.

#### Layout

- `configs/`: one YAML file for each experiment
  - `lizard-1b.yaml`: the default, with `mse_factor: 1`
  - `lizard-1b-mse1000.yaml`: the same with `mse_factor: 1000`, the scale of lolcats
  - `lizard-1b-triton.yaml`, `lizard-1b-mse1000-triton.yaml`: the same with the Triton kernels
- `src/config.py`: the `Config` dataclass and its YAML loader
- `src/lizard_attention.py`: the Lizard attention layer
- `src/kernels/`: Triton kernels of the gated branch (`gla.py`) and the window branch (`swa.py`)
- `src/benchmark.py`: latency of the GLA kernel against flash-linear-attention
- `src/check.py`: checks of Hugging Face and wandb before training
- `src/train.py`: stage 1 and stage 2 training
- `src/evaluate.py`: lm-eval benchmarks of the paper
- `tests/test_lizard.py`: tests against slow loop versions of each formula
- `tests/test_evaluate.py`: tests of the scoring rules
- `tests/test_config.py`: tests of the config loader
- `tests/test_check.py`: tests of the checks before training
- `tests/test_kernels.py`: the Triton kernels against the PyTorch implementation
