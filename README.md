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
make xai             # XAI figures of the PyTorch and the Triton checkpoints against the teacher
make evaluate        # paper metrics (Table 9) for the trained Lizard model
make evaluate-llama  # the same metrics for the original model
make mmlu            # only the full MMLU 5-shot, with the counts of the answer letters
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

#### XAI figures

`src/xai.py` compares trained checkpoints with the teacher, for the thesis. Every figure shows the PyTorch run and the Triton run side by side, with one panel for each stage. It needs a GPU (the Triton model runs its kernels) and matplotlib. No training.

```bash
make xai-curves       # optional: training losses from wandb, before make xai
make xai              # 4 x compute (PyTorch and Triton, stage 1 and stage 2), then plot
make xai XAI_SEQUENCES=4 XAI_DIR=xai-4   # fewer held-out sequences (default 16), another folder
```

- `compute` loads the teacher and one checkpoint (downloaded by name from `hf_repo` if it is not in `checkpoints/`). It runs held-out Alpaca sequences: the examples after the first `num_examples` of the training shuffle, packed to `seq_len`. Each Lizard layer gets the input of the same teacher layer. For each layer it saves:
  - the relative MSE of the layer output against the teacher output (the stage 1 target), for query positions 0-127, 128-511, 512-1023, 1024-2047 and all;
  - for each head, the attention weights against the teacher. The Lizard weights are `A_gla + alpha * A_window`, with each row normalized. The metrics are the total variation (TV) distance, the mass outside the window for the teacher and for Lizard, and the share of the window branch;
  - the mean gate, the median gate memory `1 / (1 - gamma)` in tokens, `alpha` and the mean sink logit;
  - the first 256 x 256 attention weights of the teacher and of Lizard for `--maps` (default `2:0 8:5 15:14`, layer:head).

  The weights are rebuilt with the PyTorch functions. A check compares `A @ v` with the layer output (`weights_check`, near 1e-6).
- `compute` also writes the three JSON formats of the XAI scripts of lizard-in-lolcats to `XAI_DIR/thesis/` (`xai-layer-wise-mse/`, `xai-position-mse/`, `xai-branch-fit/`), with the same definitions. The thesis plots read these files, so its figures can show the LoLCATs runs and these runs side by side:
  - the MSE of the attention output before `o_proj`, against the teacher softmax attention with RoPE, for each layer and head, also for the query positions 0-127, 128-511 and 512-2047. 1000 x the mean over the layers is the `loss` (the scale of the lolcats stage 1 loss);
  - the branch fit: for each head and position range, the least-squares weights of the gated branch G, the window branch W (without alpha) and an oracle WR (softmax with RoPE over the window, without sinks) that best give the teacher output, and the remaining error. `trained` is the error of the checkpoint output itself.

  The default of 16 sequences of 2048 tokens (32,768 tokens) is the size of the lolcats validation data of these scripts. The data are not the same: lolcats used its validation split, this script the held-out examples.
- `weights` (`make xai-weights`) writes the attention weight crops of one checkpoint in the format of `scripts/attention_weights.py` of lizard-in-lolcats, to `XAI_DIR/weights/`. It uses the same sample: the first 200 examples of Alpaca-cleaned with the LoLCATs prompt template, packed to 1024 tokens. Each crop is the last 32 queries against the first and the last 32 keys, for layers 0, 4, 8, 12 and 15 and heads 0-3, for the teacher, Lizard (`A_gla + alpha * A_window`) and the gated branch. Thus one figure can show the teacher and the checkpoints of both pipelines on the same text.
- `curves` reads the stage 1 and stage 2 losses of the four runs from wandb.
- `plot` writes these files to `XAI_DIR` (default `xai/`):

| File | Contents |
|---|---|
| `layer_mse.png` | Relative MSE of each layer |
| `position_mse.png` | Relative MSE by query position, mean over layers |
| `attention_tv.png` | TV distance of the attention weights to the teacher, for each layer |
| `position_tv.png` | TV distance by query position |
| `window_share.png` | Share of the window branch in the Lizard weights, for each layer |
| `gate_memory.png` | Median gate memory of each layer, in tokens |
| `attention_maps.png` | Attention weights of the teacher and of the stage 2 models |
| `branch_fit.png` | Branch fit at positions 0-127, one panel for each checkpoint |
| `evaluation.png` | Table 9: teacher, PyTorch and Triton models, and Lizard in the paper (from the `make evaluate` JSON files) |
| `training_loss.png` | Training loss of both stages (after `make xai-curves`) |
| `xai.md` | All numbers as Markdown tables |

The make variables `TORCH_STAGE1`, `TORCH_STAGE2`, `TRITON_STAGE1` and `TRITON_STAGE2` name the runs. Their defaults are the runs of the thesis. The two stage 2 checkpoints need approximately 5.2 GB of disk space.

#### Checks before training

`make train` first runs `make check` (`src/check.py`), and stops if one check fails. The checks are:

- a Hugging Face login;
- read access to `model_name` (Llama is gated) and to `dataset_name`;
- write access to `hf_repo`, which is created as a private repo if it does not exist;
- a wandb login. This check is skipped when `WANDB_MODE` is `disabled` or `offline`;
- Triton and a CUDA GPU, only when the config has `backend: triton`.

#### Checkpoints

Each run gets the name of its config file and the start time, for example `lizard-1b-20261024-203343`. At the start, `src/train.py` saves a copy of the config as `<run>.yaml`. After each stage, it saves `<run>-stage1.pt` and `<run>-stage2.pt`. It uploads all three files to the private Hugging Face repo `hf_repo` and keeps them in `out_dir`. If a checkpoint cannot be written, for example because the disk is full, `src/train.py` deletes the partial file and uploads the checkpoint from memory. Thus the trained weights are not lost. A stage 2 checkpoint is approximately 2.6 GB, so check the free space (`df -h`) before a run, and delete local checkpoints after their upload. The wandb run has the same name. `STAGE=stage2` trains only stage 2 from the stage 1 checkpoint in `stage1_ckpt`. `make evaluate` loads the newest `<config name>-*-stage2.pt` in `out_dir`.

#### Start stage 2 from a checkpoint

Pass a stage 1 checkpoint and train only stage 2:

```bash
make train STAGE=stage2 STAGE1_CKPT=checkpoints/lizard-1b-20261024-203343-stage1.pt
```

Without make: `python src/train.py configs/lizard-1b.yaml --stage stage2 --stage1-ckpt <path>`. You can also set `stage1_ckpt` in the config, and `STAGE1_CKPT` overrides it. If the file is not on this machine, it is downloaded by its name from `hf_repo` into `out_dir`. Thus the name of a checkpoint of an earlier run, for example `lizard-1b-20261024-203343-stage1.pt`, is enough. Stage 2 gets a new run name, and its checkpoint is saved and uploaded as usual.

#### Evaluation

`src/evaluate.py` runs LM Evaluation Harness with the metrics of Table 9 of the paper: PIQA (acc), ARC-Easy (acc), ARC-Challenge (acc_norm), HellaSwag (acc_norm) and WinoGrande (acc) zero-shot, and MMLU 5-shot. As in lolcats, MMLU is the unweighted mean over its 57 subjects, the prompts have no BOS token, and Avg. is the mean of the five zero-shot tasks. It prints the scores next to the paper values and saves them to `<out_dir>/<config name>-eval-<model>-all.json`. Set `eval_limit` in the config for a quick run on a few questions per task. `make mmlu-fast` runs only MMLU 5-shot on the first 5 test questions of each of the 57 subjects (285 questions), as `./eval.sh --limit 5` in lolcats, and saves `<config name>-eval-<model>-mmlu-fast.json`. `make mmlu` runs only the full MMLU (`--tasks mmlu`). Each JSON also has `mmlu_letters`: how often the model chose A, B, C and D (`pred`), and how often each letter is the right answer (`gold`).

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
- `src/xai.py`: XAI figures of checkpoints against the teacher
- `src/check.py`: checks of Hugging Face and wandb before training
- `src/train.py`: stage 1 and stage 2 training
- `src/evaluate.py`: lm-eval benchmarks of the paper
- `tests/test_lizard.py`: tests against slow loop versions of each formula
- `tests/test_evaluate.py`: tests of the scoring rules
- `tests/test_config.py`: tests of the config loader
- `tests/test_check.py`: tests of the checks before training
- `tests/test_xai.py`: the XAI metrics and figures with a tiny model
- `tests/test_kernels.py`: the Triton kernels against the PyTorch implementation
