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
make check           # check Hugging Face and wandb access (make train runs it first)
make train           # stage 1 and stage 2, needs a GPU, saves and uploads each stage
make train STAGE=stage1  # only stage 1 (STAGE=stage2 for only stage 2)
make evaluate        # paper metrics (Table 9) for the trained Lizard model
make evaluate-llama  # the same metrics for the original model
make mmlu-fast       # quick MMLU 5-shot, 5 questions per subject (MODEL=llama for the original model)
make push            # upload the code and the config to Hugging Face
```

Every command uses `CONFIG=configs/lizard-1b.yaml` by default. Pass another config with `make train CONFIG=configs/<name>.yaml`.

#### Configs

Each experiment is one YAML file in `configs/`. To add one, copy `configs/lizard-1b.yaml`, give it a new name and change the values. `src/config.py` loads a file into a typed `Config` dataclass, with the attention shape in a nested `Lizard` dataclass. A missing or unknown key is an error. Without make: `python src/train.py configs/<name>.yaml --stage stage1`. Each script prints its arguments with `--help`.

#### Checks before training

`make train` first runs `make check` (`src/check.py`), and stops if one check fails. The checks are:

- a Hugging Face login;
- read access to `model_name` (Llama is gated) and to `dataset_name`;
- write access to `hf_repo`, which is created as a private repo if it does not exist;
- a wandb login. This check is skipped when `WANDB_MODE` is `disabled` or `offline`.

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

#### Layout

- `configs/`: one YAML file for each experiment
  - `lizard-1b.yaml`: the default, with `mse_factor: 1`
  - `lizard-1b-mse1000.yaml`: the same with `mse_factor: 1000`, the scale of lolcats
- `src/config.py`: the `Config` dataclass and its YAML loader
- `src/lizard_attention.py`: the Lizard attention layer
- `src/check.py`: checks of Hugging Face and wandb before training
- `src/train.py`: stage 1 and stage 2 training
- `src/evaluate.py`: lm-eval benchmarks of the paper
- `tests/test_lizard.py`: tests against slow loop versions of each formula
- `tests/test_evaluate.py`: tests of the scoring rules
- `tests/test_config.py`: tests of the config loader
- `tests/test_check.py`: tests of the checks before training
