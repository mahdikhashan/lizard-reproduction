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
make train           # stage 1 and stage 2, needs a GPU, saves and uploads each stage
make train STAGE=stage1  # only stage 1 (STAGE=stage2 for only stage 2)
make evaluate        # paper metrics (Table 9) for the trained Lizard model
make evaluate-llama  # the same metrics for the original model
make push            # upload the code to Hugging Face
```

All settings are in `src/config.py`.

#### Checkpoints

Each run gets a name from the Lizard config, the dtype and the start time, for example `lizard-window128-num_meta4-feature_dim128-bfloat16-20261024-203343`. After each stage, `src/train.py` saves `checkpoints/<run>-stage1.pt` and `checkpoints/<run>-stage2.pt` and uploads them to the private Hugging Face repo `HF_REPO`. The wandb run has the same name. `STAGE=stage2` trains only stage 2 from the stage 1 checkpoint in `STAGE1_CKPT`. Without make: `python src/train.py stage1|stage2|both`. `make evaluate` loads the newest `*-stage2.pt` in `checkpoints/`.

#### Evaluation

`src/evaluate.py` runs LM Evaluation Harness with the metrics of Table 9 of the paper: PIQA (acc), ARC-Easy (acc), ARC-Challenge (acc_norm), HellaSwag (acc_norm) and WinoGrande (acc) zero-shot, and MMLU 5-shot. As in lolcats, MMLU is the unweighted mean over its 57 subjects, the prompts have no BOS token, and Avg. is the mean of the five zero-shot tasks. It prints the scores next to the paper values and saves them to `checkpoints/eval_<model>.json`. Set `EVAL_LIMIT` in `src/config.py` for a quick run on a few questions per task.

#### Layout

- `src/lizard_attention.py`: the Lizard attention layer
- `src/train.py`: stage 1 and stage 2 training
- `src/evaluate.py`: lm-eval benchmarks of the paper
- `tests/test_lizard.py`: tests against slow loop versions of each formula
- `tests/test_evaluate.py`: tests of the scoring rules
