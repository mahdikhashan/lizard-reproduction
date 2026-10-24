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
make train           # stage 1 and stage 2, needs a GPU, saves to checkpoints/
make evaluate        # lm-eval on the trained Lizard model
make evaluate-llama  # lm-eval on the original model
make push            # upload checkpoints and code to Hugging Face
```

All settings are in `src/config.py`.

#### Layout

- `src/lizard_attention.py`: the Lizard attention layer
- `src/train.py`: stage 1 and stage 2 training
- `src/evaluate.py`: lm-eval benchmarks
- `tests/test_lizard.py`: tests against slow loop versions of each formula
