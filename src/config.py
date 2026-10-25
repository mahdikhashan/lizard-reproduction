from dataclasses import dataclass, fields
from pathlib import Path

import torch
import yaml

LIZARD_PARAMS = ("phi_q", "phi_k", "W_gamma", "meta_tokens", "alpha_blend")


@dataclass(frozen=True)
class Lizard:
    window: int
    num_meta: int
    feature_dim: int
    backend: str


@dataclass(frozen=True)
class Config:
    name: str
    path: Path
    model_name: str
    dataset_name: str
    num_examples: int
    seq_len: int
    dtype: str
    device: str
    seed: int
    lizard: Lizard
    micro_batch: int
    grad_accum: int
    epochs: int
    warmup_ratio: float
    min_lr_ratio: float
    grad_clip: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    stage1_lr: float
    stage2_lr: float
    mse_factor: float
    lora_rank: int
    lora_alpha: int
    lora_targets: tuple[str, ...]
    gradient_checkpointing: bool
    out_dir: Path
    stage1_ckpt: Path | None
    wandb_project: str
    eval_batch_size: int
    eval_limit: int | None
    hf_repo: str

    @property
    def torch_dtype(self):
        return getattr(torch, self.dtype)


def load(path):
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    for field in fields(Config):
        if field.type is float:
            data[field.name] = float(data[field.name])
    data["lizard"] = Lizard(**data["lizard"])
    data["betas"] = tuple(float(beta) for beta in data["betas"])
    data["lora_targets"] = tuple(data["lora_targets"])
    data["out_dir"] = Path(data["out_dir"])
    if data["stage1_ckpt"] is not None:
        data["stage1_ckpt"] = Path(data["stage1_ckpt"])
    return Config(name=path.stem, path=path, **data)
