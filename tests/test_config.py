from pathlib import Path

import pytest
import torch

import config


def test_default_config_loads_with_types():
    cfg = config.load("configs/lizard-1b.yaml")
    assert cfg.name == "lizard-1b"
    assert cfg.lizard == config.Lizard(window=128, num_meta=4,
                                       feature_dim=128)
    assert cfg.torch_dtype is torch.bfloat16
    assert cfg.stage1_lr == 1e-3 and isinstance(cfg.eps, float)
    assert cfg.betas == (0.9, 0.99)
    assert cfg.lora_targets == ("q_proj", "k_proj", "v_proj")
    assert cfg.out_dir == Path("checkpoints") and cfg.stage1_ckpt is None


def test_unknown_key_is_an_error(tmp_path):
    path = tmp_path / "bad.yaml"
    text = Path("configs/lizard-1b.yaml").read_text() + "typo_lr: 1.0\n"
    path.write_text(text)
    with pytest.raises(TypeError):
        config.load(path)
