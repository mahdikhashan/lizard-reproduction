import copy
import json
from dataclasses import replace

import pytest
import torch
import transformers
from transformers import LlamaForCausalLM

import config
import evaluate
import train
import xai
from lizard_attention import from_llama
from test_lizard import tiny_config

LENGTH = 40


def tiny_models(seed=0):
    torch.manual_seed(seed)
    teacher = LlamaForCausalLM(tiny_config(2)).eval()
    student = copy.deepcopy(teacher)
    for block in student.model.layers:
        block.self_attn = from_llama(block.self_attn, student.config,
                                     window=16, feature_dim=8)
        with torch.no_grad():
            block.self_attn.W_gamma.weight.normal_(0, 0.5)
            block.self_attn.meta_tokens.normal_(0, 0.5)
    return teacher, student.eval()


def test_lizard_weights_reproduce_the_layer_output():
    teacher, student = tiny_models()
    ids = torch.randint(0, 100, (1, LENGTH))
    found = xai.capture(teacher, ids)
    with torch.no_grad():
        stats = xai.layer_stats(student.model.layers[0].self_attn,
                                teacher.model.layers[0].self_attn,
                                *found[0], teacher.config, {(0, 1)}, 0)
    assert stats["check"] < 1e-5
    tv = stats["per_head"]["tv"]
    assert tv.shape == (4, LENGTH) and (tv >= 0).all() and (tv <= 1).all()
    assert set(stats["crops"]) == {"0:1"}


def test_identical_weights_have_no_distance():
    teacher = torch.rand(2, 6, 6).tril()
    teacher = teacher / teacher.sum(-1, keepdim=True)
    values, _ = xai.weight_metrics(teacher, teacher, torch.zeros_like(teacher),
                                   torch.tensor(1.0), 3)
    assert values["tv"].abs().max() < 1e-6
    assert torch.allclose(values["teacher_outside"], values["lizard_outside"])
    assert values["window_share"].abs().max() == 0


@pytest.fixture
def tiny_pipeline(monkeypatch):
    teacher, student = tiny_models()
    monkeypatch.setattr(train, "load_llama",
                        lambda cfg: copy.deepcopy(teacher))
    monkeypatch.setattr(evaluate, "load",
                        lambda cfg, which, checkpoint: copy.deepcopy(student))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda name: None)
    monkeypatch.setattr(xai, "held_out", lambda tokenizer, cfg, n:
                        torch.randint(0, 100, (n, LENGTH)))
    cfg = config.load("configs/lizard-1b.yaml")
    return replace(cfg, device="cpu",
                   lizard=replace(cfg.lizard, window=16, feature_dim=8))


def test_compute_and_plot_both_versions(tiny_pipeline, tmp_path):
    paths = []
    for label in ("PyTorch", "Triton"):
        for stage in ("stage1", "stage2"):
            out = tmp_path / f"{label}-{stage}.json"
            xai.compute(tiny_pipeline, tmp_path / "x.pt", label, stage, out,
                        2, {(1, 2)})
            paths.append(out)
    result = json.loads(paths[0].read_text())
    assert result["spans"] == ["0-127", "all"]
    assert len(result["layers"]) == 2 and "1:2" in result["maps"]
    scores = {t: 50.0 for t in xai.TASKS}
    evals = []
    for label in ("Teacher", "PyTorch", "Triton"):
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps({"scores": scores}))
        evals.append(f"{label}={path}")
    curves = tmp_path / "curves.json"
    curves.write_text(json.dumps({"stage1": {
        "PyTorch": {"run": "a", "runtime": 1, "loss": [3.0, 2.0]},
        "Triton": {"run": "b", "runtime": 1, "loss": [3.1, 2.1]}}}))
    figures = tmp_path / "figures"
    xai.plot([str(p) for p in paths], evals, str(curves), figures)
    names = {p.name for p in figures.iterdir()}
    assert {"layer_mse.png", "position_mse.png", "attention_tv.png",
            "position_tv.png", "window_share.png", "gate_memory.png",
            "attention_maps.png", "evaluation.png", "training_loss.png",
            "xai.md"} <= names
    assert "| Triton | stage2 |" in (figures / "xai.md").read_text()
