import copy
from dataclasses import replace

import pytest
import torch
from transformers import LlamaForCausalLM

import config
import evaluate
import train
from test_lizard import tiny_config


def test_summarize_uses_the_metrics_of_the_paper():
    results = {"piqa": {"acc,none": 0.7, "acc_norm,none": 0.0},
               "arc_easy": {"acc,none": 0.6, "acc_norm,none": 0.0},
               "arc_challenge": {"acc,none": 0.0, "acc_norm,none": 0.3},
               "hellaswag": {"acc,none": 0.0, "acc_norm,none": 0.5},
               "winogrande": {"acc,none": 0.4},
               "mmlu": {"acc,none": 0.9},
               "mmlu_stem": {"acc,none": 0.9},
               "mmlu_anatomy": {"acc,none": 0.2},
               "mmlu_virology": {"acc,none": 0.4}}
    scores = evaluate.summarize(results, ["mmlu_anatomy", "mmlu_virology"])
    assert list(scores) == ["piqa", "arc_easy", "arc_challenge", "hellaswag",
                            "winogrande", "mmlu", "avg"]
    assert scores["arc_challenge"] == pytest.approx(30)
    assert scores["hellaswag"] == pytest.approx(50)
    assert scores["mmlu"] == pytest.approx(30)
    assert scores["avg"] == pytest.approx(50)


def test_paper_average_excludes_mmlu():
    for paper in evaluate.PAPER.values():
        tasks = [paper[t] for t in evaluate.ZERO_SHOT]
        assert sum(tasks) / len(tasks) == pytest.approx(paper["avg"], abs=0.05)


def test_load_takes_stage1_and_stage2_checkpoints(tmp_path, monkeypatch):
    cfg = replace(config.load("configs/lizard-1b.yaml"), out_dir=tmp_path,
                  lizard=config.Lizard(window=8, num_meta=4, feature_dim=8,
                                       backend="torch"))
    torch.manual_seed(0)
    teacher = LlamaForCausalLM(tiny_config(layers=2))
    monkeypatch.setattr(evaluate, "load_llama",
                        lambda cfg: copy.deepcopy(teacher))
    trained = train.to_lizard(copy.deepcopy(teacher), cfg)
    with torch.no_grad():
        for p in trained.parameters():
            p.normal_()
    full = trained.state_dict()
    stage1 = {k: v for k, v in full.items() if train.is_lizard(k)}
    torch.save(stage1, tmp_path / "run-stage1.pt")
    torch.save(full, tmp_path / "run-stage2.pt")

    got = evaluate.load(cfg, "lizard", tmp_path / "run-stage1.pt").state_dict()
    assert all(torch.equal(got[k], v) for k, v in stage1.items())
    q = "model.layers.0.self_attn.q_proj.weight"
    assert torch.equal(got[q], teacher.state_dict()[q])

    got = evaluate.load(cfg, "lizard", tmp_path / "run-stage2.pt").state_dict()
    assert all(torch.equal(got[k], v) for k, v in full.items())
