import pytest

import evaluate


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
