import argparse
import json

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer

import config
from train import load_llama, to_lizard

FAST_MMLU_LIMIT = 5
ZERO_SHOT = {"piqa": "acc", "arc_easy": "acc", "arc_challenge": "acc_norm",
             "hellaswag": "acc_norm", "winogrande": "acc"}
PAPER = {
    "llama": {"piqa": 74.1, "arc_easy": 65.4, "arc_challenge": 36.4,
              "hellaswag": 63.8, "winogrande": 60.0, "mmlu": 31.0,
              "avg": 59.9},
    "lizard": {"piqa": 74.8, "arc_easy": 65.6, "arc_challenge": 36.5,
               "hellaswag": 64.1, "winogrande": 59.7, "mmlu": 29.8,
               "avg": 60.1},
}


def load(cfg, which):
    model = load_llama(cfg)
    if which == "lizard":
        path = max(cfg.out_dir.glob(f"{cfg.name}-*-stage2.pt"),
                   key=lambda p: p.stat().st_mtime)
        state = torch.load(path, map_location="cpu")
        to_lizard(model, cfg).load_state_dict(state, strict=True)
    return model.eval()


def mmlu_score(results, subjects):
    accs = [results[task]["acc,none"] for task in subjects]
    return 100 * sum(accs) / len(accs)


def summarize(results, subjects):
    scores = {task: 100 * results[task][f"{metric},none"]
              for task, metric in ZERO_SHOT.items()}
    avg = sum(scores.values()) / len(scores)
    scores["mmlu"] = mmlu_score(results, subjects)
    scores["avg"] = avg
    return scores


def run_mmlu(lm, limit):
    out = simple_evaluate(model=lm, tasks=["mmlu"], num_fewshot=5,
                          limit=limit)
    subjects = [t for t in out["configs"] if t.startswith("mmlu_")]
    return out["results"], subjects


def main(cfg, which, tasks):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    lm = HFLM(pretrained=load(cfg, which), tokenizer=tokenizer,
              batch_size=cfg.eval_batch_size, add_bos_token=False)
    if tasks == "mmlu-fast":
        results, subjects = run_mmlu(lm, FAST_MMLU_LIMIT)
        scores = {"mmlu": mmlu_score(results, subjects)}
    else:
        zero = simple_evaluate(model=lm, tasks=list(ZERO_SHOT),
                               num_fewshot=0, limit=cfg.eval_limit)
        results, subjects = run_mmlu(lm, cfg.eval_limit)
        results = {**zero["results"], **results}
        scores = summarize(results, subjects)
    print(f"{'task':<15}{which:>8}{'paper':>8}{'teacher':>9}")
    for task, value in scores.items():
        print(f"{task:<15}{value:>8.1f}{PAPER['lizard'][task]:>8.1f}"
              f"{PAPER['llama'][task]:>9.1f}")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{cfg.name}-eval-{which}-{tasks}.json"
    with open(cfg.out_dir / name, "w") as f:
        json.dump({"scores": scores, "results": results}, f, indent=2,
                  default=str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--model", choices=["lizard", "llama"],
                        default="lizard")
    parser.add_argument("--tasks", choices=["all", "mmlu-fast"],
                        default="all")
    args = parser.parse_args()
    main(config.load(args.config), args.model, args.tasks)
