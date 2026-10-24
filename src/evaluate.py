import json
import sys

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer

import config as C
from train import load_llama, to_lizard

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


def load(which):
    model = load_llama()
    if which == "lizard":
        state = torch.load(C.STAGE2_CKPT, map_location="cpu")
        to_lizard(model).load_state_dict(state, strict=True)
    return model.eval()


def summarize(results, subjects):
    scores = {task: 100 * results[task][f"{metric},none"]
              for task, metric in ZERO_SHOT.items()}
    avg = sum(scores.values()) / len(scores)
    accs = [results[task]["acc,none"] for task in subjects]
    scores["mmlu"] = 100 * sum(accs) / len(accs)
    scores["avg"] = avg
    return scores


def main(which):
    tokenizer = AutoTokenizer.from_pretrained(C.MODEL_NAME)
    lm = HFLM(pretrained=load(which), tokenizer=tokenizer,
              batch_size=C.EVAL_BATCH_SIZE, add_bos_token=False)
    zero = simple_evaluate(model=lm, tasks=list(ZERO_SHOT), num_fewshot=0,
                           limit=C.EVAL_LIMIT)
    mmlu = simple_evaluate(model=lm, tasks=["mmlu"], num_fewshot=5,
                           limit=C.EVAL_LIMIT)
    results = {**zero["results"], **mmlu["results"]}
    scores = summarize(results, [t for t in mmlu["configs"]
                                 if t.startswith("mmlu_")])
    print(f"{'task':<15}{which:>8}{'paper':>8}{'teacher':>9}")
    for task, value in scores.items():
        print(f"{task:<15}{value:>8.1f}{PAPER['lizard'][task]:>8.1f}"
              f"{PAPER['llama'][task]:>9.1f}")
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.OUT_DIR / f"eval_{which}.json", "w") as f:
        json.dump({"scores": scores, "results": results}, f, indent=2,
                  default=str)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lizard")
