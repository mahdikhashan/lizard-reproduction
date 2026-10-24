import os
import sys

import wandb
from huggingface_hub import HfApi

import config


def check_hf_login(api, cfg):
    return f"logged in as {api.whoami()['name']}"


def check_hf_model(api, cfg):
    api.auth_check(cfg.model_name)
    return f"can read {cfg.model_name}"


def check_hf_dataset(api, cfg):
    api.auth_check(cfg.dataset_name, repo_type="dataset")
    return f"can read {cfg.dataset_name}"


def check_hf_upload(api, cfg):
    api.create_repo(cfg.hf_repo, private=True, exist_ok=True)
    api.auth_check(cfg.hf_repo, write=True)
    return f"can write {cfg.hf_repo}"


def check_wandb(api, cfg):
    mode = os.environ.get("WANDB_MODE", "online")
    if mode in ("disabled", "offline"):
        return f"skipped, WANDB_MODE={mode}"
    viewer = wandb.Api(timeout=30).viewer
    return f"logged in as {viewer.username}, project {cfg.wandb_project}"


CHECKS = [check_hf_login, check_hf_model, check_hf_dataset, check_hf_upload,
          check_wandb]


def main(cfg):
    api = HfApi()
    failed = 0
    for check in CHECKS:
        name = check.__name__.removeprefix("check_")
        try:
            print(f"[ok]   {name}: {check(api, cfg)}")
        except Exception as error:
            failed += 1
            print(f"[fail] {name}: {type(error).__name__}: {error}")
    return failed


if __name__ == "__main__":
    sys.exit(1 if main(config.load(sys.argv[1])) else 0)
