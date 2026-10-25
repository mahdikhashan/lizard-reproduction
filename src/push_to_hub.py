import argparse
from pathlib import Path

from huggingface_hub import HfApi

import config

CODE = ["src/config.py", "src/lizard_attention.py", "src/train.py",
        "src/evaluate.py"]


def upload(path, repo_id, data=None):
    path = Path(path)
    api = HfApi()
    api.create_repo(repo_id, private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=path if data is None else data,
                    path_in_repo=path.name,
                    repo_id=repo_id, commit_message=f"upload {path.name}")
    print(f"uploaded {path} to {repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()
    cfg = config.load(args.config)
    for path in args.files or CODE + [cfg.path]:
        upload(path, cfg.hf_repo)
