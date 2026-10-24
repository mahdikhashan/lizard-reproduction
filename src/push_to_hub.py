import sys
from pathlib import Path

from huggingface_hub import HfApi

import config

CODE = ["src/config.py", "src/lizard_attention.py", "src/train.py",
        "src/evaluate.py"]


def upload(path, repo_id):
    path = Path(path)
    api = HfApi()
    api.create_repo(repo_id, private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=path, path_in_repo=path.name,
                    repo_id=repo_id, commit_message=f"upload {path.name}")
    print(f"uploaded {path} to {repo_id}")


if __name__ == "__main__":
    cfg = config.load(sys.argv[1])
    for path in sys.argv[2:] or CODE + [cfg.path]:
        upload(path, cfg.hf_repo)
