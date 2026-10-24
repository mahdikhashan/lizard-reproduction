import sys
from pathlib import Path

from huggingface_hub import HfApi

import config as C

CODE = ["src/config.py", "src/lizard_attention.py", "src/train.py",
        "src/evaluate.py"]


def upload(path):
    path = Path(path)
    api = HfApi()
    api.create_repo(C.HF_REPO, private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=path, path_in_repo=path.name,
                    repo_id=C.HF_REPO, commit_message=f"upload {path.name}")
    print(f"uploaded {path} to {C.HF_REPO}")


if __name__ == "__main__":
    for path in sys.argv[1:] or CODE:
        upload(path)
