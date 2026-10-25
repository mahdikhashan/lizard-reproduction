import io
from types import SimpleNamespace

import torch

import train


def test_save_uploads_from_memory_when_the_disk_write_fails(tmp_path,
                                                             monkeypatch):
    real_save = torch.save
    uploads = []

    def save(obj, f):
        if not isinstance(f, io.BytesIO):
            open(f, "wb").write(b"partial")
            raise RuntimeError("basic_ios::clear: iostream error")
        real_save(obj, f)

    def upload(path, repo_id, data=None):
        uploads.append((path.name, repo_id, data.read()))

    monkeypatch.setattr(torch, "save", save)
    monkeypatch.setattr(train, "upload", upload)
    path = tmp_path / "run-stage2.pt"
    state = {"w": torch.arange(4.0)}
    train.save(state, path, SimpleNamespace(hf_repo="user/repo"))
    assert not path.exists()
    [(name, repo, data)] = uploads
    assert (name, repo) == ("run-stage2.pt", "user/repo")
    assert torch.equal(torch.load(io.BytesIO(data))["w"], state["w"])


def test_save_writes_and_uploads_the_file(tmp_path, monkeypatch):
    uploads = []
    monkeypatch.setattr(train, "upload",
                        lambda path, repo_id: uploads.append(path))
    path = tmp_path / "run-stage1.pt"
    train.save({"w": torch.ones(2)}, path, SimpleNamespace(hf_repo="r"))
    assert uploads == [path] and torch.load(path)["w"].sum() == 2
