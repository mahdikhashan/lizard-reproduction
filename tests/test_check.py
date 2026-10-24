from types import SimpleNamespace

import check
import config


class FakeHub:
    def __init__(self, denied=()):
        self.denied = denied

    def whoami(self):
        return {"name": "me"}

    def create_repo(self, repo_id, **kwargs):
        pass

    def auth_check(self, repo_id, repo_type=None, write=False):
        if repo_id in self.denied:
            raise PermissionError(repo_id)


def test_all_checks_pass(monkeypatch, capsys):
    viewer = SimpleNamespace(username="me")
    monkeypatch.setattr(check, "HfApi", FakeHub)
    monkeypatch.setattr(check.wandb, "Api",
                        lambda timeout: SimpleNamespace(viewer=viewer))
    monkeypatch.delenv("WANDB_MODE", raising=False)
    assert check.main(config.load("configs/lizard-1b.yaml")) == 0
    assert "[fail]" not in capsys.readouterr().out


def test_failures_are_counted_and_wandb_can_be_skipped(monkeypatch, capsys):
    cfg = config.load("configs/lizard-1b.yaml")
    monkeypatch.setattr(check, "HfApi",
                        lambda: FakeHub(denied=(cfg.model_name, cfg.hf_repo)))
    monkeypatch.setenv("WANDB_MODE", "disabled")
    assert check.main(cfg) == 2
    out = capsys.readouterr().out
    assert "[fail] hf_model" in out and "[fail] hf_upload" in out
    assert "[ok]   wandb: skipped, WANDB_MODE=disabled" in out
