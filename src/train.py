import math
import shutil
import sys
import time
from dataclasses import asdict

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from lizard_attention import from_llama
from push_to_hub import upload


def is_lizard(name):
    return any(key in name.split(".") for key in config.LIZARD_PARAMS)


def load_llama(cfg):
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name,
                                                 dtype=cfg.torch_dtype)
    return model.to(cfg.device)


def to_lizard(model, cfg):
    for block in model.model.layers:
        block.self_attn = from_llama(block.self_attn, model.config,
                                     **asdict(cfg.lizard))
    return model


def format_example(example):
    extra = ""
    if example["input"]:
        extra = f"### Input:\n{example['input']}\n\n"
    return (f"### Instruction:\n{example['instruction']}\n\n{extra}"
            f"### Response:\n{example['output']}")


def pack(token_lists, length):
    ids = [token for tokens in token_lists for token in tokens]
    n = len(ids) // length
    return torch.tensor(ids[:n * length]).view(n, length)


def make_loader(tokenizer, cfg):
    data = load_dataset(cfg.dataset_name, split="train")
    data = data.shuffle(seed=cfg.seed).select(range(cfg.num_examples))
    tokens = tokenizer([format_example(e) for e in data])["input_ids"]
    chunks = pack([t + [tokenizer.eos_token_id] for t in tokens], cfg.seq_len)
    return DataLoader(chunks, batch_size=cfg.micro_batch, shuffle=True,
                      drop_last=True,
                      generator=torch.Generator().manual_seed(cfg.seed))


def lr_factor(step, total, cfg):
    warmup = max(1, int(cfg.warmup_ratio * total))
    if step < warmup:
        return step / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine


def log(stage, loss, named):
    values = {"loss": loss}
    for key in ("alpha_blend", "meta_tokens"):
        means = [p.detach().mean() for n, p in named if n.endswith(key)]
        values[key] = torch.stack(means).mean().item()
    print(stage, values)
    if wandb.run is not None:
        wandb.log({f"{stage}/{k}": v for k, v in values.items()})


def train(named, step_fn, loader, lr, stage, cfg):
    params = [p for _, p in named]
    total = cfg.epochs * (len(loader) // cfg.grad_accum)
    optimizer = torch.optim.AdamW(params, lr=lr, betas=cfg.betas,
                                  eps=cfg.eps, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, total, cfg))
    for _ in range(cfg.epochs):
        optimizer.zero_grad()
        loss = 0.0
        for i, batch in enumerate(loader):
            loss += step_fn(batch.to(cfg.device))
            if (i + 1) % cfg.grad_accum == 0:
                norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                if torch.isfinite(norm):
                    optimizer.step()
                else:
                    print(stage, "skipped step: gradient is not finite")
                scheduler.step()
                optimizer.zero_grad()
                log(stage, loss, named)
                loss = 0.0


def stage1(teacher, loader, cfg):
    teacher.eval().requires_grad_(False)
    students = [from_llama(block.self_attn, teacher.config,
                           **asdict(cfg.lizard))
                for block in teacher.model.layers]
    for student in students:
        for name, p in student.named_parameters():
            p.requires_grad = is_lizard(name)
    captured = {}
    for i, block in enumerate(teacher.model.layers):
        def hook(module, args, kwargs, output, i=i):
            x = args[0] if args else kwargs["hidden_states"]
            captured[i] = (x, kwargs["position_embeddings"], output[0])
        block.self_attn.register_forward_hook(hook, with_kwargs=True)

    def step_fn(ids):
        with torch.no_grad():
            teacher.model(input_ids=ids, use_cache=False)
        total = 0.0
        for i, student in enumerate(students):
            x, rope, target = captured[i]
            loss = (student(x, rope)[0].float() - target.float()).pow(2).sum()
            loss = loss / len(students) / cfg.grad_accum
            loss.backward()
            total += loss.item()
        return total

    named = [(n, p) for s in students for n, p in s.named_parameters()
             if p.requires_grad]
    train(named, step_fn, loader, cfg.stage1_lr, "stage1", cfg)
    return {f"model.layers.{i}.self_attn.{n}": p.detach().cpu()
            for i, s in enumerate(students)
            for n, p in s.named_parameters() if is_lizard(n)}


def stage2(model, lizard_state, loader, cfg):
    model = to_lizard(model, cfg)
    _, unexpected = model.load_state_dict(lizard_state, strict=False)
    assert not unexpected, unexpected
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                      lora_dropout=0.0, target_modules=list(cfg.lora_targets),
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora).train()
    for name, p in model.named_parameters():
        if is_lizard(name):
            p.requires_grad = True

    def step_fn(ids):
        out = model(input_ids=ids, labels=ids, use_cache=False)
        loss = out.loss / cfg.grad_accum
        loss.backward()
        return loss.item()

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    train(named, step_fn, loader, cfg.stage2_lr, "stage2", cfg)
    return model.merge_and_unload()


def save(state, path, cfg):
    torch.save(state, path)
    upload(path, cfg.hf_repo)


def main(cfg, stage):
    assert stage in ("stage1", "stage2", "both"), stage
    assert stage != "stage2" or cfg.stage1_ckpt, "set stage1_ckpt for stage2"
    torch.manual_seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    run = f"{cfg.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy(cfg.path, cfg.out_dir / f"{run}.yaml")
    upload(cfg.out_dir / f"{run}.yaml", cfg.hf_repo)
    wandb.init(project=cfg.wandb_project, name=run)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    loader = make_loader(tokenizer, cfg)
    if stage == "stage2":
        lizard_state = torch.load(cfg.stage1_ckpt)
    else:
        lizard_state = stage1(load_llama(cfg), loader, cfg)
        save(lizard_state, cfg.out_dir / f"{run}-stage1.pt", cfg)
    if stage != "stage1":
        model = stage2(load_llama(cfg), lizard_state, loader, cfg)
        save(model.state_dict(), cfg.out_dir / f"{run}-stage2.pt", cfg)
    wandb.finish()


if __name__ == "__main__":
    stage = sys.argv[2] if len(sys.argv) > 2 else "both"
    main(config.load(sys.argv[1]), stage)
