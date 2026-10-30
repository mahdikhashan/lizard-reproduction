PYTHON ?= python
CONFIG ?= configs/lizard-1b.yaml
KERNEL_CONFIG ?= configs/lizard-1b-mse1000-triton.yaml
STAGE ?= both
MODEL ?= lizard
STAGE1_CKPT ?=
XAI_DIR ?= xai
XAI_SEQUENCES ?= 16
TORCH_STAGE1 ?= lizard-1b-mse1000-20261024-232317
TORCH_STAGE2 ?= lizard-1b-mse1000-20261025-023701
TRITON_STAGE1 ?= lizard-1b-mse1000-triton-20261025-201733
TRITON_STAGE2 ?= lizard-1b-mse1000-triton-20261025-223935
XAI_COMPUTE = $(PYTHON) src/xai.py compute --sequences $(XAI_SEQUENCES)
CKPT ?=

.PHONY: env test test-kernels check train train-kernel benchmark evaluate evaluate-llama mmlu mmlu-fast push xai xai-compute xai-curves xai-plot

env:
	conda env create -f envs/distill_lizard_llama_3_2_1B.yaml

test:
	PYTHONPATH=src $(PYTHON) -m pytest tests

test-kernels:
	PYTHONPATH=src $(PYTHON) -m pytest tests/test_kernels.py

check:
	$(PYTHON) src/check.py $(CONFIG)

train: check
	$(PYTHON) src/train.py $(CONFIG) --stage $(STAGE) $(if $(STAGE1_CKPT),--stage1-ckpt $(STAGE1_CKPT))

train-kernel:
	$(PYTHON) src/check.py $(KERNEL_CONFIG)
	$(PYTHON) src/train.py $(KERNEL_CONFIG) --stage $(STAGE) $(if $(STAGE1_CKPT),--stage1-ckpt $(STAGE1_CKPT))

benchmark:
	$(PYTHON) src/benchmark.py

evaluate:
	$(PYTHON) src/evaluate.py $(CONFIG) --model lizard $(if $(CKPT),--checkpoint $(CKPT))

evaluate-llama:
	$(PYTHON) src/evaluate.py $(CONFIG) --model llama

mmlu:
	$(PYTHON) src/evaluate.py $(CONFIG) --model $(MODEL) --tasks mmlu $(if $(CKPT),--checkpoint $(CKPT))

mmlu-fast:
	$(PYTHON) src/evaluate.py $(CONFIG) --model $(MODEL) --tasks mmlu-fast $(if $(CKPT),--checkpoint $(CKPT))

push:
	$(PYTHON) src/push_to_hub.py $(CONFIG)

xai: xai-compute xai-plot

xai-compute:
	$(XAI_COMPUTE) configs/lizard-1b-mse1000.yaml --label PyTorch --stage stage1 --checkpoint checkpoints/$(TORCH_STAGE1)-stage1.pt --out $(XAI_DIR)/pytorch-stage1.json
	$(XAI_COMPUTE) configs/lizard-1b-mse1000.yaml --label PyTorch --stage stage2 --checkpoint checkpoints/$(TORCH_STAGE2)-stage2.pt --out $(XAI_DIR)/pytorch-stage2.json
	$(XAI_COMPUTE) configs/lizard-1b-mse1000-triton.yaml --label Triton --stage stage1 --checkpoint checkpoints/$(TRITON_STAGE1)-stage1.pt --out $(XAI_DIR)/triton-stage1.json
	$(XAI_COMPUTE) configs/lizard-1b-mse1000-triton.yaml --label Triton --stage stage2 --checkpoint checkpoints/$(TRITON_STAGE2)-stage2.pt --out $(XAI_DIR)/triton-stage2.json

xai-curves:
	$(PYTHON) src/xai.py curves $(XAI_DIR)/curves.json --runs PyTorch:stage1:$(TORCH_STAGE1) PyTorch:stage2:$(TORCH_STAGE2) Triton:stage1:$(TRITON_STAGE1) Triton:stage2:$(TRITON_STAGE2)

xai-plot:
	$(PYTHON) src/xai.py plot $(XAI_DIR)/pytorch-stage1.json $(XAI_DIR)/pytorch-stage2.json $(XAI_DIR)/triton-stage1.json $(XAI_DIR)/triton-stage2.json --out $(XAI_DIR) \
		--eval Teacher=checkpoints/lizard-1b-mse1000-eval-llama-all.json PyTorch=checkpoints/$(TORCH_STAGE2)-stage2-eval-lizard-all.json Triton=checkpoints/$(TRITON_STAGE2)-stage2-eval-lizard-all.json \
		$(if $(wildcard $(XAI_DIR)/curves.json),--curves $(XAI_DIR)/curves.json)
