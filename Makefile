PYTHON ?= python
CONFIG ?= configs/lizard-1b.yaml
KERNEL_CONFIG ?= configs/lizard-1b-mse1000-triton.yaml
STAGE ?= both
MODEL ?= lizard
STAGE1_CKPT ?=
CKPT ?=

.PHONY: env test test-kernels check train train-kernel benchmark evaluate evaluate-llama mmlu-fast push

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

mmlu-fast:
	$(PYTHON) src/evaluate.py $(CONFIG) --model $(MODEL) --tasks mmlu-fast $(if $(CKPT),--checkpoint $(CKPT))

push:
	$(PYTHON) src/push_to_hub.py $(CONFIG)
