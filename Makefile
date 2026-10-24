PYTHON ?= python
CONFIG ?= configs/lizard-1b.yaml
STAGE ?= both
MODEL ?= lizard
STAGE1_CKPT ?=

.PHONY: env test check train evaluate evaluate-llama mmlu-fast push

env:
	conda env create -f envs/distill_lizard_llama_3_2_1B.yaml

test:
	PYTHONPATH=src $(PYTHON) -m pytest tests

check:
	$(PYTHON) src/check.py $(CONFIG)

train: check
	$(PYTHON) src/train.py $(CONFIG) --stage $(STAGE) $(if $(STAGE1_CKPT),--stage1-ckpt $(STAGE1_CKPT))

evaluate:
	$(PYTHON) src/evaluate.py $(CONFIG) --model lizard

evaluate-llama:
	$(PYTHON) src/evaluate.py $(CONFIG) --model llama

mmlu-fast:
	$(PYTHON) src/evaluate.py $(CONFIG) --model $(MODEL) --tasks mmlu-fast

push:
	$(PYTHON) src/push_to_hub.py $(CONFIG)
