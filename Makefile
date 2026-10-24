PYTHON ?= python
CONFIG ?= configs/lizard-1b.yaml
STAGE ?= both

.PHONY: env test train evaluate evaluate-llama push

env:
	conda env create -f envs/distill_lizard_llama_3_2_1B.yaml

test:
	PYTHONPATH=src $(PYTHON) -m pytest tests

train:
	$(PYTHON) src/train.py $(CONFIG) $(STAGE)

evaluate:
	$(PYTHON) src/evaluate.py $(CONFIG) lizard

evaluate-llama:
	$(PYTHON) src/evaluate.py $(CONFIG) llama

push:
	$(PYTHON) src/push_to_hub.py $(CONFIG)
