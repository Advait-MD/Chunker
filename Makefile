# APEX eval entry points (spec §2.8). On Windows without make, run the python commands directly.
CONFIG ?= configs/base.yaml
DATASETS ?= qasper,musique,quality,hotpotqa,2wiki,custom
SPLIT ?= test
RUN_ID ?=

.PHONY: data index eval judge test

data:            ## download pinned benchmark files and freeze the custom gold set
	python scripts/fetch_benchmarks.py
	python scripts/freeze_custom_gold.py

index:           ## build corpora and per-dataset indexes (splits are committed in apex/eval/splits/)
	python -m apex.eval.suite index qasper musique quality hotpotqa 2wiki

eval:            ## make eval CONFIG=configs/full.yaml → results/<run_id>/{metrics.json,table.md,traces/}
	python -m apex.eval.suite run --config $(CONFIG) --datasets $(DATASETS) --split $(SPLIT) $(if $(RUN_ID),--run-id $(RUN_ID),)

judge:           ## fill judge scores into an existing run: make judge RUN=results/<run_id>
	python -m apex.eval.suite judge $(RUN)

test:
	python -m pytest -q
