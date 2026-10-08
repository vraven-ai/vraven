.PHONY: install dev test compile inventory quality demo benchmark wheel release clean

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	bash scripts/run_tests.sh

compile:
	$(PYTHON) -m compileall -q src tests examples benchmarks scripts

inventory:
	$(PYTHON) scripts/generate_code_inventory.py

quality:
	bash scripts/quality_check.sh

demo:
	PYTHONPATH=src $(PYTHON) -m vraven.cli demo --mode gradient --output-dir vraven_demo

benchmark:
	PYTHONPATH=src $(PYTHON) benchmarks/benchmark_capture.py --repeats 5 --batch 16 --width 128

wheel:
	bash scripts/build_package.sh

release:
	$(PYTHON) scripts/create_release.py --output-dir release

clean:
	rm -rf build dist .pytest_cache src/vraven.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
	rm -rf vraven_demo vraven_attention_demo vraven_causal_demo vraven_dataset_demo \
		vraven_quickstart vraven_visual_gallery vraven_visual_modes
