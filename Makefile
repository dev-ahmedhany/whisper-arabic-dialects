.PHONY: help install install-bench test fmt lint docker-build docker-run baseline tables clean

help:
	@echo "install         install full training+eval deps"
	@echo "install-bench   install CPU-benchmark deps only"
	@echo "test            run unit tests"
	@echo "fmt             black + isort"
	@echo "docker-build    build CPU benchmark image"
	@echo "docker-run      run benchmark image (use ARGS=... to pass flags)"
	@echo "baseline        run zero-shot baseline (tiny smoke run on local machine)"
	@echo "tables          rebuild paper tables from runs/results.jsonl"
	@echo "clean           remove caches and build artifacts"

install:
	pip install -r requirements.txt
	pip install -e .

install-bench:
	pip install -r requirements-bench.txt
	pip install -e .

test:
	pytest tests/

fmt:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

docker-build:
	docker build -t whisper-arabic-bench:latest .

docker-run:
	docker run --rm \
	  -v $(PWD)/runs:/app/runs \
	  -v $(PWD)/test_sets:/app/test_sets \
	  -v $(HOME)/.cache/huggingface:/root/.cache/huggingface \
	  -e HF_TOKEN=$$HF_TOKEN \
	  whisper-arabic-bench:latest $(ARGS)

baseline:
	python -m scripts.run_zero_shot_baseline --tiny

tables:
	python -m src.build_results_tables runs/results.jsonl paper/paper.md

clean:
	rm -rf __pycache__ .pytest_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
