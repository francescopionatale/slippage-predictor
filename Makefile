PYTHON := .venv/bin/python
PIP    := .venv/bin/pip

.PHONY: venv install data train eval test clean

venv:
	python3.13 -m venv .venv || python3.12 -m venv .venv
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -e ".[dev]"

data:
	$(PYTHON) data/download.py

train:
	$(PYTHON) -m slippage.train

eval:
	$(PYTHON) -m slippage.evaluate

test:
	.venv/bin/pytest tests/ -v

clean:
	rm -rf data/raw/* data/processed/* results/figures/* results/metrics.json
