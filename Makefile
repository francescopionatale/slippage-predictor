PYTHON := .venv/bin/python
PIP    := .venv/bin/pip

.PHONY: venv install data train eval test report all clean

venv:
	python3.13 -m venv .venv || python3.12 -m venv .venv
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -e ".[dev]"

data:
	$(PYTHON) data/download.py

train:
	$(PYTHON) -m slippage.training.trainer

eval:
	$(PYTHON) -m slippage.evaluation

test:
	.venv/bin/pytest tests/ -v

report:
	$(PYTHON) scripts/run_experiments.py
	$(PYTHON) scripts/compare_runs.py
	$(PYTHON) scripts/run_walk_forward.py
	$(PYTHON) scripts/plot_pred_vs_actual.py
	$(PYTHON) scripts/generate_residual_plots.py
	$(PYTHON) scripts/generate_diversification_plots.py
	$(PYTHON) scripts/build_report.py

all: train eval report

clean:
	rm -rf data/raw/* data/processed/* results/figures/* results/metrics.json
