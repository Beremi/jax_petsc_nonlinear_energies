PYTHON := ../.venv/bin/python
LATEXMK := latexmk
MAIN := main.tex

.PHONY: all layout figures tables literature validate pdf clean \
	plasticity3d-validation-run plasticity3d-validation-assets \
	plasticity3d-derivative-ablation-run plasticity3d-derivative-ablation-assets \
	jax-fem-baseline-run jax-fem-baseline-assets reproducibility

all: pdf

layout:
	$(PYTHON) scripts/measure_layout.py

figures: layout
	$(PYTHON) scripts/generate_paper_figures.py

tables: layout
	$(PYTHON) scripts/generate_paper_tables.py

literature:
	$(PYTHON) scripts/generate_literature_sources.py

plasticity3d-validation-run:
	$(PYTHON) ../experiments/runners/run_plasticity3d_validation.py

plasticity3d-validation-assets:
	$(PYTHON) ../experiments/analysis/generate_plasticity3d_validation_assets.py

plasticity3d-derivative-ablation-run:
	$(PYTHON) ../experiments/runners/run_plasticity3d_derivative_ablation.py

plasticity3d-derivative-ablation-assets:
	$(PYTHON) ../experiments/analysis/generate_plasticity3d_derivative_ablation_assets.py

jax-fem-baseline-run:
	$(PYTHON) ../experiments/runners/run_jax_fem_hyperelastic_baseline.py

jax-fem-baseline-assets:
	$(PYTHON) ../experiments/analysis/generate_jax_fem_hyperelastic_baseline_assets.py

reproducibility:
	$(PYTHON) scripts/generate_reproducibility_note.py

validate: figures tables
	$(PYTHON) scripts/validate_paper_assets.py

pdf: validate
	$(LATEXMK) -pdf $(MAIN)

clean:
	rm -rf build
	rm -f figures/generated/*
	rm -f tables/generated/*
