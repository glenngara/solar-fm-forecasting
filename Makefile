PYTHON = .venv/bin/python
SRC = src

.PHONY: all setup data prepare eda zero-shot baselines finetune-chronos finetune-ttm finetune-all eval-finetuned eval-all eval-ablation data-efficiency sensitivity figures clean

# Run the entire pipeline
all: data prepare eda zero-shot baselines finetune-all eval-all eval-ablation data-efficiency sensitivity figures
	@echo "\n=== Pipeline complete ==="
	@echo "Results in: results/tables/ and results/figures/"
	@echo "Logs in: results/logs/"

# Setup Python environment
setup:
	python3.11 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install "uni2ts @ git+https://github.com/SalesforceAIResearch/uni2ts.git" || echo "Moirai install failed (optional)"
	$(PYTHON) -m pip install git+https://github.com/google-research/timesfm.git || echo "TimesFM install failed (optional)"

# Step 1: Download NASA POWER data
data: data/raw/nasa_power_laguna_de_bay_2020_2025.csv
data/raw/nasa_power_laguna_de_bay_2020_2025.csv:
	$(PYTHON) $(SRC)/data/download_nasa_power.py

# Step 2: Prepare train/val/test splits
prepare: data/processed/train.csv
data/processed/train.csv: data/raw/nasa_power_laguna_de_bay_2020_2025.csv
	$(PYTHON) $(SRC)/data/prepare_data.py

# Step 3: Exploratory data analysis
eda: results/figures/fig1_annual_irradiance_profile.png
results/figures/fig1_annual_irradiance_profile.png: data/raw/nasa_power_laguna_de_bay_2020_2025.csv
	$(PYTHON) $(SRC)/figures/eda.py

# Step 4: Zero-shot FM evaluation
zero-shot: results/tables/zero_shot_results.csv
results/tables/zero_shot_results.csv: data/processed/train.csv
	$(PYTHON) $(SRC)/eval/zero_shot.py

# Step 5: Traditional baselines (XGBoost, LSTM)
baselines: results/tables/baseline_results.csv
results/tables/baseline_results.csv: data/processed/train.csv
	$(PYTHON) $(SRC)/eval/baselines.py

# Step 6-7: Fine-tune foundation models
finetune-chronos: models/ft-chronos-t5-small models/ft-chronos-t5-base
models/ft-chronos-t5-small models/ft-chronos-t5-base: data/processed/train.csv
	$(PYTHON) $(SRC)/finetune/chronos_ft.py

finetune-ttm: models/ft-ttm-r2
models/ft-ttm-r2: data/processed/train.csv
	$(PYTHON) $(SRC)/finetune/ttm_ft.py

finetune-all: finetune-chronos finetune-ttm

# Step 8: Evaluate fine-tuned Chronos
eval-finetuned: results/tables/finetuned_comparison.csv
results/tables/finetuned_comparison.csv: models/ft-chronos-t5-small models/ft-chronos-t5-base
	$(PYTHON) $(SRC)/eval/finetuned.py

# Step 9: Full comparison
eval-all: results/tables/all_models_comparison.csv
results/tables/all_models_comparison.csv: results/tables/zero_shot_results.csv results/tables/baseline_results.csv
	$(PYTHON) $(SRC)/eval/all_finetuned.py

# Step 10: Ablation study
eval-ablation: results/tables/ablation_steps.csv
results/tables/ablation_steps.csv: models/ft-chronos-t5-base
	$(PYTHON) $(SRC)/eval/ablation.py

# Step 11: Data efficiency experiment
data-efficiency: results/tables/data_efficiency_results.csv
results/tables/data_efficiency_results.csv: data/processed/train.csv
	$(PYTHON) $(SRC)/experiments/data_efficiency.py

# Step 12: Sensitivity analysis (hyperparameter sweep)
sensitivity: results/tables/sensitivity_results.csv
results/tables/sensitivity_results.csv: data/processed/train.csv
	$(PYTHON) $(SRC)/experiments/sensitivity_analysis.py

# Step 13: Generate paper figures
figures: results/tables/all_models_comparison.csv
	$(PYTHON) $(SRC)/figures/generate.py

# Clean generated outputs (keeps raw data)
clean:
	rm -rf data/processed/ results/ models/
	@echo "Cleaned processed data, results, and models."

# Clean everything including raw data
clean-all: clean
	rm -rf data/raw/
	@echo "Cleaned all data."
