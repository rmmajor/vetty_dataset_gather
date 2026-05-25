.PHONY: help setup install-training install-all load-tasks record convert train all clean clean-trajectories clean-training \
       setup-pipeline build-sandbox pipeline-discover pipeline-filter pipeline-score pipeline-overnight pipeline-status

# Configurable variables
TASKS_FILE    ?= tasks.jsonl
TRAJ_DIR      ?= trajectories
TRAIN_DIR     ?= training_data
CONFIG        ?= config.yaml
TASK_LIMIT    ?=
MAX_STEPS     ?= 15
VIEWPORT      ?= 1280x720
VAL_SPLIT     ?= 0.1
EPOCHS        ?=
LR            ?=
BATCH_SIZE    ?=
PIPELINE_BUDGET ?= 30
PIPELINE_WORKERS ?= 5
PIPELINE_LIMIT ?=

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------- Setup ----------

setup: ## Install dependencies and Playwright chromium
	pip install -r requirements.txt
	playwright install chromium

install-training: ## Install ms-swift for training (GPU machine)
	pip install "ms-swift[llm]"

install-all: ## Install everything for a fresh GPU instance
	pip install -r requirements.txt
	pip install "ms-swift[llm]"

# ---------- Pipeline steps ----------

load-tasks: $(TASKS_FILE) ## Load Mind2Web tasks into TASKS_FILE
$(TASKS_FILE):
	python load_tasks.py --output $(TASKS_FILE) $(if $(TASK_LIMIT),--limit $(TASK_LIMIT))

record: $(TASKS_FILE) ## Record trajectories via GPT-5.4 CUA
	@test -n "$$OPENAI_API_KEY" || (echo "ERROR: OPENAI_API_KEY not set" && exit 1)
	python record_trajectories_from_gpt_api.py \
		--tasks $(TASKS_FILE) \
		--output $(TRAJ_DIR)/ \
		--max-steps $(MAX_STEPS) \
		--viewport $(VIEWPORT) \
		--resume

convert: ## Convert trajectories to ms-swift SFT format
	python train_qwen.py convert \
		--input $(TRAJ_DIR)/ \
		--output $(TRAIN_DIR)/ \
		--val-split $(VAL_SPLIT)

train: $(TRAIN_DIR)/sft_dataset.jsonl ## Launch ms-swift LoRA fine-tuning
	python train_qwen.py train \
		--config $(CONFIG) \
		--dataset $(TRAIN_DIR)/sft_dataset.jsonl \
		$(if $(EPOCHS),--epochs $(EPOCHS)) \
		$(if $(LR),--lr $(LR)) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE))

all: setup load-tasks record convert train ## Run full pipeline end-to-end

# ---------- Cleanup ----------

clean-trajectories: ## Remove all recorded trajectories
	rm -rf $(TRAJ_DIR)

clean-training: ## Remove converted training data
	rm -rf $(TRAIN_DIR)

clean: clean-trajectories clean-training ## Remove all generated data
	rm -f $(TASKS_FILE)

# ---------- Repo Pipeline (QA repo trajectory generation) ----------

setup-pipeline: ## Install pipeline dependencies
	pip install -r requirements-pipeline.txt
	playwright install chromium

build-sandbox: ## Build Docker sandbox image for pipeline
	docker build -t repo-pipeline-sandbox:latest -f docker/Dockerfile.sandbox docker/

test-sandbox: build-sandbox ## Test sandbox container lifecycle
	python -m repo_pipeline.sandbox --test

pipeline-discover: ## Stage 1: Discover QA repos on GitHub
	@test -n "$$GITHUB_TOKEN" || (echo "ERROR: GITHUB_TOKEN not set" && exit 1)
	python -m repo_pipeline.orchestrator --stage 1

pipeline-filter: ## Stage 2: Filter repos by metadata
	python -m repo_pipeline.orchestrator --stage 2

pipeline-score: ## Stage 3: Shallow score repos with Claude
	python -m repo_pipeline.orchestrator --stage 3 \
		--workers $(PIPELINE_WORKERS) \
		--budget $(PIPELINE_BUDGET) \
		$(if $(PIPELINE_LIMIT),--limit $(PIPELINE_LIMIT))

pipeline-overnight: ## Run stages 3-6 overnight with Telegram notification
	python -m repo_pipeline.orchestrator --overnight \
		--workers $(PIPELINE_WORKERS) \
		--budget $(PIPELINE_BUDGET) \
		$(if $(PIPELINE_LIMIT),--limit $(PIPELINE_LIMIT))

pipeline-status: ## Show pipeline registry stats
	python -m repo_pipeline.orchestrator --status

test-notifier: ## Test Telegram bot notification
	python -m repo_pipeline.notifier --test
