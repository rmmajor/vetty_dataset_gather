.PHONY: help setup install-training install-all load-tasks record convert train all clean clean-trajectories clean-training

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
