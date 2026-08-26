.PHONY: help setup install-training install-all load-tasks record convert train all clean clean-trajectories clean-training \
       setup-pipeline build-sandbox pipeline-discover pipeline-filter pipeline-score pipeline-overnight pipeline-until-limit pipeline-think pipeline-status dashboard \
       install-eval eval-screenspot test-eval

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
PIPELINE_LIMIT_PCT ?= 100
EVAL_MODELS      ?= eval_models.yaml
EVAL_DATASET     ?= os-copilot/ScreenSpot-v2
EVAL_SPLIT       ?= test
EVAL_BACKEND     ?= swift
EVAL_COORD_SPACE ?= viewport_abs
EVAL_VIEWPORT    ?= 1280x720
EVAL_OUTPUT_DIR  ?= eval_results
EVAL_LIMIT       ?=

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

convert: ## Convert trajectories to ms-swift SFT format (clean cut: drops meandering trajectories + waste-click steps)
	python train_qwen.py convert \
		--input $(TRAJ_DIR)/ \
		--output $(TRAIN_DIR)/ \
		--val-split $(VAL_SPLIT) \
		--exclude-waste-steps \
		--exclude-meandering

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

build-sandbox: ## Build Docker sandbox image (Playwright, arch-native)
	docker build -t repo-pipeline-sandbox:latest -f docker/Dockerfile.sandbox docker/

build-sandbox-selenium: ## Build Selenium sandbox image (linux/amd64; needed for Selenium repos)
	docker build --platform=linux/amd64 -t repo-pipeline-sandbox-selenium:latest -f docker/Dockerfile.sandbox-selenium docker/

test-sandbox: build-sandbox ## Test sandbox container lifecycle
	python -m repo_pipeline.sandbox --test

test-driver: ## Scripted (no-LLM) smoke test of the agent-capture browser driver
	python -m repo_pipeline.driver_smoke

audit-corpus: ## Mechanical quality gate over trainable trajectories (exit 1 on defects)
	python -m repo_pipeline.corpus_audit trajectories/

capture-public: ## Teacher capture on curated public sites (real-web diversity lane)
	python -m repo_pipeline.public_capture $(if $(PUBLIC_SITES),--sites $(PUBLIC_SITES)) $(if $(PUBLIC_MAX_TASKS),--max-tasks $(PUBLIC_MAX_TASKS))

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

pipeline-overnight: ## Run stages 3-7 overnight with Telegram notification
	python -m repo_pipeline.orchestrator --overnight \
		--workers $(PIPELINE_WORKERS) \
		--budget $(PIPELINE_BUDGET) \
		$(if $(PIPELINE_LIMIT),--limit $(PIPELINE_LIMIT))

pipeline-until-limit: ## Loop stages 3-7 until Claude usage hits PIPELINE_LIMIT_PCT% (default 100)
	python -m repo_pipeline.orchestrator --until-limit $(PIPELINE_LIMIT_PCT) \
		--workers $(PIPELINE_WORKERS) \
		--budget $(PIPELINE_BUDGET) \
		$(if $(PIPELINE_LIMIT),--limit $(PIPELINE_LIMIT))

pipeline-think: ## Stage 7: fill <think> blocks for captured repos (decoupled from capture)
	python -m repo_pipeline.orchestrator --stage 7 \
		--workers $(PIPELINE_WORKERS) \
		--budget $(PIPELINE_BUDGET) \
		$(if $(PIPELINE_LIMIT),--limit $(PIPELINE_LIMIT))

pipeline-status: ## Show pipeline registry stats
	python -m repo_pipeline.orchestrator --status

test-notifier: ## Test Telegram bot notification
	python -m repo_pipeline.notifier --test

# ---------- Evaluation (ScreenSpot grounding) ----------

install-eval: ## Install ScreenSpot eval dependencies (GPU machine)
	pip install -r requirements-eval.txt

eval-screenspot: ## Eval grounding for all models in EVAL_MODELS (before/after SFT)
	python eval_screenspot.py run \
		--models-config $(EVAL_MODELS) \
		--backend $(EVAL_BACKEND) \
		--dataset $(EVAL_DATASET) \
		--split $(EVAL_SPLIT) \
		--coord-space $(EVAL_COORD_SPACE) \
		--viewport $(EVAL_VIEWPORT) \
		--output-dir $(EVAL_OUTPUT_DIR) \
		$(if $(EVAL_LIMIT),--limit $(EVAL_LIMIT))

test-eval: ## Smoke-test the eval harness (no model/GPU/network)
	python eval_screenspot.py run --smoke-test
# ---------- Dashboard ----------

dashboard: ## Launch trajectory SFT-readiness dashboard (Streamlit, localhost:8501)
	streamlit run repo_pipeline/dashboard/app.py
