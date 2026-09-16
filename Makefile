# Every target is phony except the $(TASKS_FILE) file rule, whose `$(...)` name this pattern cannot match.
.PHONY: $(shell grep -oE '^[A-Za-z0-9_-]+:' $(lastword $(MAKEFILE_LIST)) | tr -d ':' | sort -u)

# $(call require,VAR[,message]) tests the shell's $VAR, never the expanded value, so `make -n` never prints a secret.
define require
@test -n "$$$(1)" || (echo 'ERROR: $(if $(2),$(2),$(1) not set)' && exit 1)
endef

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
GATE_SAMPLE   ?=
# ScreenSpot grounding eval
EVAL_MODELS      ?= eval_models.yaml
EVAL_DATASET     ?= rootsautomation/ScreenSpot
EVAL_SPLIT       ?= test
EVAL_BACKEND     ?= swift
EVAL_VIEWPORT    ?= 1280x720
EVAL_OUTPUT_DIR  ?= eval_results
# Student episode eval
EVAL_ENDPOINT    ?=
EVAL_MODEL       ?=
EVAL_ADAPTER     ?=
EVAL_API_KEY     ?=
EVAL_LABEL       ?= baseline
EVAL_RUNS        ?=
EVAL_ARGS        ?=
# Shared by both eval lanes.
# Empty coord space: each ScreenSpot spec's own coord_space in EVAL_MODELS wins.
EVAL_COORD_SPACE ?=
EVAL_LIMIT       ?=

help: ## Show this help
	@grep -E '^[A-Za-z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------- Setup ----------

setup: ## Install dependencies and Playwright chromium
	pip install -e ".[path_a]"
	playwright install chromium

install-training: ## Install ms-swift for training (GPU machine)
	pip install "ms-swift[llm]"

install-all: setup install-training ## Install everything for a fresh GPU instance

# ---------- Pipeline steps ----------

load-tasks: $(TASKS_FILE) ## Load Mind2Web tasks into TASKS_FILE
$(TASKS_FILE):
	python -m cua.path_a.load_mind2web_tasks --output $(TASKS_FILE) $(if $(TASK_LIMIT),--limit $(TASK_LIMIT))

record: $(TASKS_FILE) ## Record trajectories via GPT-5.4 CUA
	$(call require,OPENAI_API_KEY)
	python -m cua.path_a.recorder \
		--tasks $(TASKS_FILE) \
		--output $(TRAJ_DIR)/ \
		--max-steps $(MAX_STEPS) \
		--viewport $(VIEWPORT) \
		--resume

convert: ## Convert trajectories to ms-swift SFT format (waste clicks dropped, meandering subsampled, health-gated)
	python -m cua.sft convert \
		--input $(TRAJ_DIR)/ \
		--output $(TRAIN_DIR)/ \
		--val-split $(VAL_SPLIT) \
		--exclude-waste-steps \
		--meandering-mode subsample

train: convert ## Launch ms-swift LoRA fine-tuning
	python -m cua.sft train \
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
	pip install -e ".[pipeline,dashboard]"
	playwright install chromium

build-sandbox: ## Build Docker sandbox image (Playwright, arch-native)
	docker build -t repo-pipeline-sandbox:latest -f docker/Dockerfile.sandbox docker/

build-sandbox-selenium: ## Build Selenium sandbox image (linux/amd64; needed for Selenium repos)
	docker build --platform=linux/amd64 -t repo-pipeline-sandbox-selenium:latest -f docker/Dockerfile.sandbox-selenium docker/

test-sandbox: build-sandbox ## Test sandbox container lifecycle
	python -m cua.pipeline.sandbox --test

test-driver: ## Docker-only scripted (no-LLM) smoke test of the agent-capture browser driver
	python -m cua.pipeline.smoke.driver

test-auth-flow: ## Docker-only scripted (no-LLM) check of auth-flow capture: mail stack, masked secrets, session reuse
	python -m cua.pipeline.smoke.auth_flow

test-verify-gate: ## Scripted (no-LLM/Docker/network) check that the capture success gate never rejects a verified trajectory
	python -m cua.pipeline.smoke.verify_gate

test-qa-verify: ## Scripted (no-LLM/Docker/network) check that the qa_verify style reaches proposer, teacher, verifier, metadata and config
	python -m cua.pipeline.smoke.qa_verify

test-rl-unit: ## Unit tests for the RL rollout harness (no Docker, no LLM)
	python -m tests

test-corpus-audit: ## Scripted (no-LLM/Docker/network) check of the shared trainability gate and audit checks
	python -m cua.pipeline.smoke.corpus_audit

test-instruction-style: ## Scripted (no-LLM/Docker/network) check of the instruction-style pass
	python -m cua.pipeline.smoke.instruction_style

test-outcome-relabel: ## Scripted (no-LLM/Docker/network) check of the outcome-relabel pass
	python -m cua.pipeline.smoke.outcome_relabel

fixtures-build: ## Rewrite the behaviour-regression fixtures under tests/fixtures/
	python -m tests.regression_fixtures build

fixtures-check: ## Fail if today's behaviour differs from the stored regression fixtures
	python -m tests.regression_fixtures check

check: test-rl-unit test-eval test-verify-gate test-qa-verify test-remote test-corpus-audit test-instruction-style test-outcome-relabel registry-check conventions-check fixtures-check lint ## Tier-1 gate: every no-Docker, no-LLM, no-network self-test plus the regression fixtures

test-rl: ## Docker-only end-to-end smoke of the RL harness (static page + scripted policy, no LLM)
	python -m cua.rl.smoke

rl-tasks: ## Build + check the RL task suite (disjoint from anything trained on; RL_TRAJ_DIR overrides trajectories/)
	python -m cua.rl.tasks build $(if $(RL_TRAJ_DIR),--trajectories $(RL_TRAJ_DIR)) && python -m cua.rl.tasks check $(if $(RL_TRAJ_DIR),--trajectories $(RL_TRAJ_DIR))

rl-candidates: ## Build the oracle-checked frontier candidate pool for the pass@k sweep
	python -m cua.rl.tasks build-candidates $(if $(RL_TRAJ_DIR),--trajectories $(RL_TRAJ_DIR))

rl-analyze: ## Diagnose a rollout run (RL_RUN=rl_runs/<id>/iter_000); use `python -m cua.rl.analysis bnorm` for adapter norms
	python -m cua.rl.analysis rollouts $(RL_RUN)

rl-serve: ## Start the RL env service for the GPU trainer (127.0.0.1:8900; RL_SPLIT, RL_WORKERS, RL_ARGS)
	python -m cua.rl.service --split $(or $(RL_SPLIT),train) --workers $(or $(RL_WORKERS),3) $(RL_ARGS)

rl-rollout: ## Local rollouts against an OpenAI-compatible endpoint (RL_ENDPOINT, RL_MODEL, RL_K, RL_ARGS)
	python -m cua.rl.rollout --endpoint $(or $(RL_ENDPOINT),http://127.0.0.1:8001/v1) --model $(or $(RL_MODEL),policy) --k $(or $(RL_K),4) $(RL_ARGS)

rl-eval: ## E3 student eval on held-out tasks (RL_ENDPOINT, RL_MODEL, RL_BASELINE=report.json, RL_ARGS)
	python -m cua.rl.eval --endpoint $(or $(RL_ENDPOINT),http://127.0.0.1:8001/v1) --model $(or $(RL_MODEL),policy) $(if $(RL_BASELINE),--baseline $(RL_BASELINE)) $(RL_ARGS)

audit-corpus: ## Mechanical quality gate over trainable trajectories (exit 1 on defects)
	python -m cua.pipeline.corpus.audit trajectories/

capture-public: ## Teacher capture on curated public sites (real-web diversity lane)
	python -m cua.pipeline.lanes.public $(if $(PUBLIC_SITES),--sites $(PUBLIC_SITES)) $(if $(PUBLIC_MAX_TASKS),--max-tasks $(PUBLIC_MAX_TASKS))

provision-human: ## Human-in-the-loop: headed browser to sign a burner into one login-gated SITE (SITE=name)
	$(call require,SITE,set SITE=<site-name> (a site from the indie/logingated config))
	python -m playwright install chromium
	python -m cua.pipeline.lanes.provision_human $(SITE) $(if $(SRC_CONFIG),--config $(SRC_CONFIG))

capture-human: ## Capture tasks on human-provisioned sites (reuses saved sessions; SITES=a,b optional)
	python -m cua.pipeline.lanes.public --config cua/pipeline/sites/public_sites_human.json --no-propose $(if $(SITES),--sites $(SITES))

capture-search: ## Search-goal capture lane (SFT v2 gap 2: constrained info-seeking on mainstream sites)
	python -m cua.pipeline.lanes.public --config cua/pipeline/sites/public_sites_search.json $(if $(SITES),--sites $(SITES)) $(if $(PUBLIC_MAX_TASKS),--max-tasks $(PUBLIC_MAX_TASKS))

capture-qa-verify: ## QA-verification capture lane (SFT v3: test-objective tasks that change, re-open/reload, then confirm on signed-in sites)
	python -m cua.pipeline.lanes.public --config cua/pipeline/sites/public_sites_qa_verify.json $(if $(SITES),--sites $(SITES)) $(if $(PUBLIC_MAX_TASKS),--max-tasks $(PUBLIC_MAX_TASKS))

polish-pass: ## Stage-7 thought polish for trainable teacher_raw dirs outside the registry flow
	python -m cua.pipeline.corpus.polish trajectories/

outcome-relabel: ## Outcome-checked thought relabel over trainable dirs (dry-run by default; APPLY=1 to write; ESTIMATE=1 to price)
	python -m cua.pipeline.corpus.outcome_relabel trajectories/ $(if $(APPLY),--apply,--dry-run) $(if $(ESTIMATE),--estimate,)

relabel-recoverable: ## Audited hindsight-relabel pass (dry-run by default; APPLY=1 to write)
	python -m cua.pipeline.corpus.relabel_recoverable trajectories/ $(if $(APPLY),--apply,--dry-run)

wall-demos: ## Synthesize terminal wall supervision from blocked runs (dry-run by default; APPLY=1)
	python -m cua.pipeline.corpus.wall_supervision trajectories/ $(if $(APPLY),--apply,--dry-run)

instruction-style: ## Rewrite procedure-flavored instructions into goals (dry-run by default; APPLY=1)
	python -m cua.pipeline.corpus.instruction_style trajectories/ $(if $(APPLY),--apply,--dry-run)

instruction-qa-voice: ## Restate re-checking trajectories' instructions as QA test objectives (dry-run previews old -> new; APPLY=1 writes)
	python -m cua.pipeline.corpus.instruction_style trajectories/ --mode qa_verify $(if $(APPLY),--apply,--dry-run)

pipeline-discover: ## Stage 1: Discover QA repos on GitHub
	$(call require,GITHUB_TOKEN)
	python -m cua.pipeline.orchestrator --stage 1

pipeline-filter: ## Stage 2: Filter repos by metadata
	python -m cua.pipeline.orchestrator --stage 2

pipeline-score: ## Stage 3: Shallow score repos with Claude
	python -m cua.pipeline.orchestrator --stage 3 \
		--workers $(PIPELINE_WORKERS) \
		--budget $(PIPELINE_BUDGET) \
		$(if $(PIPELINE_LIMIT),--limit $(PIPELINE_LIMIT))

pipeline-overnight: ## Run stages 3-7 overnight with Telegram notification
	python -m cua.pipeline.orchestrator --overnight \
		--workers $(PIPELINE_WORKERS) \
		--budget $(PIPELINE_BUDGET) \
		$(if $(PIPELINE_LIMIT),--limit $(PIPELINE_LIMIT))

pipeline-until-limit: ## Loop stages 3-7 until Claude usage hits PIPELINE_LIMIT_PCT% (default 100)
	python -m cua.pipeline.orchestrator --until-limit $(PIPELINE_LIMIT_PCT) \
		--workers $(PIPELINE_WORKERS) \
		--budget $(PIPELINE_BUDGET) \
		$(if $(PIPELINE_LIMIT),--limit $(PIPELINE_LIMIT))

pipeline-think: ## Stage 7: fill <think> blocks for captured repos (decoupled from capture)
	python -m cua.pipeline.orchestrator --stage 7 \
		--workers $(PIPELINE_WORKERS) \
		--budget $(PIPELINE_BUDGET) \
		$(if $(PIPELINE_LIMIT),--limit $(PIPELINE_LIMIT))

pipeline-status: ## Show pipeline registry stats
	python -m cua.pipeline.orchestrator --status

gate-validate: ## Measure the bootable-app gate's precision floor and recall (GATE_SAMPLE=N, no LLM)
	python -m cua.pipeline.tools.gate_tools validate $(GATE_SAMPLE)

gate-regate: ## Re-gate the stage-3/4 backlog through the bootable-app gate (dry-run by default; APPLY=1 to reject)
	python -m cua.pipeline.tools.gate_tools regate $(GATE_SAMPLE) $(if $(APPLY),--apply)

test-notifier: ## Test Telegram bot notification
	python -m cua.pipeline.notifier --test

# ---------- Student eval (before/after a training run) ----------

# $(call eval-student-recipe,task-source,endpoint-hint[,limit-extra-flags]) — the one student-eval body both lanes run.
define eval-student-recipe
$(call require,EVAL_ENDPOINT,$(2))
python -m cua.eval.student run \
	--task-source $(1) \
	--endpoint $(EVAL_ENDPOINT) \
	$(if $(EVAL_MODEL),--model $(EVAL_MODEL)) \
	$(if $(EVAL_ADAPTER),--adapter $(EVAL_ADAPTER)) \
	$(if $(EVAL_API_KEY),--api-key $(EVAL_API_KEY)) \
	--label $(EVAL_LABEL) \
	$(if $(EVAL_COORD_SPACE),--coord-space $(EVAL_COORD_SPACE)) \
	$(if $(EVAL_LIMIT),--limit $(EVAL_LIMIT) $(3)) \
	$(EVAL_ARGS)
endef

eval-online-mind2web: ## Run Online-Mind2Web against a served model (EVAL_ENDPOINT=, EVAL_MODEL=, EVAL_LABEL=before|after)
	$(call eval-student-recipe,online_mind2web,set EVAL_ENDPOINT=http://<host>:8000/v1 (the served model; the browser runs locally),--stratify)

eval-held-out: ## Same harness over our own never-captured public-site tasks
	$(call eval-student-recipe,held_out,set EVAL_ENDPOINT=http://<host>:8000/v1)

eval-compare: ## Before/after table from two or more runs (EVAL_RUNS="eval_runs/before_... eval_runs/after_...")
	$(call require,EVAL_RUNS,set EVAL_RUNS="eval_runs/<before_id> eval_runs/<after_id>")
	python -m cua.eval.student compare $(EVAL_RUNS)

eval-sources: ## List the registered eval task sources
	python -m cua.eval.student sources

test-student-eval: ## Self-test the eval harness (parser round-trip + mock replay episode + judge); the backend check needs Docker
	python -m cua.eval.student_smoke $(EVAL_ARGS)

# ---------- Evaluation (ScreenSpot grounding) ----------

install-eval: ## Install ScreenSpot eval dependencies (GPU machine)
	pip install -e ".[eval]"

eval-screenspot: ## Eval grounding for all models in EVAL_MODELS (before/after SFT)
	python -m cua.eval.screenspot run \
		--models-config $(EVAL_MODELS) \
		--backend $(EVAL_BACKEND) \
		--dataset $(EVAL_DATASET) \
		--split $(EVAL_SPLIT) \
		--viewport $(EVAL_VIEWPORT) \
		--output-dir $(EVAL_OUTPUT_DIR) \
		$(if $(EVAL_COORD_SPACE),--coord-space $(EVAL_COORD_SPACE)) \
		$(if $(EVAL_LIMIT),--limit $(EVAL_LIMIT))

test-eval: ## Smoke-test the eval harness (no model/GPU/network)
	python -m cua.eval.screenspot run --smoke-test

# ---------- Dashboard ----------

dashboard: ## Launch trajectory SFT-readiness dashboard (Streamlit, localhost:8501)
	streamlit run cua/pipeline/dashboard/app.py

# ---------- Remote training (Vast.ai; provider-pluggable) ----------

install-remote: ## Install the vastai CLI (local controller deps)
	pip install vastai

remote-train: ## Rent a GPU, train, pull adapter, and ALWAYS destroy the instance (needs VAST_API_KEY)
	$(call require,VAST_API_KEY)
	python -m cua.cloud.vast train $(if $(REMOTE_ARGS),$(REMOTE_ARGS))

remote-eval: ## Benchmark model(s) on a rented GPU, no training (EVAL_MODEL=... EVAL_LIMIT=...)
	$(call require,VAST_API_KEY)
	python -m cua.cloud.vast eval \
		$(foreach m,$(EVAL_MODEL),--model $(m)) \
		$(if $(EVAL_LIMIT),--limit $(EVAL_LIMIT)) \
		$(if $(REMOTE_ARGS),$(REMOTE_ARGS))

remote-train-dry: ## Dry run: print the cheapest geo-allowed offer, provision nothing
	$(call require,VAST_API_KEY)
	python -m cua.cloud.vast search

remote-train-cleanup: ## Destroy instances abandoned by dead runs (live sibling runs are spared; --all forces)
	$(call require,VAST_API_KEY)
	python -m cua.cloud.vast cleanup

test-remote: ## No-cost self-test of the remote-train lifecycle (mock provider; asserts teardown)
	python -m cua.cloud.vast run --smoke

# ---------- GRPO (RL) on the rented GPU ----------

remote-grpo: ## Rent a GPU, run GRPO against the local env service (make rl-serve first), pull adapters, ALWAYS destroy
	$(call require,VAST_API_KEY)
	python -m cua.cloud.vast grpo $(if $(REMOTE_ARGS),$(REMOTE_ARGS))

remote-serve: ## Rent a GPU and serve the student on 127.0.0.1:8001 through a tunnel (SERVE_ADAPTER=output, SERVE_SECONDS=0)
	$(call require,VAST_API_KEY)
	python -m cua.cloud.vast serve $(if $(SERVE_ADAPTER),--adapter $(SERVE_ADAPTER)) $(if $(SERVE_SECONDS),--for-seconds $(SERVE_SECONDS)) $(if $(REMOTE_ARGS),$(REMOTE_ARGS))

test-grpo: test-remote ## No-cost GRPO self-test: mock provider + mock env (teardown, tunnel, phase handshake) + command builders + unit tests
	python -m cua.rl.train print-commands --config rl_config.yaml
	python -m tests test_trainer_core test_rl_train test_tunnel

grpo-dataset: ## Write rl_tasks.jsonl (train split) for the GRPO trainer
	python -m cua.rl.train make-dataset --tasks cua/rl/rl_tasks.json --out rl_tasks.jsonl

rl-expose: ## Expose the local env service (127.0.0.1:8900) to Modal through a public ingress (RL_INGRESS_CMD overrides ngrok)
	$(call require,RL_ENV_TOKEN,set RL_ENV_TOKEN (the same value `make rl-serve` runs with))
	$(or $(RL_INGRESS_CMD),ngrok http 127.0.0.1:8900)

modal-grpo-smoke: ## Modal: merge + one GRPO step against the in-container mock env (validates pins, plugin, weight sync; ~$$2)
	modal run cua/cloud/modal/grpo.py --smoke --run-name $(or $(GRPO_RUN),grpo-smoke)

modal-grpo: ## Modal: full GRPO run against the exposed env service (RL_ENV_URL=https://... from rl-expose, RL_ENV_TOKEN)
	$(call require,RL_ENV_URL,set RL_ENV_URL=https://<ingress-host> (from make rl-expose))
	$(call require,RL_ENV_TOKEN,set RL_ENV_TOKEN)
	modal run cua/cloud/modal/grpo.py --run-name $(or $(GRPO_RUN),grpo-v1) --env-url $(RL_ENV_URL) $(if $(GRPO_MAX_STEPS),--max-steps $(GRPO_MAX_STEPS))

# ---------- Experiment registry ----------

registry: ## Re-render experiments/*.csv into the README registry block
	python experiments/render_registry.py --fill-ci
	python experiments/render_registry.py

registry-check: ## Fail if the README registry block is out of date
	python experiments/render_registry.py --check

conventions-check: ## Fail if the environment table in docs/conventions.md no longer matches cua/settings.py
	python -m cua.settings --check

# ---------- Lint / format ----------

LINT_PATHS     ?= .
ENV_READ_ALLOW ?= cua/settings.py
# Empty: no path is exempt from the convention checks; a finding anywhere fails `make lint`.
LINT_PENDING   ?=

lint: ## Ruff plus the repo's own comment/env/import conventions, all three enforcing
	ruff check .
	ruff format --check .
	python scripts/check_comments.py $(LINT_PATHS) --quiet --pending $(LINT_PENDING)
	python scripts/check_env_reads.py $(LINT_PATHS) --quiet --allow $(ENV_READ_ALLOW)
	python scripts/check_nested_imports.py $(LINT_PATHS) --quiet --pending $(LINT_PENDING)

fmt: ## Apply ruff formatting plus the safe import/upgrade autofixes
	ruff format .
	ruff check --fix .
