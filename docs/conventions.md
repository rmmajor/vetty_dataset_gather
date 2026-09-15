# Conventions

## Environment variables

Every variable below is read exactly once, at import of `cua/settings.py`, into a frozen section
object. No other module reads the environment **except** the marked runtime reads: a handful of
files that execute where the `cua` package is not importable — the injected conftests, the program
string `auth_flow_smoke` runs inside the sandbox, and the Modal serve/merge images — keep their own
read behind a `# env-read: <reason>` marker, which `scripts/check_env_reads.py` enforces. Those
variables are therefore not in this table; the marker line names each one.

Regenerate this table with `python -m cua.settings --table`.

| variable | default | settings field | used by |
| --- | --- | --- | --- |
| `SANDBOX_MEM_LIMIT` | `2g` | `SANDBOX.mem_limit` | cua.pipeline.sandbox |
| `SANDBOX_MEMSWAP_LIMIT` | `SANDBOX.mem_limit` | `SANDBOX.memswap_limit` | cua.pipeline.sandbox |
| `SANDBOX_PIDS_LIMIT` | `1024` | `SANDBOX.pids_limit` | cua.pipeline.sandbox |
| `SANDBOX_CPUS` | `2` | `SANDBOX.cpus` | cua.pipeline.sandbox |
| `SANDBOX_TMP_SIZE` | `1g` | `SANDBOX.tmp_size` | cua.pipeline.sandbox |
| `SANDBOX_SHM_SIZE` | `256m` | `SANDBOX.shm_size` | cua.pipeline.sandbox |
| `SANDBOX_VM_RESERVE` | `1.5g` | `SANDBOX.vm_reserve` | cua.pipeline.sandbox |
| `SANDBOX_APP_READY_TIMEOUT` | `60` | `SANDBOX.app_ready_timeout` | cua.pipeline.app_runner, cua.pipeline.driver_session |
| `SANDBOX_APP_BUILD_TIMEOUT` | `420` | `SANDBOX.app_build_timeout` | cua.pipeline.app_runner |
| `PIPELINE_MAX_RETRIES` | `3` | `PIPELINE.max_retries` | cua.pipeline._processing |
| `PIPELINE_RETRY_BACKOFF` | `5` | `PIPELINE.retry_backoff` | cua.pipeline._processing |
| `PIPELINE_CAPTURE_MODE` | `agent` | `PIPELINE.capture_mode` | cua.pipeline.schemas |
| `PIPELINE_AUTH_FLOW_CAPTURE` | `0` | `PIPELINE.auth_flow_capture` | cua.pipeline.stage6_generate |
| `PIPELINE_AUTH_DISCOVERED_RATIO` | `0.25` | `PIPELINE.auth_discovered_ratio` | cua.pipeline.stage6_generate |
| `PIPELINE_LANGUAGE` | — | `PIPELINE.language` | cua.pipeline.registry |
| `PIPELINE_CREDENTIALS_DIR` | `credentials` | `PIPELINE.credentials_dir` | cua.pipeline.app_secrets |
| `STAGE3_BUDGET_USD` | `2.0` | `PIPELINE.stage3_budget_usd` | cua.pipeline.stage3_shallow_score |
| `STAGE3_GATE_APP_MIN` | `3` | `PIPELINE.stage3_gate_app_min` | cua.pipeline.stage3_shallow_score, cua.pipeline.dashboard.pages.4_Rejections |
| `STAGE3_GATE_RUNNABILITY_MIN` | `2` | `PIPELINE.stage3_gate_runnability_min` | cua.pipeline.stage3_shallow_score, cua.pipeline.dashboard.pages.4_Rejections |
| `STAGE3_GATE_INTERACTION_MIN` | `2` | `PIPELINE.stage3_gate_interaction_min` | cua.pipeline.stage3_shallow_score, cua.pipeline.dashboard.pages.4_Rejections |
| `STAGE4_BUDGET_USD` | `5.0` | `PIPELINE.stage4_budget_usd` | cua.pipeline.stage4_deep_analyze |
| `STAGE4_TOP_K_PER_PASS` | `15` | `PIPELINE.stage4_top_k_per_pass` | cua.pipeline.stage4_deep_analyze |
| `STAGE4_DEEP_SCORE_THRESHOLD` | `4` | `PIPELINE.stage4_deep_score_threshold` | cua.pipeline.stage4_deep_analyze, cua.pipeline.dashboard.pages.4_Rejections |
| `STAGE4_MIN_EST_TRAJECTORIES` | `3` | `PIPELINE.stage4_min_est_trajectories` | cua.pipeline.stage4_deep_analyze |
| `STAGE4_TOY_COMPLEXITY_MAX` | `2` | `PIPELINE.stage4_toy_complexity_max` | cua.pipeline.stage4_deep_analyze |
| `STAGE4_WORKERS` | `3` | `PIPELINE.stage4_workers` | cua.pipeline.orchestrator |
| `STAGE5_WORKERS` | `2` | `PIPELINE.stage5_workers` | cua.pipeline.orchestrator |
| `STAGE6_WORKERS` | `2` | `PIPELINE.stage6_workers` | cua.pipeline.orchestrator |
| `BOOTABLE_GATE` | `1` | `PIPELINE.bootable_gate` | cua.pipeline.stage2_filter |
| `BOOTABLE_GATE_AGGRESSIVE` | `0` | `PIPELINE.bootable_gate_aggressive` | cua.pipeline.stage2_filter |
| `BOOTABLE_GATE_WORKERS` | `8` | `PIPELINE.bootable_gate_workers` | cua.pipeline.stage2_filter |
| `BOOT_GATE_MIN_ELEMENTS` | `5` | `PIPELINE.boot_gate_min_elements` | cua.pipeline.stage5_trial_run |
| `AGENT_TASKS_PER_REPO` | `8` | `PIPELINE.tasks_per_repo` | cua.pipeline.task_proposer |
| `TASK_PROPOSE_BUDGET_USD` | `2.0` | `PIPELINE.task_propose_budget_usd` | cua.pipeline.task_proposer |
| `INSTRUCTION_BUDGET_USD` | `0.30` | `PIPELINE.instruction_budget_usd` | cua.pipeline.stage6_generate |
| `TASK_TYPE_BUDGET_USD` | `0.30` | `PIPELINE.task_type_budget_usd` | cua.pipeline.stage6_generate |
| `THINK_BUDGET_USD` | `0.30` | `PIPELINE.think_budget_usd` | cua.pipeline.think_generator |
| `THINK_SESSION_BUDGET_USD` | `5.0` | `PIPELINE.think_session_budget_usd` | cua.pipeline.think_generator |
| `POLISH_BUDGET_USD` | `0.50` | `PIPELINE.polish_budget_usd` | cua.pipeline.think_generator |
| `QA_VOICE_BUDGET_USD` | `0.50` | `PIPELINE.qa_voice_budget_usd` | cua.pipeline.instruction_style_pass |
| `OUTCOME_WINDOW_STEPS` | `12` | `PIPELINE.outcome_window_steps` | cua.pipeline.outcome_relabel |
| `OUTCOME_BUDGET_USD` | `1.00` | `PIPELINE.outcome_budget_usd` | cua.pipeline.outcome_relabel, cua.pipeline.outcome_relabel_smoke |
| `OUTCOME_IMAGE_TOKENS` | `350` | `PIPELINE.outcome_image_tokens` | cua.pipeline.outcome_relabel |
| `OUTCOME_PRICE_IN_PER_M` | `2.0` | `PIPELINE.outcome_price_in_per_m` | cua.pipeline.outcome_relabel |
| `OUTCOME_PRICE_OUT_PER_M` | `10.0` | `PIPELINE.outcome_price_out_per_m` | cua.pipeline.outcome_relabel |
| `OUTCOME_CALL_OVERHEAD_USD` | `0.08` | `PIPELINE.outcome_call_overhead_usd` | cua.pipeline.outcome_relabel |
| `PUBLIC_TASKS_PER_SITE` | `6` | `PIPELINE.public_tasks_per_site` | cua.pipeline.public_capture |
| `DRIVER_PORT` | `8765` | `PIPELINE.driver_port` | cua.pipeline.driver_session, cua.rl.backends |
| `DRIVER_PLAYWRIGHT_CORE_VERSION` | `1.55.0` | `PIPELINE.driver_playwright_core_version` | cua.pipeline.driver_session |
| `DRIVER_READY_TIMEOUT` | `60` | `PIPELINE.driver_ready_timeout` | cua.pipeline.driver_session |
| `AGENT_MAX_ACTIONS` | `60` | `CAPTURE.max_actions_per_task` | cua.pipeline.agent_capture |
| `AGENT_TASK_BUDGET_USD` | `25.0` | `CAPTURE.task_budget_usd` | cua.pipeline.agent_capture |
| `AGENT_MAX_TURNS` | `150` | `CAPTURE.max_turns` | cua.pipeline.agent_capture |
| `VERIFY_BUDGET_USD` | `0.30` | `CAPTURE.verify_budget_usd` | cua.pipeline.agent_capture |
| `AGENT_RELABEL_MIN_STEPS` | `3` | `CAPTURE.relabel_min_steps` | cua.pipeline.relabel_recoverable |
| `AGENT_DEGENERATE_LOOP_FRACTION` | `0.6` | `CAPTURE.degenerate_loop_fraction` | cua.pipeline.agent_capture |
| `AGENT_DEGENERATE_MIN_REPEATS` | `3` | `CAPTURE.degenerate_min_repeats` | cua.pipeline.agent_capture |
| `AGENT_CONTAINER_TIMEOUT` | `10800` | `CAPTURE.container_timeout` | cua.pipeline.stage6_generate |
| `AGENT_TASK_TIME_RESERVE` | `900` | `CAPTURE.task_time_reserve` | cua.pipeline.stage6_generate |
| `AGENT_MAX_KEPT_PER_REPO` | `10` | `CAPTURE.max_kept_per_repo` | cua.pipeline.stage6_generate |
| `AGENT_MAX_CONSECUTIVE_FAILS` | `3` | `CAPTURE.max_consecutive_fails` | cua.pipeline.stage6_generate |
| `CAPTURE_MAX_ATTEMPTS` | `2` | `CAPTURE.max_attempts` | cua.pipeline.public_capture, cua.pipeline.stage6_generate |
| `PUBLIC_CONTAINER_TIMEOUT` | `10800` | `CAPTURE.public_container_timeout` | cua.pipeline.public_capture |
| `WATCHDOG_POLICY` | `llm` | `CAPTURE.watchdog_policy` | cua.pipeline.safety_watchdog |
| `WATCHDOG_BUDGET_USD` | `0.20` | `CAPTURE.watchdog_budget_usd` | cua.pipeline.safety_watchdog |
| `PERSONA_EMAIL_DOMAIN` | `example.com` | `CAPTURE.persona_email_domain` | cua.pipeline.persona |
| `PERSONA_PHONE_PREFIX` | `+1202555` | `CAPTURE.persona_phone_prefix` | cua.pipeline.persona |
| `MAIL_HOST_SMTP_PORT` | `13025` | `CAPTURE.mail_host_smtp_port` | cua.pipeline.mail_service |
| `MAIL_HOST_IMAP_PORT` | `13143` | `CAPTURE.mail_host_imap_port` | cua.pipeline.mail_service |
| `MAIL_HOST_WEBMAIL_PORT` | `18080` | `CAPTURE.mail_host_webmail_port` | cua.pipeline.mail_service |
| `OPERATOR_PII` | `""` | `CAPTURE.operator_pii` | cua.pipeline.operator_pii |
| `OPERATOR_PII_FILE` | — | `CAPTURE.operator_pii_file` | cua.pipeline.operator_pii |
| `SDK_QUERY_TIMEOUT` | `2400` | `SDK.query_timeout_s` | cua.pipeline.agent_runner, cua.pipeline.sdk_runner |
| `CLAUDE_CLI_PATH` | — | `SDK.claude_cli_path` | cua.pipeline.sdk_runner |
| `PREV_IMAGE_SIZE` | `640x360` | `PROMPT.prev_image_size` | cua.prompt.effects, cua.prompt.format |
| `EFFECT_NO_CHANGE_MAX_FRACTION` | `0.0005` | `PROMPT.effect_no_change_max_fraction` | cua.prompt.effects |
| `EFFECT_PIXEL_DELTA` | `16` | `PROMPT.effect_pixel_delta` | cua.prompt.effects |
| `EFFECT_PATH_MAX_CHARS` | `40` | `PROMPT.effect_path_max_chars` | cua.prompt.effects |
| `WASTE_CLICK_RADIUS_PX` | `10` | `CORPUS.waste_click_radius_px` | cua.sft.corpus_filters |
| `WASTE_LOOKBACK_STEPS` | `4` | `CORPUS.waste_lookback_steps` | cua.sft.corpus_filters |
| `MEANDER_KEEP_HEAD` | `5` | `CORPUS.meander_keep_head` | cua.sft.corpus_filters |
| `MEANDER_KEEP_TAIL` | `5` | `CORPUS.meander_keep_tail` | cua.sft.corpus_filters |
| `MEANDER_MID_KEEP_EVERY` | `6` | `CORPUS.meander_mid_keep_every` | cua.sft.corpus_filters |
| `KEYPRESS_MAX_REPEAT` | `6` | `CORPUS.keypress_max_repeat` | cua.sft.corpus_filters |
| `GROUNDING_SMALL_OVERSAMPLE` | `3` | `GROUNDING.small_oversample` | cua.sft.grounding |
| `GROUNDING_PER_GROUP_CAP` | `40` | `GROUNDING.per_group_cap` | cua.sft.grounding |
| `GROUNDING_SEED` | `0` | `GROUNDING.seed` | cua.sft.grounding |
| `EVAL_RUNS_DIR` | `eval_runs` | `EVAL.runs_dir` | cua.eval.student |
| `EVAL_EPISODE_TIME_RESERVE` | `900` | `EVAL.episode_time_reserve` | cua.eval.student |
| `EVAL_CONTAINER_TIMEOUT` | `10800` | `EVAL.container_timeout` | cua.eval.student |
| `EVAL_MAX_PARSE_FAILURES` | `3` | `EVAL.max_parse_failures` | cua.eval.student |
| `EVAL_MAX_UNPRODUCTIVE_STEPS` | `5` | `EVAL.max_unproductive_steps` | cua.eval.student |
| `EVAL_COORD_SPACE_SUSPECT_RATE` | `0.25` | `EVAL.coord_space_suspect_rate` | cua.eval.student |
| `EVAL_ENDPOINT` | `""` | `EVAL.endpoint` | cua.eval.student |
| `EVAL_MODEL` | `""` | `EVAL.model` | cua.eval.student |
| `EVAL_ADAPTER` | `""` | `EVAL.adapter` | cua.eval.student |
| `EVAL_API_KEY` | `""` | `EVAL.api_key` | cua.eval.student |
| `EVAL_COORD_SPACE` | `normalized_1000` | `EVAL.coord_space` | cua.eval.student |
| `EVAL_ENV_FILE` | `""` | `EVAL.env_file` | cua.eval.task_sources |
| `VERIFY_MID_SCREENSHOTS` | `2` | `EVAL.verify_mid_screenshots` | cua.pipeline.agent_capture |
| `HF_ENDPOINT` | `https://huggingface.co` | `EVAL.hf_endpoint` | cua.eval.task_sources |
| `HF_TOKEN` | — | `EVAL.hf_token` | cua.cloud.vast.config |
| `HUGGING_FACE_HUB_TOKEN` | — | `EVAL.hugging_face_hub_token` | — |
| `HUGGINGFACE_HUB_TOKEN` | — | `EVAL.huggingface_hub_token` | — |
| `HF_READ_TOKEN` | — | `EVAL.hf_read_token` | — |
| `SCREENSPOT_IMAGE_SIZE` | `native` | `EVAL.screenspot_image_size` | cua.eval.screenspot |
| `SCREENSPOT_ENDPOINT` | `""` | `EVAL.screenspot_endpoint` | cua.eval.screenspot |
| `SCREENSPOT_API_KEY` | `""` | `EVAL.screenspot_api_key` | cua.eval.screenspot |
| `SCREENSPOT_WORKERS` | `8` | `EVAL.screenspot_workers` | cua.eval.screenspot |
| `SCREENSPOT_CLAUDE_WORKERS` | `4` | `EVAL.screenspot_claude_workers` | cua.eval.screenspot |
| `SCREENSPOT_CLAUDE_BUDGET_USD` | `0.30` | `EVAL.screenspot_claude_budget_usd` | cua.eval.screenspot |
| `CUA_TRAIN_GPU` | `A100-80GB` | `CLOUD.train_gpu` | cua.cloud.modal.train |
| `CUA_RUN_NAME` | `sft-v1` | `CLOUD.run_name` | cua.cloud.modal.train |
| `CUA_TRAIN_TIMEOUT` | `36000` | `CLOUD.train_timeout` | cua.cloud.modal.train |
| `CUA_TRAIN_COMMIT_EVERY_S` | `600` | `CLOUD.train_commit_every_s` | cua.cloud.modal.train |
| `CUA_GRPO_GPU` | `A100-80GB:2` | `CLOUD.grpo_gpu` | cua.cloud.modal.grpo |
| `CUA_GRPO_TIMEOUT` | `32400` | `CLOUD.grpo_timeout` | cua.cloud.modal.grpo |
| `CUA_GRPO_COMMIT_EVERY_S` | `600` | `CLOUD.grpo_commit_every_s` | cua.cloud.modal.grpo |
| `VAST_API_KEY` | — | `CLOUD.vast_api_key` | cua.cloud.vast.config |
| `REMOTE_TRAIN_STATE_FILE` | `.remote_train_state.json` | `CLOUD.remote_train_state_file` | cua.cloud.vast.state |
| `REMOTE_TRAIN_BAD_OFFERS` | `.remote_train_bad_offers.json` | `CLOUD.remote_train_bad_offers` | cua.cloud.vast.state |
| `REMOTE_TRAIN_EXCLUDE_COUNTRIES` | `""` | `CLOUD.remote_train_exclude_countries` | cua.cloud.vast.config |
| `REMOTE_TRAIN_INSTANCE_LABEL` | `""` | `CLOUD.remote_train_instance_label` | cua.cloud.vast.config |
| `RL_ENV_HOST` | `127.0.0.1` | `RL.env_host` | cua.rl.service |
| `RL_ENV_PORT` | `8900` | `RL.env_port` | cua.rl.service |
| `RL_ENV_TOKEN` | `""` | `RL.env_token` | cua.cloud.modal.grpo, cua.rl.service, cua.rl.trainer.swift_plugin |
| `RL_ENV_URL` | `http://127.0.0.1:8900` | `RL.env_url` | cua.rl.trainer.swift_plugin |
| `RL_SERVE_PORT` | `8001` | `RL.serve_port` | cua.rl.rollout |
| `RL_RUN_DIR` | `/workspace/rl_runs/current` | `RL.run_dir` | cua.rl.trainer.swift_plugin |
| `RL_RUN_ID` | `grpo` | `RL.run_id` | cua.rl.trainer.swift_plugin |
| `RL_RESET_TIMEOUT_S` | `45` | `RL.reset_timeout_s` | cua.rl.service |
| `RL_EPISODE_IDLE_S` | `600` | `RL.episode_idle_s` | cua.rl.service |
| `RL_CONTAINER_TIMEOUT` | `14400` | `RL.container_timeout` | cua.rl.backends |
| `RL_RECYCLE_RESERVE_S` | `1200` | `RL.recycle_reserve_s` | cua.rl.backends |
| `RL_DRIVERS_PER_SANDBOX` | `1` | `RL.drivers_per_sandbox` | cua.rl.backends |
| `RL_LOOP_GUARD_REPEATS` | `5` | `RL.loop_guard_repeats` | cua.rl.backends, cua.rl.rollout, cua.rl.service, tests.test_backends |
| `RL_RESET_WAIT_S` | `1800` | `RL.reset_wait_s` | cua.rl.trainer.core |
| `RL_STEP_WAIT_S` | `2400` | `RL.step_wait_s` | cua.rl.trainer.core |
| `RL_VERIFIER_CONCURRENCY` | `2` | `RL.verifier_concurrency` | cua.rl.judge |
| `RL_VERIFY_TIMEOUT_S` | `2700` | `RL.verify_timeout_s` | cua.rl.judge |
| `RL_VERIFY_RETRY_S` | `300` | `RL.verify_retry_s` | cua.rl.judge, tests.test_judge |
| `RL_VERIFY_WAIT_S` | `1800` | `RL.verify_wait_s` | cua.rl.judge |
| `RL_POLICY_BACKOFF_S` | `5` | `RL.policy_backoff_s` | cua.rl.policies |
| `RL_POLICY_WAIT_S` | `600` | `RL.policy_wait_s` | cua.rl.policies |
| `RL_POLICY_API_KEY` | `""` | `RL.policy_api_key` | cua.rl.policies |
| `CUDA_VISIBLE_DEVICES` | `0` | `RL.cuda_visible_devices` | cua.rl.train |
| `TELEGRAM_BOT_TOKEN` | `""` | `NOTIFY.telegram_bot_token` | cua.pipeline.notifier |
| `TELEGRAM_CHAT_ID` | `""` | `NOTIFY.telegram_chat_id` | cua.pipeline.notifier |
| `GITHUB_TOKEN` | `""` | `GITHUB.token` | cua.pipeline.orchestrator |
