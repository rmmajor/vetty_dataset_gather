"""Every environment knob the package reads, resolved once at import into frozen sections."""

import argparse
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

# Not an environment knob: ms-swift READS MAX_PIXELS from the child env, the pipeline only ever writes it.
MAX_PIXELS_DEFAULT = 1003520


def parse_viewport(text):
    """`1280x720` -> (1280, 720); also the env cast for PREV_IMAGE_SIZE, re-exported by cua.prompt.coords."""
    width, height = text.lower().split("x")
    return int(width), int(height)


def flag_unless_falsy(text):
    """On unless `0`, empty or `false`/`False` — the auth-flow switch's convention."""
    return text not in ("0", "", "false", "False")


def flag_unless_zero(text):
    """On unless exactly `0` — the bootable gate's convention, where `false` still reads as on."""
    return text != "0"


def flag_if_one(text):
    """Off unless exactly `1` — the aggressive-gate opt-in convention."""
    return text == "1"


def _env(name, default, cast=str):
    """The one environment read in the package; a None default means "unset" and skips the cast."""
    value = os.environ.get(name, default)
    return value if value is None else cast(value)


def env_field(name, default=None, cast=str, source=""):
    """A field bound to one environment variable; `source` names a sibling field whose value is the default."""
    return field(metadata={"env": name, "default": default, "cast": cast, "source": source})


def load(section):
    """Instantiate a settings section, reading each field's variable exactly once, in field order."""
    values = {}
    for entry in fields(section):
        source = entry.metadata["source"]
        default = values[source] if source else entry.metadata["default"]
        values[entry.name] = _env(entry.metadata["env"], default, entry.metadata["cast"])
    return section(**values)


def env_values(section):
    """This section's values keyed by environment-variable name, for callers that own a resolution order."""
    return {entry.metadata["env"]: getattr(section, entry.name) for entry in fields(section)}


@dataclass(frozen=True)
class SandboxSettings:
    mem_limit: str = env_field("SANDBOX_MEM_LIMIT", "2g")
    memswap_limit: str = env_field("SANDBOX_MEMSWAP_LIMIT", source="mem_limit")
    pids_limit: int = env_field("SANDBOX_PIDS_LIMIT", "1024", int)
    cpus: float = env_field("SANDBOX_CPUS", "2", float)
    tmp_size: str = env_field("SANDBOX_TMP_SIZE", "1g")
    shm_size: str = env_field("SANDBOX_SHM_SIZE", "256m")
    vm_reserve: str = env_field("SANDBOX_VM_RESERVE", "1.5g")
    app_ready_timeout: int = env_field("SANDBOX_APP_READY_TIMEOUT", "60", int)
    app_build_timeout: int = env_field("SANDBOX_APP_BUILD_TIMEOUT", "420", int)


@dataclass(frozen=True)
class PipelineSettings:
    max_retries: int = env_field("PIPELINE_MAX_RETRIES", "3", int)
    retry_backoff: float = env_field("PIPELINE_RETRY_BACKOFF", "5", float)
    capture_mode: str = env_field("PIPELINE_CAPTURE_MODE", "agent")
    auth_flow_capture: bool = env_field("PIPELINE_AUTH_FLOW_CAPTURE", "0", flag_unless_falsy)
    auth_discovered_ratio: float = env_field("PIPELINE_AUTH_DISCOVERED_RATIO", "0.25", float)
    language: str = env_field("PIPELINE_LANGUAGE")
    credentials_dir: str = env_field("PIPELINE_CREDENTIALS_DIR", "credentials")
    stage3_budget_usd: float = env_field("STAGE3_BUDGET_USD", "2.0", float)
    stage3_gate_app_min: int = env_field("STAGE3_GATE_APP_MIN", "3", int)
    stage3_gate_runnability_min: int = env_field("STAGE3_GATE_RUNNABILITY_MIN", "2", int)
    stage3_gate_interaction_min: int = env_field("STAGE3_GATE_INTERACTION_MIN", "2", int)
    stage4_budget_usd: float = env_field("STAGE4_BUDGET_USD", "5.0", float)
    stage4_top_k_per_pass: int = env_field("STAGE4_TOP_K_PER_PASS", "15", int)
    stage4_deep_score_threshold: int = env_field("STAGE4_DEEP_SCORE_THRESHOLD", "4", int)
    stage4_min_est_trajectories: int = env_field("STAGE4_MIN_EST_TRAJECTORIES", "3", int)
    stage4_toy_complexity_max: int = env_field("STAGE4_TOY_COMPLEXITY_MAX", "2", int)
    stage4_workers: int = env_field("STAGE4_WORKERS", "3", int)
    stage5_workers: int = env_field("STAGE5_WORKERS", "2", int)
    stage6_workers: int = env_field("STAGE6_WORKERS", "2", int)
    bootable_gate: bool = env_field("BOOTABLE_GATE", "1", flag_unless_zero)
    bootable_gate_aggressive: bool = env_field("BOOTABLE_GATE_AGGRESSIVE", "0", flag_if_one)
    bootable_gate_workers: int = env_field("BOOTABLE_GATE_WORKERS", "8", int)
    boot_gate_min_elements: int = env_field("BOOT_GATE_MIN_ELEMENTS", "5", int)
    tasks_per_repo: int = env_field("AGENT_TASKS_PER_REPO", "8", int)
    task_propose_budget_usd: float = env_field("TASK_PROPOSE_BUDGET_USD", "2.0", float)
    instruction_budget_usd: float = env_field("INSTRUCTION_BUDGET_USD", "0.30", float)
    task_type_budget_usd: float = env_field("TASK_TYPE_BUDGET_USD", "0.30", float)
    think_budget_usd: float = env_field("THINK_BUDGET_USD", "0.30", float)
    think_session_budget_usd: float = env_field("THINK_SESSION_BUDGET_USD", "5.0", float)
    polish_budget_usd: float = env_field("POLISH_BUDGET_USD", "0.50", float)
    qa_voice_budget_usd: float = env_field("QA_VOICE_BUDGET_USD", "0.50", float)
    outcome_window_steps: int = env_field("OUTCOME_WINDOW_STEPS", "12", int)
    outcome_budget_usd: float = env_field("OUTCOME_BUDGET_USD", "1.00", float)
    outcome_image_tokens: int = env_field("OUTCOME_IMAGE_TOKENS", "350", int)
    outcome_price_in_per_m: float = env_field("OUTCOME_PRICE_IN_PER_M", "2.0", float)
    outcome_price_out_per_m: float = env_field("OUTCOME_PRICE_OUT_PER_M", "10.0", float)
    outcome_call_overhead_usd: float = env_field("OUTCOME_CALL_OVERHEAD_USD", "0.08", float)
    public_tasks_per_site: int = env_field("PUBLIC_TASKS_PER_SITE", "6", int)
    driver_port: int = env_field("DRIVER_PORT", "8765", int)
    driver_playwright_core_version: str = env_field("DRIVER_PLAYWRIGHT_CORE_VERSION", "1.55.0")
    driver_ready_timeout: int = env_field("DRIVER_READY_TIMEOUT", "60", int)


@dataclass(frozen=True)
class CaptureSettings:
    max_actions_per_task: int = env_field("AGENT_MAX_ACTIONS", "60", int)
    task_budget_usd: float = env_field("AGENT_TASK_BUDGET_USD", "25.0", float)
    max_turns: int = env_field("AGENT_MAX_TURNS", "150", int)
    verify_budget_usd: float = env_field("VERIFY_BUDGET_USD", "0.30", float)
    relabel_min_steps: int = env_field("AGENT_RELABEL_MIN_STEPS", "3", int)
    degenerate_loop_fraction: float = env_field("AGENT_DEGENERATE_LOOP_FRACTION", "0.6", float)
    degenerate_min_repeats: int = env_field("AGENT_DEGENERATE_MIN_REPEATS", "3", int)
    container_timeout: int = env_field("AGENT_CONTAINER_TIMEOUT", "10800", int)
    task_time_reserve: int = env_field("AGENT_TASK_TIME_RESERVE", "900", int)
    max_kept_per_repo: int = env_field("AGENT_MAX_KEPT_PER_REPO", "10", int)
    max_consecutive_fails: int = env_field("AGENT_MAX_CONSECUTIVE_FAILS", "3", int)
    max_attempts: int = env_field("CAPTURE_MAX_ATTEMPTS", "2", int)
    public_container_timeout: int = env_field("PUBLIC_CONTAINER_TIMEOUT", "10800", int)
    logingated_container_timeout: int = env_field("LOGINGATED_CONTAINER_TIMEOUT", "5400", int)
    register_container_timeout: int = env_field("REGISTER_CONTAINER_TIMEOUT", "5400", int)
    probe_container_timeout: int = env_field("PROBE_CONTAINER_TIMEOUT", "1800", int)
    watchdog_policy: str = env_field("WATCHDOG_POLICY", "llm")
    watchdog_budget_usd: float = env_field("WATCHDOG_BUDGET_USD", "0.20", float)
    persona_email_domain: str = env_field("PERSONA_EMAIL_DOMAIN", "example.com")
    persona_phone_prefix: str = env_field("PERSONA_PHONE_PREFIX", "+1202555")
    mail_host_smtp_port: int = env_field("MAIL_HOST_SMTP_PORT", "13025", int)
    mail_host_imap_port: int = env_field("MAIL_HOST_IMAP_PORT", "13143", int)
    mail_host_webmail_port: int = env_field("MAIL_HOST_WEBMAIL_PORT", "18080", int)
    operator_pii: str = env_field("OPERATOR_PII", "")
    operator_pii_file: str = env_field("OPERATOR_PII_FILE")


@dataclass(frozen=True)
class SdkSettings:
    query_timeout_s: int = env_field("SDK_QUERY_TIMEOUT", "2400", int)
    claude_cli_path: str = env_field("CLAUDE_CLI_PATH")


@dataclass(frozen=True)
class PromptSettings:
    prev_image_size: tuple = env_field("PREV_IMAGE_SIZE", "640x360", parse_viewport)
    effect_no_change_max_fraction: float = env_field("EFFECT_NO_CHANGE_MAX_FRACTION", "0.0005", float)
    effect_pixel_delta: int = env_field("EFFECT_PIXEL_DELTA", "16", int)
    effect_path_max_chars: int = env_field("EFFECT_PATH_MAX_CHARS", "40", int)


@dataclass(frozen=True)
class CorpusSettings:
    waste_click_radius_px: int = env_field("WASTE_CLICK_RADIUS_PX", "10", int)
    waste_lookback_steps: int = env_field("WASTE_LOOKBACK_STEPS", "4", int)
    meander_keep_head: int = env_field("MEANDER_KEEP_HEAD", "5", int)
    meander_keep_tail: int = env_field("MEANDER_KEEP_TAIL", "5", int)
    meander_mid_keep_every: int = env_field("MEANDER_MID_KEEP_EVERY", "6", int)
    keypress_max_repeat: int = env_field("KEYPRESS_MAX_REPEAT", "6", int)


@dataclass(frozen=True)
class GroundingSettings:
    small_oversample: int = env_field("GROUNDING_SMALL_OVERSAMPLE", "3", int)
    per_group_cap: int = env_field("GROUNDING_PER_GROUP_CAP", "40", int)
    seed: int = env_field("GROUNDING_SEED", "0", int)


@dataclass(frozen=True)
class EvalSettings:
    runs_dir: str = env_field("EVAL_RUNS_DIR", "eval_runs")
    episode_time_reserve: int = env_field("EVAL_EPISODE_TIME_RESERVE", "900", int)
    container_timeout: int = env_field("EVAL_CONTAINER_TIMEOUT", "10800", int)
    max_parse_failures: int = env_field("EVAL_MAX_PARSE_FAILURES", "3", int)
    max_unproductive_steps: int = env_field("EVAL_MAX_UNPRODUCTIVE_STEPS", "5", int)
    coord_space_suspect_rate: float = env_field("EVAL_COORD_SPACE_SUSPECT_RATE", "0.25", float)
    endpoint: str = env_field("EVAL_ENDPOINT", "")
    model: str = env_field("EVAL_MODEL", "")
    adapter: str = env_field("EVAL_ADAPTER", "")
    api_key: str = env_field("EVAL_API_KEY", "")
    coord_space: str = env_field("EVAL_COORD_SPACE", "normalized_1000")
    env_file: str = env_field("EVAL_ENV_FILE", "")
    verify_mid_screenshots: int = env_field("VERIFY_MID_SCREENSHOTS", "2", int)
    hf_endpoint: str = env_field("HF_ENDPOINT", "https://huggingface.co")
    hf_token: str = env_field("HF_TOKEN")
    hugging_face_hub_token: str = env_field("HUGGING_FACE_HUB_TOKEN")
    huggingface_hub_token: str = env_field("HUGGINGFACE_HUB_TOKEN")
    hf_read_token: str = env_field("HF_READ_TOKEN")
    screenspot_image_size: str = env_field("SCREENSPOT_IMAGE_SIZE", "native")
    screenspot_endpoint: str = env_field("SCREENSPOT_ENDPOINT", "")
    screenspot_api_key: str = env_field("SCREENSPOT_API_KEY", "")
    screenspot_workers: int = env_field("SCREENSPOT_WORKERS", "8", int)
    screenspot_claude_workers: int = env_field("SCREENSPOT_CLAUDE_WORKERS", "4", int)
    screenspot_claude_budget_usd: float = env_field("SCREENSPOT_CLAUDE_BUDGET_USD", "0.30", float)


@dataclass(frozen=True)
class CloudSettings:
    train_gpu: str = env_field("CUA_TRAIN_GPU", "A100-80GB")
    run_name: str = env_field("CUA_RUN_NAME", "sft-v1")
    train_timeout: int = env_field("CUA_TRAIN_TIMEOUT", str(10 * 3600), int)
    train_commit_every_s: int = env_field("CUA_TRAIN_COMMIT_EVERY_S", "600", int)
    grpo_gpu: str = env_field("CUA_GRPO_GPU", "A100-80GB:2")
    grpo_timeout: int = env_field("CUA_GRPO_TIMEOUT", str(9 * 3600), int)
    grpo_commit_every_s: int = env_field("CUA_GRPO_COMMIT_EVERY_S", "600", int)
    vast_api_key: str = env_field("VAST_API_KEY")
    remote_train_state_file: str = env_field("REMOTE_TRAIN_STATE_FILE", ".remote_train_state.json")
    remote_train_bad_offers: str = env_field("REMOTE_TRAIN_BAD_OFFERS", ".remote_train_bad_offers.json")
    remote_train_exclude_countries: str = env_field("REMOTE_TRAIN_EXCLUDE_COUNTRIES", "")
    remote_train_instance_label: str = env_field("REMOTE_TRAIN_INSTANCE_LABEL", "")


@dataclass(frozen=True)
class RlSettings:
    env_host: str = env_field("RL_ENV_HOST", "127.0.0.1")
    env_port: int = env_field("RL_ENV_PORT", "8900", int)
    env_token: str = env_field("RL_ENV_TOKEN", "")
    env_url: str = env_field("RL_ENV_URL", "http://127.0.0.1:8900")
    serve_port: int = env_field("RL_SERVE_PORT", "8001", int)
    run_dir: str = env_field("RL_RUN_DIR", "/workspace/rl_runs/current")
    run_id: str = env_field("RL_RUN_ID", "grpo")
    reset_timeout_s: float = env_field("RL_RESET_TIMEOUT_S", "45", float)
    episode_idle_s: float = env_field("RL_EPISODE_IDLE_S", "600", float)
    container_timeout: int = env_field("RL_CONTAINER_TIMEOUT", str(4 * 3600), int)
    recycle_reserve_s: int = env_field("RL_RECYCLE_RESERVE_S", "1200", int)
    drivers_per_sandbox: int = env_field("RL_DRIVERS_PER_SANDBOX", "1", int)
    loop_guard_repeats: int = env_field("RL_LOOP_GUARD_REPEATS", "5", int)
    reset_wait_s: float = env_field("RL_RESET_WAIT_S", "1800", float)
    step_wait_s: float = env_field("RL_STEP_WAIT_S", "2400", float)
    verifier_concurrency: int = env_field("RL_VERIFIER_CONCURRENCY", "2", int)
    verify_timeout_s: int = env_field("RL_VERIFY_TIMEOUT_S", "2700", int)
    verify_retry_s: float = env_field("RL_VERIFY_RETRY_S", "300", float)
    verify_wait_s: float = env_field("RL_VERIFY_WAIT_S", "1800", float)
    policy_backoff_s: float = env_field("RL_POLICY_BACKOFF_S", "5", float)
    policy_wait_s: float = env_field("RL_POLICY_WAIT_S", "600", float)
    policy_api_key: str = env_field("RL_POLICY_API_KEY", "")
    cuda_visible_devices: str = env_field("CUDA_VISIBLE_DEVICES", "0")


@dataclass(frozen=True)
class NotifySettings:
    telegram_bot_token: str = env_field("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = env_field("TELEGRAM_CHAT_ID", "")


@dataclass(frozen=True)
class GithubSettings:
    token: str = env_field("GITHUB_TOKEN", "")


SANDBOX = load(SandboxSettings)
PIPELINE = load(PipelineSettings)
CAPTURE = load(CaptureSettings)
SDK = load(SdkSettings)
PROMPT = load(PromptSettings)
CORPUS = load(CorpusSettings)
GROUNDING = load(GroundingSettings)
EVAL = load(EvalSettings)
CLOUD = load(CloudSettings)
RL = load(RlSettings)
NOTIFY = load(NotifySettings)
GITHUB = load(GithubSettings)

SECTIONS = {
    "SANDBOX": SANDBOX,
    "PIPELINE": PIPELINE,
    "CAPTURE": CAPTURE,
    "SDK": SDK,
    "PROMPT": PROMPT,
    "CORPUS": CORPUS,
    "GROUNDING": GROUNDING,
    "EVAL": EVAL,
    "CLOUD": CLOUD,
    "RL": RL,
    "NOTIFY": NOTIFY,
    "GITHUB": GITHUB,
}

REPO_ROOT = Path(__file__).resolve().parents[1]
# Generated, vendored and gitignored data trees, pruned so the table scan stays a source-tree walk.
SKIP_DIRS = {
    "__pycache__",
    "build",
    "credentials",
    "dist",
    "eval_archive",
    "eval_runs",
    "models",
    "node_modules",
    "output",
    "pipeline_data",
    "rl_runs",
    "trajectories",
    "training_data",
}


def _checkout_sources():
    """Every checkout module as (dotted name, text), read once so the table's `used by` column cannot go stale."""
    for directory, subdirs, filenames in os.walk(REPO_ROOT):
        subdirs[:] = sorted(name for name in subdirs if name not in SKIP_DIRS and not name.startswith("."))
        for name in sorted(filenames):
            path = Path(directory) / name
            if not name.endswith(".py") or path == Path(__file__).resolve():
                continue
            dotted = path.relative_to(REPO_ROOT).with_suffix("").as_posix().replace("/", ".")
            yield dotted, path.read_text(encoding="utf-8", errors="ignore")


def _shown(value):
    """A default rendered for the table: unset reads as an em dash, empty as a quoted blank."""
    return "—" if value is None else '`""`' if value == "" else f"`{value}`"


def markdown_table():
    """The environment-variable table in docs/conventions.md, derived from the section field metadata."""
    sources = list(_checkout_sources())
    rows = ["| variable | default | settings field | used by |", "| --- | --- | --- | --- |"]
    for section_name, section in SECTIONS.items():
        for entry in fields(section):
            reference = f"{section_name}.{entry.name}"
            source = entry.metadata["source"]
            default = f"`{section_name}.{source}`" if source else _shown(entry.metadata["default"])
            used_by = [name for name, text in sources if reference in text] or ["—"]
            rows.append(f"| `{entry.metadata['env']}` | {default} | `{reference}` | {', '.join(used_by)} |")
    return "\n".join(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Print the environment variables this package reads")
    parser.add_argument("--table", action="store_true", help="print the Markdown table for docs/conventions.md")
    if not parser.parse_args(argv).table:
        parser.error("nothing to print: pass --table")
    print(markdown_table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
