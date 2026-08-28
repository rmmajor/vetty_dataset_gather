import json
import tempfile
from pathlib import Path

from remote_train.config import RemoteTrainConfig
from remote_train.jobs import JOBS, GrpoJob, ServeJob
from remote_train.mock import MockProvider
from rl_train import (PhaseDriver, build_deploy_command, build_env, build_merge_command, build_rlhf_command,
                      build_rollout_command, derive_run_config, load_rl_config, make_dataset)

CFG = load_rl_config("rl_config.yaml")


def test_config_mirrors_sft_base_and_ports():
    assert CFG["base_model"] == "Qwen/Qwen3.5-9B" and CFG["rollout_port"] == 8000 and CFG["compute"]["serve_port"] == 8001
    assert CFG["env_url"].endswith(":8900") and CFG["compute"]["min_vram_gb"] == 80


def test_commands_are_argv_lists_with_expected_flags():
    merge = build_merge_command(CFG)
    assert merge[:2] == ["swift", "export"] and "--merge_lora" in merge and "/workspace/sft_adapter" in merge
    rollout = build_rollout_command(CFG)
    assert rollout[:2] == ["swift", "rollout"] and "--use_gym_env" in rollout and rollout[rollout.index("--gym_env") + 1] == "browser_env"
    assert rollout[rollout.index("--multi_turn_scheduler") + 1] == "browser_scheduler" and "--vllm_enable_lora" in rollout
    rlhf = build_rlhf_command(CFG)
    assert rlhf[:2] == ["swift", "rlhf"] and rlhf[rlhf.index("--rlhf_type") + 1] == "grpo" and rlhf[rlhf.index("--vllm_mode") + 1] == "server"
    assert "--reward_weights" not in rlhf and "--reward_funcs" not in rlhf
    weighted = build_rlhf_command(dict(CFG, reward_funcs="action_format", reward_weights="0.2 1.0"))
    assert weighted[weighted.index("--reward_weights"):weighted.index("--reward_weights") + 3] == ["--reward_weights", "0.2", "1.0"]
    assert rlhf[rlhf.index("--num_generations") + 1] == "4" and rlhf[rlhf.index("--freeze_vit") + 1] == "true"
    deploy = build_deploy_command(CFG, adapters="/workspace/rl_output/latest")
    assert deploy[:2] == ["swift", "deploy"] and deploy[deploy.index("--adapters") + 1] == "/workspace/rl_output/latest" and "--port" in deploy
    assert "--adapters" not in build_deploy_command(CFG)
    for argv in (rollout, rlhf, deploy):
        assert argv[argv.index("--response_prefix") + 1] == ""


def test_env_carries_pixels_and_service_url():
    env = build_env(CFG, "/workspace/rl_runs/r1")
    assert env["MAX_PIXELS"] == "1003520" and env["RL_ENV_URL"] == CFG["env_url"] and env["RL_RUN_DIR"] == "/workspace/rl_runs/r1"
    assert "RL_ENV_TOKEN" not in env or env["RL_ENV_TOKEN"] == ""
    assert build_env(dict(CFG, env_token="s3cret"), "/x")["RL_ENV_TOKEN"] == "s3cret"
    assert build_env(CFG, "/x", workdir="/workspace")["PYTHONPATH"].split(":")[0] == "/workspace"


def test_derive_run_config_merges_compute_and_ignores_none():
    derived = derive_run_config(CFG, {"env_url": "https://x.ngrok.app", "max_steps": 1, "eval_before": False, "template": None,
                                      "compute": {"serve_port": 9001}})
    assert derived["env_url"] == "https://x.ngrok.app" and derived["max_steps"] == 1 and derived["eval_before"] is False
    assert derived["template"] == CFG["template"] and derived["compute"]["serve_port"] == 9001 and derived["compute"]["min_vram_gb"] == 80
    assert CFG["max_steps"] == 30 and CFG["compute"]["serve_port"] == 8001


def test_make_dataset_rows():
    with tempfile.TemporaryDirectory() as tmp:
        suite = Path(tmp) / "suite.json"
        suite.write_text(json.dumps({"suite_sha256": "x", "tasks": [{"task_id": "a", "split": "train", "expected_actions": 12},
                                                                    {"task_id": "b", "split": "eval_unseen_site"},
                                                                    {"task_id": "c", "split": "train", "expected_actions": 30}]}))
        out = Path(tmp) / "rl_tasks.jsonl"
        assert make_dataset(suite, out, split="train", repeats=2) == 4
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert rows[0]["messages"] == [{"role": "user", "content": "<placeholder:a>"}]
        assert rows[0]["env_config"] == {"name": "browser_env", "task_id": "a", "sample_idx": 0} and rows[2]["env_config"]["sample_idx"] == 1
        assert make_dataset(suite, out, split="train", repeats=1, max_expected_actions=18) == 1
        assert json.loads(out.read_text())["env_config"]["task_id"] == "a"


class FakeProc:
    def __init__(self, argv):
        self.argv, self.killed, self.returncode = argv, False, 0

    def poll(self):
        return None if not self.killed else self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.killed = True

    def kill(self):
        self.killed = True


class FakeRun:
    returncode = 0
    stdout = "torch==2.11.0\nnumpy==2.0\n"


def test_phase_driver_runs_all_phases_with_handshake():
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        launched, calls, acks, devices = [], [], [], []

        def launcher(argv, **kw):
            launched.append(FakeProc(argv))
            devices.append((argv[1], kw["env"]["CUDA_VISIBLE_DEVICES"]))
            return launched[-1]

        cfg = dict(CFG, merged_model=str(workdir / "merged"), output_dir=str(workdir / "rl_output"), sft_adapter_remote=str(workdir / "sft_adapter"))
        driver = PhaseDriver(cfg, str(workdir / "run"), str(workdir), launcher=launcher, run=lambda argv, **kw: FakeRun(), http_ok=lambda url: True, sleep=lambda s: None)

        def run_foreground(argv, env):
            calls.append(argv[:2])
            devices.append((argv[1], env["CUDA_VISIBLE_DEVICES"]))
            if argv[1] == "rlhf":
                (Path(cfg["output_dir"]) / "checkpoint-5").mkdir(parents=True)
                (Path(cfg["output_dir"]) / "checkpoint-5" / "adapter_config.json").write_text("{}")
            return 0

        driver.run_foreground = run_foreground
        driver.wait_for_ack = lambda phase: acks.append(phase) or True
        assert driver.run(mode="swift") == 0
        assert calls == [["swift", "export"], ["swift", "rlhf"]]
        assert acks == ["EVAL_BEFORE_READY", "EVAL_AFTER_READY"]
        assert (Path(cfg["output_dir"]) / "latest" / "adapter_config.json").exists()
        assert (workdir / "PHASE").read_text().strip() == "DONE"
        assert [p.argv[1] for p in launched] == ["deploy", "rollout", "deploy"] and all(p.killed for p in launched)
        assert dict(devices) == {"export": "0", "deploy": "0", "rollout": "0", "rlhf": "1"}
        assert [p.argv[p.argv.index("--adapters") + 1] for p in launched if "--adapters" in p.argv] == [str(Path(cfg["output_dir"]) / "latest")]
        versions = json.loads((workdir / "run" / "versions.json").read_text())
        assert versions["mode"] == "swift" and versions["packages"] == ["torch==2.11.0"]


def test_fallback_mode_is_explicitly_unimplemented():
    with tempfile.TemporaryDirectory() as tmp:
        driver = PhaseDriver(dict(CFG), tmp, tmp, launcher=None, run=None, http_ok=lambda url: True, sleep=lambda s: None)
        try:
            driver.run(mode="fallback")
            raise AssertionError("fallback must raise")
        except NotImplementedError as e:
            assert "3.4" in str(e)


def _grpo_cfg(tmp):
    cfg = RemoteTrainConfig()
    cfg.rl_config = "rl_config.yaml"
    cfg.rl_runs_dir = str(Path(tmp) / "rl_runs")
    cfg.output_dir = str(Path(tmp) / "output")
    cfg.env_url = "mock://"
    cfg.poll_interval_s = 0
    cfg.collect_every_polls = 2
    Path(cfg.output_dir).mkdir(parents=True)
    (Path(cfg.output_dir) / "adapter_config.json").write_text("{}")
    (Path(cfg.output_dir) / "adapter_model.safetensors").write_bytes(b"0" * 16)
    return cfg


def test_grpo_job_registered_and_preflight_applies_compute():
    assert set(JOBS) >= {"train", "eval", "grpo", "serve"}
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _grpo_cfg(tmp)
        job = GrpoJob(cfg)
        job.preflight(cfg)
        assert cfg.min_vram_gb == 80 and cfg.disk_gb == 120 and cfg.max_runtime_hours == 8
        header = json.loads((Path(cfg.rl_runs_dir) / job.run_id / "run_header.json").read_text())
        assert header["sft_adapter_sha256"] and header["suite_sha256"] and header["rl_config_sha256"]
        assert Path(job.dataset_path).read_text().count("\n") > 0
        cmd = job.remote_command(cfg)
        assert "requirements-rl.txt" in cmd and "rl_train.py run" in cmd and "--run-dir /workspace/rl_runs/" in cmd


def test_grpo_job_on_poll_handles_handshake_and_collects_adapter():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _grpo_cfg(tmp)
        job = GrpoJob(cfg)
        job.preflight(cfg)
        job.run_eval = lambda phase: {"phase": phase}
        provider = MockProvider(cfg)
        instance = provider.provision(provider.search_offer(cfg), cfg)
        provider.wait_ready(instance, 1)
        stop = [job.on_poll(provider, instance, cfg) for _ in range(4)]
        assert stop == [False, False, False, False]
        assert job.evals_done == ["EVAL_BEFORE_READY", "EVAL_AFTER_READY"]
        assert any("PHASE_ACK" in c for c in provider.exec_log)
        assert any("DOWNLOAD /workspace/rl_output/latest/" in c for c in provider.exec_log)


def test_serve_job_stops_after_serve_seconds():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _grpo_cfg(tmp)
        cfg.serve_adapter = cfg.output_dir
        cfg.serve_seconds = 1
        job = ServeJob(cfg)
        job.preflight(cfg)
        job.clock = iter([0, 2.0]).__next__
        job.http_ok = lambda url: True
        provider = MockProvider(cfg)
        instance = provider.provision(provider.search_offer(cfg), cfg)
        assert job.on_poll(provider, instance, cfg) is False
        assert job.on_poll(provider, instance, cfg) is True
        assert "swift deploy" in job.remote_command(cfg) and "--adapters" in job.remote_command(cfg)
