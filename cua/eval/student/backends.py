"""Where the model is served and where the browser runs: two registries, so a new stack is one entry."""

import json
import time
from pathlib import Path

from cua.eval.student.config import EvalConfig
from cua.pipeline import driver_session, remote_sandbox
from cua.pipeline.driver_session import DriverClient, run_blocking
from cua.pipeline.sandbox import DockerSandbox
from cua.settings import EVAL
from cua.trajectory.actions import format_action_for_training


class StudentError(Exception):
    """The served model could not be reached or returned an unusable response."""


class StudentClient:
    def complete(self, messages: list) -> str:
        raise NotImplementedError


class OpenAIChatStudent(StudentClient):
    """Any OpenAI-compatible /chat/completions endpoint (vLLM, `swift deploy`, SGLang, a hosted API)."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str = "",
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout_s: int = 180,
        retries: int = 2,
        retry_backoff_s: int = 90,
    ):
        self.url = self._chat_url(endpoint)
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.retries = retries
        self.retry_backoff_s = retry_backoff_s

    @staticmethod
    def _chat_url(endpoint: str) -> str:
        endpoint = endpoint.rstrip("/")
        if endpoint.endswith("/chat/completions"):
            return endpoint
        return f"{endpoint}/chat/completions"

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, messages: list) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _content(self, response) -> str:
        try:
            choice = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as e:
            raise StudentError(f"unexpected response shape from {self.url}: {e}")
        if isinstance(choice, list):
            choice = "".join(part.get("text", "") for part in choice if isinstance(part, dict))
        return choice or ""

    def complete(self, messages: list) -> str:
        # lazy: requests is an optional extra, resolved at call time — preserved from the pre-split module.
        import requests

        payload = self._payload(messages)
        headers = self._headers()
        last_error = None
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(self.retry_backoff_s * attempt)
            try:
                response = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout_s)
            except requests.RequestException as e:
                last_error = f"transport error: {e}"
                continue
            if response.status_code >= 500:
                # A preempted or restarting model answers 5xx for minutes: a retry, not a verdict on the episode.
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                continue
            if response.status_code != 200:
                raise StudentError(f"{self.url} returned HTTP {response.status_code}: {response.text[:300]}")
            return self._content(response)
        raise StudentError(f"{self.url} unreachable after {self.retries + 1} attempts ({last_error})")


class ReplayStudent(StudentClient):
    """Zero-model student replaying a recorded trajectory's actions in the training wire format."""

    def __init__(self, trajectory: dict):
        self.steps = trajectory.get("steps", [])
        self.index = 0

    def complete(self, messages: list) -> str:
        if self.index >= len(self.steps):
            return "<think>The recorded run is finished and the final state is on screen.</think>\naction: done()"
        step = self.steps[self.index]
        self.index += 1
        thought = step.get("thought") or "Replaying the next recorded action."
        actions = step.get("actions") or []
        lines = "\n".join(format_action_for_training(a) for a in actions)
        return f"<think>{thought}</think>\n{lines}"


def build_openai_student(cfg: EvalConfig) -> StudentClient:
    if not cfg.endpoint:
        raise StudentError("--endpoint is required for the 'openai' student backend")
    return OpenAIChatStudent(
        endpoint=cfg.endpoint,
        model=cfg.model,
        api_key=cfg.api_key,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        timeout_s=cfg.student_timeout_s,
        retries=cfg.student_retries,
    )


def build_replay_student(cfg: EvalConfig) -> StudentClient:
    if not cfg.mock_trajectory:
        raise StudentError("--mock-trajectory PATH is required for the 'mock' student backend")
    return ReplayStudent(json.loads(Path(cfg.mock_trajectory).read_text()))


STUDENT_BACKENDS: dict = {
    "openai": build_openai_student,
    "mock": build_replay_student,
}


def build_student(cfg: EvalConfig) -> StudentClient:
    if cfg.student not in STUDENT_BACKENDS:
        raise StudentError(f"Unknown student backend {cfg.student!r}. Available: {sorted(STUDENT_BACKENDS)}")
    return STUDENT_BACKENDS[cfg.student](cfg)


def _docker_browser(cfg):
    kwargs = {"container_timeout": EVAL.container_timeout}
    if cfg.sandbox_image:
        kwargs["image"] = cfg.sandbox_image
    return DockerSandbox(**kwargs)


def _remote_ssh_browser(cfg):
    host, _, port = cfg.browser_host.partition(":")
    return remote_sandbox.RemoteSandbox(host=host, port=int(port or 22), container_timeout=EVAL.container_timeout)


def _vast_residential_browser(cfg):
    return remote_sandbox.VastResidentialSandbox(container_timeout=EVAL.container_timeout)


# Where the browser runs, i.e. what IP live sites see; an egress is a registry entry, as in TASK_SOURCES.
BROWSER_BACKENDS = {
    "docker": _docker_browser,
    "remote_ssh": _remote_ssh_browser,
    "vast_residential": _vast_residential_browser,
}


class EvalBrowser:
    """One sandbox holding one browser driver, reused across episodes and recycled before self-expiry."""

    def __init__(self, cfg: EvalConfig, name: str):
        self.cfg = cfg
        self.name = name
        self.sandbox: DockerSandbox | None = None
        self.client: DriverClient | None = None

    async def ensure(self):
        if self.sandbox is not None and self.sandbox.remaining_seconds() > EVAL.episode_time_reserve:
            return
        await self.stop()
        self.sandbox = BROWSER_BACKENDS[self.cfg.browser_backend](self.cfg)
        await run_blocking(self.sandbox.start, f"eval/{self.name}")
        # The fingerprint must name the ASN, not just the backend: two residential rentals are different IPs.
        egress = getattr(self.sandbox, "egress", None)
        if egress and not self.cfg.egress_note:
            self.cfg.egress_note = f"{egress.get('country', '?')}/{egress.get('city', '?')}/{egress.get('org', '?')}"
        self.client = await driver_session.boot_driver(
            self.sandbox, {"DRIVER_LOOP_GUARD_REPEATS": str(self.cfg.loop_guard_repeats)}
        )

    async def stop(self):
        if self.sandbox is not None:
            await run_blocking(self.sandbox.stop)
        self.sandbox = None
        self.client = None
