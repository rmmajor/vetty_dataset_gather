"""Where the model is served and where the browser runs: two registries, so a new stack is one entry."""

import json
from pathlib import Path

from cua.eval.student.config import EvalConfig
from cua.pipeline import driver_session, remote_sandbox
from cua.pipeline.driver_session import DriverClient, run_blocking
from cua.pipeline.sandbox import DockerSandbox
from cua.settings import EVAL
from cua.student.client import (
    OpenAICompatibleClient,
    RetryPolicy,
    TransportError,
    choice_text_parts,
    requests_transport,
)
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
        self.client = OpenAICompatibleClient(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            retry=RetryPolicy(
                attempts=retries + 1,
                backoff=lambda attempt: retry_backoff_s * attempt,
                retry_error=lambda error: isinstance(error, TransportError),
                fatal=_refused,
                exhausted=_unreachable,
            ),
            content=self._content,
            transport=requests_transport,
        )
        self.url = self.client.url

    def _content(self, body) -> str:
        try:
            return choice_text_parts(body)
        except (KeyError, IndexError, ValueError) as e:
            raise StudentError(f"unexpected response shape from {self.url}: {e}") from e

    def complete(self, messages: list) -> str:
        return self.client.chat(messages)


def _refused(url: str, response) -> StudentError | None:
    """A non-200 that is not a 5xx is the endpoint's verdict on the request, not a blip worth retrying."""
    if response.status == 200:
        return None
    return StudentError(f"{url} returned HTTP {response.status}: {response.text[:300]}")


def _unreachable(url: str, attempts: int, failure):
    detail = (
        f"transport error: {failure}"
        if isinstance(failure, TransportError)
        # A preempted or restarting model answers 5xx for minutes: retried, not a verdict on the episode.
        else f"HTTP {failure.status}: {failure.text[:200]}"
    )
    raise StudentError(f"{url} unreachable after {attempts} attempts ({detail})")


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
