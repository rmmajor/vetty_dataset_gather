"""Policies turn an observation into the student's response text (registry keyed by kind)."""

import json
import logging
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

from cua.rl.actions import encode_step
from cua.rl.observation import to_openai_messages
from cua.settings import RL
from cua.student.client import OpenAICompatibleClient, RetryPolicy, choice_text_or_empty, urllib_transport

logger = logging.getLogger(__name__)
TRANSIENT = (ssl.SSLError, urllib.error.URLError, ConnectionError, TimeoutError)


def _retryable(error):
    """A 5xx or a dropped connection is worth another try; a 4xx and anything unexpected end the rollout."""
    if isinstance(error, urllib.error.HTTPError):
        return error.code >= 500
    return isinstance(error, TRANSIENT)


def _log_retry(attempt, failure, remaining_s):
    logger.warning(
        f"policy call failed ({type(failure).__name__}: {str(failure)[:120]}); "
        f"retrying, {round(remaining_s)}s of budget left"
    )


class OpenAICompatiblePolicy:
    def __init__(
        self,
        endpoint,
        model,
        temperature=0.0,
        max_tokens=512,
        timeout_s=180,
        api_key=None,
        wait_s=RL.policy_wait_s,
        backoff_s=RL.policy_backoff_s,
        opener=urllib.request.urlopen,
        sleep=time.sleep,
        clock=time.monotonic,
        **_,
    ):
        self.endpoint, self.model, self.temperature, self.max_tokens, self.timeout_s = (
            endpoint.rstrip("/"),
            model,
            temperature,
            max_tokens,
            timeout_s,
        )
        self.api_key = api_key or RL.policy_api_key
        self.wait_s, self.backoff_s, self.opener, self.sleep, self.clock = wait_s, backoff_s, opener, sleep, clock
        self.client = OpenAICompatibleClient(
            endpoint=self.endpoint,
            model=model,
            api_key=self.api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            retry=RetryPolicy(
                deadline_s=wait_s,
                backoff=lambda attempt: min(backoff_s * attempt, 30),
                retry_error=_retryable,
                exhausted=self._unreachable,
                on_retry=_log_retry,
            ),
            content=choice_text_or_empty,
            transport=urllib_transport(opener),
            sleep=sleep,
            clock=clock,
        )

    def _unreachable(self, url, attempts, failure):
        raise ConnectionError(f"policy endpoint unreachable after {attempts} attempts in {self.wait_s}s: {failure}")

    def act(self, observation):
        return self.client.chat(to_openai_messages(observation))

    def served_model_id(self):
        try:
            with urllib.request.urlopen(f"{self.endpoint}/models", timeout=30) as response:
                return ",".join(m["id"] for m in json.loads(response.read()).get("data", []))
        except Exception:
            return self.model


class MockReplayPolicy:
    """Replays a recorded trajectory's action steps verbatim, then done()."""

    def __init__(self, traj_dir, **_):
        traj = json.loads((Path(traj_dir) / "trajectory.json").read_text())
        self.steps = [s for s in traj["steps"] if s.get("actions")]
        self.index = 0

    def act(self, observation):
        if self.index >= len(self.steps):
            return encode_step("Nothing left to replay.", [{"type": "done"}])
        step = self.steps[self.index]
        self.index += 1
        return encode_step(step.get("thought") or "replay", step["actions"])


class ScriptedPolicy:
    def __init__(self, texts, **_):
        self.texts, self.index = list(texts), 0

    def act(self, observation):
        text = self.texts[min(self.index, len(self.texts) - 1)]
        self.index += 1
        return text


POLICIES = {"openai": OpenAICompatiblePolicy, "mock": MockReplayPolicy, "scripted": ScriptedPolicy}


def make_policy(spec):
    return POLICIES[spec["kind"]](**{k: v for k, v in spec.items() if k != "kind"})
