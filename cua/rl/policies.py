"""Policies turn an observation into the student's response text (registry keyed by kind)."""

import json
import logging
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

from cua.rl.actions import encode_step
from cua.rl.observation import to_openai_messages

logger = logging.getLogger(__name__)
# A served endpoint drops connections and cold-starts; the request carries no side effect, so transient failures are retried.
POLICY_BACKOFF_S = float(os.environ.get("RL_POLICY_BACKOFF_S", "5"))
# A scale-to-zero endpoint takes minutes to answer its first request, so the budget is a deadline, not a retry count.
POLICY_WAIT_S = float(os.environ.get("RL_POLICY_WAIT_S", "600"))
TRANSIENT = (ssl.SSLError, urllib.error.URLError, ConnectionError, TimeoutError)


class OpenAICompatiblePolicy:
    def __init__(
        self,
        endpoint,
        model,
        temperature=0.0,
        max_tokens=512,
        timeout_s=180,
        api_key=None,
        wait_s=POLICY_WAIT_S,
        backoff_s=POLICY_BACKOFF_S,
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
        self.api_key = api_key or os.environ.get("RL_POLICY_API_KEY", "")
        self.wait_s, self.backoff_s, self.opener, self.sleep, self.clock = wait_s, backoff_s, opener, sleep, clock

    def act(self, observation):
        body = {
            "model": self.model,
            "messages": to_openai_messages(observation),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        deadline, last, attempt = self.clock() + self.wait_s, None, 0
        while True:
            request = urllib.request.Request(
                f"{self.endpoint}/chat/completions", data=json.dumps(body).encode(), headers=headers
            )
            try:
                with self.opener(request, timeout=self.timeout_s) as response:
                    return json.loads(response.read())["choices"][0]["message"].get("content") or ""
            except urllib.error.HTTPError as e:
                if e.code < 500:
                    raise
                last = e
            except TRANSIENT as e:
                last = e
            attempt += 1
            if self.clock() >= deadline:
                break
            logger.warning(
                f"policy call failed ({type(last).__name__}: {str(last)[:120]}); retrying, {round(deadline - self.clock())}s of budget left"
            )
            self.sleep(min(self.backoff_s * attempt, 30))
        raise ConnectionError(f"policy endpoint unreachable after {attempt} attempts in {self.wait_s}s: {last}")

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
