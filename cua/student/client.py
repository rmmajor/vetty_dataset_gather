"""One OpenAI-compatible /chat/completions client: shared URL, headers, payload, transport; per-caller retry."""

import json
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple


class TransportError(Exception):
    """The HTTP call did not complete; whether that ends the run is the caller's retry policy."""


class ChatResponse(NamedTuple):
    status: int
    # Always decoded text: every transport normalises here, so no reading has to handle bytes.
    text: str


def chat_url(endpoint: str) -> str:
    """The /chat/completions URL, whether the endpoint is given as a base url or as the full path."""
    endpoint = endpoint.rstrip("/")
    return endpoint if endpoint.endswith("/chat/completions") else f"{endpoint}/chat/completions"


def chat_headers(api_key: str, send_empty_auth: bool = False) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key or send_empty_auth:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def chat_payload(model: str, messages: list, temperature: float, max_tokens: int) -> dict:
    return {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}


def choice_text(body) -> str:
    """The assistant text of a response body; a body that is not a chat completion raises, and the policy decides."""
    return json.loads(body)["choices"][0]["message"]["content"] or ""


def choice_text_or_empty(body) -> str:
    """The RL policy's reading: a completion that carries no content is an empty answer, not a broken response."""
    return json.loads(body)["choices"][0]["message"].get("content") or ""


def choice_text_parts(body) -> str:
    """The student eval's reading: a content list (some servers segment the answer) is joined into one string."""
    content = json.loads(body)["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def urllib_transport(opener=None) -> Callable:
    """A stdlib round trip, for a lane that must add no HTTP dependency or that injects its own opener."""

    def send(url, payload, headers, timeout_s):
        request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
        # Resolved per call, so patching urllib.request.urlopen still reaches a client that was built earlier.
        with (opener or urllib.request.urlopen)(request, timeout=timeout_s) as response:
            return ChatResponse(getattr(response, "status", 200) or 200, response.read().decode("utf-8"))

    return send


def requests_transport(url, payload, headers, timeout_s) -> ChatResponse:
    """The `requests` round trip; a failed call becomes a TransportError, so no policy needs a requests import."""
    # requests is an optional extra, resolved at call time — preserved from the pre-split module.
    import requests

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
    except requests.RequestException as e:
        raise TransportError(str(e)) from e
    return ChatResponse(response.status_code, response.text)


@dataclass(frozen=True)
class RetryPolicy:
    """What a caller retries, how long it waits between tries, and what a spent budget means."""

    attempts: int = 3
    # Set at all — 0 included — retries until the time budget is spent instead of counting attempts.
    deadline_s: float | None = None
    backoff: Callable = lambda attempt: 0.0
    retry_error: Callable = lambda error: True
    retry_status: Callable = lambda status: status >= 500
    # A status this caller refuses outright: returns the exception to raise, or None to read the body.
    fatal: Callable = lambda url, response: None
    exhausted: Callable = lambda url, attempts, failure: ""
    on_retry: Callable = lambda attempt, failure, remaining_s: None


DEFAULT_RETRY = RetryPolicy()


class OpenAICompatibleClient:
    """Any OpenAI-compatible /chat/completions endpoint (vLLM, `swift deploy`, SGLang, a hosted API)."""

    def __init__(
        self,
        endpoint: str,
        model: str = "",
        api_key: str = "",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_s: int = 180,
        retry: RetryPolicy = DEFAULT_RETRY,
        send_empty_auth: bool = False,
        content: Callable = choice_text,
        transport: Callable = None,
        sleep: Callable = time.sleep,
        clock: Callable = time.monotonic,
    ):
        self.url = chat_url(endpoint)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.retry = retry
        self.headers = chat_headers(api_key, send_empty_auth)
        self.content = content
        self.transport = transport or urllib_transport()
        self.sleep = sleep
        self.clock = clock

    def _attempt(self, payload):
        """(answer, None) for a completed call, or (None, failure) for one this policy retries; anything else raises."""
        try:
            response = self.transport(self.url, payload, self.headers, self.timeout_s)
        except Exception as e:
            if self.retry.retry_error(e):
                return None, e
            raise
        if self.retry.retry_status(response.status):
            return None, response
        fatal = self.retry.fatal(self.url, response)
        if fatal is not None:
            raise fatal
        try:
            return self.content(response.text), None
        except Exception as e:
            if self.retry.retry_error(e):
                return None, e
            raise

    def _keep_going(self, attempt: int, deadline: float) -> bool:
        return self.clock() < deadline if self.retry.deadline_s is not None else attempt < self.retry.attempts

    def chat(self, messages: list, model: str = None, temperature: float = None, max_tokens: int = None) -> str:
        payload = chat_payload(
            model or self.model,
            messages,
            self.temperature if temperature is None else temperature,
            self.max_tokens if max_tokens is None else max_tokens,
        )
        deadline = self.clock() + self.retry.deadline_s if self.retry.deadline_s is not None else 0.0
        attempt, failure = 0, None
        while True:
            attempt += 1
            answer, failure = self._attempt(payload)
            if failure is None:
                return answer
            if not self._keep_going(attempt, deadline):
                return self.retry.exhausted(self.url, attempt, failure)
            self.retry.on_retry(attempt, failure, deadline - self.clock() if self.retry.deadline_s is not None else 0.0)
            self.sleep(self.retry.backoff(attempt))
