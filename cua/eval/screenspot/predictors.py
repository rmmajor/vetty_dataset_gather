"""One registry entry per way of asking a model where to click; the served arms get training's grounding prompt."""

import asyncio
import concurrent.futures
import io
import json
import os
import subprocess
import tempfile
from pathlib import Path

from cua.eval.screenspot.dataset import prepare_sample_image
from cua.prompt.format import (
    GROUNDING_SYSTEM_PROMPT,
    build_grounding_observation,
    build_step_messages,
    png_b64,
    to_claude_content,
    to_openai_content,
)
from cua.settings import EVAL
from cua.student.client import OpenAICompatibleClient, RetryPolicy

# The element description handed to the grounding observation; a ScreenSpot instruction names the target.
GROUNDING_DESCRIPTION_TEMPLATE = 'the element described as "{instruction}"'

# An unreachable endpoint scores as a parse failure instead of aborting: one dead arm loses no other's numbers.
SPENT_ATTEMPTS_SCORE_AS_PARSE_FAILURE = RetryPolicy(attempts=3, backoff=lambda attempt: 2 * attempt)

CLAUDE_GROUNDING_SYSTEM_PROMPT = (
    "You are a GUI grounding model. Given a screenshot and a described element, reply with the "
    "pixel coordinates of that element's centre in the image exactly as shown, as a single line "
    "`click(x, y)`. x and y are absolute pixels of this image, with (0, 0) at the top left. "
    "Reply with the click line only, no reasoning."
)


def build_messages(instruction, image_ref):
    """Training's grounding slice verbatim, in the content-list form `swift infer` consumes."""
    observation = build_grounding_observation(GROUNDING_DESCRIPTION_TEMPLATE.format(instruction=instruction), image_ref)
    return build_step_messages(GROUNDING_SYSTEM_PROMPT, observation)


def build_openai_messages(instruction, image_b64):
    """The same messages with the one image inlined as a PNG data URL for /chat/completions."""
    messages = build_messages(instruction, "cur")
    messages[1]["content"] = to_openai_content(messages[1]["content"], lambda ref: image_b64)
    return messages


def build_claude_content(instruction, image_b64):
    """The same user turn as Anthropic content blocks; the system prompt rides on ClaudeAgentOptions."""
    messages = build_messages(instruction, "cur")
    return to_claude_content(messages[1]["content"], lambda ref: image_b64)


def encode_png(image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return png_b64(buffer.getvalue())


class Predictor:
    def predict_batch(self, samples, cfg, coord_space):
        raise NotImplementedError


class MockPredictor(Predictor):
    def __init__(self, response_fn):
        self.response_fn = response_fn

    def predict_batch(self, samples, cfg, coord_space):
        return [self.response_fn(sample, cfg, coord_space) for sample in samples]


def default_mock_response(sample, cfg, coord_space):
    center_x = cfg.viewport_w / 2
    center_y = cfg.viewport_h / 2
    return f"<think>mock prediction</think>\naction: click(<ref>({int(center_x)}, {int(center_y)})</ref>, button=left)"


def extract_response_text(obj):
    if not isinstance(obj, dict):
        return ""
    for key in ("response", "generated_text", "completion"):
        if key in obj and obj[key] is not None:
            return str(obj[key])
    messages = obj.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content", ""))
    return ""


def read_swift_results(result_path, expected):
    responses = []
    with open(result_path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                responses.append(extract_response_text(json.loads(line)))
    if len(responses) < expected:
        responses += [""] * (expected - len(responses))
    return responses[:expected]


class SwiftPredictor(Predictor):
    """Runs `swift infer` once per model spec over all samples, shelling out the way cua.sft.swift_launch does."""

    def __init__(self, model, adapters, cfg):
        self.model = model
        self.adapters = adapters
        self.cfg = cfg

    def _write_batch(self, samples, tmp_dir):
        infer_path = tmp_dir / "infer.jsonl"
        with open(infer_path, "w") as handle:
            for index, sample in enumerate(samples):
                prepared = prepare_sample_image(sample.image)
                image_path = tmp_dir / f"image_{index:06d}.png"
                prepared.save(image_path)
                handle.write(json.dumps({"messages": build_messages(sample.instruction, str(image_path))}) + "\n")
        return infer_path

    def _command(self, infer_path, result_path, cfg):
        cmd = ["swift", "infer", "--model", self.model]
        if self.adapters:
            cmd += ["--adapters", self.adapters]
        return cmd + [
            "--val_dataset",
            str(infer_path),
            "--result_path",
            str(result_path),
            "--max_new_tokens",
            str(cfg.max_tokens),
            "--temperature",
            str(cfg.temperature),
        ]

    def predict_batch(self, samples, cfg, coord_space):
        with tempfile.TemporaryDirectory(prefix="screenspot_infer_") as tmp:
            tmp_dir = Path(tmp)
            result_path = tmp_dir / "result.jsonl"
            cmd = self._command(self._write_batch(samples, tmp_dir), result_path, cfg)

            env = os.environ.copy()  # The `swift infer` child inherits this process's environment
            if cfg.max_pixels:
                env["MAX_PIXELS"] = str(cfg.max_pixels)

            print("Running:", " ".join(cmd))
            subprocess.run(cmd, env=env, check=True)
            return read_swift_results(result_path, len(samples))


class OpenAIPredictor(Predictor):
    """Any OpenAI-compatible /chat/completions endpoint, same wire format the student eval uses."""

    def __init__(self, model, cfg):
        self.model = model
        self.workers = EVAL.screenspot_workers
        endpoint = EVAL.screenspot_endpoint.rstrip("/")
        if not endpoint:
            raise ValueError("set SCREENSPOT_ENDPOINT to the /v1 base url of the served model")
        self.client = OpenAICompatibleClient(
            endpoint=endpoint,
            model=model,
            api_key=EVAL.screenspot_api_key,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            retry=SPENT_ATTEMPTS_SCORE_AS_PARSE_FAILURE,
            send_empty_auth=True,
        )
        self.cfg = cfg

    def _one(self, sample, coord_space):
        image = prepare_sample_image(sample.image)
        messages = build_openai_messages(sample.instruction, encode_png(image))
        return self.client.chat(messages)

    def predict_batch(self, samples, cfg, coord_space):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(lambda s: self._one(s, coord_space), samples))


class ClaudePredictor(Predictor):
    """The frontier arm, through the logged-in `claude` CLI and never an API key; grounding is the fair lane."""

    def __init__(self, model, cfg):
        self.model = model
        self.workers = EVAL.screenspot_claude_workers
        # A runaway guard, not a throttle: a cap near the true per-call cost returns "" instead of an answer.
        self.budget_usd = EVAL.screenspot_claude_budget_usd
        self.cfg = cfg

    def _one(self, sample, coord_space):
        # claude_agent_sdk ships only in the pipeline extra, absent from the `.[eval]` box.
        from claude_agent_sdk import ClaudeAgentOptions

        # cua.pipeline is not shipped to the eval box, so this arm's deps load only when it runs.
        from cua.pipeline.agent_runner import run_agent

        # Same, for the centrally configured model id.
        from cua.pipeline.llm_config import MODEL_FOR_REASONING

        content = build_claude_content(sample.instruction, encode_png(prepare_sample_image(sample.image)))

        async def stream():
            yield {
                "type": "user",
                "message": {"role": "user", "content": content},
                "parent_tool_use_id": None,
                "session_id": "",
            }

        options = ClaudeAgentOptions(
            model=self.model or MODEL_FOR_REASONING,
            system_prompt=CLAUDE_GROUNDING_SYSTEM_PROMPT,
            max_turns=1,
            max_budget_usd=self.budget_usd,
            permission_mode="bypassPermissions",
        )
        return asyncio.run(run_agent(stream(), options, purpose="screenspot", raise_on_error=False))

    def predict_batch(self, samples, cfg, coord_space):
        # Threads, not asyncio.gather: each call owns a CLI subprocess and its own timeout in a fresh loop.
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(lambda s: self._one(s, coord_space), samples))


def build_openai_predictor(spec, cfg):
    return OpenAIPredictor(model=spec.model, cfg=cfg)


def build_claude_predictor(spec, cfg):
    return ClaudePredictor(model=spec.model, cfg=cfg)


def build_swift_predictor(spec, cfg):
    return SwiftPredictor(model=spec.model, adapters=spec.adapters, cfg=cfg)


def build_mock_predictor(spec, cfg):
    return MockPredictor(default_mock_response)


PREDICTOR_BACKENDS: dict = {
    "swift": build_swift_predictor,
    "openai": build_openai_predictor,
    "claude": build_claude_predictor,
    "mock": build_mock_predictor,
}


def build_predictor(spec, cfg):
    backend = spec.backend or cfg.backend
    if backend not in PREDICTOR_BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}. Available: {sorted(PREDICTOR_BACKENDS)}")
    return PREDICTOR_BACKENDS[backend](spec, cfg)
