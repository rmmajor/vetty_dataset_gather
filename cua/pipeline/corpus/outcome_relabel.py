"""Outcome-checked thought relabel: every thought opens with what the previous action did, judged from consecutive screens."""

import asyncio
import base64
import json
import logging
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions

from cua.pipeline.agent_runner import run_agent
from cua.pipeline.corpus.cli import add_shared, build_parser
from cua.pipeline.corpus.pass_base import CorpusPass, Outcome
from cua.pipeline.llm_config import MODEL_FOR_GROUNDED_GENERATION
from cua.pipeline.sdk_runner import UsageLimitReached
from cua.prompt.coords import to_model_action
from cua.prompt.effects import downscale_prev, step_effects
from cua.prompt.format import PREV_IMAGE_SIZE, history_entries_for_step
from cua.settings import PIPELINE
from cua.trajectory.actions import format_action_summary
from cua.trajectory.io import TRAJECTORY_FILE, TrajectoryDir

logger = logging.getLogger(__name__)

THOUGHT_SOURCE = "teacher_outcome"
BACKUP_NAME = "trajectory_pre_outcome.json"
IMAGE_SIZE = PREV_IMAGE_SIZE
NO_PREVIOUS_ACTION = "none (this is the starting screen)"
# Estimate knobs: image tokens per downscaled screen on the judge model, list prices per 1M tokens, SDK per-call floor.
TEXT_TOKENS_PER_STEP = 130
PROMPT_TOKENS = 450
OUTPUT_TOKENS_PER_STEP = 70

OUTCOME_PROMPT = """Above are the screens an agent saw while carrying out a web task, one per step, labelled
"Screen at step k" (shown at reduced size; where text is too small to read, trust the step's
current thought for what it says). Below, each step lists the action that came BEFORE the screen
(with the note the model will be given about it), the action taken next, and the thought written
at the time. Rewrite each step's thought into training supervision for a computer-use model: the
first-person, present-tense inner monologue of someone who looks at the screen, checks what the
last move did, and decides the next move.

Rules for every rewritten thought:
- 1-3 short sentences, first person, present tense.
- The FIRST sentence states the outcome of the previous action, judged by comparing the previous
  screen with the current one — e.g. "Clicking Save worked: a 'Settings saved' banner is visible."
  or "Nothing changed after that click; the menu is still closed." For step 1 (no previous action)
  describe what the starting screen shows instead.
- The bracketed note after a previous action is what the model will be told: [no change] means the
  screen stayed the same, -> /path means the page moved to that path. Agree with it and say what it
  means on screen.
- Then say what to do next and why, tied to the goal.
- Keep the original's observations and any visible UI text; you are adding the outcome check and
  polishing voice, not inventing content.
- Remove ALL traces of tooling: never mention refs, snapshots, screenshots, tools, selectors,
  elements by number, coordinates, "the page state", or being an AI or agent under instruction.
  Say "the screen" or "the page".
- For a step whose action is done(): cite each visible condition of the goal that the current screen
  satisfies — concrete on-screen evidence, one per condition. Never "the goal is accomplished"
  without evidence.
- For a step whose action is handoff(...) or confirm(...), or that ends without reaching the goal:
  state what is seen and why stopping (or asking) here is right.
- A step marked CONTEXT ONLY is already final: use it to understand the situation, but do not
  output a thought for it.

Goal of the task: {goal}
Starting URL: {start_url}

Steps:
{steps_block}

Output ONLY a JSON array of {n} strings — the rewritten thought for each step that is not context-only,
in step order. No markdown fence, no prose."""


def needs_relabel(traj_dir: Path) -> bool:
    return TrajectoryDir.load(traj_dir).metadata.get("thought_source") != THOUGHT_SOURCE


def step_windows(n_steps: int, size: int = PIPELINE.outcome_window_steps) -> list[tuple[int | None, list[int]]]:
    """(context index or None, target indices) per window; the context is the previous window's last target."""
    windows = []
    start = 0
    while start < n_steps:
        targets = list(range(start, min(start + size, n_steps)))
        windows.append((start - 1 if start else None, targets))
        start += size
    return windows


def _actions_summary(step: dict) -> str:
    actions = step.get("actions") or ([step["action"]] if "action" in step else [])
    return ", ".join(format_action_summary(to_model_action(a)) for a in actions) or "(none)"


def render_steps_block(steps: list[dict], effects: list, context_index: int | None, target_indices: list[int]) -> str:
    lines = []
    for index in ([context_index] if context_index is not None else []) + target_indices:
        step = steps[index]
        previous = (
            ", ".join(history_entries_for_step(steps[index - 1], effects[index - 1])) if index else NO_PREVIOUS_ACTION
        )
        header = f'Step {step["step"]} (see "Screen at step {step["step"]}")'
        if index == context_index:
            header += " — CONTEXT ONLY, thought already final, do not output it"
        lines.append(header)
        lines.append(f"- previous action: {previous}")
        lines.append(f"- action taken next: {_actions_summary(step)}")
        lines.append(f"- current thought: {(step.get('thought') or '').strip() or '(empty)'}")
        lines.append("")
    return "\n".join(lines).rstrip()


def build_window_blocks(
    traj_dir: Path, trajectory: dict, effects: list, context_index: int | None, target_indices: list[int]
) -> list[dict]:
    """Content blocks for one window: labelled downscaled screens, then the prompt with the steps block."""
    steps = trajectory["steps"]
    blocks = []
    for index in ([context_index] if context_index is not None else []) + target_indices:
        step = steps[index]
        png = downscale_prev((traj_dir / step["screenshot"]).read_bytes(), IMAGE_SIZE)
        blocks.append({"type": "text", "text": f"Screen at step {step['step']}"})
        blocks.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(png).decode("utf-8")},
            }
        )
    blocks.append(
        {
            "type": "text",
            "text": OUTCOME_PROMPT.format(
                goal=(trajectory.get("instruction") or "(goal not recorded)").strip(),
                start_url=trajectory.get("start_url") or "(not recorded)",
                steps_block=render_steps_block(steps, effects, context_index, target_indices),
                n=len(target_indices),
            ),
        }
    )
    return blocks


def parse_thought_array(text: str, n: int) -> list[str] | None:
    """The JSON array between the first [ and the last ]; None unless it holds exactly n non-empty strings."""
    start, end = text.find("["), text.rfind("]") + 1
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or len(parsed) != n or not all(isinstance(t, str) and t.strip() for t in parsed):
        return None
    return [t.strip() for t in parsed]


async def llm_call(blocks: list[dict], full_name: str) -> str:
    async def prompt_stream():
        yield {
            "type": "user",
            "message": {"role": "user", "content": blocks},
            "parent_tool_use_id": None,
            "session_id": "",
        }

    return await run_agent(
        prompt_stream(),
        # tools=[] removes every built-in tool: a tool call would consume the single turn and the SDK raises error_max_turns.
        ClaudeAgentOptions(
            model=MODEL_FOR_GROUNDED_GENERATION,
            max_turns=1,
            max_budget_usd=PIPELINE.outcome_budget_usd,
            permission_mode="bypassPermissions",
            tools=[],
        ),
        full_name=full_name,
        stage=7,
        purpose="outcome_relabel",
        raise_on_error=False,
    )


def write_relabel(
    traj_dir: Path, trajectory: dict, original_bytes: bytes, windows: int, unchanged_windows: int
) -> None:
    """Back up the original once, replace the thoughts where convert reads them, and record provenance."""
    backup = traj_dir / BACKUP_NAME
    if not backup.exists():
        backup.write_bytes(original_bytes)
    recorded = TrajectoryDir.load(traj_dir)
    recorded.save_trajectory(trajectory)
    metadata = recorded.metadata
    metadata["thought_source"] = THOUGHT_SOURCE
    metadata["outcome_relabel"] = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": MODEL_FOR_GROUNDED_GENERATION,
        "windows": windows,
        "unchanged_windows": unchanged_windows,
    }
    recorded.save_metadata()


async def relabel_trajectory(traj_dir: Path, call_llm=llm_call, apply: bool = False) -> dict:
    """Rewrite one trajectory's thoughts window by window; returns status, window counts and (step, old, new) changes."""
    trajectory_path = traj_dir / TRAJECTORY_FILE
    original_bytes = trajectory_path.read_bytes()
    trajectory = json.loads(original_bytes)
    steps = trajectory.get("steps", [])
    if not steps:
        return {"status": "empty", "windows": 0, "unchanged_windows": 0, "changes": []}
    effects = step_effects(traj_dir, steps, TrajectoryDir.load(traj_dir).capture_meta)
    windows = step_windows(len(steps))
    unchanged_windows = 0
    changes = []
    for number, (context_index, target_indices) in enumerate(windows, start=1):
        blocks = build_window_blocks(traj_dir, trajectory, effects, context_index, target_indices)
        # Windows run in order so each later window sees the already rewritten context thought.
        try:
            text = await call_llm(blocks, f"outcome_relabel:{traj_dir.name}:w{number}")
        except UsageLimitReached:
            raise
        except Exception as e:
            logger.warning(
                f"  {traj_dir.name}: window {number}/{len(windows)} call failed ({e}); leaving its thoughts unchanged"
            )
            text = ""
        rewritten = parse_thought_array(text or "", len(target_indices))
        if rewritten is None:
            logger.warning(
                f"  {traj_dir.name}: window {number}/{len(windows)} returned no usable "
                f"{len(target_indices)}-thought array; leaving its thoughts unchanged"
            )
            unchanged_windows += 1
            changes.extend(
                (steps[i]["step"], steps[i].get("thought", ""), steps[i].get("thought", "")) for i in target_indices
            )
            continue
        for index, thought in zip(target_indices, rewritten, strict=False):
            changes.append((steps[index]["step"], steps[index].get("thought", ""), thought))
            steps[index]["thought"] = thought
    if unchanged_windows == len(windows):
        return {
            "status": "unchanged",
            "windows": len(windows),
            "unchanged_windows": unchanged_windows,
            "changes": changes,
        }
    if apply:
        write_relabel(traj_dir, trajectory, original_bytes, len(windows), unchanged_windows)
    return {"status": "relabeled", "windows": len(windows), "unchanged_windows": unchanged_windows, "changes": changes}


def estimate_cost(targets: list[Path]) -> dict:
    """Window/image/token counts for the given targets priced at the module's list-price knobs."""
    trajectories = steps = windows = images = 0
    for traj_dir in targets:
        n_steps = len(TrajectoryDir.load(traj_dir).trajectory.get("steps", []))
        if not n_steps:
            continue
        trajectories += 1
        steps += n_steps
        for context_index, target_indices in step_windows(n_steps):
            windows += 1
            images += len(target_indices) + (context_index is not None)
    input_tokens = images * PIPELINE.outcome_image_tokens + steps * TEXT_TOKENS_PER_STEP + windows * PROMPT_TOKENS
    output_tokens = steps * OUTPUT_TOKENS_PER_STEP
    usd = (
        input_tokens / 1e6 * PIPELINE.outcome_price_in_per_m
        + output_tokens / 1e6 * PIPELINE.outcome_price_out_per_m
        + windows * PIPELINE.outcome_call_overhead_usd
    )
    return {
        "trajectories": trajectories,
        "steps": steps,
        "windows": windows,
        "images": images,
        "image_tokens": images * PIPELINE.outcome_image_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "usd": usd,
    }


def print_changes(corpus_dir: Path, traj_dir: Path, result: dict, show: int) -> None:
    print(
        f"\n{traj_dir.relative_to(corpus_dir)} [{result['status']}] "
        f"{len(result['changes'])} steps, {result['windows']} windows, {result['unchanged_windows']} unchanged"
    )
    for step_number, old, new in result["changes"][:show]:
        print(f"step {step_number}: {old}\n     → {new}")


def render_report(corpus_dir: Path, rows: list, apply: bool) -> str:
    lines = [
        f"# Outcome relabel report ({'applied' if apply else 'dry run'})",
        "",
        "| trajectory | step | old | new |",
        "| --- | --- | --- | --- |",
    ]
    for traj_dir, result in sorted(rows, key=lambda r: str(r[0])):
        for step_number, old, new in result["changes"]:
            old, new = (s.replace("|", "/").replace("\n", " ") for s in (old, new))
            lines.append(f"| {traj_dir.relative_to(corpus_dir)} | {step_number} | {old} | {new} |")
    return "\n".join(lines) + "\n"


class OutcomeRelabelPass(CorpusPass):
    """Windowed vision relabel of every thought; `writes` is what --apply flips, so a dry run still generates and prints old -> new."""

    name = "outcome_relabel"

    def __init__(self, writes: bool, show: int = 5, max_steps=None, sample=None, seed: int = 0, report_path=None):
        self.writes = writes
        self.show = show
        self.max_steps = max_steps
        self.sample = sample
        self.seed = seed
        self.report_path = report_path

    def select(self, recorded: TrajectoryDir) -> bool:
        if not needs_relabel(recorded.path) or not recorded.is_trainable():
            return False
        return self.max_steps is None or len(recorded.trajectory.get("steps", [])) <= self.max_steps

    def candidates(self, corpus: Path, limit: int | None = None) -> list[TrajectoryDir]:
        """Trainable, not-yet-relabeled dirs; --sample draws from the whole selection and --limit cuts what remains."""
        targets = super().candidates(corpus)
        if self.sample:
            drawn = random.Random(self.seed).sample(targets, min(self.sample, len(targets)))
            targets = sorted(drawn, key=lambda recorded: recorded.path)
        return targets[:limit] if limit else targets

    async def apply(self, recorded: TrajectoryDir) -> Outcome:
        try:
            result = await relabel_trajectory(recorded.path, apply=self.writes)
        except UsageLimitReached:
            raise
        except Exception as e:
            logger.warning(f"  {recorded.path.name}: relabel failed: {e}")
            result = {"status": "failed", "windows": 0, "unchanged_windows": 0, "changes": []}
        return Outcome(changed=result["status"] == "relabeled", note=result["status"], detail=result)

    def summary(self, corpus: Path, candidates: list) -> str:
        return f"{len(candidates)} trajectories to relabel under {corpus} ({'apply' if self.writes else 'dry run'})"

    def observe(self, corpus: Path, recorded: TrajectoryDir, outcome: Outcome) -> None:
        result = outcome.detail
        logger.info(
            f"  {recorded.path.name}: {result['status']} "
            f"({result['unchanged_windows']}/{result['windows']} windows unchanged)"
        )
        print_changes(corpus, recorded.path, result, self.show)

    def report(self, corpus: Path, outcomes: list) -> str:
        return render_report(corpus, [(outcome.path, outcome.detail) for outcome in outcomes], self.writes)

    def write_report(self, corpus: Path, outcomes: list) -> None:
        # The report file is a --report-path opt-in here, and an empty run leaves no table behind.
        if self.report_path is None or not outcomes:
            return
        self.report_path.write_text(self.report(corpus, outcomes))
        logger.info(f"report: {self.report_path}")


def print_estimate(estimate: dict) -> None:
    print(
        f"targets: {estimate['trajectories']} trajectories, {estimate['steps']} steps, {estimate['windows']} windows "
        f"({PIPELINE.outcome_window_steps} steps each), {estimate['images']} images"
    )
    print(
        f"tokens: ~{estimate['input_tokens']:,} in ({estimate['image_tokens']:,} image @ {PIPELINE.outcome_image_tokens}/image) "
        f"+ ~{estimate['output_tokens']:,} out"
    )
    print(
        f"estimate: ~${estimate['usd']:.2f} at ${PIPELINE.outcome_price_in_per_m}/M in, "
        f"${PIPELINE.outcome_price_out_per_m}/M out, "
        f"${PIPELINE.outcome_call_overhead_usd} SDK floor per call ({estimate['windows']} calls)"
    )


def main():
    parser = build_parser(
        __doc__,
        "corpus",
        "apply",
        "dry_run",
        apply={"help": "Write the rewrites (default is a dry run)"},
        dry_run={"help": "Print old -> new and write nothing (default)"},
    )
    add_shared(parser, "workers", "limit")
    parser.add_argument("--sample", type=int, default=None, help="Random subset of the targets (before --limit)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None, help="Only trajectories with at most N steps")
    parser.add_argument("--show", type=int, default=5, help="Steps printed per trajectory as OLD -> NEW")
    parser.add_argument("--report-path", type=Path, default=None, help="Also write a Markdown old/new table here")
    parser.add_argument("--estimate", action="store_true", help="Print the full-run cost estimate and exit")
    args = parser.parse_args()
    if args.apply and args.dry_run:
        parser.error("--apply and --dry-run are mutually exclusive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    corpus_pass = OutcomeRelabelPass(
        args.apply,
        show=args.show,
        max_steps=args.max_steps,
        sample=args.sample,
        seed=args.seed,
        report_path=args.report_path,
    )
    # Selected here, not inside the runner, because --estimate prices the same selection without running it.
    candidates = corpus_pass.candidates(args.corpus, args.limit)
    if args.estimate:
        print_estimate(estimate_cost([recorded.path for recorded in candidates]))
        return

    logger.info(corpus_pass.summary(args.corpus, candidates))
    counts = asyncio.run(corpus_pass.run(args.corpus, candidates=candidates, workers=args.workers))
    print(f"\n{'applied' if args.apply else 'dry run'}: {counts}")
    sys.exit(1 if counts.get("failed") and not counts.get("relabeled") else 0)


if __name__ == "__main__":
    main()
