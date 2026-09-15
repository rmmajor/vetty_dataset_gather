"""Rewrite trainable-corpus instructions into a registered target voice (STYLE_MODES: goal | qa_verify), resumable via metadata.instruction_style."""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions

from cua.pipeline.agent_runner import run_agent
from cua.pipeline.corpus_audit import QA_VERIFY_STYLE, demonstrates_recheck, in_qa_verify_voice, is_trainable
from cua.pipeline.llm_config import MODEL_FOR_GROUNDED_GENERATION
from cua.pipeline.relabel_recoverable import NOT_GOAL_SHAPED
from cua.pipeline.sdk_runner import UsageLimitReached

logger = logging.getLogger(__name__)

REWRITE_BUDGET_USD = 0.10
# The qa_verify prompt carries the run's thoughts, so it sits well above the SDK's per-call floor.
QA_VOICE_BUDGET_USD = float(os.environ.get("QA_VOICE_BUDGET_USD", "0.50"))
QA_VERIFY_MAX_SHARE = 0.25
THOUGHT_CHARS = 160
FINAL_THOUGHT_CHARS = 400
# Long runs are shown head + tail so the prompt stays bounded; the final thought is always passed in full.
MAX_PROMPT_THOUGHTS = 40

_STARTS_WITH_ROUTE = re.compile(r"^(open|go to|navigate|visit|click|scroll)\b", re.IGNORECASE)
# Route verbs mid-instruction; content specs ("a post with five items, each…") must NOT fire.
_ROUTE_VERBS = re.compile(
    r"\b(click|clicking|navigate|navigating|scroll|scrolling|go to|open the|advancing through|"
    r"proceed(ing)? (to|through)|then (type|select|press|choose))\b",
    re.IGNORECASE,
)

REWRITE_PROMPT = """Rewrite this browser-task instruction as the terse GOAL a real user would give.
Keep EVERY concrete parameter (names, titles, values, filters, quantities, dates) and the same
final outcome; drop only the route — no "open/click/scroll/navigate", no step lists, no UI tour.
One sentence when possible, two at most.

Instruction: {instruction}

Reply with ONLY the rewritten instruction."""

QA_VERIFY_PROMPT = """Restate this browser-task instruction as a QA test objective — the way a tester phrases an
acceptance check — for the SAME goal on the SAME site.

HARD CONSTRAINT: the instruction describes the EXPECTED application behaviour a tester would state
before running anything. It must never mention errors, retries, warnings, workarounds, or anything
that happened during the run; if the run hit an obstacle, the instruction still reads as if the
feature simply works.
  Bad:  "Inviting a user as Admin succeeds despite a misleading role-format error. Verify …"
  Good: "Inviting a user as Admin adds them to the Invited Users list. Verify …"

Shape: at most two sentences and at most 45 words.
  1. A short assertion-style title stating the behaviour that should hold.
  2. A sentence starting with "Verify" stating what should hold and how it shows on screen.
Keep every concrete value the original instruction specifies (names, titles, amounts, the exact
setting or filter): the task must stay reproducible from the instruction alone, so cut description
before cutting a value. Where the original left a choice open (which result, which option), keep it
open. No numbered steps, no element selectors, no URLs, no credentials, no "click"/"navigate" tours.

Example of the voice (for a different task): "Toggling Email notifications checkbox in Settings
persists after save and reload. Verify unchecking Email notifications and saving retains that state
on page reload."

Current instruction: {instruction}

What the run did (step thoughts, in order) — evidence of what the feature does, NOT text to echo:
{thoughts}

Final thought: {final_thought}

Reply with ONLY a JSON object: {{"instruction": "..."}}"""


@dataclass(frozen=True)
class StyleMode:
    tag: str
    provenance_key: str
    purpose: str
    report_name: str
    selects: Callable[[dict, dict], bool]
    already_styled: Callable[[dict], bool]
    build_prompt: Callable[[dict], str]
    parse_reply: Callable[[str], str | None]
    budget_usd: float = REWRITE_BUDGET_USD
    max_share: float | None = None
    dry_run_previews: bool = False


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def is_procedure_flavored(instruction: str) -> bool:
    text = (instruction or "").strip()
    if _STARTS_WITH_ROUTE.match(text):
        return True
    return len(_ROUTE_VERBS.findall(text)) >= 2


def _trim(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _thought_lines(steps: list[dict]) -> str:
    thoughts = [(step.get("thought") or "") for step in steps]
    indices = list(range(len(thoughts)))
    if len(indices) > MAX_PROMPT_THOUGHTS:
        half = MAX_PROMPT_THOUGHTS // 2
        indices = indices[:half] + [None] + indices[-half:]
    lines = []
    for index in indices:
        lines.append("  …" if index is None else f"  {index + 1}. {_trim(thoughts[index], THOUGHT_CHARS)}")
    return "\n".join(lines) or "  (no recorded thoughts)"


def build_qa_verify_prompt(trajectory: dict) -> str:
    steps = trajectory.get("steps", [])
    final_thought = _trim(steps[-1].get("thought") or "", FINAL_THOUGHT_CHARS) if steps else ""
    return QA_VERIFY_PROMPT.format(
        instruction=trajectory.get("instruction", ""), thoughts=_thought_lines(steps), final_thought=final_thought
    )


def parse_goal_reply(text: str) -> str | None:
    new = (text or "").strip().strip('"')
    if not new or len(new) > 300 or "\n" in new or NOT_GOAL_SHAPED.search(new):
        return None
    return new


def parse_json_instruction(text: str) -> str | None:
    """The `instruction` field of the first JSON object in the reply, or None when there is none to read."""
    start = (text or "").find("{")
    if start == -1:
        return None
    snippet = text[start:]
    try:
        parsed = json.JSONDecoder().raw_decode(snippet)[0]
    except json.JSONDecodeError:
        try:
            parsed = json.JSONDecoder().raw_decode(re.sub(r",(\s*[}\]])", r"\1", snippet))[0]
        except json.JSONDecodeError:
            return None
    instruction = parsed.get("instruction") if isinstance(parsed, dict) else None
    return " ".join(instruction.split()) or None if isinstance(instruction, str) else None


STYLE_MODES: dict[str, StyleMode] = {
    "goal": StyleMode(
        tag="goal_rewrite_v1",
        provenance_key="original_instruction",
        purpose="instruction_style",
        report_name="instruction_style_report.md",
        selects=lambda trajectory, meta: is_procedure_flavored(trajectory.get("instruction", "")),
        already_styled=lambda meta: meta.get("instruction_style") == "goal_rewrite_v1",
        build_prompt=lambda trajectory: REWRITE_PROMPT.format(instruction=trajectory.get("instruction", "")),
        parse_reply=parse_goal_reply,
    ),
    "qa_verify": StyleMode(
        tag=QA_VERIFY_STYLE,
        provenance_key="instruction_pre_qa_voice",
        purpose="instruction_qa_voice",
        report_name="instruction_qa_voice_report.md",
        selects=lambda trajectory, meta: (
            not meta.get("wall_demo") and demonstrates_recheck(trajectory.get("steps", []))
        ),
        already_styled=in_qa_verify_voice,
        build_prompt=build_qa_verify_prompt,
        parse_reply=parse_json_instruction,
        budget_usd=QA_VOICE_BUDGET_USD,
        max_share=QA_VERIFY_MAX_SHARE,
        dry_run_previews=True,
    ),
}


def find_targets(corpus_dir: Path, mode: StyleMode) -> list[Path]:
    """Trainable dirs the mode selects, in sorted order; already-styled dirs count against the mode's cap."""
    trainable = [
        d for d in sorted(corpus_dir.glob("**/task_*")) if (d / "trajectory.json").exists() and is_trainable(d)
    ]
    styled = 0
    targets = []
    for traj_dir in trainable:
        meta = _load(traj_dir / "metadata.json")
        if mode.already_styled(meta):
            styled += 1
        elif mode.selects(_load(traj_dir / "trajectory.json"), meta):
            targets.append(traj_dir)
    if mode.max_share is not None:
        targets = targets[: max(int(mode.max_share * len(trainable)) - styled, 0)]
    return targets


def write_rewrite(traj_dir: Path, mode: StyleMode, old: str, new: str) -> None:
    """Update the instruction where convert reads it (trajectory.json) and keep metadata.json consistent."""
    traj_file = traj_dir / "trajectory.json"
    traj = _load(traj_file)
    traj["instruction"] = new
    traj_file.write_text(json.dumps(traj, indent=2))
    meta_file = traj_dir / "metadata.json"
    meta = _load(meta_file)
    meta.update(
        {"instruction": new, mode.provenance_key: meta.get(mode.provenance_key) or old, "instruction_style": mode.tag}
    )
    meta_file.write_text(json.dumps(meta, indent=2))


async def rewrite_one(traj_dir: Path, mode: StyleMode, apply: bool) -> tuple[str, str, str]:
    """(status, old, new) — status is rewritten | unchanged | empty | unparseable; writes only when apply is set."""
    traj = _load(traj_dir / "trajectory.json")
    old = traj.get("instruction", "")
    text = await run_agent(
        mode.build_prompt(traj),
        ClaudeAgentOptions(
            model=MODEL_FOR_GROUNDED_GENERATION,
            max_turns=1,
            max_budget_usd=mode.budget_usd,
            permission_mode="bypassPermissions",
        ),
        full_name=f"{mode.tag}:{traj_dir.name}",
        stage=7,
        purpose=mode.purpose,
        raise_on_error=False,
    )
    if not (text or "").strip():
        return "empty", old, ""
    new = mode.parse_reply(text)
    if new is None:
        return "unparseable", old, text.strip()
    if new == old:
        return "unchanged", old, new
    if apply:
        write_rewrite(traj_dir, mode, old, new)
    return "rewritten", old, new


def _write_report(corpus_dir: Path, mode: StyleMode, rows: list, apply: bool, report_dir: Path) -> None:
    lines = [
        f"# Instruction rewrite report — mode {mode.tag} ({'applied' if apply else 'dry run'})",
        "",
        "| trajectory | status | old | new |",
        "| --- | --- | --- | --- |",
    ]
    for traj_dir, status, old, new in sorted(rows, key=lambda r: str(r[0])):
        old, new = (s.replace("|", "/").replace("\n", " ")[:300] for s in (old, new))
        lines.append(f"| {traj_dir.relative_to(corpus_dir)} | {status} | {old} | {new} |")
    report = report_dir / mode.report_name
    report.write_text("\n".join(lines) + "\n")
    logger.info(f"report: {report} ({len(rows)} rows)")


async def run_pass(
    corpus_dir: Path, mode: StyleMode, workers: int, limit: int | None, apply: bool, report_dir: Path = Path(".")
) -> dict:
    targets = find_targets(corpus_dir, mode)
    if limit:
        targets = targets[:limit]
    logger.info(f"{len(targets)} instructions selected for mode {mode.tag} ({'apply' if apply else 'dry run'})")
    semaphore = asyncio.Semaphore(workers)
    outcomes: dict = {}
    rows: list = []

    async def guarded(traj_dir):
        async with semaphore:
            status, old, new = await rewrite_one(traj_dir, mode, apply)
        outcomes[status] = outcomes.get(status, 0) + 1
        rows.append((traj_dir, status, old, new))
        logger.info(f"  {traj_dir.name}: {status}")
        print(f"\n{traj_dir.relative_to(corpus_dir)} [{status}]\n  old: {old}\n  new: {new}")

    try:
        await asyncio.gather(*(guarded(d) for d in targets))
    except UsageLimitReached:
        logger.warning("usage limit reached — unstyled targets remain selectable; re-run to resume")

    if rows:
        _write_report(corpus_dir, mode, rows, apply, report_dir)
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", type=Path, nargs="?", default=Path("trajectories/"))
    parser.add_argument("--mode", choices=sorted(STYLE_MODES), default="goal")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the selection and exit (default); qa_verify also previews old -> new without writing",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    mode = STYLE_MODES[args.mode]

    if args.apply or mode.dry_run_previews:
        print(asyncio.run(run_pass(args.corpus, mode, args.workers, args.limit, apply=args.apply)))
    else:
        targets = find_targets(args.corpus, mode)
        for traj_dir in targets:
            instr = _load(traj_dir / "trajectory.json").get("instruction", "")
            print(f"{traj_dir.relative_to(args.corpus)}: {instr[:140]}")
        print(f"\n{len(targets)} instructions selected for mode {mode.tag}")
    sys.exit(0)


if __name__ == "__main__":
    main()
