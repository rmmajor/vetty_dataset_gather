"""Rewrite trainable-corpus instructions into a registered target voice (STYLE_MODES: goal | qa_verify), resumable via metadata.instruction_style."""

import asyncio
import json
import logging
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions

from cua.pipeline.agent_runner import run_agent
from cua.pipeline.corpus.audit import QA_VERIFY_STYLE, demonstrates_recheck, in_qa_verify_voice
from cua.pipeline.corpus.cli import add_shared, build_parser
from cua.pipeline.corpus.pass_base import CorpusPass, Outcome
from cua.pipeline.corpus.relabel_recoverable import NOT_GOAL_SHAPED
from cua.pipeline.llm_config import MODEL_FOR_GROUNDED_GENERATION
from cua.settings import PIPELINE
from cua.trajectory.io import TrajectoryDir, iter_trajectory_dirs

logger = logging.getLogger(__name__)

REWRITE_BUDGET_USD = 0.10
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
        budget_usd=PIPELINE.qa_voice_budget_usd,
        max_share=QA_VERIFY_MAX_SHARE,
        dry_run_previews=True,
    ),
}


class InstructionStylePass(CorpusPass):
    """One registered voice over the trainable corpus: `writes` is what --apply flips, so a dry run can still preview old -> new."""

    name = "instruction_style"

    def __init__(self, mode: StyleMode, writes: bool, report_dir: Path = Path(".")):
        self.mode = mode
        self.writes = writes
        self.report_path = report_dir / mode.report_name

    def select(self, recorded: TrajectoryDir) -> bool:
        return not self.mode.already_styled(recorded.metadata) and self.mode.selects(
            recorded.trajectory, recorded.metadata
        )

    def candidates(self, corpus: Path, limit: int | None = None) -> list[TrajectoryDir]:
        """Trainable dirs the mode selects, in sorted order; already-styled dirs count against the mode's cap."""
        trainable = [r for r in (TrajectoryDir.load(p) for p in iter_trajectory_dirs(corpus)) if r.is_trainable()]
        styled = sum(self.mode.already_styled(recorded.metadata) for recorded in trainable)
        targets = [recorded for recorded in trainable if self.select(recorded)]
        if self.mode.max_share is not None:
            targets = targets[: max(int(self.mode.max_share * len(trainable)) - styled, 0)]
        return targets[:limit] if limit else targets

    async def apply(self, recorded: TrajectoryDir) -> Outcome:
        status, old, new = await rewrite_one(recorded.path, self.mode, self.writes)
        return Outcome(changed=status == "rewritten", note=status, detail=(old, new))

    def summary(self, corpus: Path, candidates: list) -> str:
        return f"{len(candidates)} instructions selected for mode {self.mode.tag}"

    def describe(self, corpus: Path, recorded: TrajectoryDir) -> str:
        return f"{recorded.path.relative_to(corpus)}: {recorded.trajectory.get('instruction', '')[:140]}"

    def print_dry_run(self, corpus: Path, candidates: list) -> None:
        for recorded in candidates:
            print(self.describe(corpus, recorded))
        print(f"\n{self.summary(corpus, candidates)}")

    def observe(self, corpus: Path, recorded: TrajectoryDir, outcome: Outcome) -> None:
        old, new = outcome.detail
        logger.info(f"  {recorded.path.name}: {outcome.note}")
        print(f"\n{recorded.path.relative_to(corpus)} [{outcome.note}]\n  old: {old}\n  new: {new}")

    def report(self, corpus: Path, outcomes: list) -> str:
        lines = [
            f"# Instruction rewrite report — mode {self.mode.tag} ({'applied' if self.writes else 'dry run'})",
            "",
            "| trajectory | status | old | new |",
            "| --- | --- | --- | --- |",
        ]
        for outcome in sorted(outcomes, key=lambda o: str(o.path)):
            old, new = (s.replace("|", "/").replace("\n", " ")[:300] for s in outcome.detail)
            lines.append(f"| {outcome.path.relative_to(corpus)} | {outcome.note} | {old} | {new} |")
        return "\n".join(lines) + "\n"

    def write_report(self, corpus: Path, outcomes: list) -> None:
        # A run that selected nothing leaves no report behind, so a previous run's table is never read as this one's.
        if outcomes:
            super().write_report(corpus, outcomes)


def write_rewrite(traj_dir: Path, mode: StyleMode, old: str, new: str) -> None:
    """Update the instruction where convert reads it (trajectory.json) and keep metadata.json consistent."""
    recorded = TrajectoryDir.load(traj_dir)
    recorded.trajectory["instruction"] = new
    recorded.save_trajectory()
    meta = recorded.metadata
    meta.update(
        {"instruction": new, mode.provenance_key: meta.get(mode.provenance_key) or old, "instruction_style": mode.tag}
    )
    recorded.save_metadata()


async def rewrite_one(traj_dir: Path, mode: StyleMode, apply: bool) -> tuple[str, str, str]:
    """(status, old, new) — status is rewritten | unchanged | empty | unparseable; writes only when apply is set."""
    traj = TrajectoryDir.load(traj_dir).trajectory
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


def main():
    parser = build_parser(__doc__, "corpus")
    parser.add_argument("--mode", choices=sorted(STYLE_MODES), default="goal")
    add_shared(parser, "apply")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the selection and exit (default); qa_verify also previews old -> new without writing",
    )
    add_shared(parser, "workers", "limit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    mode = STYLE_MODES[args.mode]

    corpus_pass = InstructionStylePass(mode, args.apply)
    if args.apply or mode.dry_run_previews:
        print(asyncio.run(corpus_pass.run(args.corpus, limit=args.limit, workers=args.workers)))
    else:
        asyncio.run(corpus_pass.run(args.corpus, apply=False))
    sys.exit(0)


if __name__ == "__main__":
    main()
