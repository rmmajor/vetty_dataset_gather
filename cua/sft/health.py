"""The convert-time corpus report plus the two fail-closed gates that can refuse a whole dataset."""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from cua.pipeline import operator_pii
from cua.pipeline.corpus_audit import corpus_health_violations
from cua.prompt.coords import COORD_SPACE
from cua.prompt.format import PREV_IMAGE_SIZE, PROMPT_FORMAT_VERSION

# High-entropy literals that cannot be legitimate page content; masked credentials ship as `<secret:NAME>`, so a literal is a real leak.
_SECRET_PATTERNS = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("github_token", re.compile(r"\bghp_[A-Za-z0-9]{30,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
]


@dataclass(frozen=True)
class CorpusSummary:
    """The convert-run numbers the report and the health gate both read."""

    meandering_mode: str
    grounding_share: float
    grounding_seed: int
    agent_examples: int
    total_examples: int
    grounding_examples: int

    @property
    def grounding_share_actual(self):
        return self.grounding_examples / self.agent_examples if self.agent_examples else 0.0


def scan_text_for_secrets(text):
    return [name for name, pat in _SECRET_PATTERNS if pat.search(text)] + operator_pii.scan_text(text)


def scan_examples_for_secrets(examples):
    """Distinct findings across every message of one trajectory's examples — the text that would ship."""
    findings = []
    for example in examples:
        for message in example["messages"]:
            content = message["content"]
            text = (
                content
                if isinstance(content, str)
                else " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            )
            for finding in scan_text_for_secrets(text):
                if finding not in findings:
                    findings.append(finding)
    return findings


def print_corpus_overview(report, summary):
    """The at-a-glance health check that catches a Booking.com-style regression at convert time, not weeks later."""
    lengths = sorted(report["lengths"])
    total_actions = sum(report["actions"].values()) or 1
    total_targets = sum(report["target_actions"].values()) or 1
    effect_steps = report["effect_steps"] or 1
    print("\n=== Corpus report ===")
    print(
        f"Prompt format: {PROMPT_FORMAT_VERSION} ({COORD_SPACE}, prev image {PREV_IMAGE_SIZE[0]}x{PREV_IMAGE_SIZE[1]})"
    )
    print(
        f"Trajectories trained on: {report['trajectories']}  (examples: {summary.total_examples} = "
        f"{summary.agent_examples} agent steps + {summary.grounding_examples} grounding)"
    )
    print(f"By source: {report['by_source']}")
    print(f"Autonomy: {report['autonomy']}  Auth modes: {report['auth_mode']}")
    print(f"Thought sources: {report['thought_sources']}")
    if lengths:
        mid = lengths[len(lengths) // 2]
        print(
            f"Steps/trajectory: min={lengths[0]} median={mid} p90={lengths[int(len(lengths) * 0.9)]} max={lengths[-1]}"
        )
    print(
        "Action mix: "
        + ", ".join(
            f"{t}={n} ({n / total_actions:.0%})" for t, n in sorted(report["actions"].items(), key=lambda kv: -kv[1])
        )
    )
    print(
        "Supervision-target mix: "
        + ", ".join(
            f"{t}={n} ({n / total_targets:.0%})"
            for t, n in sorted(report["target_actions"].items(), key=lambda kv: -kv[1])
        )
    )
    print(
        f"Terminal done() steps: {report['done_steps']}/{report['trajectories']} trajectories "
        f"(done in emitted targets: {report['done_trajectories']})"
    )
    print(
        f"Meandering (mode={summary.meandering_mode}): {report['meandering']} trajectories, "
        f"{report['meandering_examples']} examples kept, {report['subsample_dropped']} step targets subsampled away"
    )
    print(f"Waste steps dropped as targets (--exclude-waste-steps): {report['waste_steps_dropped']}")
    print(
        f"History effects: [no change] on {report['effect_no_change']}/{report['effect_steps']} annotated steps "
        f"({report['effect_no_change'] / effect_steps:.1%}), URL path change on "
        f"{report['effect_url_change']}/{report['effect_steps']} ({report['effect_url_change'] / effect_steps:.1%})"
    )


def print_corpus_footer(report, skipped):
    print("Domains: " + ", ".join(f"{d}={n}" for d, n in sorted(report["domains"].items(), key=lambda kv: -kv[1])[:10]))
    print(f"Skipped: {skipped}")
    print("======================\n")
    print(
        "Operator-identity patterns: "
        + (
            ", ".join(operator_pii.mask(v) for v in operator_pii.configured_identities())
            or "NONE — the PII gate is inert; set OPERATOR_PII or credentials/operator_pii.txt"
        )
    )


def health_stats(report, examples_by_group, summary, grounding):
    total_targets = sum(report["target_actions"].values()) or 1
    max_group, max_group_examples = max(examples_by_group.items(), key=lambda kv: len(kv[1]), default=("", []))
    return {
        "trainable": report["trajectories"],
        "done_coverage": (report["done_trajectories"] / report["trajectories"] if report["trajectories"] else 0.0),
        "teacher_raw": report["teacher_raw"],
        "meandering_example_share": (
            report["meandering_examples"] / summary.agent_examples if summary.agent_examples else 0.0
        ),
        "max_group": max_group,
        "max_group_example_share": (
            len(max_group_examples) / summary.total_examples if summary.total_examples else 0.0
        ),
        # Agent step targets only: grounding clicks would otherwise inflate the click share past every floor.
        "example_action_share": {t: n / total_targets for t, n in report["target_actions"].items()},
        "grounding_examples": grounding["grounding_examples"],
        "grounding_share": summary.grounding_share_actual,
        "grounding_example_share": (
            grounding["grounding_examples"] / summary.total_examples if summary.total_examples else 0.0
        ),
        "grounding_small_target_share": grounding["grounding_small_target_share"],
    }


def leak_gate(leak_sources):
    """FAIL CLOSED: any literal secret or operator identity refuses the whole dataset."""
    if not leak_sources:
        return
    print(f"SECRET/PII LEAK DETECTED in {len(leak_sources)} trajectories — refusing to write dataset.")
    for traj_dir, findings in leak_sources:
        print(f"  {traj_dir}: {', '.join(findings)}")
    print(
        "Quarantine each one (metadata.json excluded_reason) or re-capture: credentials belong in "
        "<secret:NAME> placeholders, and a real person's identity belongs nowhere in the corpus."
    )
    sys.exit(1)


def health_gate(stats, health_config, skip_health_gate):
    """FAIL CLOSED against corpus_health.json: the regressions that produced sft-v1's gaps never ship again."""
    health_path = Path(health_config)
    if not health_path.is_file():
        print(f"Corpus health gate: SKIPPED — no thresholds file at {health_path}")
        return
    violations = corpus_health_violations(stats, json.loads(health_path.read_text()))
    if not violations:
        print(f"Corpus health gate: PASS ({health_path})")
        return
    print(
        f"CORPUS HEALTH GATE {'(bypassed with --skip-health-gate)' if skip_health_gate else 'FAILED'} "
        f"— {len(violations)} violation(s) against {health_path}:"
    )
    for violation in violations:
        print(f"  - {violation}")
    if not skip_health_gate:
        print("Fix the corpus (or consciously bypass with --skip-health-gate) before shipping a dataset.")
        sys.exit(2)
