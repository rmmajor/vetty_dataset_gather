"""The two rendered artefacts: one run's report.md, and the before/after comparison with its harness-parity check."""

import json

from cua.eval.student.fingerprint import EXPECTED_TO_DIFFER_KEYS, PARITY_EXEMPT_KEYS, coord_space_diagnosis

LEVEL_ORDER = ("easy", "medium", "hard")


def _format_deviations(deviations: dict) -> str:
    if not deviations:
        return "none"
    return ", ".join(f"{kind} {count}" for kind, count in deviations.items())


def _headline(results: dict) -> list:
    metrics = results["metrics"]
    source_name = results["harness"]["task_source"]
    return [
        f"# Student eval — `{results['label']}`",
        "",
        (
            f"**verified task success: {metrics['verified_success_rate'] * 100:.1f}% across "
            f"{metrics['episodes']} {source_name} tasks, "
            f"parse-failure rate {metrics['parse_failure_rate'] * 100:.1f}%, "
            f"site-blocked {metrics['site_blocked_rate'] * 100:.1f}%**"
        ),
        "",
        (
            f"Success is over the SCORED episodes ({metrics['successes']} verified of "
            f"{metrics['scored_episodes']}); site-blocked / unreachable / harness-error episodes are "
            f"excluded from that denominator and reported separately, because folding bot-blocking "
            f"into the success rate measures the network, not the model. Over all "
            f"{metrics['episodes']} episodes the rate is "
            f"{metrics['verified_success_rate_all_episodes'] * 100:.1f}%."
        ),
        "",
    ]


def _run_table(results: dict) -> list:
    cfg = results["config"]
    fingerprint = results["harness"]
    return [
        "## Run",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| run id | `{results['run_id']}` |",
        f"| label | `{results['label']}` |",
        f"| started (UTC) | {results['started_at']} |",
        f"| finished (UTC) | {results['finished_at']} |",
        f"| endpoint | `{cfg['endpoint'] or '(none — ' + cfg['student'] + ' backend)'}` |",
        f"| model | `{cfg['model'] or '(n/a)'}` |",
        f"| adapter / checkpoint | `{cfg['adapter'] or '(none declared)'}` |",
        f"| task source | `{fingerprint['task_source']}` — {fingerprint['task_source_description']} |",
        f"| dataset | `{fingerprint['dataset']}` |",
        f"| dataset revision | `{fingerprint['dataset_revision'] or '(n/a)'}` |",
        f"| coordinate space | `{fingerprint['coord_space']}` |",
        f"| step cap | {fingerprint['step_cap_policy']} |",
        (
            f"| abort rules | {fingerprint['max_unproductive_steps']} unproductive steps / "
            f"{fingerprint['max_parse_failures']} parse failures |"
        ),
        f"| task subset | {fingerprint['subset_strategy']} |",
        f"| temperature / max tokens | {fingerprint['temperature']} / {fingerprint['max_tokens']} |",
        f"| observation format | {fingerprint['screenshot_format']} @ 1280x720 |",
        f"| prompt format | `{fingerprint['prompt_format']}` |",
        f"| previous-screenshot size | {fingerprint['prev_image_size']} (from step 2 on) |",
        f"| effect annotations | {'on' if fingerprint['effect_annotations'] else 'off'} |",
        f"| type mode | {fingerprint['type_mode']} |",
        (
            f"| judge | {fingerprint['judge_model'] or 'disabled'} "
            f"(final + {fingerprint['judge_mid_screenshots']} mid screenshots) |"
        ),
        f"| history window | {fingerprint['history_window']} |",
        f"| driver loop guard | {fingerprint['driver_loop_guard_repeats'] or 'disabled'} |",
        "| browser | local DockerSandbox on the invoking host |",
        f"| repo commit | `{fingerprint['git_commit']}` |",
        "",
    ]


def _caveats(results: dict) -> list:
    """The banners that decide whether the numbers below may be read at all."""
    lines = []
    if results.get("aborted_reason"):
        lines += [
            (
                f"> **Run aborted early: {results['aborted_reason']}** — the numbers below "
                f"cover only the episodes that ran."
            ),
            "",
        ]
    warning = coord_space_diagnosis(results["harness"], results["metrics"])
    if warning:
        lines += [f"> **{warning}**", ""]
    if results.get("task_provenance"):
        lines += [
            "<details><summary>task provenance</summary>",
            "",
            "```json",
            json.dumps(results["task_provenance"], indent=2),
            "```",
            "",
            "</details>",
            "",
        ]
    return lines


def _metrics_table(metrics: dict) -> list:
    return [
        "## Metrics",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| episodes | {metrics['episodes']} |",
        f"| scored episodes | {metrics['scored_episodes']} |",
        f"| judge deferred (usage limit) | {metrics.get('judge_deferred_episodes', 0)} |",
        f"| verified success (scored) | {metrics['verified_success_rate'] * 100:.1f}% |",
        f"| verified success (all episodes) | {metrics['verified_success_rate_all_episodes'] * 100:.1f}% |",
        f"| verified AND non-meandering | {metrics['efficient_success_rate'] * 100:.1f}% |",
        f"| site-blocked | {metrics['site_blocked_rate'] * 100:.1f}% |",
        f"| parse-failure rate (steps) | {metrics['parse_failure_rate'] * 100:.1f}% |",
        f"| malformed action lines | {metrics['malformed_action_lines']} |",
        (
            f"| canonical action form | {metrics.get('canonical_form_rate', 0) * 100:.1f}% "
            f"({metrics.get('actions_canonical', 0)}/{metrics.get('actions_parsed', 0)} actions) |"
        ),
        f"| form deviations | {_format_deviations(metrics.get('action_form_deviations'))} |",
        f"| done() emission rate | {metrics['done_rate'] * 100:.1f}% |",
        f"| navigate actions executed | {metrics.get('navigate_actions', 0)} |",
        f"| episodes ended by handoff/confirm | {metrics.get('handoff_episodes', 0)} |",
        f"| <think> emission rate (steps) | {metrics['think_rate'] * 100:.1f}% |",
        f"| driver-refused action rate | {metrics['driver_error_rate'] * 100:.1f}% |",
        (
            f"| emitted coords outside the declared coordinate space | {metrics.get('emitted_out_of_viewport', 0)}"
            f"/{metrics.get('emitted_coords', 0)} "
            f"({metrics.get('emitted_out_of_viewport_rate', 0) * 100:.1f}%) |"
        ),
        f"| out-of-viewport actions refused | {metrics['out_of_bounds_actions']} |",
        f"| episodes abandoned as stuck | {metrics.get('stuck_episodes', 0)} |",
        f"| mean steps on success | {metrics['mean_steps_on_success']} |",
        f"| mean steps / reference_length on success | {metrics['mean_steps_over_reference_on_success']}x |",
        f"| mean steps (all) | {metrics['mean_steps_all']} |",
        f"| episodes that hit the step cap | {metrics['step_cap_hits']} |",
        "",
        (
            "`verified` is the judge's verdict on the ORIGINAL goal after its own criteria checklist is "
            "enforced (`judged` is the raw claim); the efficiency grade is reported beside it rather than "
            "folded into it. `steps / reference_length` compares against the "
            "benchmark's own human step count, which is a more defensible efficiency denominator than "
            "our raw step counts."
        ),
        "",
        (
            "`canonical action form` is the share of parsed actions spelled exactly as "
            "`format_action_for_training` emits them. The parser accepts any UNAMBIGUOUS spelling of a "
            "call (optional `action:`, bare or `<ref>`-wrapped coordinates, `wait` with or without "
            "`()`, omitted keyword arguments, single or double quotes, bare key names) and rejects "
            "genuinely ambiguous ones. A base model has only seen the grammar described in the system "
            "prompt, so charging surface deviation to its success rate would measure imitation of an "
            "unfamiliar convention rather than browsing. The deviation is reported here instead, and "
            "the tolerance is a harness parity key so both sides of a delta are parsed identically."
        ),
        "",
    ]


def _level_section(metrics: dict) -> list:
    lines = [
        "## Per level",
        "",
        "| level | n | scored | verified | success rate | steps/ref | blocked |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for level in sorted(metrics["by_level"], key=lambda k: LEVEL_ORDER.index(k) if k in LEVEL_ORDER else 99):
        bucket = metrics["by_level"][level]
        lines.append(
            f"| {level} | {bucket['n']} | {bucket['scored']} | {bucket['success']} | "
            f"{bucket['success_rate'] * 100:.1f}% | "
            f"{bucket['mean_steps_over_reference']}x | {bucket['blocked']} |"
        )
    return lines


def _outcome_section(metrics: dict) -> list:
    lines = ["", "## Outcomes", "", "| outcome | episodes |", "| --- | --- |"]
    for outcome, count in sorted(metrics["outcomes"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {outcome} | {count} |")
    return lines


def _domain_section(metrics: dict) -> list:
    lines = [
        "",
        "## Per-domain",
        "",
        "| domain | n | scored | verified | success rate | blocked |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for domain, bucket in sorted(metrics["by_domain"].items(), key=lambda kv: -kv[1]["n"]):
        lines.append(
            f"| {domain} | {bucket['n']} | {bucket['scored']} | {bucket['success']} | "
            f"{bucket['success_rate'] * 100:.1f}% | {bucket['blocked']} |"
        )
    return lines


def _episode_section(episodes: list) -> list:
    lines = [
        "",
        "## Episodes",
        "",
        "| task | level | outcome | verified | efficiency | steps | cap | ref | parse fails | driver errors | note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in episodes:
        note = r["block_reason"] or r["error"] or (r["judge_reasoning"] or "")[:120]
        note = note.replace("|", "/").replace("\n", " ")
        lines.append(
            f"| `{r['task_id'][:40]}` | {r.get('level') or '-'} | {r['outcome']} | "
            f"{'yes' if r['verified_success'] else 'no'} | {r['efficiency'] or '-'} | "
            f"{r['model_steps']} | {r.get('step_cap', 0)} | {r['reference_length']} | "
            f"{r['parse_failure_steps']} | {r['driver_errors']} | {note} |"
        )
    return lines


def _training_exclusion_footer() -> list:
    return [
        "",
        "---",
        "",
        (
            "Episodes are recorded under this run directory and are excluded from training by "
            "construction (outside `trajectories/`, and each carries "
            "`metadata.excluded_reason=eval_episode`)."
        ),
        "",
    ]


def render_report(results: dict) -> str:
    metrics = results["metrics"]
    lines = _headline(results) + _run_table(results) + _caveats(results) + _metrics_table(metrics)
    lines += _level_section(metrics)
    lines += _outcome_section(metrics)
    lines += _domain_section(metrics)
    lines += _episode_section(results["episodes"])
    lines += _training_exclusion_footer()
    return "\n".join(lines)


def _comparison_summary(runs: list) -> list:
    lines = [
        "# Student eval — comparison",
        "",
        (
            "| label | verified success | all-episode success | parse-failure | canonical form | "
            "done() rate | mean steps (success) | steps/ref | site-blocked | n |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        m = run["metrics"]
        lines.append(
            f"| `{run['label']}` | {m['verified_success_rate'] * 100:.1f}% | "
            f"{m['verified_success_rate_all_episodes'] * 100:.1f}% | "
            f"{m['parse_failure_rate'] * 100:.1f}% | "
            f"{m.get('canonical_form_rate', 0) * 100:.1f}% | {m['done_rate'] * 100:.1f}% | "
            f"{m['mean_steps_on_success']} | "
            f"{m.get('mean_steps_over_reference_on_success', 0)}x | "
            f"{m['site_blocked_rate'] * 100:.1f}% | {m['episodes']} |"
        )
    return lines


def _comparison_levels(runs: list, levels: list) -> list:
    if not levels:
        return []
    lines = [
        "",
        "## By level",
        "",
        "| label | " + " | ".join(levels) + " |",
        "| --- | " + " | ".join("---" for _ in levels) + " |",
    ]
    for run in runs:
        cells = []
        for level in levels:
            bucket = run["metrics"].get("by_level", {}).get(level)
            cells.append(
                f"{bucket['success_rate'] * 100:.1f}% ({bucket['success']}/{bucket['scored']})" if bucket else "-"
            )
        lines.append(f"| `{run['label']}` | " + " | ".join(cells) + " |")
    return lines


def _comparison_deltas(runs: list, levels: list) -> list:
    if len(runs) <= 1:
        return []
    base = runs[0]
    lines = [
        "",
        f"## Delta vs `{base['label']}`",
        "",
        "| label | verified success | parse-failure | canonical form | done() rate | "
        + " | ".join(f"{level} success" for level in levels)
        + " |",
        "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in levels) + " |",
    ]
    for run in runs[1:]:
        a, b = base["metrics"], run["metrics"]
        level_cells = []
        for level in levels:
            before = (a.get("by_level", {}).get(level) or {}).get("success_rate")
            after = (b.get("by_level", {}).get(level) or {}).get("success_rate")
            level_cells.append(f"{(after - before) * 100:+.1f} pp" if before is not None and after is not None else "-")
        lines.append(
            f"| `{run['label']}` | "
            f"{(b['verified_success_rate'] - a['verified_success_rate']) * 100:+.1f} pp | "
            f"{(b['parse_failure_rate'] - a['parse_failure_rate']) * 100:+.1f} pp | "
            f"{(b.get('canonical_form_rate', 0) - a.get('canonical_form_rate', 0)) * 100:+.1f} pp | "
            f"{(b['done_rate'] - a['done_rate']) * 100:+.1f} pp | " + " | ".join(level_cells) + " |"
        )
    return lines


def _parity_differences(runs: list) -> list:
    """Every harness key the runs disagree on, which is what makes a delta unattributable to the model."""
    base = runs[0]
    differences = []
    for key, value in base["harness"].items():
        if key in PARITY_EXEMPT_KEYS:
            continue
        for run in runs[1:]:
            if run["harness"].get(key) != value:
                differences.append(
                    f"- `{key}`: `{base['label']}`={value!r} vs `{run['label']}`={run['harness'].get(key)!r}"
                )
    return differences


def _comparison_parity(runs: list) -> list:
    annotated = []
    for key in EXPECTED_TO_DIFFER_KEYS:
        values = {run["label"]: run["harness"].get(key) for run in runs}
        if len(set(map(repr, values.values()))) > 1:
            annotated.append(f"- `{key}`: " + ", ".join(f"`{k}`={v!r}" for k, v in values.items()))

    differences = _parity_differences(runs)
    lines = ["", "## Harness parity", ""]
    if differences:
        lines += [
            "**These runs are NOT the same harness — the delta above is not attributable to the model alone:**",
            "",
        ] + differences
    else:
        lines.append(
            "All compared runs share the same task subset, step-cap and abort policy, "
            "prompt settings and judge configuration."
        )
    if annotated:
        lines += [
            "",
            "Expected model-side differences (not parity violations — each model is measured in its own convention):",
            "",
        ] + annotated
    commits = {run["label"]: run["harness"].get("git_commit") for run in runs}
    return lines + ["", f"Repo commits: {json.dumps(commits)}", ""]


def _comparison_domains(runs: list) -> list:
    domains = sorted({d for run in runs for d in run["metrics"]["by_domain"]})
    lines = [
        "## Per-domain success rate",
        "",
        "| domain | " + " | ".join(f"`{run['label']}`" for run in runs) + " |",
        "| --- | " + " | ".join("---" for _ in runs) + " |",
    ]
    for domain in domains:
        cells = []
        for run in runs:
            bucket = run["metrics"]["by_domain"].get(domain)
            cells.append(
                f"{bucket['success_rate'] * 100:.0f}% ({bucket['success']}/{bucket['scored']})" if bucket else "-"
            )
        lines.append(f"| {domain} | " + " | ".join(cells) + " |")
    return lines + [""]


def render_comparison(runs: list[dict]) -> str:
    """Before/after table over two or more runs, with the parity check and per-level and per-domain diffs."""
    levels = [
        level
        for level in (*LEVEL_ORDER, "ungraded")
        if any(level in run["metrics"].get("by_level", {}) for run in runs)
    ]
    lines = _comparison_summary(runs)
    lines += _comparison_levels(runs, levels)
    lines += _comparison_deltas(runs, levels)
    lines += _comparison_parity(runs)
    lines += _comparison_domains(runs)
    return "\n".join(lines)
