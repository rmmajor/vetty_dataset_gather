"""Reporting: the per-spec results file, the comparison file cua.cloud.vast summarises, and the side-by-side table."""

import json

from cua.eval.screenspot.config import PREPARED_IMAGE_SIZE
from cua.eval.screenspot.scoring import resolve_coord_space
from cua.prompt.format import PROMPT_FORMAT_VERSION


def write_results(path, spec, cfg, summary, records):
    payload = {
        "label": spec.label,
        "model": spec.model,
        "adapters": spec.adapters,
        "backend": spec.backend or cfg.backend,
        "dataset": cfg.dataset,
        "split": cfg.split,
        "prompt_format": PROMPT_FORMAT_VERSION,
        "coord_space": resolve_coord_space(cfg, spec),
        "bbox_format": cfg.bbox_format,
        "bbox_normalized": cfg.bbox_normalized,
        "viewport": [cfg.viewport_w, cfg.viewport_h],
        "prepared_image_size": PREPARED_IMAGE_SIZE,
        "summary": summary,
        "records": [
            {
                "instruction": r.instruction,
                "data_type": r.data_type,
                "data_source": r.data_source,
                "parsed_point": r.parsed_point,
                "mapped_point": r.mapped_point,
                "bbox_xyxy": r.bbox_xyxy,
                "hit": r.hit,
                "orig_w": r.orig_w,
                "orig_h": r.orig_h,
                "raw": r.raw,
            }
            for r in records
        ],
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def write_comparison(path, cfg, summaries):
    payload = {
        "dataset": cfg.dataset,
        "split": cfg.split,
        "prompt_format": PROMPT_FORMAT_VERSION,
        "coord_space": cfg.coord_space,
        "summaries": summaries,
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def format_group(group):
    if not group:
        return "-"
    return f"{group['accuracy'] * 100:.1f}% ({group['hits']}/{group['total']})"


def comparison_rows(summaries, labels):
    """Every table row as (name, one cell per label), headline metrics first and the data-type/source breakdowns under them."""
    rows = [("overall accuracy", [f"{summaries[label]['accuracy'] * 100:.1f}%" for label in labels])]
    rows.append(("parse rate", [f"{summaries[label]['parse_rate'] * 100:.1f}%" for label in labels]))
    rows.append(("samples", [str(summaries[label]["total"]) for label in labels]))

    data_types = sorted({t for label in labels for t in summaries[label]["by_data_type"]})
    for data_type in data_types:
        rows.append(
            (f"  type:{data_type}", [format_group(summaries[label]["by_data_type"].get(data_type)) for label in labels])
        )

    data_sources = sorted({s for label in labels for s in summaries[label]["by_data_source"]})
    for data_source in data_sources:
        rows.append(
            (
                f"  source:{data_source}",
                [format_group(summaries[label]["by_data_source"].get(data_source)) for label in labels],
            )
        )
    return rows


def print_comparison(summaries):
    labels = list(summaries.keys())
    rows = comparison_rows(summaries, labels)

    label_width = max([len("metric")] + [len(name) for name, _ in rows])
    col_widths = [max(len(label), *(len(row[1][i]) for row in rows)) for i, label in enumerate(labels)]

    header = (
        "metric".ljust(label_width) + "  " + "  ".join(label.ljust(col_widths[i]) for i, label in enumerate(labels))
    )
    print(header)
    print("-" * len(header))
    for name, cells in rows:
        print(name.ljust(label_width) + "  " + "  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(cells)))
