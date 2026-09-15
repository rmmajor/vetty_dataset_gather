"""Scoring: read a point out of the model's text, map it into source-image pixels through the spec's coordinate space, hit-test it."""

import re
from dataclasses import dataclass


@dataclass
class Record:
    instruction: str
    data_type: str
    data_source: str
    raw: str
    parsed_point: tuple | None  # coordinate as the model emitted it
    mapped_point: tuple | None  # mapped into original-image pixels
    bbox_xyxy: tuple
    hit: bool
    # Without the source dimensions a saved run cannot be re-scored, the only way to tell grounding failure from mismatch.
    orig_w: int = 0
    orig_h: int = 0


class CoordSpace:
    def to_original(self, x, y, orig_w, orig_h):
        raise NotImplementedError

    def to_model(self, x, y, orig_w, orig_h):
        raise NotImplementedError


class ViewportAbsCoordSpace(CoordSpace):
    def __init__(self, viewport_w, viewport_h):
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h

    def to_original(self, x, y, orig_w, orig_h):
        return (x * orig_w / self.viewport_w, y * orig_h / self.viewport_h)

    def to_model(self, x, y, orig_w, orig_h):
        return (x * self.viewport_w / orig_w, y * self.viewport_h / orig_h)


class Normalized1000CoordSpace(CoordSpace):
    def to_original(self, x, y, orig_w, orig_h):
        return (x / 1000.0 * orig_w, y / 1000.0 * orig_h)

    def to_model(self, x, y, orig_w, orig_h):
        return (x / orig_w * 1000.0, y / orig_h * 1000.0)


class NativeAbsCoordSpace(CoordSpace):
    def to_original(self, x, y, orig_w, orig_h):
        return (x, y)

    def to_model(self, x, y, orig_w, orig_h):
        return (x, y)


# A space maps points both directions and NEVER touches the image; a new convention is another entry, not a caller branch.
COORD_SPACES: dict = {
    "viewport_abs": lambda cfg: ViewportAbsCoordSpace(cfg.viewport_w, cfg.viewport_h),
    "normalized_1000": lambda cfg: Normalized1000CoordSpace(),
    "native_abs": lambda cfg: NativeAbsCoordSpace(),
}


def resolve_coord_space(cfg, spec=None) -> str:
    """The coord space for one spec: its own if set, else the global default."""
    return spec.coord_space if spec is not None and spec.coord_space else cfg.coord_space


def build_coord_space(cfg, spec=None):
    name = resolve_coord_space(cfg, spec)
    if name not in COORD_SPACES:
        raise ValueError(f"Unknown coord space {name!r}. Available: {sorted(COORD_SPACES)}")
    return COORD_SPACES[name](cfg)


def parse_point(text, patterns):
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        try:
            return (float(match.group(1)), float(match.group(2)))
        except (ValueError, IndexError):
            continue
    return None


def point_in_bbox(point, bbox_xyxy):
    if point is None:
        return False
    x, y = point
    x0, y0, x1, y1 = bbox_xyxy
    return x0 <= x <= x1 and y0 <= y <= y1


def evaluate(samples, predictor, cfg, coord_space):
    raws = predictor.predict_batch(samples, cfg, coord_space)
    records = []
    for sample, raw in zip(samples, raws, strict=False):
        parsed = parse_point(raw, cfg.parse_patterns)
        if parsed is None:
            mapped = None
            hit = False
        else:
            mapped = coord_space.to_original(parsed[0], parsed[1], sample.orig_w, sample.orig_h)
            hit = point_in_bbox(mapped, sample.bbox_xyxy)
        records.append(
            Record(
                instruction=sample.instruction,
                data_type=sample.data_type,
                data_source=sample.data_source,
                raw=raw,
                parsed_point=parsed,
                mapped_point=mapped,
                bbox_xyxy=sample.bbox_xyxy,
                hit=hit,
                orig_w=sample.orig_w,
                orig_h=sample.orig_h,
            )
        )
    return records


def group_accuracy(records, key):
    groups = {}
    for record in records:
        bucket = groups.setdefault(key(record), {"total": 0, "hits": 0})
        bucket["total"] += 1
        bucket["hits"] += 1 if record.hit else 0
    return {
        name: {
            "total": bucket["total"],
            "hits": bucket["hits"],
            "accuracy": bucket["hits"] / bucket["total"],
        }
        for name, bucket in sorted(groups.items())
    }


def aggregate(records):
    total = len(records)
    hits = sum(1 for record in records if record.hit)
    parsed = sum(1 for record in records if record.parsed_point is not None)
    return {
        "total": total,
        "hits": hits,
        "accuracy": hits / total if total else 0.0,
        "parse_rate": parsed / total if total else 0.0,
        "by_data_type": group_accuracy(records, lambda r: r.data_type),
        "by_data_source": group_accuracy(records, lambda r: r.data_source),
    }
