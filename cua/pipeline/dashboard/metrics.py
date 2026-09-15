"""Descriptive-stat helpers and the histogram labelling bands the loader and charts share, which never score."""

from collections import Counter
from math import log

import pandas as pd

COORD_ACTION_TYPES = ("click", "type", "drag")

PLACEHOLDER_THOUGHT_SUBSTRINGS = (
    "I need to see the screenshot",
    "Could you provide",
    "As an AI",
    "please provide",
    "I don't have access to",
)


def shannon_entropy_nats(counts: dict | Counter | pd.Series) -> float | None:
    if isinstance(counts, pd.Series):
        counts = counts.value_counts().to_dict()
    elif isinstance(counts, Counter):
        counts = dict(counts)
    total = sum(counts.values()) if counts else 0
    if total == 0:
        return None
    entropy = 0.0
    for n in counts.values():
        if n <= 0:
            continue
        p = n / total
        entropy -= p * log(p)
    return entropy
