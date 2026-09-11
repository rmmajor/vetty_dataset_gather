"""A rental whose job failed is blocklisted, so a deterministic cheapest-first retry stops re-renting the same broken host."""

import json
import tempfile
from pathlib import Path

from remote_train import state


def test_a_failed_job_blocklists_its_offer_so_selection_moves_on():
    with tempfile.TemporaryDirectory() as tmp:
        original = state.BAD_OFFER_FILE
        state.BAD_OFFER_FILE = Path(tmp) / "bad_offers.json"
        try:
            state.record_bad_offer("45242143", "serve reported FAILED: unsupported display driver")
            recorded = json.loads(state.BAD_OFFER_FILE.read_text())
            assert "45242143" in recorded
            assert "FAILED" in recorded["45242143"]["reason"]
            assert "45242143" in state.bad_offer_ids()
            assert "99999999" not in state.bad_offer_ids()
        finally:
            state.BAD_OFFER_FILE = original
