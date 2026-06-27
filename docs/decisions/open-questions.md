# Open questions / deferred decisions

Running list of decisions surfaced during development, parked for later action.

## Meandering threshold vs. completed long trajectories (raised 2026-06-13)

**Tension:** The verifier blocks `meandering` trajectories from training (built to kill
Booking.com-style stuck loops). But it also drops *goal-completing* long trajectories that
exceed ~2× expected actions. Concrete case: gad-gui `book-shop-order-expensive-book` — 43 steps,
**goal fully achieved** (Order #6 confirmed), but 2.8× the expected 15 actions → graded
meandering → rejected. Long completing trajectories are exactly the "long, hard" data the
project wants, and they're rare.

Note: the *other* two long gad-gui rejects (41-step comment-and-like, 25-step publish-article)
were correct rejects — they ended on an empty list / error page, i.e. genuine failures, not
inefficiency. So this only concerns runs where `verifier.success == true` but efficiency ==
meandering.

**Options:**
- (a) Keep as-is — efficiency matters for clean supervision.
- (b) Train completed trajectories regardless of efficiency, tag them, keep as a minority.
- (c) Raise the meandering threshold for verified-SUCCESS runs only (e.g. allow up to ~3.5×
  expected actions when the goal was met). Stuck loops never complete, so they stay blocked by
  the success check regardless. **Current lean: (c).**

Where to change it: `repo_pipeline/agent_capture.py` — `VERIFY_PROMPT` efficiency rubric +
`verify_trajectory` logic that maps `efficiency == "meandering"` to `success = False`.
