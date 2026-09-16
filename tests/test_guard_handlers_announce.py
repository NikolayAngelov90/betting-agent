"""A handler that guards a guarantee, a measurement or a gate must ANNOUNCE.

    NOTHING LOGGED AT DEBUG EXISTS IN PRODUCTION.

`setup_logger()` calls `logger.remove()` and installs sinks at
`log_cfg.get("level", "INFO")`, so every `logger.debug()` after it is discarded.
That is LOG-1, and it made five ledger claims unfounded — each of the form
"X did not happen, because the line was absent", where the line could not be
emitted.

THE SORTING RULE, applied 2026-09-16 (Stage 25):

    degrading a GUARANTEE, a MEASUREMENT or a GATE  -> WARNING
    degrading a CONVENIENCE                          -> DEBUG is correct

MEASURED BEFORE SHIPPING, because a sorting that produces a double-digit daily
WARNING count fails the way `fixtures_zero_active` did — it fired 21 times a day
and became noise four days after it was written. **Every guarded operation below
currently SUCCEEDS in production**, so the measured firing rate is **0 per run**:

    team_former_names   125 rows        api_budget           45 rows
    pick_observations  1700 rows        injury_observations 3366 rows
    odds_snapshots    97986 rows        calibration/ev_threshold/weights present

These are EXCEPTION handlers, not assertions. `fixtures_zero_active` fired on a
condition that was routinely false; these fire only when an operation that
currently works stops working. **A healthy run gains zero lines, and any line is
a real degradation.**

THE PATTERN WAS ALREADY IN THE TREE, TWICE. `coverage_checks.find_unpriced_
fixtures` and `fixture_plausibility.find_implausible_attributions` both log at
DEBUG and return None, and their CALLERS emit `CHECK DID NOT RUN` at WARNING.
Those two are correct as they stand and are exempt below. The other twenty-one
had no such caller, so the announcement is made where the failure is.
"""

import ast
import pathlib

import pytest

#: `file: [(function, what degrades if this handler fires)]`.
#: Every one guards a guarantee, a measurement or a gate.
MUST_ANNOUNCE = {
    "src/data/team_resolution.py": [
        ("lookup_former_name", "step 2 — every resolution reverts to pre-Stage-23"),
        ("record_former_name", "a merge removes a name without recording it"),
    ],
    "src/data/history_mirror.py": [
        ("invalidate", "a failed invalidation leaves the stale mirror serving"),
    ],
    "src/data/api_budget.py": [
        ("available", "the credit gate falls back to per-process counting"),
        ("release", "claimed budget is leaked"),
    ],
    "src/data/database.py": [
        ("_register_numpy_psycopg2_adapters",
         "numpy scalars reach SQL; surfaces later as an unreadable schema error"),
        ("_migrate_missing_columns", "a column is silently absent"),
        ("_migrate_missing_indexes", "an index is silently absent"),
    ],
    "src/data/price_history.py": [
        ("record_price", "a CLV price observation is lost"),
        ("stamp_first_seen", "first_seen_at is never set"),
        ("record_injury", "an injury observation is lost"),
    ],
    "src/scrapers/theodds_scraper.py": [
        ("_save_game_odds", "odds rows are silently not written"),
        ("_upsert_odds", "an odds upsert is silently dropped"),
    ],
}

#: Correct at DEBUG because the CALLER announces. The pattern, not an exception.
CALLER_ANNOUNCES = {
    ("src/data/coverage_checks.py", "find_unpriced_fixtures"):
        "report_unpriced_fixtures logs UNPRICED FIXTURE CHECK DID NOT RUN",
    ("src/data/fixture_plausibility.py", "find_implausible_attributions"):
        "report_implausible_attributions logs IMPLAUSIBLE ATTRIBUTION CHECK DID NOT RUN",
}

_VISIBLE = {"warning", "error", "critical", "exception"}


def _handler_levels(path: str, func: str):
    """Log levels used by every `except` inside `func`, or None if absent."""
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != func:
            continue
        out = []
        for h in [n for n in ast.walk(node) if isinstance(n, ast.ExceptHandler)]:
            for sub in ast.walk(h):
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                        and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id == "logger"):
                    out.append(sub.func.attr)
        return out
    return None


@pytest.mark.parametrize(
    "path,func,why",
    [(p, f, w) for p, sites in MUST_ANNOUNCE.items() for f, w in sites],
    ids=[f"{p.split('/')[-1]}:{f}" for p, s in MUST_ANNOUNCE.items() for f, _ in s])
def test_a_guard_handler_announces_at_warning(path, func, why):
    levels = _handler_levels(path, func)
    assert levels is not None, f"{path}:{func} no longer exists — re-point this pin"
    assert levels, (
        f"{path}:{func} swallows without logging at all. If it fires, "
        f"{why} — and nothing anywhere records it.")
    assert set(levels) & _VISIBLE, (
        f"{path}:{func} logs only at {sorted(set(levels))}. Nothing logged at "
        f"DEBUG exists in production (setup_logger installs INFO sinks), so if "
        f"it fires, {why} — silently.\n\n"
        f"Raise it to WARNING, or move the announcement to a caller that names "
        f"what degraded, as coverage_checks and fixture_plausibility do.")


@pytest.mark.parametrize("path,func", sorted(CALLER_ANNOUNCES))
def test_the_caller_announces_pattern_still_holds(path, func):
    """These two are correct at DEBUG ONLY BECAUSE their callers warn.

    If a caller stops warning, the inner DEBUG becomes silence and the check
    joins the class this file exists to empty.
    """
    levels = _handler_levels(path, func)
    assert levels and set(levels) <= {"debug"}, (
        f"{path}:{func} changed level — this pin records that it is DEBUG *by "
        f"design*, because its caller announces. Update both or neither.")
    src = pathlib.Path(path).read_text(encoding="utf-8")
    assert "CHECK DID NOT RUN" in src, (
        f"{path} no longer emits 'CHECK DID NOT RUN' — the caller-announces "
        f"half of the pattern is gone, so the DEBUG half is now silence")


def test_the_debug_only_population_does_not_grow_silently():
    """A census, pinned. Stage 25 took it from 66 to 45; 158 log nothing.

    Not a target — a tripwire. A new DEBUG-only handler is fine when it guards a
    convenience, and this fails so that the judgement is made rather than
    inherited.
    """
    debug_only = 0
    for p in sorted(list(pathlib.Path("src").rglob("*.py"))
                    + list(pathlib.Path("scripts").rglob("*.py"))):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for h in [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]:
            lv = [s.func.attr for s in ast.walk(h)
                  if isinstance(s, ast.Call) and isinstance(s.func, ast.Attribute)
                  and isinstance(s.func.value, ast.Name) and s.func.value.id == "logger"]
            if lv and set(lv) <= {"debug"}:
                debug_only += 1
    assert debug_only <= 45, (
        f"{debug_only} DEBUG-only exception handlers, up from the 45 Stage 25 "
        f"left. Nothing logged at DEBUG exists in production: classify the new "
        f"one as guarantee/measurement/gate (-> WARNING, and add it to "
        f"MUST_ANNOUNCE) or as a convenience (-> raise this number with the "
        f"reason).")
