"""Truthiness on quantities that can legitimately be ZERO — 2026-10-19.

    Zero is a legitimate value. A guard using truthiness on a quantity that
    can legitimately be zero fails on the one day it is zero, and midnight is
    that day for every cron.

THE INSTANCE. `ci_audit.missing_scheduled_runs()` guarded on
`if not any(crons.values())`. A cron is stored as MINUTES PAST MIDNIGHT, so
`0 0` is the integer `0` and `any([0, 0])` is `False`. It survived review
because the weekday cron was `0 4` (240) at the time — the weekend zero was
carried by the weekday value. The revert to `0 0` at `566c927` made both zero
and the function returned `UNEVALUATED` on **every call for a day**, silently,
because UNEVALUATED is a legitimate state and its consumer skips on it. The
detector it killed was the one built for the 422 window.

THE CENSUS, `measured` 2026-10-19 over `scripts/` and `src/`:

    hits                20 sites at 18 distinct keys
    NOT                  4   `if not n`
    TRUTHY              14   `if count`
    OR_DEFAULT           2   `x or <default>`
    ANY_ALL              0   (the motivating shape, now repaired)
    registered REACHABLE 4   still in the tree: 0
    registered EXEMPT   18   still in the tree: 18

FOUR REACHABLE, ALL FIXED:

  1. `ci_audit.py` `any(crons.values())` — the instance above.
  2. `run_baseline.py` `ws = w.sum() or 1.0` — zero exactly when `train` is
     empty, with NO minimum-train guard at the call site. Returned
     `gh = ga = 0.0` and an empty strength table without raising, because the
     loops that divide by them never ran. This harness gates every model
     change. REACHABLE BY CONSTRUCTION, NOT OBSERVED.
  3. `run_clean_baseline.py` — the SAME LINE in the second harness. The fix
     was needed twice, which is what THE HABIT costs.
  4. `schedule_deadline.py` `ref.get('n_usable') or 0` — mine, written the
     same day. Behaviour was already correct because the fallback equals the
     falsy value; rewritten because the idiom reads as if zero were absent.

SIXTEEN EXEMPT, every one read. What makes truthiness CORRECT there is that
`0` and `absent` mean the same thing to the consumer: nothing to log, nothing
to divide by, nothing to update. What made it WRONG above is that `0` was a
MEASURED VALUE and `absent` a different fact.
"""

import pathlib

import scripts.falsy_zero_scan as fz

#: The function AS WRITTEN before the repair. The known-wrong case, kept
#: verbatim so the detector is measured against the instance that motivated it
#: rather than against a paraphrase of it.
THE_LINE_THAT_KILLED_THE_DETECTOR = '''import datetime as _dt


def missing_scheduled_runs(workflow="daily-picks.yml", days=14, today=None):
    today = today or _dt.date.today()
    crons = current_crons_by_daytype() or {}
    if not any(crons.values()):
        return {"state": "UNEVALUATED",
                "reason": "no cron resolved for either day type"}
    return {"state": "CHECKED"}
'''


def _scan_text(body: str, tmp_path, name="probe.py"):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return fz.scan_file(p)


# ── THE KNOWN-WRONG CONTROL ──────────────────────────────────────────────────

def test_KNOWN_WRONG_the_line_that_killed_the_detector_IS_CAUGHT(tmp_path):
    """A detector that cannot catch the instance that motivated it has no
    measured coverage. This is that instance, verbatim."""
    hits = _scan_text(THE_LINE_THAT_KILLED_THE_DETECTOR, tmp_path)
    anyall = [h for h in hits if h.shape == fz.SHAPE_ANY_ALL]
    assert anyall, (
        f"`if not any(crons.values())` is NOT flagged. The sweep has no "
        f"measured coverage. Hits were: "
        f"{[(h.shape, h.expr) for h in hits]}")
    assert anyall[0].expr == "crons"
    assert "any(crons.values())" in anyall[0].src


def test_the_ANY_ALL_shape_looks_THROUGH_values_to_the_mapping(tmp_path):
    """The discrimination that makes the control work.

    `crons` is in `COLLECTION_EXCEPTIONS` because `if not crons:` on the DICT
    is correct — an empty mapping really is "no mapping". But
    `any(crons.values())` tests the VALUES, which are minutes, and zero is a
    legitimate minute. Same identifier, opposite verdict, decided by the
    shape it appears in.
    """
    assert "crons" in fz.COLLECTION_EXCEPTIONS
    bare = _scan_text("def f(crons):\n    if not crons:\n        return 1\n",
                      tmp_path, "bare.py")
    assert not bare, f"`if not crons:` on the dict should NOT flag: {bare}"
    vals = _scan_text(
        "def f(crons):\n    if not any(crons.values()):\n        return 1\n",
        tmp_path, "vals.py")
    assert [h.shape for h in vals] == [fz.SHAPE_ANY_ALL]


def test_POSITIVE_CONTROL_each_shape_is_detectable(tmp_path):
    """All five shapes, driven. Without this the scan could be finding one
    shape and reporting five."""
    cases = {
        fz.SHAPE_NOT: "def f(cron_min):\n    if not cron_min:\n        return 1\n",
        fz.SHAPE_TRUTHY: "def f(n_count):\n    if n_count:\n        return 1\n",
        fz.SHAPE_OR_DEFAULT: "def f(offset):\n    x = offset or 99\n    return x\n",
        fz.SHAPE_ANY_ALL: "def f(margins):\n    return any(margins.values())\n",
        fz.SHAPE_ASSERT: "def f(total_minutes):\n    assert total_minutes\n",
    }
    for shape, body in cases.items():
        hits = _scan_text(body, tmp_path, f"s_{shape.lower()}.py")
        assert shape in {h.shape for h in hits}, (
            f"shape {shape} is not detected; body was:\n{body}")


def test_NEGATIVE_CONTROL_a_COLLECTION_test_is_not_flagged(tmp_path):
    """`if not rows:` is correct and must stay quiet, or the census becomes
    noise and nobody reads it."""
    for body in ("def f(rows):\n    if not rows:\n        return 1\n",
                 "def f(ctx=None):\n    ctx = ctx or {}\n    return ctx\n",
                 "def f(items):\n    if not len(items):\n        return 1\n"):
        hits = _scan_text(body, tmp_path, "neg.py")
        assert not hits, f"false positive on:\n{body}\n-> {hits}"


# ── THE REPAIRS, EACH PINNED ────────────────────────────────────────────────

def test_the_FOUR_REACHABLE_sites_are_REGISTERED_and_GONE():
    """The census reports 0 reachable in the tree, which is also what an empty
    scan reports. The register is what distinguishes them."""
    hits = fz.triage(fz.scan())
    reachable = [k for k, (v, _w) in fz.TRIAGE.items() if v == fz.REACHABLE]
    assert len(reachable) == 4, (
        f"the reachable register moved to {len(reachable)}; if a fifth was "
        f"found, say so in the ledger rather than only here")
    assert fz.regressions(hits) == [], (
        f"a site registered REACHABLE is back in the tree: "
        f"{fz.regressions(hits)}")
    for k in reachable:
        _v, why = fz.TRIAGE[k]
        assert len(why) > 80, f"{k} has no real reason"


def test_BOTH_baseline_harnesses_REFUSE_an_empty_training_set():
    """The duplicated defect, fixed in both. A refusal in one and a silent
    0.0 in the other is worse than either, because the two harnesses are
    compared against each other."""
    import datetime as dt

    import pytest

    import scripts.run_baseline as rb
    import scripts.run_clean_baseline as rcb

    for mod in (rb, rcb):
        with pytest.raises(ValueError, match="nothing to fit"):
            mod.fit_poisson([], dt.date(2026, 1, 1))


def test_the_refusal_is_a_NO_OP_for_a_NON_EMPTY_training_set():
    """No baseline number moves.

    The change replaces `or 1.0` with a refusal, and for any non-empty train
    the weight sum is strictly positive — `exp()` is never zero and
    `np.ones(n).sum()` is `n`. So every result that was computable before is
    computed identically, and nothing in the evidence record shifts.
    """
    import datetime as dt
    import math
    import types

    import numpy as np

    import scripts.run_baseline as rb

    train = [types.SimpleNamespace(
        match_date=dt.date(2026, 1, 1) - dt.timedelta(days=d),
        home_goals=1, away_goals=2, league="x", home_team_id=1, away_team_id=2)
        for d in (1, 10, 400)]
    for half_life in (180, 540, None):
        dr = math.log(2) / half_life if half_life else 0.0
        w = (np.array([math.exp(-dr * max(0, (dt.date(2026, 1, 1) - m.match_date).days))
                       for m in train]) if half_life else np.ones(len(train)))
        assert float(w.sum()) > 0, (
            "a non-empty train produced a zero weight sum, so the refusal is "
            "NOT a no-op and baselines must be re-run")
    state = rb.fit_poisson(train, dt.date(2026, 1, 1))
    assert state is not None


def test_the_MOTIVATING_instance_is_still_fixed_in_ci_audit():
    """Belt and braces with `tests/test_absent_run_and_unparseable.py`: that
    file drives the behaviour, this one pins the shape.

    BY THE DETECTOR, NOT BY TEXT. The first version of this test asserted
    `"if not any(crons.values())" not in src` and failed on the COMMENT that
    documents the repair -- the fourth comment-blind text scan of the day,
    which is what prompted `conftest.code_of`. Here the scanner itself is the
    check, which is better than either: it is AST-based by construction, and
    using it dogfoods the instrument on the file it was built for.
    """
    hits = fz.scan_file(pathlib.Path("scripts/ci_audit.py"))
    bad = [h for h in hits
           if h.shape == fz.SHAPE_ANY_ALL and h.expr == "crons"]
    assert not bad, f"the motivating instance is back: {bad}"

    from tests.conftest import code_of

    code = code_of(pathlib.Path("scripts/ci_audit.py")
                   .read_text(encoding="utf-8"))
    assert "if not any(crons.values())" not in code
    assert "all(v is None for v in crons.values())" in code


# ── THE CENSUS IS REPORTED, NOT ASSERTED AWAY ───────────────────────────────

def test_the_census_is_FULLY_TRIAGED_and_the_counts_check_out():
    hits = fz.triage(fz.scan())
    assert fz.untriaged(hits) == [], (
        f"untriaged falsy-zero sites: "
        f"{[(h.path, h.line, h.shape, h.expr) for h in fz.untriaged(hits)]}. "
        f"Each must be REACHABLE (and fixed), UNREACHABLE or EXEMPT with a "
        f"reason.")
    text = fz.format_census(hits)
    assert "distinct key(s)" in text, (
        "the census no longer distinguishes sites from register keys; 20 "
        "sites at 18 keys otherwise reads as an arithmetic error")
    # RULE 3: the count is checked against its own arithmetic.
    n_sites = len(hits)
    n_keys = len({h.key for h in hits})
    assert n_keys <= n_sites, "more keys than sites is impossible"
    assert sum(1 for h in hits if h.verdict) == n_sites


def test_the_scan_itself_is_PURELY_STATIC():
    """Same rule as the calendar scan: a detector whose verdict can change
    with no code change must not be trusted to gate anything."""
    import ast

    tree = ast.parse(pathlib.Path("scripts/falsy_zero_scan.py")
                     .read_text(encoding="utf-8"))
    banned = {"today", "now", "utcnow", "urlopen", "check_output", "run"}
    called = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        if name in banned:
            called.add(name)
    assert not called, (
        f"the scan calls {sorted(called)}; its verdict could then change with "
        f"no code change")
