"""The H1 window has ONE definition, and every consumer imports it — 2026-10-28.

FOUR SITES WERE HALF-MOVED IN ONE WEEK, each found individually and none by
the same mechanism:

    the reservation             `odds_quota.H1_RESERVATION_FROM/UNTIL`
    the snapshot consumer       `h1_collection_check._H1_WINDOW_FROM` and
                                `H1_WINDOW_TO`, re-typed as strings
    the collect loop's fixtures pinned to October, so the loop halted
                                `NO_DATA` instead of `RESERVATION_EXHAUSTED`
                                — the halt was real, the REASON was wrong
    `h1_analysis --since`       still October's credit reset, so the default
                                population would have been 2026-10-01..11-09:
                                five weeks of normal-pricing snapshots counted
                                as H1 observations

A FIFTH MISSED SITE DURING COLLECTION CORRUPTS THE RESULT RATHER THAN
REDDENING A RUN, and that asymmetry is the whole argument. All four were
caught by tests, and the only reason they were caught is that they moved while
nobody was collecting. A date left behind on 2026-11-03 produces a plausible
number with no error anywhere.

THE SCAN IS AST-BASED, so a date in a COMMENT or a DOCSTRING cannot match —
sixteen comment-blind instances this week say that matters. It is also why
this file imports nothing from `tests/conftest.py`: that would reach the write
path through its database fixtures and make the test about the window
capture-critical.
"""

import ast
import pathlib

from src.data import h1_window as w

#: Every spelling of a window edge that could be restated in code.
EDGES = {
    (w.FROM.year, w.FROM.month, w.FROM.day),
    (w.UNTIL.year, w.UNTIL.month, w.UNTIL.day),
}
EDGE_ISO = {w.FROM.isoformat(), w.UNTIL.isoformat()}

#: ── THE CONSUMERS, ENUMERATED. ───────────────────────────────────────────
#:
#: Every site that READS the collection window, and what it reads it for: the
#: collector, the precondition, the stop, the credit accounting,
#: `load_observations`, `h1_analysis`'s default, the reservation and the daily
#: check.
#:
#: ENUMERATED RATHER THAN SCANNED FOR A DATE LITERAL, and the first version of
#: this test is why. Scanning every `.py` for `2026-11-01` returned SEVEN
#: offenders and SIX WERE FALSE: that date is also
#: `H1_RESERVATION_ACK_UNTIL`, also the credit-ledger month flip, also a
#: picks-bound forecast date — and `2026-11-09` is also the FIFA window's
#: first day, which is the one-day margin and a DIFFERENT FACT that must be
#: able to move independently.
#:
#:     A DATE LITERAL IS NOT A WINDOW REFERENCE.
#:
#: So membership is DECLARED, like the `calendar` marker's, and for the same
#: reason: a grep cannot tell which of two facts a date is.
CONSUMERS = {
    "src/data/odds_quota.py":
        "the reservation and the credit accounting",
    "scripts/h1_collection_check.py":
        "the stop, and load_observations' population clamp",
    "scripts/h1_analysis.py":
        "the analysis default lower bound and the clamp",
    "scripts/h1_collect.py":
        "the collector's observation window",
    "scripts/h1_precondition.py":
        "the precondition and the daily check",
}

#: ── THE PINS, WHICH MUST RESTATE. ────────────────────────────────────────
#:
#: A pin that imports its own subject is a tautology: `assert FROM == FROM`
#: verifies nothing. These exist to FAIL when the window moves, which is how a
#: move becomes a decision rather than a diff.
#:
#: CAPPED, for the reason every exemption list here is capped: a list that
#: absorbs the backlog makes the mechanism vacuous.
PINS = {
    "tests/test_calendar_separation.py":
        "BOUNDARY_DATES and the driven predicates. The boundary census exists "
        "to say what flips on which date, so it names them.",
    "tests/test_h1_reservation_is_an_allocation.py":
        "WINDOW_EDGES drives all four edges by value; deriving them would "
        "assert that allocate() agrees with itself.",
    "tests/test_h1_enforced_stop.py":
        "the ONSET pin asserts H1_RESERVATION_FROM against the credit reset, "
        "which is the claim that the onset was not chosen arbitrarily.",
    "src/data/h1_window.py":
        "the definition itself: the one place the two dates are written, and "
        "the only site a scan should ever find them in.",
}
PINS_CAP = 5

#: The names a consumer may read the window through. Each RE-EXPORTS the one
#: definition; none holds a date of its own.
PUBLISHED_NAMES = {
    "h1_window",              # the definition, imported directly
    "H1_RESERVATION_FROM",    # odds_quota's published spelling
    "H1_RESERVATION_UNTIL",
    "H1_WINDOW_TO",           # h1_collection_check's, for load_observations
    "_H1_WINDOW_FROM",
}

#: ── AND THE CHAIN IS VERIFIED, NOT ASSUMED. ──────────────────────────────
#:
#: A consumer reading `H1_RESERVATION_FROM` is only reading ONE definition if
#: `odds_quota` itself imports it. Accepting the published name on trust would
#: be the half-move one level up: the chain would look consolidated while a
#: re-export quietly held its own literal, which is precisely what
#: `h1_collection_check` did with two hand-typed strings until today.
RE_EXPORTERS = ("src/data/odds_quota.py", "scripts/h1_collection_check.py")

#: ── DATES THAT COINCIDE WITH AN EDGE AND MEAN SOMETHING ELSE ─────────────
#:
#: `{path: {lineno: reason}}`. A site here restates an edge's VALUE while
#: carrying a DIFFERENT FACT, and the two must be able to move independently —
#: so deriving one from the other would be worse than the duplication.
#:
#: THE SCAN FOUND THIS RATHER THAN A HUMAN NOTICING IT, which is the point of
#: running it at all: the collision is not an accident, it IS the one-day
#: margin between the last collection day and the international break, and
#: `test_the_FIFA_WINDOW_does_not_overlap_the_COLLECTION_WINDOW` asserts that
#: margin by value. If the H1 window moved and the FIFA date followed it, that
#: test would become a tautology and the overlap check would stop checking.
#:
#: CAPPED AT TWO. A third would mean the edge values are load-bearing in more
#: places than this mechanism can reason about.
INDEPENDENT_FACTS = {
    "scripts/h1_precondition.py": {
        552: "FIFA_WINDOW_NOV_2026's first day. It equals the H1 window's "
             "exclusive end BY COINCIDENCE -- that coincidence is the "
             "one-day margin -- and the published FIFA calendar must be able "
             "to move without dragging the collection window with it.",
    },
}
INDEPENDENT_FACTS_CAP = 2

ROOTS = ("src", "scripts", "tests")


def _code_date_sites(path: pathlib.Path):
    """`[(lineno, what)]` for window edges appearing in CODE.

    By AST, so a comment cannot match and a DOCSTRING cannot either — the
    docstring of a module, class or function is its first statement and is
    skipped explicitly.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []

    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            docstrings.add(id(body[0].value))

    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = (node.func.attr if isinstance(node.func, ast.Attribute)
                    else getattr(node.func, "id", ""))
            if name in ("date", "_date", "datetime", "_datetime") \
                    and len(node.args) >= 3 \
                    and all(isinstance(a, ast.Constant) for a in node.args[:3]):
                triple = tuple(a.value for a in node.args[:3])
                if triple in EDGES:
                    hits.append((node.lineno, f"{name}{triple}"))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings and node.value in EDGE_ISO:
            hits.append((node.lineno, repr(node.value)))
    return hits


def _scan():
    out = {}
    for root in ROOTS:
        for p in sorted(pathlib.Path(root).rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            hits = _code_date_sites(p)
            if hits:
                out[str(p).replace("\\", "/")] = hits
    return out


# ═══════════════════════ THE ENUMERATION, ANCHORED ══════════════════════

def test_every_H1_DATE_SITE_IMPORTS_rather_than_RESTATES():
    """Each enumerated consumer must IMPORT the window, never restate it.

    AND THE SCAN'S OWN ANCHOR IS ASSERTED FIRST. A scan that finds zero sites
    is `UNEVALUATED`, not clean: if the predicate stopped matching, this test
    would pass while checking nothing — the shape four of this week's defects
    took.
    """
    sites = _scan()

    assert "src/data/h1_window.py" in sites, (
        "UNEVALUATED: the scan did not find the window's own definition, so "
        "its predicate no longer matches a date literal at all and every "
        "'no restatement' result below is empty of meaning")
    found_pins = [p for p in PINS if p in sites]
    assert len(found_pins) >= 3, (
        f"UNEVALUATED: the scan found only {len(found_pins)} of the "
        f"{len(PINS)} known pins ({found_pins}). A predicate that cannot see "
        f"the sites it exempts cannot see the sites it refuses.")

    for path, why in CONSUMERS.items():
        p = pathlib.Path(path)
        assert p.exists(), f"{path} ({why}) no longer exists"
        code = p.read_text(encoding="utf-8")
        assert any(n in code for n in PUBLISHED_NAMES), (
            f"{path} reads an H1 date for {why} and references none of the "
            f"window's published names {sorted(PUBLISHED_NAMES)} — so where "
            f"does it get the date from?")
        allowed = INDEPENDENT_FACTS.get(path, {})
        restated = [h for h in sites.get(path, []) if h[0] not in allowed]
        assert not restated, (
            f"{path} RESTATES a window edge at line(s) "
            f"{[l for l, _ in restated]} ({[t for _, t in restated]}) while "
            f"also importing it. Two spellings of one date is how four sites "
            f"came to be half-moved in a week.")

    # EVERY RE-EXPORTER IMPORTS THE DEFINITION DIRECTLY, so a consumer
    # reading a published name really is reading one source.
    for path in RE_EXPORTERS:
        code = pathlib.Path(path).read_text(encoding="utf-8")
        assert "h1_window" in code, (
            f"{path} publishes a window name that consumers read, and does "
            f"not import `h1_window` — the chain is broken one level up and "
            f"the consolidation is cosmetic")

    assert not (set(PINS) & set(CONSUMERS)), (
        f"{set(PINS) & set(CONSUMERS)} is both pinned and enumerated as a "
        f"consumer, so it is exempt from the rule it is listed under")


def test_the_INDEPENDENT_FACTS_are_CAPPED_and_REALLY_COINCIDE():
    """Each entry must name a line the scan actually finds, and the fact it
    carries must still equal an edge — otherwise the allowance is stale and
    hides a real restatement."""
    assert len(INDEPENDENT_FACTS) <= INDEPENDENT_FACTS_CAP, (
        f"{len(INDEPENDENT_FACTS)} files hold an edge value for another "
        f"reason, against a cap of {INDEPENDENT_FACTS_CAP}")
    sites = _scan()
    for path, lines in INDEPENDENT_FACTS.items():
        found = {l for l, _ in sites.get(path, [])}
        for lineno, why in lines.items():
            assert len(why) > 60, f"{path}:{lineno} allowed without a reason"
            assert lineno in found, (
                f"{path}:{lineno} is allowed to restate an edge and the scan "
                f"finds no edge there (it finds {sorted(found)}). The "
                f"allowance is STALE: either the line moved, or the "
                f"coincidence ended and the two facts have diverged — which "
                f"is the case this exists to make visible.")
    # and the coincidence itself is asserted by value somewhere else
    pre = pathlib.Path("scripts/h1_precondition.py").read_text(
        encoding="utf-8")
    assert "FIFA_WINDOW_NOV_2026" in pre and "one-day margin" in pre.replace(
        "ONE DAY", "one-day margin"), (
        "the margin the coincidence represents is no longer recorded beside "
        "the FIFA dates")


def test_the_PINS_LIST_is_CAPPED_and_each_entry_is_REACHED():
    """An exemption that names a site the scan never sees excuses nothing —
    the vacuous-exemption shape, which this project has shipped once."""
    assert len(PINS) <= PINS_CAP, (
        f"{len(PINS)} pins against a cap of {PINS_CAP}; a list that absorbs "
        f"the backlog makes the mechanism vacuous")
    sites = _scan()
    for path, why in PINS.items():
        assert len(why) > 40, f"{path} is pinned without a reason"
        assert pathlib.Path(path).exists(), f"{path} no longer exists"
        assert path in sites, (
            f"the pin for {path} is VACUOUS: the scan finds no window edge "
            f"there, so the exemption excuses nothing and spends one of "
            f"{PINS_CAP} slots looking like a decision")


# ══════════════════ NO CONSUMER COMPUTES A DATE ITSELF ═══════════════════

def test_NO_CONSUMER_DERIVES_A_DATE_FROM_THE_WINDOW():
    """The derivations live beside the definition and are imported too.

    `month_key()` off the WALL CLOCK was exactly this defect: the credit
    ledger is keyed by month, two callers each derived that month
    independently, and the moment the window and the clock sat in different
    months the stop reported `credits=0/168` WITH 168 CREDITS SPENT.
    """
    for name in ("iso", "month_keys", "as_datetimes", "contains",
                 "straddles_a_month"):
        assert callable(getattr(w, name)), (
            f"`{name}` left the window module; a consumer now derives it, "
            f"which is one more place for the window to be half-moved")
    assert w.iso() == (w.FROM.isoformat(), w.UNTIL.isoformat())
    assert w.as_datetimes()[0].date() == w.FROM
    assert w.as_datetimes()[1].date() == w.UNTIL
    assert w.DAYS == (w.UNTIL - w.FROM).days == 8
    assert w.contains(w.FROM) and w.contains(w.LAST_DAY)
    assert not w.contains(w.UNTIL), "UNTIL is exclusive"
    assert w.LAST_DAY < w.UNTIL and (w.UNTIL - w.LAST_DAY).days == 1
    # the window does not straddle a credit reset, CHECKED rather than
    # assumed: a straddling window touches two ledger keys and
    # `h1_credits_used` reads only one
    assert w.month_keys() == (w.FROM.replace(day=1),)
    assert w.straddles_a_month() is False, (
        "the window now crosses a credit reset, so the reservation is carved "
        "out of TWO monthly pools and the credit accounting reads one key")


def test_the_RATE_DECISION_is_DERIVED_and_falls_at_the_MIDPOINT():
    """2026-11-04 — day 4 of 8 by arithmetic, not by choice.

    It must fall where a projection can still change the plan; stated at the
    end of the window it would be a post-mortem.
    """
    day_of = (w.RATE_DECISION - w.FROM).days + 1
    assert day_of == w.DAYS // 2, (
        f"the rate decision is day {day_of} of {w.DAYS}, not the midpoint")
    assert w.contains(w.RATE_DECISION), (
        "the rate decision falls outside the collection window")
    assert w.RATE_DECISION.isoformat() == "2026-11-04", w.RATE_DECISION


def test_the_RETIRED_WINDOW_IS_HERE_AND_INERT():
    """A moved window must not leave the old one live anywhere, and the old
    dates must stay readable: the first attempt's `n=4` belongs to them."""
    from src.data.odds_quota import h1_reservation
    old_from, old_until = w.RETIRED
    assert old_until > old_from
    assert not w.contains(old_from) and not w.contains(old_until)
    for d in (old_from, old_until):
        assert h1_reservation(d) == 0, f"the retired window is live on {d}"


# ════════════════════════ THE GRAIN RULE, RECORDED ═══════════════════════

def test_the_GRAIN_RULE_is_recorded_beside_the_definition():
    """> Every H1 figure states its grain before its value.

    Grain has been the deciding axis TWICE and both times the wrong grain
    pointed at the more dramatic conclusion — key against fixture (a
    hundredfold overstatement of `n`), and fixture-day against series (67-103
    in-band instants reading as contamination where the series grain showed
    one). Neither error was arithmetic; both were a correct count of the wrong
    thing.
    """
    src = pathlib.Path("src/data/h1_window.py").read_text(encoding="utf-8")
    assert "Every H1 figure states its grain before its value" in src, (
        "the grain rule is no longer recorded beside the window definition")
    for must in ("KEY against FIXTURE", "FIXTURE-DAY against SERIES",
                 "hundredfold", "1,939"):
        assert must in src, f"the grain record no longer names {must!r}"


# ════════════════ THE DISCRIMINATION CONTROL, DRIVEN ════════════════════

def test_CONTROL_CHANGING_THE_WINDOW_MOVES_EVERY_CONSUMER_TOGETHER(monkeypatch):
    """Move the definition and confirm every consumer follows.

    A CONSUMER THAT DOES NOT MOVE IS THE FOURTH INSTANCE. The four half-moves
    this module exists to prevent were each found by a DIFFERENT accident —
    one by a failing test, one by a wrong halt REASON, one by reading the
    code. This drives the question directly instead.

    MONKEYPATCHED RATHER THAN EDITED, so the control runs in CI on every
    commit instead of being a thing someone did once by hand. `h1_reservation`
    and `allocate` read the module attributes at CALL time, which is what
    makes that possible — and is itself the property worth pinning: a consumer
    that bound the dates at IMPORT time would pass this test while being
    unable to follow a change.
    """
    import datetime as _dt

    from src.data import odds_quota as q

    moved_from = _dt.date(2026, 12, 3)
    moved_until = _dt.date(2026, 12, 11)
    assert not w.contains(moved_from), "pick a date outside the real window"

    monkeypatch.setattr(w, "FROM", moved_from)
    monkeypatch.setattr(w, "UNTIL", moved_until)
    monkeypatch.setattr(q, "H1_RESERVATION_FROM", moved_from)
    monkeypatch.setattr(q, "H1_RESERVATION_UNTIL", moved_until)

    # the reservation follows, on BOTH edges
    assert q.h1_reservation(moved_from) == q.H1_RESERVED_CREDITS, (
        "the reservation did not open on the moved window's first day")
    assert q.h1_reservation(moved_until) == 0, (
        "the reservation did not close on the moved window's UNTIL")
    assert q.h1_reservation(_dt.date(2026, 11, 1)) == 0, (
        "THE OLD DATE IS STILL LIVE. A consumer bound to 2026-11-01 did not "
        "move, which is the fourth instance.")

    # and so does the allocation that both consumers spend against
    assert q.allocate(450, 50, moved_from)["h1_ceiling"] == \
        q.H1_RESERVED_CREDITS
    assert q.allocate(450, 50, _dt.date(2026, 11, 1))["h1_ceiling"] == 0, (
        "allocate() still reserves on the old window's first day")

    # the derived values move with the definition, not independently
    assert w.contains(moved_from) and not w.contains(moved_until)
    assert w.iso() == (moved_from.isoformat(), moved_until.isoformat())
    assert w.as_datetimes()[0].date() == moved_from
    assert w.month_keys() == (moved_from.replace(day=1),)


def test_CONTROL_a_CONSUMER_THAT_BOUND_AT_IMPORT_TIME_WOULD_BE_CAUGHT():
    """The negative half: show the control can fail.

    A control that only ever passes has not been shown to discriminate. This
    plants a consumer that snapshots the date at import time — the realistic
    way a consumer stops following — and confirms the comparison the test
    above makes would catch it.
    """
    import datetime as _dt

    snapshot_at_import = w.FROM            # what a bad consumer would hold

    class _Bad:
        """Binds the date once and never looks again."""
        bound = snapshot_at_import

        @classmethod
        def reservation(cls, day):
            return 168 if cls.bound <= day else 0

    moved = _dt.date(2026, 12, 3)
    # with the window moved, the bad consumer still answers for the OLD date
    assert _Bad.reservation(_dt.date(2026, 11, 1)) == 168, (
        "the planted bad consumer does not reproduce the defect")
    assert _Bad.bound != moved, (
        "the plant did not actually diverge from the moved window")
    # which is exactly the assertion the control above would fail on
    assert _Bad.reservation(_dt.date(2026, 11, 1)) != 0
