"""Truthiness tests on quantities that can legitimately be ZERO.

    Zero is a legitimate value. A guard using truthiness on a quantity that
    can legitimately be zero fails on the one day it is zero, and midnight is
    that day for every cron.

THE INSTANCE THAT MOTIVATED IT. `ci_audit.missing_scheduled_runs()` guarded on
`if not any(crons.values())`. A cron is stored as MINUTES PAST MIDNIGHT, so
`0 0` is the integer `0` and `any([0, 0])` is `False`. It passed review because
the weekday cron was `0 4` (240) at the time -- the weekend zero was carried by
the weekday value. The revert to `0 0` made both zero and the function returned
`UNEVALUATED` on **every call** for a day, silently, because UNEVALUATED is a
legitimate state and its consumer skips on it. The detector it killed was the
one built for the 422 window.

WHY NAME-BASED. Reachability cannot be decided syntactically: whether a count
can be zero depends on what produces it. So the predicate flags by the
VOCABULARY of quantities this project measures -- minutes, counts, offsets,
credits, margins, n, lags, deltas -- and every hit is triaged by hand into
`REACHABLE` (fixed) or `UNREACHABLE` (listed with a reason). A scanner that
claimed to decide reachability would be the confident-wrong-answer shape.

WHAT IS DELIBERATELY NOT FLAGGED. `if not rows:` on a COLLECTION is correct --
an empty list is genuinely falsy and means "nothing". `x = x or {}` on a
collection is the standard default idiom. Only NUMERIC quantities are flagged,
because for those `0` and `absent` are different facts.

    python -m scripts.falsy_zero_scan             # the census
    python -m scripts.falsy_zero_scan --untriaged # only what is not registered
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys
from dataclasses import dataclass, field

#: Directories scanned. `tests/` is excluded: a test asserting `not n` is
#: usually asserting about a fixture it just built, where zero is not a
#: legitimate value but the point.
ROOTS = ("scripts", "src")

#: The vocabulary of quantities that can legitimately be ZERO in this project.
#: Each entry is a regex matched against the WHOLE identifier, lowercased.
ZERO_LEGITIMATE = (
    r".*_min", r".*_minutes", r"minutes?", r"cron(_min(utes)?)?", r".*_cron",
    r"crons",
    r".*_offset", r"offset", r".*_count", r"count", r"counts", r"n",
    r"n_.*", r".*_n",
    r".*credits?", r".*_margin", r"margin", r"margins", r".*_delay",
    r"delays?", r".*_lag", r"lags?", r".*_delta", r"delta", r".*_total",
    r"total", r".*_sum", r"sum", r".*_index", r"index", r".*_days",
    r".*_secs?", r".*_seconds", r".*_pct", r".*_score", r".*_balance",
    r".*_stake", r".*_ev", r".*_roi", r".*_streak", r".*_pages",
    r"pages_fetched", r".*_fetched", r".*_spent", r".*_remaining",
)
_ZERO_RE = re.compile("|".join(f"(?:{p})" for p in ZERO_LEGITIMATE))

#: Names that LOOK numeric by the rules above but are collections here, so a
#: falsy test on them is correct. Measured by reading each one.
COLLECTION_EXCEPTIONS = {
    "crons",        # dict; `not crons` means "no mapping at all" - correct
    "margins",      # dict of per-day-type margins
    "counts", "delays", "lags", "minutes",
}

SHAPE_NOT = "NOT"                 #: `if not x`
SHAPE_TRUTHY = "TRUTHY"           #: `if x`
SHAPE_OR_DEFAULT = "OR_DEFAULT"   #: `x or <default>`
SHAPE_ANY_ALL = "ANY_ALL"         #: `any(...)` / `all(...)` over values
SHAPE_ASSERT = "ASSERT"           #: `assert x`
SHAPES = (SHAPE_NOT, SHAPE_TRUTHY, SHAPE_OR_DEFAULT, SHAPE_ANY_ALL,
          SHAPE_ASSERT)

#: Triage verdicts.
REACHABLE = "REACHABLE"           #: zero occurs; the guard is wrong
UNREACHABLE = "UNREACHABLE"       #: zero cannot occur here, with a reason
EXEMPT = "EXEMPT"                 #: zero occurs and the behaviour is correct
VERDICTS = (REACHABLE, UNREACHABLE, EXEMPT)


@dataclass
class Hit:
    path: str
    line: int
    shape: str
    expr: str
    src: str = ""
    verdict: str = ""
    why: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.path}:{self.expr}:{self.shape}"


def _name_of(node) -> str:
    """The identifier a test is applied to, or `''`."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _name_of(node.value)
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in ("get", "pop"):
            # `d.get("cron_min")` - the KEY carries the meaning
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    return a.value
            return _name_of(f.value)
        if isinstance(f, ast.Name) and f.id in ("len", "int", "float",
                                                "round", "abs", "sum"):
            return ""        # len() of a collection is correctly falsy at 0
        return _name_of(f)
    return ""


def _is_zero_legitimate(name: str) -> bool:
    if not name:
        return False
    low = name.lower().lstrip("_")
    if low in COLLECTION_EXCEPTIONS:
        return False
    return bool(_ZERO_RE.fullmatch(low))


def _iter_any_all_names(node):
    """Names reached by `any(x.values())`, `all(x)`, `any(a for a in b)`."""
    for arg in node.args:
        if isinstance(arg, ast.Call):
            yield _name_of(arg.func if isinstance(arg.func, ast.Attribute)
                           else arg)
            if isinstance(arg.func, ast.Attribute) and \
                    arg.func.attr in ("values", "keys"):
                yield _name_of(arg.func.value)
        elif isinstance(arg, (ast.GeneratorExp, ast.ListComp)):
            yield _name_of(arg.elt)
        else:
            yield _name_of(arg)


def scan_file(path: pathlib.Path) -> list:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()
    rel = path.as_posix()
    out = []

    def add(node, shape, name):
        ln = getattr(node, "lineno", 0)
        out.append(Hit(rel, ln, shape, name,
                       src=(lines[ln - 1].strip() if 0 < ln <= len(lines)
                            else "")))

    for node in ast.walk(tree):
        # `any(...)` / `all(...)` - the shape that killed the detector
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("any", "all"):
            for nm in _iter_any_all_names(node):
                if _is_zero_legitimate(nm) or nm in COLLECTION_EXCEPTIONS:
                    add(node, SHAPE_ANY_ALL, nm)
                    break
            continue
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            nm = _name_of(node.operand)
            if _is_zero_legitimate(nm):
                add(node, SHAPE_NOT, nm)
            continue
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            nm = _name_of(node.values[0])
            if _is_zero_legitimate(nm):
                add(node, SHAPE_OR_DEFAULT, nm)
            continue
        if isinstance(node, ast.If):
            nm = _name_of(node.test)
            if _is_zero_legitimate(nm):
                add(node, SHAPE_TRUTHY, nm)
            continue
        if isinstance(node, ast.Assert):
            nm = _name_of(node.test)
            if _is_zero_legitimate(nm):
                add(node, SHAPE_ASSERT, nm)
    return out


def scan(roots=ROOTS) -> list:
    hits = []
    for root in roots:
        for p in sorted(pathlib.Path(root).rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            hits.extend(scan_file(p))
    return hits


# ─────────────────────────────────────────────────────────────────────────────
# THE REGISTER. Every hit is triaged BY HAND, because reachability is a fact
# about the producer and not about the syntax. `REACHABLE` entries are fixed;
# `UNREACHABLE` and `EXEMPT` entries carry the reason they are safe.
#
# Keyed by `path:expr:shape`.
TRIAGE = {
    # ── REACHABLE, AND FIXED 2026-10-19 ─────────────────────────────────────
    #
    # The motivating instance is listed although its code is already repaired,
    # because the known-wrong control re-scans it: a detector that cannot
    # catch the instance that motivated it has no measured coverage.
    "scripts/ci_audit.py:crons:ANY_ALL": (
        REACHABLE,
        "THE MOTIVATING INSTANCE. `if not any(crons.values())` over MINUTES "
        "PAST MIDNIGHT: `0 0` is the integer 0, both day types became 0 at "
        "566c927, and the detector returned UNEVALUATED on every call for a "
        "day. It passed review only because the weekday cron was `0 4` (240) "
        "at the time. FIXED to `all(v is None for v in crons.values())`, with "
        "the unreadable-tree case separated. Caught by the known-wrong "
        "control in tests/test_falsy_zero_swept.py."),
    "scripts/run_baseline.py:sum:OR_DEFAULT": (
        REACHABLE,
        "`ws = w.sum() or 1.0`. Zero exactly when `train` is empty -- both "
        "branches give 0.0 -- and there is NO minimum-train guard at the call "
        "site. The substitution returned gh=ga=0.0 and an empty strength "
        "table WITHOUT raising, because the loops that divide by them never "
        "ran. This harness gates every model change. FIXED: it refuses. "
        "REACHABLE BY CONSTRUCTION, NOT OBSERVED."),
    "scripts/run_clean_baseline.py:sum:OR_DEFAULT": (
        REACHABLE,
        "The SAME LINE in the second harness -- `fit_poisson` is duplicated, "
        "so the fix was needed twice, which is what THE HABIT costs. This is "
        "the harness the clean-dataset rule points experiments at, so a "
        "degenerate fit here is a corrupt evidence run. FIXED identically."),
    "scripts/schedule_deadline.py:n_usable:OR_DEFAULT": (
        REACHABLE,
        "Mine, written the same day. `ref.get('n_usable') or 0` -- the "
        "fallback happens to equal the falsy value so behaviour was already "
        "correct, but the idiom is the one that killed the detector above and "
        "reads as if zero were absent. REWRITTEN as `ref.get(k, 0)`."),

    # ── EXEMPT: zero IS reachable and the behaviour is correct ──────────────
    #
    # Every entry below was read. The pattern that makes truthiness correct
    # here is that `0` and `absent` mean the SAME THING to the consumer: there
    # is nothing to log, nothing to divide by, nothing to update. The pattern
    # that makes it wrong -- as above -- is that `0` is a MEASURED VALUE and
    # `absent` is a different fact.
    "scripts/capture_closing_lines.py:n:NOT": (
        EXEMPT,
        "`n = len(s); if not n: return None`. The median of an empty list is "
        "None; 0 and absent agree."),
    "scripts/ci_audit.py:n:NOT": (
        EXEMPT,
        "`if not m.get('n'): return UNEVALUATED`. n=0 means the series is "
        "empty, and an empty series IS unevaluated -- the two causes have one "
        "correct outcome here, which is the opposite of the crons case."),
    "scripts/stage22_team_identity_repair.py:n:TRUTHY": (
        EXEMPT,
        "`if n:` guards an UPDATE by the number of rows that would move. Zero "
        "rows means no statement to run."),
    "src/agent/betting_agent.py:no_match_days:TRUTHY": (
        EXEMPT,
        "FALSE POSITIVE of the name vocabulary: `no_match_days` is a LIST "
        "(the next line calls len() on it), not a count. An empty list is "
        "correctly falsy. Kept in the register rather than silenced so the "
        "predicate's precision is visible."),
    "src/agent/betting_agent.py:no_odds_count:TRUTHY": (
        EXEMPT, "Log suppression: nothing to report when zero."),
    "src/agent/betting_agent.py:total:OR_DEFAULT": (
        EXEMPT,
        "`if real.get('total', 0) or fb.get('total', 0)` -- 'did either source "
        "return rows'. Zero rows from both means nothing to report, and the "
        "explicit `, 0` default already separates absent from zero."),
    "src/agent/betting_agent.py:total:TRUTHY": (
        EXEMPT, "Same block, per-source. Reporting only."),
    "src/data/coverage_checks.py:n_odds:OR_DEFAULT": (
        EXEMPT,
        "`int(n_odds or 0)` -- the fallback equals the falsy value, so "
        "behaviour is identical for 0 and None. Left as written because it is "
        "coercing a possibly-NULL SQL aggregate, where None genuinely means "
        "'no rows' and 0 is unreachable from the query."),
    "src/data/database.py:total:TRUTHY": (
        EXEMPT, "Log suppression after pruning: 0 rows deleted, nothing to "
                "announce. The batch-cap warning above it fires separately."),
    "src/features/team_features.py:n:NOT": (
        EXEMPT, "Early return on an empty sample; 0 and absent agree."),
    "src/models/ml_models.py:n_imp_models:TRUTHY": (
        EXEMPT,
        "GUARDS A DIVISION: `avg_scores[k] /= n_imp_models`. This is the "
        "CORRECT use of truthiness on a count -- the alternative is "
        "ZeroDivisionError."),
    "src/reporting/experiment_record.py:n:NOT": (
        EXEMPT,
        "`if not s.n: 'no closing prices captured yet'`. The 'not computed "
        "this run' case is handled ABOVE and separately, so n=0 here really "
        "does mean the sample is empty."),
    "src/reporting/experiment_record.py:current_n:TRUTHY": (
        EXEMPT,
        "Skips the current-cohort line when the cohort has no settled picks. "
        "Same separation: 'record unavailable' is handled earlier."),
    "src/scrapers/flashscore_scraper.py:stats_count:TRUTHY": (
        EXEMPT, "Log suppression."),
    "src/scrapers/footballdataorg_scraper.py:total:TRUTHY": (
        EXEMPT, "Log suppression."),
    "src/scrapers/injury_scraper.py:already_fresh_count:TRUTHY": (
        EXEMPT, "Log suppression."),
    "src/scrapers/injury_scraper.py:team_fetched:TRUTHY": (
        EXEMPT, "Log suppression."),
    "src/scrapers/injury_scraper.py:old_count:TRUTHY": (
        EXEMPT, "Log suppression."),
}


def triage(hits: list) -> list:
    for h in hits:
        v = TRIAGE.get(h.key)
        if v:
            h.verdict, h.why = v
    return hits


def untriaged(hits: list) -> list:
    return [h for h in hits if not h.verdict]


def regressions(hits: list) -> list:
    """Sites registered REACHABLE that are present in the tree again.

    A fixed site VANISHES from the scan, so the only way to notice one coming
    back is to look for it by name. Without this, the repair is permanent only
    for as long as nobody reintroduces it.
    """
    present = {h.key for h in hits}
    return sorted(k for k, (v, _w) in TRIAGE.items()
                  if v == REACHABLE and k in present)


def format_census(hits: list) -> str:
    by_shape = {}
    for h in hits:
        by_shape.setdefault(h.shape, []).append(h)
    lines = ["falsy-zero census: truthiness on quantities that can be ZERO",
             f"  scanned: {', '.join(ROOTS)}  (tests/ excluded by design)",
             f"  hits: {len(hits)} site(s) at "
             f"{len({h.key for h in hits})} distinct key(s)   "
             f"untriaged: {len(untriaged(hits))}",
             # SITES != KEYS, and the difference is not an error. The register
             # is keyed by `path:expr:shape`, so two `if rec.current_n:` in one
             # file share one entry. Stated because a reader comparing 20 to 18
             # otherwise has to work out which number is wrong, and the lesson
             # from `inverted=10 of n=9` is that they check the arithmetic
             # against itself before drawing the conclusion.
             ]
    for shape in SHAPES:
        rows = by_shape.get(shape) or []
        lines.append(f"  {shape:11s} {len(rows):3d}")
    for v in VERDICTS:
        n = sum(1 for h in hits if h.verdict == v)
        lines.append(f"  verdict {v:12s} {n:3d}  (in the tree now)")
    # THE REGISTER'S OWN COUNTS, BECAUSE A FIXED HIT DISAPPEARS.
    #
    # `REACHABLE 0 in the tree` is what a repaired codebase looks like AND
    # what an empty scan looks like. Reporting only live hits would say "no
    # reachable zeros found" about a sweep that found FOUR and fixed them --
    # the always-unevaluated-detector shape, one layer up.
    present = {h.key for h in hits}
    for v in VERDICTS:
        reg = [k for k, (vv, _w) in TRIAGE.items() if vv == v]
        still = [k for k in reg if k in present]
        lines.append(f"  REGISTERED {v:12s} {len(reg):3d}"
                     f"   still in the tree: {len(still)}")
    if regressions(hits):
        lines.append("  !! REGRESSION: a site registered REACHABLE is back: "
                     + ", ".join(regressions(hits)))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--untriaged", action="store_true")
    ap.add_argument("--shape", choices=SHAPES)
    args = ap.parse_args(argv)

    hits = triage(scan())
    print(format_census(hits))
    show = untriaged(hits) if args.untriaged else hits
    if args.shape:
        show = [h for h in show if h.shape == args.shape]
    print()
    for h in sorted(show, key=lambda x: (x.path, x.line)):
        mark = h.verdict or "UNTRIAGED"
        print(f"  {mark:11s} {h.path}:{h.line}  [{h.shape}] {h.expr}")
        if h.src:
            print(f"              {h.src[:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
