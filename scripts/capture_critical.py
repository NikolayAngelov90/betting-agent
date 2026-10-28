"""Which tests may stop price capture — derived, not chosen. 2026-10-28.

> **Four cards lost. Three of the four were clock-dependent tests that say
> nothing about capture correctness.**

THE GATE'S DIRECTION IS INVERTED HERE. `Run tests` gated EVERY test and
excluded two markers by hand (`not calendar and not live`), so each new test
gated price capture by default and had to be opted OUT. That default has been
wrong four times:

    2026-10-09  the empty-card suppression's expiry + three quota tests that
                inherited the H1 allocator's first active day   -> 1 card
    2026-10-13  a rolling-window assertion over a live `gh run list`
                                                                -> 6 cards
    2026-10-21  a COUNT asserted against `finding_register.evaluate`, whose
                predicate reads an untracked path                -> 1 card
    2026-10-27  a frozen `NOW` fixture asserted against `mf.main()`, which
                reads the clock with no seam                     -> 1 card

NONE of the four says anything about whether capture writes correct rows. All
four could stop it.

So gating becomes OPT-IN: a test gates capture only if it is capture-critical.
Everything else runs in the non-gating step, reddens the run, and cannot stop
`--update`, `--settle` or `--picks`.

AND THE SET IS DERIVED FROM THE IMPORT GRAPH, NOT LISTED BY HAND. A hand list
is the `LIVE_NAMES` shape: 16 seamless clock readers in `scripts/` and 0 of
them registered, because a list is a remedy someone must remember. The
criterion below is mechanical, and `CAPTURE_PATH` is the only judgement in it.

WHAT THIS DOES NOT DO. It does not close `UNCOVERABLE`'s clock-reader gap —
"a reader nobody registered is invisible" is still true and the enumeration
stays. It makes that gap HARMLESS rather than fixed: an unregistered clock
reader can still make a test flap, but unless that test is capture-critical
the flap can no longer reach `--update`. Routed around, not closed, and the
difference matters because a capture-critical test with an unregistered clock
reader is still exactly as dangerous as before.
"""

from __future__ import annotations

import argparse
import ast
import os
import pathlib
import sys
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: ── THE CRITERION, AND IT IS THE ONLY JUDGEMENT HERE ─────────────────────
#:
#: A test is CAPTURE-CRITICAL if it imports, directly or through another test
#: helper, any module on the path that WRITES a captured row. Each entry below
#: is a module `--update` traverses between asking a provider and committing a
#: row, and each is here because a defect in it corrupts or loses data that
#: never backfills:
#:
#:   the scrapers      ask the providers and parse the answers
#:   team_resolution   the identity gate. A wrong club writes a real price
#:                     onto the wrong fixture, and the h2h/Home corruption
#:                     (28 matches, 13 picks) came through here
#:   fixture_identity  the same question for fixtures: a duplicate fixture is
#:                     two rows that disagree
#:   fixture_plausibility  rejects a fixture whose shape cannot be real
#:   market_spec       which selections a market may hold. The
#:                     "Home/Away" two-way trap corrupted 2,486 matches
#:   odds_quota        the budget claim. Over-spend is silent until the pool
#:                     is dry and the card goes unpriced
#:   api_budget        its ledger
#:   database          the write path itself
#:   models            the schema those writes go through
#:   price_history     the odds/odds_snapshots write path
#:
#: DELIBERATELY ABSENT, and each absence is a decision: the models
#: (`src/models/**`), the feature engineer, the reporting layer and the
#: evaluation harness. A defect in any of them produces a WRONG PICK, which is
#: recoverable — the row is still there and the pick can be re-made — whereas
#: a lost or corrupted odds row never backfills. The gate exists for the
#: irrecoverable half.
CAPTURE_PATH: Tuple[str, ...] = (
    "src.scrapers",
    "src.data.team_resolution",
    "src.data.fixture_identity",
    "src.data.fixture_plausibility",
    "src.data.market_spec",
    "src.data.odds_quota",
    "src.data.api_budget",
    "src.data.database",
    "src.data.models",
    "src.data.price_history",
)

#: The marker a capture-critical test carries, and the COMPLETE `-m`
#: expressions the three workflows run.
#:
#: `GATING_EXPR` WAS `MARKER` ALONE, and that was a real defect for the few
#: minutes it existed: importing it into the controls widened the gating
#: selection from 1,641 to 1,663 and pulled `calendar`-marked tests back in —
#: including `test_the_suppression_has_EXPIRED`, the one that cost the
#: 2026-10-09 card. The marker exclusions are PART of the gate, not a detail
#: the caller adds, and leaving them to the caller is how two call sites come
#: to disagree.
MARKER = "capture_critical"
GATING_EXPR = f"{MARKER} and not calendar and not live"
NON_GATING_EXPR = f"not {MARKER} and not calendar and not live"

TESTS = pathlib.Path("tests")


def _modules_imported(tree: ast.AST) -> Set[str]:
    out: Set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                out.add(a.name)
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                out.add(n.module)
                for a in n.names:
                    out.add(f"{n.module}.{a.name}")
    return out


def _on_capture_path(mods: Set[str]) -> Set[str]:
    hit = set()
    for m in mods:
        for p in CAPTURE_PATH:
            if m == p or m.startswith(p + "."):
                hit.add(p)
    return hit


def classify(paths=None) -> Dict[str, Dict]:
    """`{test_file: {"critical": bool, "via": [...], "helpers": [...]}}`.

    TRANSITIVE THROUGH TEST HELPERS ONLY, one hop. A test importing
    `tests.h1_synthetic`, which imports `src.data.models`, is capture-critical
    — the helper is how the test reaches the write path. It is NOT transitive
    through `src.` itself: almost everything in `src/` reaches `models`
    eventually, and a criterion that selects everything is the default this
    exists to replace.
    """
    files = sorted(paths or TESTS.glob("test_*.py"))
    helpers: Dict[str, Set[str]] = {}
    for h in sorted(TESTS.glob("*.py")):
        if h.name.startswith("test_"):
            continue
        try:
            helpers[f"tests.{h.stem}"] = _modules_imported(
                ast.parse(h.read_text(encoding="utf-8")))
        except SyntaxError:
            helpers[f"tests.{h.stem}"] = set()

    out: Dict[str, Dict] = {}
    for f in files:
        try:
            mods = _modules_imported(ast.parse(f.read_text(encoding="utf-8")))
        except SyntaxError:
            out[f.name] = {"critical": False, "via": [], "helpers": [],
                           "unparseable": True}
            continue
        via = _on_capture_path(mods)
        used_helpers = sorted(
            h for h in helpers if h in mods or any(
                m.startswith(h + ".") for m in mods))
        via_helpers = []
        for h in used_helpers:
            hv = _on_capture_path(helpers[h])
            if hv:
                via_helpers.append(h)
                via |= hv
        out[f.name] = {"critical": bool(via), "via": sorted(via),
                       "helpers": via_helpers}
    return out


def marked(path: pathlib.Path) -> bool:
    """Whether the file declares the marker at module level.

    MODULE LEVEL ONLY (`pytestmark`), by AST so a mention in a comment or a
    docstring cannot count — the comment-blind defect, thirteen instances.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return False
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "pytestmark"
                for t in n.targets):
            for c in ast.walk(n.value):
                if isinstance(c, ast.Attribute) and c.attr == MARKER:
                    return True
    return False


def census(paths=None) -> Dict:
    c = classify(paths)
    crit = {k: v for k, v in c.items() if v["critical"]}
    return {
        "files": len(c),
        "capture_critical": len(crit),
        "not_critical": len(c) - len(crit),
        "critical_files": sorted(crit),
        "marked_files": sorted(f.name for f in sorted(TESTS.glob("test_*.py"))
                               if marked(f)),
        "by_module": _by_module(crit),
    }


def _by_module(crit: Dict[str, Dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in crit.values():
        for m in v["via"]:
            out[m] = out.get(m, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true",
                    help="print every capture-critical file and why")
    ap.add_argument("--unmarked", action="store_true",
                    help="print capture-critical files missing the marker")
    args = ap.parse_args(argv)

    c = classify()
    cen = census()
    print(f"CAPTURE-CRITICAL CENSUS  (criterion: imports a CAPTURE_PATH "
          f"module, one hop through tests/ helpers)")
    print(f"  test files              : {cen['files']}")
    print(f"  capture-critical        : {cen['capture_critical']}")
    print(f"  not capture-critical    : {cen['not_critical']}")
    print(f"  carrying the marker     : {len(cen['marked_files'])}")
    print()
    print("  reached via:")
    for m, n in cen["by_module"].items():
        print(f"    {m:34s} {n:4d} file(s)")

    if args.list:
        print("\n  capture-critical files:")
        for f in cen["critical_files"]:
            v = c[f]
            h = f"  (via helper {v['helpers']})" if v["helpers"] else ""
            print(f"    {f:52s} {v['via']}{h}")

    missing = [f for f in cen["critical_files"]
               if f not in cen["marked_files"]]
    if args.unmarked:
        print(f"\n  capture-critical WITHOUT the marker: {len(missing)}")
        for f in missing:
            print(f"    {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
