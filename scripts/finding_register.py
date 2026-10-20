"""Diagnosed findings that produced no change — harvested, triaged, re-checked.

    The weather producer was diagnosed correctly on 2026-08-07 and sat for
    seventy-four days. The detector built to find that class could not see it.
    The ledger held the answer the whole time.

WHY A SEPARATE REGISTER AND NOT A MARKER IN THE LEDGER. Part C's question was
whether `docs/ci-audit-ledger.md` can carry a machine-readable marker for an
open finding. **It cannot, and the reason is a rule this project has held four
times: historical entries are not rewritten.** Annotating a 2026-08-07 entry
with today's status rewrites it. So this register REFERENCES doc and line and
holds the status itself; the referenced prose is never touched.

THE HARVEST PREDICATE, stated before it ran:

  1. a defect named with a MECHANISM, not a symptom
  2. a LOCATION -- file, line, symbol, table or workflow step
  3. no subsequent entry recording a fix, a deletion, a declaration or a
     deliberate exemption

QUERY WINDOW, and it is the honest limit. Only ONE document in the repository
carries findings in a form this predicate can be applied to mechanically:
`docs/predictive-audit-2026-08-07.md` lines 645-657, a thirteen-row table of
`# | File | Location | Defect`. The ledger's 29,700 lines are prose; a
mechanical harvest of "diagnosis without subsequent fix" from prose would be a
confident guess, which is why Part C's answer is a register rather than a
scan. **So every age below is the same 74 days and the median carries no
spread** -- that is a property of the source, not of the backlog, and it is
stated rather than presented as a distribution.

STILL-TRUE IS THE ONLY COLUMN THAT MATTERS, and it is the one that costs
effort: an entry fixed incidentally is not a finding, and an entry still live
is a dropped task. Each row below carries a `live` predicate evaluated against
TODAY'S TREE.

    python -m scripts.finding_register            # the register
    python -m scripts.finding_register --live     # only what is still true
"""

from __future__ import annotations

import argparse
import datetime as _dt
import pathlib
import re
import sys

#: The source. Referenced, never rewritten.
SOURCE = "docs/predictive-audit-2026-08-07.md"
SOURCE_DATE = _dt.date(2026, 8, 7)
SOURCE_LINES = (645, 657)

#: Triage outcomes — the same three the PNC-1 backlog uses, plus the two
#: terminal states a re-check can produce.
FIX = "FIX"            #: small, bounded, no cohort effect
DELETE = "DELETE"      #: producer with no consumer and no registered question
DECLARE = "DECLARE"    #: a registered consumer exists -> AWAITING_CONSUMER
FIXED = "FIXED"         #: no longer true against today's tree
OUTCOMES = (FIX, DELETE, DECLARE, FIXED)


def _txt(path: str) -> str:
    try:
        return pathlib.Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


def _fn_body(path: str, marker: str, span: int = 4000) -> str:
    s = _txt(path)
    i = s.find(marker)
    return s[i:i + span] if i >= 0 else ""


# ─────────────────────────────────────────────────────────────────────────────
# THE THIRTEEN, each with a predicate that is TRUE while the defect is live.
#
# `diagnosis` is VERBATIM from the source row. `location` is the row's own
# Location cell. Nothing here paraphrases the original.

FINDINGS = [
    {
        "id": 1, "line": 645,
        "file": "src/models/bayesian_weights.py", "location": "update() 112-147",
        "diagnosis": "League/global params updated regardless of `market` -> "
                     "Poisson double-counted (872 vs 436 obs), 1X2 and goals "
                     "accuracy mixed",
        "live": lambda: 'market: str = ""' not in _fn_body(
            "src/models/bayesian_weights.py", "def update("),
        "outcome": FIXED,
        "why": "`update()` now takes `market` and its docstring states that a "
               "falsy league updates only the market-global scope.",
    },
    {
        "id": 2, "line": 646,
        "file": "src/models/bayesian_weights.py",
        "location": "_params_to_weights() 213-221",
        "diagnosis": "Normalising near-identical accuracies compresses all "
                     "differences -> weights converge to uniform; cannot "
                     "express \"Elo > Poisson\"",
        "live": lambda: "_params_to_weights" in _txt(
            "src/models/bayesian_weights.py"),
        "outcome": FIXED,
        "why": "The symbol no longer exists; the normalisation it named went "
               "with it.",
    },
    {
        "id": 3, "line": 647,
        "file": "src/agent/betting_agent.py", "location": "3295",
        "diagnosis": "`ml_models._models` - attribute is `models`. CV "
                     "classification report has never run",
        "live": lambda: "ml_models._models" in _txt(
            "src/agent/betting_agent.py"),
        "outcome": FIXED,
        "why": "The misspelled attribute access is gone; no `_models` "
               "reference remains in `betting_agent.py`, so the CV "
               "classification report can run.",
    },
    {
        "id": 4, "line": 648,
        "file": "src/agent/betting_agent.py",
        "location": "tune_ensemble_weights 2762-2890",
        "diagnosis": "Poisson/Elo refit as-of; ML uses full-data pickles. "
                     "Asymmetric look-ahead in the weight comparison",
        # HALF ADDRESSED: a "Leak-resistant refit" now refits Poisson/Elo
        # as-of. The ML side still evaluates with the full-data pickle, which
        # is the asymmetry the row named.
        "live": lambda: (
            "Leak-resistant refit" in _fn_body(
                "src/agent/betting_agent.py",
                "async def tune_ensemble_weights", 9000)
            and "ml_models.fit(" not in _fn_body(
                "src/agent/betting_agent.py",
                "async def tune_ensemble_weights", 9000)),
        "outcome": DECLARE,
        "why": "Refitting ML as-of would change the tuned weights, which "
               "changes predictions: a cohort event, not a bounded fix. The "
               "Poisson/Elo half was fixed; the ML half is the remainder.",
        "retriage": "2026-11-23",
    },
    {
        "id": 5, "line": 649,
        "file": "src/agent/betting_agent.py", "location": "get_stats 2614-2621",
        "diagnosis": "`avg_clv` is `p - 1/odds`, not CLV. Mislabelled in every "
                     "`--stats` / `--report` output",
        "live": lambda: "closing_odds" not in _fn_body(
            "src/agent/betting_agent.py", "def get_stats", 9000),
        "outcome": FIXED,
        "why": "The mislabelled quantity is now `avg_divergence`, with the "
               "error written out beside it, and `avg_clv` is computed from "
               "stored `closing_odds`.",
    },
    {
        "id": 6, "line": 650,
        "file": "src/features/feature_engineer.py",
        "location": "499-505, 653-658",
        "diagnosis": "Injury and weather features excluded from training but "
                     "computed at predict time -> dropped by "
                     "`_align_features`. Dead work on every fixture",
        # THE KNOWN-WRONG CONTROL. The weather half was deleted 2026-10-20 --
        # 74 days after this row was written. The INJURY half is unchanged:
        # the block is still guarded by `if not for_training:` and no
        # `injury_*` name appears among the 111 fitted features.
        "live": lambda: bool(re.search(
            r"#\s*\d+\.\s*Injur[^\n]*\n(?:[^\n]*\n){0,6}?\s*if not for_training:",
            _txt("src/features/feature_engineer.py"))),
        "outcome": DELETE,
        "why": "The injury FEATURES have no consumer -- not fitted, dropped "
               "by alignment -- so they are deletable on the same gate the "
               "weather half passed. NOTE the injury SCRAPER is NOT: "
               "`analysis.injury_report` reaches the Claude review prompt.",
        "size": "small: one guarded block plus its defaults, same 8-fixture "
                "aligned-vector gate",
    },
    {
        "id": 7, "line": 651,
        "file": "src/features/team_features.py",
        "location": "_get_league_standings 647-696",
        "diagnosis": "\"League position\" is a rolling 50-match points ranking "
                     "across seasons, with unequal match counts. "
                     "`in_relegation_zone` / `title_gap` are not what they "
                     "claim",
        "live": lambda: "num_matches=50" in _fn_body(
            "src/features/team_features.py", "def _get_league_standings", 4000),
        "outcome": DECLARE,
        "why": "FIVE of the 111 fitted features come from this: "
               "home/away_league_league_position, "
               "home/away_league_relegation_gap, relegation_gap_diff. "
               "Correcting the semantics changes fitted inputs -- a cohort "
               "event.",
        "retriage": "2026-11-23",
    },
    {
        "id": 8, "line": 652,
        "file": "src/models/poisson_model.py",
        "location": "_estimate_league_rhos 325-393",
        "diagnosis": "Per-league MLE over every league x match; measured "
                     "benefit on both 1X2 and O/U 2.5 is nil",
        "live": lambda: "self._estimate_league_rhos(" in _txt(
            "src/models/poisson_model.py"),
        "outcome": DECLARE,
        "why": "Removing it changes rho, which changes every Poisson "
               "probability. 'Nil measured benefit' is not 'no numerical "
               "effect' -- a cohort event.",
        "retriage": "2026-11-23",
    },
    {
        "id": 9, "line": 653,
        "file": "config/config.yaml vs config.example.yaml", "location": "-",
        "diagnosis": "Diverged: `over_3.5` excluded in example only; "
                     "`max_total_kelly_pct: 0` (cap disabled) in live",
        "live": lambda: _config_diverges(),
        "outcome": FIXED,
        "why": "Both files now agree on `max_total_kelly_pct` and on the "
               "excluded-market list. NOTE they agree at 0, i.e. the total "
               "Kelly cap is disabled by design in both -- the DIVERGENCE is "
               "resolved, which is what the row named.",
    },
    {
        "id": 10, "line": 654,
        "file": "src/reporting/match_briefing.py",
        "location": "_recent_review_stats 784-816, "
                    "_recent_selection_stats 751-782",
        "diagnosis": "Injects 8-sample statistics into the decision prompt as "
                     "directives; contaminates the KEEP/CHANGE experiment",
        "live": lambda: ("self._recent_selection_stats()" in _txt(
            "src/reporting/match_briefing.py")
            and "self._recent_review_stats()" in _txt(
                "src/reporting/match_briefing.py")),
        "outcome": DECLARE,
        "why": "Both are still called into the prompt. Removing them changes "
               "what Claude sees and therefore the KEEP/CHANGE decisions -- "
               "value-affecting, so it is a cohort event rather than a "
               "bounded fix.",
        "retriage": "2026-11-23",
    },
    {
        "id": 11, "line": 655,
        "file": "src/agent/betting_agent.py",
        "location": "_auto_calibrate_ev_threshold 3790-3889",
        "diagnosis": "Tightens `min_ev` on cold streaks, selecting *higher* "
                     "claimed-EV picks - the cohort that loses most. n=40 is "
                     "below the noise floor",
        "live": lambda: "self._auto_calibrate_ev_threshold()" in _txt(
            "src/agent/betting_agent.py"),
        "outcome": DECLARE,
        "why": "It moves `min_ev`, which decides which picks are taken. "
               "Selection-affecting by definition.",
        "retriage": "2026-11-23",
    },
    {
        "id": 12, "line": 656,
        "file": "src/models/probability_calibration.py",
        "location": "calibrate_from_pick_outcomes",
        "diagnosis": "Fit and evaluated on the same selection-biased picks; "
                     "`calibrate_from_pick_outcomes` is direction-blind "
                     "(`1 - mean|err|`)",
        "live": lambda: "self.calibrate_from_pick_outcomes()" in _txt(
            "src/agent/betting_agent.py"),
        "outcome": DECLARE,
        "why": "It writes calibration factors the ensemble reads, so changing "
               "it changes every probability.",
        "retriage": "2026-11-23",
    },
    {
        "id": 13, "line": 657,
        "file": "data/models/", "location": "-",
        "diagnosis": "Not version-controlled; survives only via GH Actions "
                     "cache. `feature_list.json` missing -> warning on every "
                     "predict",
        "live": lambda: not pathlib.Path("data/models/feature_list.json").exists(),
        "outcome": FIX,
        "why": "Infrastructure only: no fitted input changes, so no cohort "
               "effect. The cache-only survival is the real risk -- a cache "
               "eviction loses the pickles and the next run retrains silently.",
        "size": "small: write `feature_list.json` at fit time and assert it on "
                "load; deciding whether to track the pickles is separate",
    },
]


def _config_diverges() -> bool:
    """True while the two named divergences are present."""
    # ── "COULD NOT LOOK" IS NOT "FIXED". ──────────────────────────────────
    #
    # The first version of this function returned False on ImportError, which
    # would have reported finding 9 as FIXED in the audit job -- where `yaml`
    # is not installed, because that step runs stdlib-only. A predicate that
    # answers "no defect" when it could not evaluate is the third-state
    # failure this register exists to catch, one level up. It RAISES, and
    # `evaluate()` records UNEVALUATED.
    import yaml      # ImportError propagates deliberately

    a = yaml.safe_load(_txt("config/config.yaml")) or {}
    b = yaml.safe_load(_txt("config/config.example.yaml")) or {}
    if not a or not b:
        raise ValueError(
            "one or both config files are unreadable here, so the divergence "
            "cannot be evaluated -- NOT the same as agreeing")
    ba, bb = (a.get("betting") or {}), (b.get("betting") or {})
    if ba.get("max_total_kelly_pct") != bb.get("max_total_kelly_pct"):
        return True
    return set(ba.get("excluded_markets") or []) != set(
        bb.get("excluded_markets") or [])


def evaluate(today: _dt.date = None) -> list:
    """Each finding with its live status and age. No DB, no network."""
    today = today or _dt.date.today()
    out = []
    for f in FINDINGS:
        try:
            live = bool(f["live"]())
        except Exception as exc:                       # noqa: BLE001
            live = None                                # third state
            f = dict(f, eval_error=f"{type(exc).__name__}: {exc}")
        out.append(dict(f, is_live=live,
                        age_days=(today - SOURCE_DATE).days))
    return out


def untriaged(rows: list) -> list:
    """Live findings with no outcome, or a DECLARE with no re-triage date.

    An unmarked entry is how seventy-four days happened.
    """
    bad = []
    for r in rows:
        if r["is_live"] is not True:
            continue
        if r.get("outcome") not in (FIX, DELETE, DECLARE):
            bad.append(r)
        elif r["outcome"] == DECLARE and not r.get("retriage"):
            bad.append(r)
    return bad


def format_register(rows: list) -> str:
    import statistics

    live = [r for r in rows if r["is_live"] is True]
    unev = [r for r in rows if r["is_live"] is None]
    ages = [r["age_days"] for r in live]
    lines = [
        "diagnosed-and-unactioned register",
        f"  source     : {SOURCE} lines {SOURCE_LINES[0]}-{SOURCE_LINES[1]}"
        f" ({SOURCE_DATE}) -- REFERENCED, never rewritten",
        f"  entries    : {len(rows)}",
        f"  still live : {len(live)}",
        f"  fixed      : {sum(1 for r in rows if r['is_live'] is False)}",
        f"  UNEVALUATED: {len(unev)}   (predicate raised -- not 'fixed')",
        f"  median age of the still-live: "
        f"{statistics.median(ages) if ages else 0:.0f} days"
        f"   (ALL from one dated source, so the distribution is a point mass "
        f"and the median carries NO SPREAD)",
        f"  untriaged  : {len(untriaged(rows))}"
        f"   (an unmarked entry is how 74 days happened)",
        "",
        f"  {'#':>2s} {'live':>6s} {'outcome':8s} {'age':>5s}  location",
    ]
    for r in sorted(rows, key=lambda x: x["id"]):
        mark = {True: "YES", False: "no", None: "UNEVAL"}[r["is_live"]]
        lines.append(f"  {r['id']:2d} {mark:>6s} {r['outcome']:8s} "
                     f"{r['age_days']:4d}d  {r['file']} :: {r['location']}")
    by_outcome = {}
    for r in live:
        by_outcome.setdefault(r["outcome"], []).append(r["id"])
    lines.append("")
    for o in (FIX, DELETE, DECLARE):
        ids = sorted(by_outcome.get(o, []))
        lines.append(f"  {o:8s} {len(ids):2d}  {ids}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true",
                    help="only findings still true against today's tree")
    args = ap.parse_args(argv)

    rows = evaluate()
    print(format_register(rows))
    show = [r for r in rows if r["is_live"] is not False] if args.live else rows
    print()
    for r in sorted(show, key=lambda x: x["id"]):
        mark = {True: "STILL LIVE", False: "fixed", None: "UNEVALUATED"}[
            r["is_live"]]
        print(f"\n  [{r['id']}] {mark}  ({SOURCE}:{r['line']}, "
              f"{r['age_days']} days)")
        print(f"      location : {r['file']} :: {r['location']}")
        print(f"      diagnosis: {r['diagnosis']}")
        print(f"      outcome  : {r['outcome']}"
              + (f"  re-triage {r['retriage']}" if r.get("retriage") else ""))
        print(f"      why      : {r['why']}")
        if r.get("size"):
            print(f"      size     : {r['size']}")
        if r.get("eval_error"):
            print(f"      ERROR    : {r['eval_error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
