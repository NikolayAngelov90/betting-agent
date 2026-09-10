"""H5 — pre-kickoff drift. THE REGISTERED ANALYSIS, PRESERVED.

    THIS RAN ONCE, ON 2026-09-10, AT n=129.
    The verdict it produced is recorded in `docs/h5-drift-preregistration.md`
    and is FIXED. Re-running this file reproduces the computation; it does NOT
    create a new verdict, and no number in this project's record may be updated
    from a later execution of it.

The registration's stopping rule is the reason for that banner:

    "Analysis runs ONCE, when n >= 50 is reached. Not before, and not again
     after a disappointing result."

A file that recomputes a registered result on demand is an invitation to run it
until it says something else. It is kept anyway, because the alternative — a
record citing numbers no one can reproduce — is worse. The banner is the guard.

WHAT THIS FILE IS
-----------------
The corrected harness. The first execution was defective in two places, both
mine, both disclosed in the registration's RESULT section:

  Q2  membership was tested by comparing `odds_snapshots.market_type/selection`
      against `saved_picks.market/selection`. THE TWO TABLES USE DIFFERENT
      VOCABULARIES — snapshots say ('1X2','Home'), saved_picks says
      ('1X2','Home Win'); snapshots say ('over_under','Over 2.5'), saved_picks
      says ('Over 2.5','Over 2.5 Goals'). The intersection was empty by
      construction and Q2 reported n=0.

      Fixed with the alias declarations THAT ALREADY EXIST — market_spec's
      MARKET_SPECS.legs and capture_closing_lines' SELECTION_SPEC. No third
      mapping table: a third copy is how the first two came to disagree, and
      this data layer has recorded that habit five times.

  Q1  the control was built with `WHERE sp.disposition IS NULL`, so fixtures
      whose picks had been CONSOLIDATED counted as never-picked. Fixed — a
      consolidated row was still a stake that was taken. Correcting it made the
      control smaller, from 1 to 0.

n=0 on Q2 was treated as evidence about the MEASUREMENT, not as a null result.
Q3 is untouched by both faults and is therefore the control on the correction:
it must reproduce the first execution exactly, and it did (6/25 strata,
identical price and lead-time cells).

WHY THE CLUSTER BOOTSTRAP IS NOT OPTIONAL HERE
----------------------------------------------
Q2's observations are (fixture x bookmaker x selection). One fixture's price
move appears once per bookmaker, so 814 observations are not 814 draws — the
measured design effect is ~11 and the effective n is 51.7. Quoting the naive
interval would present 61 fixtures as 814 independent draws, which is the error
direction that makes a null look significant. `src.evaluation.clv._boot` is the
same estimator the CLV report uses, and it DECLINES below five clusters — which
is why taken 1X2 Away (86 observations across 4 fixtures) is printed and
refused on the same line rather than quietly reported.

WHY THIS LIVES IN `analysis/` AND NOT `scripts/`
------------------------------------------------
DO NOT MOVE IT BACK. `tests/test_price_history_accumulates.py::
test_nothing_reads_the_snapshot_table_yet` fails any file under `src/` or
`scripts/` that names `odds_snapshots`, on the grounds that a model reading
price history is a COHORT EVENT and needs a decision, not a deploy.

This file reads that table, and it is the sanctioned reader — the pin's own
words are "Stage 18 stored it so Stage 19 could STUDY it, not so the model
could use it." But the guard cannot tell a study from a learner, and the fix
that suggests itself — an exemption marker on this file — is the wrong trade:
a marker that fits this file fits every file, which is precisely how an
exemption gets pasted past review. `test_valid_evidence_gate` records that
failure mode in its own docstring.

So the guard keeps its teeth and the analysis moves out of its blast radius.
Run it with `python -m analysis.h5_drift_analysis` from the repo root.

Read-only. Touches no learner, writes nothing.
"""
from __future__ import annotations

import importlib.util
import pathlib
import statistics as st
from collections import defaultdict

from dotenv import load_dotenv
from sqlalchemy import text

from src.data.database import DatabaseManager
from src.data.market_spec import MARKET_SPECS
from src.evaluation.clv import _boot, _effective_n

load_dotenv(".env")

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "_ccl", _ROOT / "scripts" / "capture_closing_lines.py")
_ccl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ccl)
SELECTION_SPEC = _ccl.SELECTION_SPEC

#: Registered thresholds. Named, not inlined, so a reader can see that none of
#: them was chosen after the numbers were known.
BREAK_EVEN = 1.85          # best-line break-even, %
MIN_FIXTURES = 50          # registered minimum sample
MIN_GAP_MIN = 30           # registered separation between the two observations
BROAD_FRACTION = 0.60      # >= this share of strata above BREAK_EVEN = BROAD


def snapshot_key(pick_selection):
    """SavedPick.selection -> (odds_snapshots.market_type, .selection).

    Derived from the two EXISTING declarations rather than a new table:
      * SELECTION_SPEC gives (market_type, line, side, leg).
      * MARKET_SPECS[mt].legs[leg][0] is that leg's canonical alias, which is
        the label the snapshot table carries.
    Line- and side-qualified markets compose their label the same way the
    scraper composed it when it wrote the row.
    """
    spec = SELECTION_SPEC.get(pick_selection)
    if spec is None:
        return None
    mt, line, side, leg = spec
    if mt == "over_under":
        return (mt, ("Over " if leg == 0 else "Under ") + line)
    if mt == "team_goals":
        return (mt, side + (" Over " if leg == 0 else " Under ") + line)
    return (mt, MARKET_SPECS[mt].legs[leg][0])


#: Phantom rows carry `match_date == created_at` with non-zero microseconds;
#: the EXTRACT clause is this project's standing exclusion for them.
SQL = """
SELECT o.match_id, o.bookmaker, o.market_type, o.selection,
       o.odds_value, o.observed_at, m.league, m.match_date
FROM odds_snapshots o JOIN matches m ON m.id = o.match_id
WHERE o.observed_at < m.match_date
  AND EXTRACT(microsecond FROM m.match_date) = 0
ORDER BY o.match_id, o.bookmaker, o.market_type, o.selection, o.observed_at
"""


class Pair:
    __slots__ = ("match_id", "bookmaker", "market_type", "selection",
                 "f", "l", "gap", "league", "match_date", "lead_h")


def _pct(a, b):
    return 100.0 * (float(b) - float(a)) / float(a)


def rep(vals, label):
    if len(vals) < 2:
        return "    " + label.ljust(34) + " n=" + str(len(vals)) + " - too few"
    m, sd = st.mean(vals), st.stdev(vals)
    se = sd / (len(vals) ** 0.5)
    return ("    " + label.ljust(34) + " n=" + str(len(vals)).rjust(4)
            + "  mean " + format(m, "+7.3f") + "%  sd " + format(sd, "6.3f")
            + "  95% CI [" + format(m - 1.96 * se, "+.3f") + "%, "
            + format(m + 1.96 * se, "+.3f") + "%]")


def clustered(label, vals, clusters):
    """Report a mean with the CLUSTER interval, and refuse it below 5 clusters."""
    xs = list(vals)
    if len(xs) < 2:
        print("    " + label + ": too few")
        return
    mean, sd = st.mean(xs), st.stdev(xs)
    naive_se = sd / (len(xs) ** 0.5)
    lo, hi = _boot(xs, clusters)
    _n, _k, deff_wc, eff = _effective_n(clusters)
    print("    " + label)
    print("        n=" + str(len(xs)) + "  fixtures=" + str(len(set(clusters)))
          + "  effective n=" + format(eff, ".1f")
          + "  (worst-case deff " + format(deff_wc, ".2f") + ")")
    print("        mean " + format(mean, "+.3f") + "%")
    print("        naive 95% CI   [" + format(mean - 1.96 * naive_se, "+.3f")
          + "%, " + format(mean + 1.96 * naive_se, "+.3f") + "%]")
    if lo is None:
        print("        CLUSTER 95% CI: _boot DECLINED - fewer than 5 fixtures.")
        print("        => this mean rests on " + str(len(set(clusters)))
              + " independent fixtures and is NOT interpretable.")
        return
    half = (hi - lo) / 2.0
    deff = (half / (1.96 * naive_se)) ** 2 if naive_se else float("nan")
    print("        CLUSTER 95% CI [" + format(lo, "+.3f") + "%, "
          + format(hi, "+.3f") + "%]   design effect ~" + format(deff, ".2f"))
    print("        crosses +" + format(BREAK_EVEN, ".2f") + "%? "
          + ("YES" if hi > BREAK_EVEN else "NO"))


def main():
    db = DatabaseManager()
    with db.get_session() as s:
        raw = s.execute(text(SQL)).fetchall()
        # disposition IGNORED: a consolidated row was still a stake taken.
        pickrows = s.execute(text(
            "SELECT match_id, market, selection FROM saved_picks")).fetchall()

    taken_keys, unmapped = set(), defaultdict(int)
    for mid, mk, sel in pickrows:
        k = snapshot_key(sel)
        if k is None:
            unmapped[(mk, sel)] += 1
            continue
        taken_keys.add((mid, k[0], k[1]))
    picked_fixtures = {mid for mid, _mk, _s in pickrows}

    groups = defaultdict(list)
    for r in raw:
        groups[(r.match_id, r.bookmaker, r.market_type, r.selection)].append(r)

    rows, atleast2 = [], set()
    for key, obs in groups.items():
        if (len(obs) >= 2 and key[2] == "1X2" and key[3] == "Home"
                and (obs[-1].observed_at - obs[0].observed_at
                     ).total_seconds() / 60.0 >= MIN_GAP_MIN):
            atleast2.add(key[0])
        if len(obs) != 2:              # registered: a TWO-point observation
            continue
        a, b = obs
        gap = (b.observed_at - a.observed_at).total_seconds() / 60.0
        if gap < MIN_GAP_MIN:
            continue
        p = Pair()
        (p.match_id, p.bookmaker, p.market_type, p.selection) = key
        p.f, p.l, p.gap = a.odds_value, b.odds_value, gap
        p.league, p.match_date = a.league, a.match_date
        p.lead_h = (a.match_date - a.observed_at).total_seconds() / 3600.0
        rows.append(p)

    def mv(r):
        return _pct(r.f, r.l)

    print("=" * 78)
    print("H5 - PRE-KICKOFF DRIFT.  Registered 2026-09-03.  RAN ONCE 2026-09-10.")
    print("=" * 78)

    if unmapped:
        print("\n!! pick selections with NO snapshot mapping - Q2 would DROP these:")
        for k, n in sorted(unmapped.items(), key=lambda x: -x[1]):
            print("     " + k[0].ljust(16) + " " + k[1].ljust(24) + " x" + str(n))
    else:
        print("\n  vocabulary check: every saved_picks selection maps to a "
              "snapshot key (0 unmapped)")

    home = [r for r in rows if r.market_type == "1X2" and r.selection == "Home"]
    by = defaultdict(list)
    for r in home:
        by[r.match_id].append(mv(r))
    fx = {m: st.median(v) for m, v in by.items()}

    print("\nSAMPLE: " + str(len(fx)) + " fixtures with a two-point 1X2-Home "
          "observation >=" + str(MIN_GAP_MIN) + " min apart")
    print("        registered minimum n=" + str(MIN_FIXTURES) + " - "
          + ("MET" if len(fx) >= MIN_FIXTURES else "NOT MET"))
    print("        sensitivity: reading 'two-point' as AT LEAST two gives n="
          + str(len(atleast2)))
    print(rep(list(fx.values()), "  fixture-level 1X2 Home drift"))
    print("    sigma re-derived (registered task): "
          + format(st.stdev(list(fx.values())), ".3f")
          + "%   (registration carried 9.39% from n=15)")

    print("\n" + "=" * 78)
    print("Q1 - REAL, OR SELECTION BIAS?")
    print("=" * 78)
    pk = [v for m, v in fx.items() if m in picked_fixtures]
    un = [v for m, v in fx.items() if m not in picked_fixtures]
    print(rep(pk, "PICKED fixtures (1X2 Home)"))
    print(rep(un, "UNPICKED fixtures (1X2 Home)"))
    if len(pk) >= 2 and len(un) >= 2:
        gap = st.mean(pk) - st.mean(un)
        print("\n    gap (picked - unpicked): " + format(gap, "+.3f") + " pp")
        print("    ==> " + ("SELECTION BIAS" if gap >= 2.0 else
                            "MARKET EFFECT" if abs(gap) < 1.0 else "MIXED"))
    else:
        print("\n    ==> NOT EVALUABLE - the control population is EMPTY.")
        print("        refresh_and_capture.py defaults to require_pending_pick"
              "=True, so a fixture")
        print("        is re-priced BECAUSE it carries a pending pick. Waiting "
              "yields more PICKED")
        print("        fixtures. The control needs --any-fixture, which costs "
              "credits.")

    print("\n" + "=" * 78)
    print("Q2 - HOME-SPECIFIC, OR DOES EVERYTHING DRIFT THE WAY IT WAS TAKEN?")
    print("=" * 78)
    taken = defaultdict(list)          # market_type -> [(selection, move, fix)]
    for r in rows:
        if (r.match_id, r.market_type, r.selection) in taken_keys:
            taken[r.market_type].append((r.selection, mv(r), r.match_id))
    allv = [(v, f) for lst in taken.values() for _s, v, f in lst]
    for mk, lst in sorted(taken.items(), key=lambda x: -len(x[1])):
        clustered("taken: " + mk, [v for _s, v, _f in lst],
                  [f for _s, _v, f in lst])
    print()
    clustered("ALL taken selections", [v for v, _f in allv],
              [f for _v, f in allv])
    if len(allv) >= 2:
        m = st.mean([v for v, _f in allv])
        pos = {mk: st.mean([v for _s, v, _f in l]) > 0
               for mk, l in taken.items() if len(l) >= 2}
        print("\n    registered bands: >+" + format(BREAK_EVEN, ".2f")
              + "% across ALL markets = PRICING ARTEFACT | Home out & another "
                "in = HOME-SPECIFIC | below = NEITHER")
        if m > BREAK_EVEN and all(pos.values()) and len(pos) > 1:
            v2 = "PRICING ARTEFACT"
        elif any(not p for p in pos.values()) and pos.get("1X2", False):
            v2 = "HOME-SPECIFIC (mixed directions)"
        else:
            v2 = "NEITHER"
        print("    per-market mean > 0: " + str(pos))
        print("    ==> " + v2)
        sel = defaultdict(list)
        for _s, v, f in taken.get("1X2", []):
            sel[_s].append((v, f))
        for k in ("Home", "Draw", "Away"):
            if sel.get(k):
                clustered("taken 1X2 " + k, [v for v, _f in sel[k]],
                          [f for _v, f in sel[k]])

    print("\n" + "=" * 78)
    print("Q3 - BROAD, OR CONCENTRATED?  (descriptive; NO threshold fitted)")
    print("=" * 78)
    lg = defaultdict(list)
    for r in home:
        lg[r.league].append(mv(r))
    strata = {k: v for k, v in lg.items() if len(v) >= 10}
    above = sum(1 for v in strata.values() if st.mean(v) > BREAK_EVEN)
    print("    leagues with >=10 observations: " + str(len(strata)))
    if strata:
        frac = above / len(strata)
        print("    strata above +" + format(BREAK_EVEN, ".2f") + "%: "
              + str(above) + "/" + str(len(strata)) + " = "
              + format(100 * frac, ".0f") + "%")
        print("    ==> " + ("BROAD" if frac >= BROAD_FRACTION else "NOT BROAD"))

    print("\n    by price band (first price):")
    band = defaultdict(list)
    for r in home:
        o = float(r.f)
        band["<2.0" if o < 2.0
             else ("2.0-3.5" if o <= 3.5 else ">3.5")].append(mv(r))
    for k in ("<2.0", "2.0-3.5", ">3.5"):
        if band[k]:
            print(rep(band[k], "  odds " + k))

    # Lead time carries its FIXTURE count because that is the number that
    # decides whether the cell means anything: on the recorded run, 127 of 129
    # fixtures sat in one bucket and >12h was empty.
    print("\n    by lead time (first observation to kickoff):")
    lead, leadfx = defaultdict(list), defaultdict(set)
    for r in home:
        h = float(r.lead_h)
        k = "<6h" if h < 6 else ("6-12h" if h <= 12 else ">12h")
        lead[k].append(mv(r))
        leadfx[k].add(r.match_id)
    for k in ("<6h", "6-12h", ">12h"):
        if lead[k]:
            print(rep(lead[k], "  lead " + k)
                  + "  fixtures=" + str(len(leadfx[k])))
    print("\n    A schedule that fires at a fixed hour produces a near-constant"
          " lead, so this")
    print("    breakdown cannot test the Stage 21 lead-time claim. See "
          "docs/stage21-schedule-prediction.md.")


if __name__ == "__main__":
    main()
