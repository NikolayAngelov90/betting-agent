"""MARGIN EXPOSURE - acknowledge a rare event against its FREQUENCY.

THE STRUCTURAL MARGIN IS NEGATIVE BY CONSTRUCTION AND NEVER CLEARS. Its floor
is the earliest kickoff EVER OBSERVED over the whole record, so the margin
asks "could we be ready before the worst day in three years" and the answer
is no, every day, for ever. A quantity like that cannot be acknowledged: an
acknowledgement is made against a measurement, and this measurement cannot
improve.

The decision quantity is the EXPOSURE:

    exposure(day) = max(0, ready(day_type) - earliest kickoff ON THAT DAY)
                  = minutes we would have been late for the first kickoff we bet

A frequency, not a worst case. It is zero on almost every day, and the
question an operator actually has -- "how often does this bite, and how
badly" -- is answerable from it.

WHY THIS IS NOT THE EXISTING `COVERAGE_EXPOSURE`. That one is
`max(0, deadline - kickoff)` and is ZERO BY CONSTRUCTION, because the
deadline IS the observed earliest kickoff. It measures whether the DEADLINE
is well chosen. This one measures whether WE ARE READY, which is the term the
margin actually bears. Two quantities, two decisions; the first is satisfied
and printed at zero, and this one is new.

UNITS. Minutes throughout. `exposure` is non-negative by definition;
`margin` and `binding` are signed, where negative means short.

    python -m scripts.margin_exposure
    python -m scripts.margin_exposure --json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import os
import re
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

#: BULGARIAN PUBLIC HOLIDAYS over the record window, LISTED rather than
#: computed, because no holiday calendar is installed (`holidays`,
#: `workalendar` both absent) and adding a dependency to stratify a
#: measurement is not a trade this stage should make silently.
#:
#: provenance: `assumed` 2026-11-07 from the published Bulgarian public
#: holiday calendar. Bulgaria is the relevant calendar and not an arbitrary
#: choice: `OPERATOR_TZ` is Europe/Sofia and BOTH earliest-kickoff anchors in
#: `schedule_deadline` come from `bulgaria/efbet-league`. Movable feasts
#: (Orthodox Easter) are included for the years in the window.
#:
#: IT IS LABELLED `assumed` AND NOT `measured`, and the stratum it produces
#: is reported with that label attached, so a reader can discount it.
BG_HOLIDAYS = {
    # fixed
    *(_dt.date(y, m, d)
      for y in (2023, 2024, 2025, 2026)
      for m, d in ((1, 1), (3, 3), (5, 1), (5, 6), (5, 24),
                   (9, 6), (9, 22), (11, 1), (12, 24), (12, 25), (12, 26))),
    # Orthodox Good Friday .. Easter Monday
    _dt.date(2023, 4, 14), _dt.date(2023, 4, 15), _dt.date(2023, 4, 16),
    _dt.date(2023, 4, 17),
    _dt.date(2024, 5, 3), _dt.date(2024, 5, 4), _dt.date(2024, 5, 5),
    _dt.date(2024, 5, 6),
    _dt.date(2025, 4, 18), _dt.date(2025, 4, 19), _dt.date(2025, 4, 20),
    _dt.date(2025, 4, 21),
    _dt.date(2026, 4, 10), _dt.date(2026, 4, 11), _dt.date(2026, 4, 12),
    _dt.date(2026, 4, 13),
}

#: The figures this re-measurement is compared against, kept labelled.
#: THEY ARE A DIFFERENT QUANTITY and the comparison says so: these are
#: `max(0, deadline - kickoff)` under the RETIRED 12:00/10:15 percentile
#: constants, measured 2026-10-19 before the deadline was re-anchored.
PRIOR_1019 = {
    "quantity": "max(0, deadline - kickoff), deadline = 12:00/10:15 constants",
    "weekday": {"n": 715, "pos": 14, "share": 0.020, "p99": 75, "max": 150,
                "over60": 8},
    "weekend": {"n": 375, "pos": 7, "share": 0.019, "p99": 15, "max": 30,
                "over60": 0},
    "wd_holiday": {"n": 22, "pos": 3, "share": 0.136, "p99": 90, "max": 90,
                   "over60": 1},
    "measured": "2026-10-19",
}

#: The registered minimum. NOT edited by this stage; it was stated 2026-10-10.
MIN_ACCEPTABLE_MARGIN_MIN = 60


def _load_env() -> None:
    """INSIDE a function. Reads `.env` and `.env.local` from the repo root."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in (".env", ".env.local"):
        try:
            with open(os.path.join(here, name), encoding="utf-8") as fh:
                for line in fh:
                    m = re.match(r"\s*([A-Z0-9_]+)\s*=\s*(.*)", line)
                    if m and not os.environ.get(m.group(1)):
                        os.environ[m.group(1)] = \
                            m.group(2).strip().strip('"').strip("'")
        except FileNotFoundError:
            continue


def _pct(vals: List[float], q: float) -> Optional[float]:
    """The `q`-quantile by nearest-rank, which is defined at every n.

    `statistics.quantiles` needs n > the number of cut points and raises
    below it, and a stratum of 22 holiday days must still report a p99
    rather than an exception.
    """
    if not vals:
        return None
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def _summarise(exposures: List[float], label: str, n_days: int) -> dict:
    pos = [e for e in exposures if e > 0]
    return {
        "stratum": label,
        "n": n_days,
        "pos": len(pos),
        "share": (len(pos) / n_days) if n_days else None,
        "median": _pct(exposures, 0.50),
        "p90": _pct(exposures, 0.90),
        "p99": _pct(exposures, 0.99),
        "max": max(exposures) if exposures else None,
        "over60": sum(1 for e in exposures if e > MIN_ACCEPTABLE_MARGIN_MIN),
        "median_when_positive": _pct(pos, 0.50) if pos else None,
    }


def measure() -> dict:
    _load_env()
    from sqlalchemy import text

    import scripts.ci_audit as ci
    from scripts.ci_audit import FULL_CARD_PICKS_MAX
    from scripts.schedule_deadline import day_type, projected_ready
    from src.data.database import get_db

    db = get_db()
    assert db.is_postgres, "refusing to measure against the SQLite fallback"

    # ── THE LIVE SERIES, under the DEPLOYED crons ────────────────────────
    series, crons = ci.collect_lag_series_by_daytype()
    unreadable = series.get("__unreadable__") or []
    durations = ci.duration_series_by_daytype() \
        if hasattr(ci, "duration_series_by_daytype") else None

    ready: Dict[str, dict] = {}
    for dtype in ("weekday", "weekend"):
        # THE SERIES IS `(date, lateness)` PAIRS, exactly as `ci_audit`
        # unpacks it at the call site. Passing the pairs through reaches
        # `statistics.mean` and raises on a tuple -- which is the right
        # failure, and it is why the unpack is written out here rather than
        # left implicit.
        delays = [lag for _d, lag in (series.get(dtype) or [])]
        durs = ci.MARGIN_RUN_DURATIONS_BY_DAYTYPE.get(dtype)
        r = projected_ready(crons[dtype], delays, durs or [FULL_CARD_PICKS_MAX])
        if not r:
            ready[dtype] = {"UNEVALUATED": "no lateness series for this "
                                           "day type, so no ready time"}
            continue
        # THE BINDING READY IS THE LATER OF THE TWO, because the margin is
        # `min(margin_max, margin_3sd)` -- the same choice, stated once.
        ready[dtype] = {
            "cron_min": crons[dtype],
            "n_delays": r["n_delays"],
            "ready_max": r["max"],
            "ready_3sd": r["mean_plus_3sd"],
            "binding_ready": max(r["max"], r["mean_plus_3sd"]),
        }

    # ── EVERY DAY'S OWN EARLIEST KICKOFF ─────────────────────────────────
    #
    # Same population rule as `discovered_deadline` EXCEPT for one clause,
    # and the exception matters: that function filters `is_fixture = true`
    # because it answers "what is on tomorrow's card". A row stops being a
    # fixture once it is played, so that clause over a HISTORICAL window
    # returns only the days still ahead -- it gave 17 days here before being
    # removed, against the ~1,090 the record holds. `world/*` excluded and
    # the PHANTOM filter applied in Python, as there: a sub-second microsecond
    # stamp is an ingestion artefact, and the one query that forgets it reads
    # 07:53.
    # ── THE CONFIGURED LEAGUES, READ FROM THE DEPLOYED CONFIG. ───────────
    #
    # `NOT LIKE 'world/%'` IS NOT THE SAME FILTER AND IT COST ME THE FIRST
    # READING. It leaves `other/world` (07:30 UTC), `other/romania` (08:00),
    # `other/andorra` (08:30) and `other/greece` (08:45) in the population --
    # leagues we do not bet -- and they drove every one of the ten worst
    # days. The weekend exposure read 22.7% before this filter and the whole
    # tail was fixtures no pick could ever have been made on.
    #
    # The 30 configured leagues bottom out at 09:30 (`bulgaria/efbet-league`)
    # and 09:59 (`finland/veikkausliiga`), which are exactly the anchors
    # `schedule_deadline` already records -- so the corrected population
    # reproduces the recorded floor instead of contradicting it.
    import yaml
    _cfg = yaml.safe_load(pathlib.Path(
        "config/config.example.yaml").read_text(encoding="utf-8"))
    configured = list(_cfg["scraping"]["flashscore_leagues"])
    assert len(configured) == 30, f"{len(configured)} configured leagues"

    with db.get_session() as s:
        rows = s.execute(text("""
            SELECT m.match_date FROM matches m
            WHERE m.league = ANY(:lg)
              AND m.match_date >= '2023-01-01'
        """), {"lg": configured}).fetchall()

    # ── TIMED, PLACEHOLDER, OR UNEVALUATED. THREE OUTCOMES, NOT TWO. ─────
    #
    # 20,704 of 35,772 configured-league rows are stored at EXACTLY 00:00 --
    # date-only placeholders carrying no kickoff time -- and 509 more carry a
    # sub-second microsecond stamp, which is an ingestion artefact.
    #
    # A DAY WHOSE EVERY ROW IS A PLACEHOLDER IS NOT A ZERO-EXPOSURE DAY. It
    # is a day whose earliest kickoff COULD NOT BE READ, and counting it as
    # zero would be the absence class again: "could not look" reported as
    # "nothing there". Those days are counted and reported as UNEVALUATED and
    # they are excluded from the denominator, so the share is a share of days
    # the record can actually answer for.
    by_day: Dict[_dt.date, List[_dt.datetime]] = defaultdict(list)
    seen_any: Dict[_dt.date, int] = defaultdict(int)
    mixed: Dict[_dt.date, int] = defaultdict(int)
    placeholders = microsecond_stamped = 0
    for (md,) in rows:
        if md is None:
            continue
        seen_any[md.date()] += 1
        if md.microsecond != 0:
            microsecond_stamped += 1
            continue
        if md.hour < 6:
            placeholders += 1
            mixed[md.date()] += 1
            continue
        by_day[md.date()].append(md)
    unevaluated = sorted(d for d in seen_any if d not in by_day)
    # days that DO have a timed kickoff but also carry a placeholder: the
    # true earliest could be hidden behind one, so the exposure on those days
    # is a LOWER BOUND. Counted rather than assumed away.
    partial = sorted(d for d in by_day if mixed.get(d))
    phantoms = placeholders + microsecond_stamped

    # ── THE EXPOSURE, PER DAY ────────────────────────────────────────────
    per_stratum: Dict[str, List[float]] = defaultdict(list)
    worst: List[tuple] = []
    for day, kos in sorted(by_day.items()):
        dtype = day_type(day)
        rd = ready.get(dtype) or {}
        if "binding_ready" not in rd:
            continue
        ko = min(kos)
        ko_min = ko.hour * 60 + ko.minute
        exposure = max(0.0, rd["binding_ready"] - ko_min)
        per_stratum[dtype].append(exposure)
        per_stratum["ALL"].append(exposure)
        if day in BG_HOLIDAYS:
            per_stratum[f"{dtype}/holiday"].append(exposure)
        if exposure > 0:
            worst.append((exposure, day.isoformat(), dtype, ko_min))
    worst.sort(reverse=True)

    # ── THE SINGLE-PERIOD MOVEMENT, MEASURED OVER PREFIXES ───────────────
    #
    # The deterioration threshold must be DERIVED, and the quantity that
    # moves it is `binding_ready`, which moves when the lateness series gains
    # a point. So the sensitivity is measured the way the margin's jitter was:
    # recompute the exposure over each PREFIX of the series and take the
    # largest single-step change in each reported figure.
    #
    # A count has no jitter band of its own -- one more day above the minimum
    # is a new fact, not noise -- so what this measures is whether the COUNT
    # is stable under the arrival of one more lateness point, which is a
    # different and answerable question.
    sensitivity: Dict[str, dict] = {}
    for dtype in ("weekday", "weekend"):
        pairs = series.get(dtype) or []
        durs = ci.MARGIN_RUN_DURATIONS_BY_DAYTYPE.get(dtype)
        steps = []
        for k in range(max(3, len(pairs) - 5), len(pairs) + 1):
            dl = [lag for _d, lag in pairs[:k]]
            rr = projected_ready(crons[dtype], dl,
                                 durs or [FULL_CARD_PICKS_MAX])
            if not rr:
                continue
            br = max(rr["max"], rr["mean_plus_3sd"])
            exps = []
            for day, kos in by_day.items():
                if day_type(day) != dtype:
                    continue
                ko = min(kos)
                exps.append(max(0.0, br - (ko.hour * 60 + ko.minute)))
            steps.append({
                "n": k, "binding_ready": br,
                "pos": sum(1 for e in exps if e > 0),
                "over60": sum(1 for e in exps
                              if e > MIN_ACCEPTABLE_MARGIN_MIN),
                "max": max(exps) if exps else None,
            })
        moves_ready = [abs(steps[i]["binding_ready"]
                           - steps[i - 1]["binding_ready"])
                       for i in range(1, len(steps))]
        moves_pos = [abs(steps[i]["pos"] - steps[i - 1]["pos"])
                     for i in range(1, len(steps))]
        moves_60 = [abs(steps[i]["over60"] - steps[i - 1]["over60"])
                    for i in range(1, len(steps))]
        sensitivity[dtype] = {
            "prefixes": steps,
            "largest_ready_move": max(moves_ready) if moves_ready else None,
            "largest_pos_move": max(moves_pos) if moves_pos else None,
            "largest_over60_move": max(moves_60) if moves_60 else None,
            "provenance": (f"measured 2026-11-07 over "
                           f"{len(steps)} prefixes of the {dtype} lateness "
                           f"series, grain=RUN"),
        }

    out = {
        "measured": "2026-11-07",
        "sensitivity": sensitivity,
        "quantity": ("max(0, binding_ready - earliest kickoff that day); "
                     "binding_ready = max(ready_max, ready_mean+3sd) from the "
                     "LIVE lateness series under the DEPLOYED crons, with "
                     "FULL_CARD_PICKS_MAX=40.35 (parallel regime, n=12)"),
        "unit": "minutes",
        "grain": "DAY",
        "phantoms_excluded": phantoms,
        "placeholder_rows_0000": placeholders,
        "microsecond_stamped_rows": microsecond_stamped,
        "configured_leagues": len(configured),
        "UNEVALUATED_days": len(unevaluated),
        "UNEVALUATED_why": ("every configured-league row on these days is a "
                            "00:00 placeholder, so the earliest kickoff "
                            "could not be read. NOT counted as zero "
                            "exposure and NOT in the denominator"),
        "days_with_any_configured_match": len(seen_any),
        "partial_days": len(partial),
        "partial_why": ("these days carry a timed kickoff AND a placeholder, "
                        "so the true earliest may sit behind the "
                        "placeholder and the exposure is a LOWER BOUND"),
        "unreadable_runs": len(unreadable),
        "ready": ready,
        "strata": {k: _summarise(v, k, len(v))
                   for k, v in sorted(per_stratum.items())},
        "worst_10": worst[:10],
        "prior": PRIOR_1019,
        "min_margin": MIN_ACCEPTABLE_MARGIN_MIN,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = measure()
    if a.json:
        print(json.dumps(r, indent=1, default=str))
        return 0

    print("=" * 78)
    print("EXPOSURE MEASURED  -  Part A")
    print("=" * 78)
    print(f"  quantity: {r['quantity']}")
    print(f"  unit: {r['unit']}   grain: {r['grain']}   "
          f"measured: {r['measured']}")
    print(f"  rows excluded: {r['placeholder_rows_0000']} at 00:00 "
          f"(date-only placeholders) + {r['microsecond_stamped_rows']} "
          f"microsecond-stamped (ingestion artefacts)")
    print(f"  days with any configured match: "
          f"{r['days_with_any_configured_match']}")
    print(f"  UNEVALUATED days: {r['UNEVALUATED_days']} -- "
          f"{r['UNEVALUATED_why']}")
    print(f"  PARTIAL days: {r['partial_days']} -- {r['partial_why']}")
    print(f"  unreadable runs in the lateness series: "
          f"{r['unreadable_runs']}")
    print()
    print("  THE READY TIME, from the live series under the deployed crons:")
    for dtype, rd in r["ready"].items():
        if "binding_ready" not in rd:
            print(f"    {dtype:8s} UNEVALUATED: {rd.get('UNEVALUATED')}")
            continue
        print(f"    {dtype:8s} cron {rd['cron_min']:>3}m  n={rd['n_delays']:>3}"
              f"  ready_max {rd['ready_max']:7.1f}  ready+3sd "
              f"{rd['ready_3sd']:7.1f}  -> binding "
              f"{rd['binding_ready']:7.1f} ({rd['binding_ready'] / 60:.2f}h)")
    print()
    hdr = (f"  {'stratum':18s} {'n':>5s} {'exp>0':>7s} {'share':>7s} "
           f"{'median':>7s} {'p90':>6s} {'p99':>6s} {'max':>7s} "
           f"{'>60m':>6s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for key in ("ALL", "weekday", "weekend", "weekday/holiday",
                "weekend/holiday"):
        v = r["strata"].get(key)
        if not v:
            continue
        print(f"  {key:18s} {v['n']:5d} {v['pos']:7d} "
              f"{100 * v['share']:6.1f}% {v['median']:7.1f} {v['p90']:6.1f} "
              f"{v['p99']:6.1f} {v['max']:7.1f} {v['over60']:6d}")
    print()
    print("  SINGLE-PERIOD SENSITIVITY, over prefixes of the series:")
    for dtype, sv in r["sensitivity"].items():
        print(f"    {dtype}: largest one-point move in binding_ready "
              f"{sv['largest_ready_move']:.2f}m, in the positive-day count "
              f"{sv['largest_pos_move']}, in the >60m count "
              f"{sv['largest_over60_move']}")
        for st in sv["prefixes"]:
            print(f"        n={st['n']:>3}  ready {st['binding_ready']:7.1f}"
                  f"  pos {st['pos']:>3}  >60m {st['over60']:>3}"
                  f"  max {st['max']:6.1f}")
    print()
    print("  the 10 worst days:")
    for e, day, dtype, ko in r["worst_10"]:
        print(f"    {day}  {dtype:8s} exposure {e:6.1f}m  "
              f"(first kickoff {ko // 60:02d}:{ko % 60:02d} UTC)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
