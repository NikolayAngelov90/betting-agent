"""Simulate The Odds API credit usage of the Stage 6 refresh design.

    python -m scripts.simulate_odds_quota [--window 120] [--min-interval 180]
                                          [--every 2]

Stage 6, Phase 13. Replays the shipped selection rules against real production
fixture and pick history, day by day, and reports what the design WOULD have
spent. Read-only; makes no API call.

Scenario labels are derived from the data, not assumed: a "high-volume weekend"
is the busiest observed Saturday, not a guess.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta

from src.data.odds_quota import (
    DEFAULT_MONTHLY_BUDGET,
    FREE_TIER_CREDITS,
    credits_for,
)

#: clv.DEFAULT_MAX_CAPTURE_LEAD — a capture earlier than this is not a close.
CAPTURE_LEAD_LIMIT_MIN = 180


def _load_env() -> None:
    """From main() only — never at import (see capture_closing_lines)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def load_history():
    """(picks with kickoff+league, fixtures with kickoff+league) — projected."""
    from src.data.database import get_db
    from src.data.models import Match, SavedPick

    db = get_db()
    with db.get_session() as s:
        picks = s.query(
            SavedPick.id, Match.match_date, Match.league,
        ).join(Match, Match.id == SavedPick.match_id).filter(
            Match.match_date.isnot(None)).all()
        fixtures = s.query(Match.id, Match.match_date, Match.league).filter(
            Match.match_date.isnot(None)).all()
    return ([(p.id, p.match_date, p.league) for p in picks],
            [(f.id, f.match_date, f.league) for f in fixtures])


def simulate(picks, window_min, min_interval_min, run_every_h, mapped):
    """Replay the selection rules. Returns per-day credits and coverage."""
    by_day = defaultdict(list)
    for pid, ko, league in picks:
        if league in mapped:
            by_day[ko.date()].append((pid, ko, league))

    per_day = {}
    covered_total = coverable_total = 0
    for day, items in by_day.items():
        last_refresh = {}
        requests = 0
        covered = set()
        for hour in range(0, 24, run_every_h):
            t = datetime.combine(day, datetime.min.time()) + timedelta(hours=hour)
            due = defaultdict(list)
            for pid, ko, league in items:
                if t < ko <= t + timedelta(minutes=window_min):
                    due[league].append((pid, ko))
            for league, entries in due.items():
                prev = last_refresh.get(league)
                if prev is not None and (t - prev) < timedelta(minutes=min_interval_min):
                    # Skipped by the dedup guard, but the earlier refresh may
                    # still be fresh enough for these picks to validate.
                    for pid, ko in entries:
                        if (ko - prev) <= timedelta(minutes=CAPTURE_LEAD_LIMIT_MIN):
                            covered.add(pid)
                    continue
                requests += 1
                last_refresh[league] = t
                for pid, ko in entries:
                    if (ko - t) <= timedelta(minutes=CAPTURE_LEAD_LIMIT_MIN):
                        covered.add(pid)
        per_day[day] = {
            "requests": requests,
            "credits": credits_for(requests),
            "picks": len(items),
            "covered": len(covered),
        }
        covered_total += len(covered)
        coverable_total += len(items)
    return per_day, (covered_total / coverable_total if coverable_total else 0.0)


def main():
    _load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--min-interval", type=int, default=180)
    ap.add_argument("--every", type=int, default=2, help="hours between runs")
    args = ap.parse_args()

    from src.scrapers.theodds_scraper import LEAGUE_TO_THEODDS_SPORT

    mapped = set(LEAGUE_TO_THEODDS_SPORT)
    picks, fixtures = load_history()

    print("=" * 92)
    print("STAGE 6 QUOTA SIMULATION")
    print("=" * 92)
    print(f"  window {args.window} min | min interval {args.min_interval} min | "
          f"runs every {args.every}h")
    print(f"  credit model: 1 x regions(1) x markets(2) = 2 per league request")
    print(f"  budget {DEFAULT_MONTHLY_BUDGET} of {FREE_TIER_CREDITS} free tier")
    print()
    mapped_picks = [p for p in picks if p[2] in mapped]
    print(f"  saved picks total          : {len(picks)}")
    print(f"  on Odds-API-covered leagues: {len(mapped_picks)} "
          f"({len(mapped_picks)/max(len(picks),1):.0%})")

    per_day, coverage = simulate(picks, args.window, args.min_interval,
                                 args.every, mapped)
    if not per_day:
        print("\n  no pick history to simulate")
        return

    days = sorted(per_day)
    credits = [per_day[d]["credits"] for d in days]
    mean = sum(credits) / len(credits)
    ordered = sorted(credits)

    print()
    print(f"  simulated days             : {len(days)} ({days[0]} .. {days[-1]})")
    print(f"  mean credits/day           : {mean:.1f}")
    print(f"  median                     : {ordered[len(ordered)//2]}")
    print(f"  p95                        : {ordered[int(len(ordered)*0.95)]}")
    print(f"  worst observed day         : {max(credits)}")
    print(f"  PROJECTED credits/month    : {mean*30:.0f}  "
          f"({'within' if mean*30 <= DEFAULT_MONTHLY_BUDGET else 'OVER'} the "
          f"{DEFAULT_MONTHLY_BUDGET} budget, free tier {FREE_TIER_CREDITS})")
    print(f"  pick coverage (capturable) : {coverage:.0%}")

    # Worst case: every day as busy as the worst observed day.
    print(f"  worst-case month (every day = worst observed): "
          f"{max(credits)*30}  "
          f"{'OVER FREE TIER' if max(credits)*30 > FREE_TIER_CREDITS else 'still within free tier'}")

    print()
    print("=" * 92)
    print("SCENARIOS (from the data, not assumed)")
    print("=" * 92)
    weekday = [d for d in days if d.weekday() < 5]
    weekend = [d for d in days if d.weekday() >= 5]

    def _row(label, subset):
        if not subset:
            print(f"  {label:<34} (none observed)")
            return
        c = [per_day[d]["credits"] for d in subset]
        pk = sum(per_day[d]["picks"] for d in subset)
        cv = sum(per_day[d]["covered"] for d in subset)
        print(f"  {label:<34}{len(subset):>5} days | "
              f"mean {sum(c)/len(c):>5.1f} cr | max {max(c):>3} cr | "
              f"picks {pk:>4} | covered {cv/pk if pk else 0:>4.0%}")

    _row("normal weekday", weekday)
    _row("normal weekend", weekend)
    busiest = max(days, key=lambda d: per_day[d]["credits"])
    _row(f"busiest day ({busiest} {busiest.strftime('%a')})", [busiest])
    euro_days = [d for d in days
                 if any(lg.startswith("europe/")
                        for _, ko, lg in picks if ko.date() == d)]
    _row("European competition day", euro_days)

    print()
    print("  busiest five days:")
    for d in sorted(days, key=lambda x: -per_day[x]["credits"])[:5]:
        v = per_day[d]
        print(f"      {d} {d.strftime('%a')}: {v['requests']:>2} requests = "
              f"{v['credits']:>3} credits, {v['picks']:>3} picks, "
              f"{v['covered']:>3} capturable")


if __name__ == "__main__":
    main()
