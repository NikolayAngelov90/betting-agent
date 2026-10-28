"""H1's input precondition, and the alarm for a silent zero. Added 2026-10-18.

    Independence of schedule is not independence of input. Two jobs that never
    collide can still have one starve the other when one writes the population
    the other samples.

WHAT HAPPENED. H1's job ran on 2026-10-13, 10-14 and 10-15, **succeeded every
time**, spent **0 TheOddsAPI credits** and wrote **0 rows**. Its own apparatus
check reported `state=RUNNING_AND_FINE alarms=False` throughout. The only trace
was one INFO line per run:

    TheOddsAPI imminent refresh: nothing to do (window 360 min) — 4 league(s) skipped

Meanwhile `daily-picks` was dead 10-13 → 10-17, so the cards on those days were
**8, 4 and 1** fixtures against a normal 20–100. The experiment collected n=4
of a target 39 and the window closed.

THE DEPENDENCY, TRACED AND STATED:

    daily-picks --update      writes `matches` rows with is_fixture = true
        |                     (weekday cron, so once per day at best)
        v
    matches                   the population H1 samples
        |
        v
    _imminent_league_fixtures selects rows with is_fixture = true AND
        |                     now < match_date <= now + window
        v
    h1_collect                requests odds for the surviving leagues

H1 never touches `daily-picks` and cannot be blocked by it. It is nonetheless
**entirely downstream of it**, and nothing in H1 said so.

THE BAND. A collection round needs the fixture to still be unkicked when the
LAST request fires. With 3 rounds at 120-minute gaps the last request is at
`T+240`, so a usable fixture sits in `(T+240, T+360]`. A fixture at `T+60` is
in the 360-minute window and is useless to H1 — it will have kicked off before
round 3. Counting the window instead of the band is what made `nothing to do`
indistinguishable from `nothing usable`.
"""

from __future__ import annotations

import datetime as _dt
import os as _os
import sys as _sys


def _sys_argv():
    return _sys.argv[1:]
from typing import List, NamedTuple, Optional

#: Rounds x gap, from the registered design. The last request fires at
#: `ROUNDS_MINUS_ONE * GAP` minutes after the start, so a fixture must still be
#: unkicked then to contribute three instants.
H1_ROUNDS = 3
H1_GAP_MINUTES = 120
BAND_LO_MIN = (H1_ROUNDS - 1) * H1_GAP_MINUTES          # 240
BAND_HI_MIN = BAND_LO_MIN + H1_GAP_MINUTES              # 360

#: Terminal states. `NO_CARD` is DISTINCT from `nothing to do`, which is what
#: the old log said for both "the card is empty" and "the card has nothing in
#: the band" and "every league was skipped for its own reason".
#: ── `OK` BECAME `READY`. 2026-10-28. ─────────────────────────────────────
#: The printed value is what a reader of the log sees, and "OK" says the
#: CHECK succeeded while `READY` says the COLLECTION may proceed - which is
#: the fact the state carries. The other three already name their own fact.
STATE_READY = "READY"
STATE_NO_CARD = "NO_CARD"
STATE_NO_FIXTURE_IN_BAND = "NO_FIXTURE_IN_BAND"
STATE_UNEVALUATED = "UNEVALUATED"

PRECONDITION_STATES = (STATE_READY, STATE_NO_CARD, STATE_NO_FIXTURE_IN_BAND,
                       STATE_UNEVALUATED)


class Precondition(NamedTuple):
    state: str
    card_total: int               #: rows for today+tomorrow, any is_fixture
    card_as_fixture: int          #: of those, is_fixture = true
    in_band: int                  #: is_fixture AND kickoff in (T+240, T+360]
    in_band_pinned: int           #: ...and in one of the pinned leagues
    band: tuple
    leagues: tuple
    reason: str

    @property
    def can_collect(self) -> bool:
        return self.state == STATE_READY


def evaluate(session, leagues, now: Optional[_dt.datetime] = None,
             band=(BAND_LO_MIN, BAND_HI_MIN)) -> Precondition:
    """Is there anything for H1 to collect? Counted BEFORE any request.

    Read-only, and it spends nothing: the point is to distinguish "the
    provider had nothing" from "we had nothing to ask about" without paying a
    credit to find out.
    """
    from sqlalchemy import text

    now = now or _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)
    lo = now + _dt.timedelta(minutes=band[0])
    hi = now + _dt.timedelta(minutes=band[1])
    day_lo = now.replace(hour=0, minute=0, second=0, microsecond=0)
    day_hi = day_lo + _dt.timedelta(days=2)
    lg = list(leagues)

    try:
        total = session.execute(text(
            "SELECT count(*) FROM matches "
            "WHERE match_date >= :a AND match_date < :b"),
            {"a": day_lo, "b": day_hi}).scalar() or 0
        as_fix = session.execute(text(
            "SELECT count(*) FROM matches "
            "WHERE match_date >= :a AND match_date < :b AND is_fixture"),
            {"a": day_lo, "b": day_hi}).scalar() or 0
        band_n = session.execute(text(
            "SELECT count(*) FROM matches "
            "WHERE is_fixture AND match_date > :lo AND match_date <= :hi"),
            {"lo": lo, "hi": hi}).scalar() or 0
        band_pinned = session.execute(text(
            "SELECT count(*) FROM matches WHERE is_fixture "
            "AND match_date > :lo AND match_date <= :hi "
            "AND league = ANY(:lg)"),
            {"lo": lo, "hi": hi, "lg": lg}).scalar() or 0
    except Exception as exc:                        # noqa: BLE001
        return Precondition(STATE_UNEVALUATED, -1, -1, -1, -1, band,
                            tuple(lg),
                            f"the precondition could not be evaluated "
                            f"({type(exc).__name__}: {str(exc).splitlines()[0][:90]}). "
                            f"That is NOT a clean precondition.")

    if as_fix == 0:
        state, why = STATE_NO_CARD, (
            f"NO CARD: {total} row(s) exist for today+tomorrow and {as_fix} "
            f"are fixtures. `matches` is written by `daily-picks --update`; "
            f"if that has not run, H1 has nothing to sample. This is NOT "
            f"'nothing to do'.")
    elif not lg:
        # ── "NOT PINNED" MEANS ALL, NOT NONE. 2026-10-28. ─────────────────
        #
        # `league = ANY(:lg)` with an empty list matches NOTHING, so an
        # unpinned run scored `band_pinned = 0` and reported
        # NO_FIXTURE_IN_BAND on a card full of in-band fixtures. The loop
        # prints "pinned leagues: ALL (not pinned)" for that case, so the two
        # disagreed about what an empty list meant.
        #
        # An empty selection is not an empty set of fixtures -- the same rule
        # the reminders classifier needed, where `0 == 0` satisfied an
        # equality and an empty selection had to be checked FIRST.
        band_pinned = band_n
        state, why = ((STATE_READY, f"{band_n} fixture(s) in "
                       f"({band[0]},{band[1]}] minutes across ALL leagues "
                       f"(none pinned), of {as_fix} on the card")
                      if band_n else
                      (STATE_NO_FIXTURE_IN_BAND,
                       f"NO FIXTURE IN BAND: {as_fix} fixture(s) on the card "
                       f"and {band_n} in ({band[0]},{band[1]}] minutes, with "
                       f"no league pinned so ALL were considered. A fixture "
                       f"nearer than {band[0]} min cannot carry {H1_ROUNDS} "
                       f"instants at {H1_GAP_MINUTES}-minute gaps."))
    elif band_pinned == 0:
        state, why = STATE_NO_FIXTURE_IN_BAND, (
            f"NO FIXTURE IN BAND: {as_fix} fixture(s) on the card, {band_n} "
            f"anywhere in ({band[0]},{band[1]}] minutes, {band_pinned} of "
            f"those in the {len(lg)} pinned league(s). A fixture nearer than "
            f"{band[0]} min cannot carry {H1_ROUNDS} instants at "
            f"{H1_GAP_MINUTES}-minute gaps, so counting the WINDOW overstates "
            f"what is collectable.")
    else:
        state, why = STATE_READY, (
            f"{band_pinned} pinned fixture(s) in ({band[0]},{band[1]}] "
            f"minutes, of {band_n} in band and {as_fix} on the card")
    return Precondition(state, int(total), int(as_fix), int(band_n),
                        int(band_pinned), band, tuple(lg), why)


def format_precondition(p: Precondition) -> str:
    head = f"H1_PRECONDITION state={p.state}"
    return (f"{head}{chr(10)}  {p.reason}{chr(10)}"
            f"  card={p.card_total} fixtures={p.card_as_fixture} "
            f"in_band={p.in_band} in_band_pinned={p.in_band_pinned} "
            f"band=({p.band[0]},{p.band[1]}]m leagues={len(p.leagues)}")


# ── THE SILENT ZERO, ALARMED ─────────────────────────────────────────────────

class ZeroDay(NamedTuple):
    day: _dt.date
    card_as_fixture: int
    trajectories: int
    state: str
    reason: str


#: Same shape as `pipeline_loss_alert`: N chosen before the firing rate was
#: known, because six identical messages there produced no action.
ESCALATE_NOT_SELF_CORRECTING = 2
ESCALATE_WINDOW_AT_RISK = 3


def zero_collection_streak(rows: List[ZeroDay]) -> List[ZeroDay]:
    """Trailing days that HAD a card and collected nothing.

    A `NO_CARD` day still counts: not having a card is the failure, and it was
    the actual failure on 10-13..10-15. A day with no fixtures anywhere is not
    H1's problem and ends the streak.
    """
    out: List[ZeroDay] = []
    for r in reversed(rows):
        if r.trajectories > 0:
            break
        if r.card_as_fixture == 0 and r.state != STATE_NO_CARD:
            break
        out.append(r)
    return list(reversed(out))


def format_zero_alarm(streak: List[ZeroDay], target_n: int, floor_n: int,
                      collected_n: int, credits_spent: int,
                      credits_reserved: int, run_url: str = "") -> str:
    """The alarm H1 did not have. The pipeline got one on 2026-10-18."""
    n = len(streak)
    if n >= ESCALATE_WINDOW_AT_RISK:
        head = (f"!!! H1 COLLECTED NOTHING FOR {n} DAYS - THE WINDOW IS AT "
                f"RISK. n={collected_n} of {target_n}, floor {floor_n}.")
    elif n >= ESCALATE_NOT_SELF_CORRECTING:
        head = (f"!! H1 COLLECTED NOTHING FOR {n} DAYS - NOT SELF-CORRECTING. "
                f"n={collected_n} of {target_n}, floor {floor_n}.")
    else:
        head = (f"H1 COLLECTED NOTHING TODAY. n={collected_n} of {target_n}, "
                f"floor {floor_n}.")

    lines = [head, "",
             "A COLLECTION DAY IS NOT RECOVERABLE: the price path it would "
             "have sampled is gone once the fixtures kick off, and the window "
             "has a fixed number of days.", "",
             "PER DAY:"]
    for r in streak:
        lines.append(f"  {r.day}  fixtures={r.card_as_fixture:4d}  "
                     f"trajectories={r.trajectories:3d}  {r.state}")
        lines.append(f"      {r.reason[:110]}")
    lines += [
        "",
        f"CREDITS: {credits_spent} spent of {credits_reserved} reserved "
        f"(TheOddsAPI) - a zero here means it never asked, not that it was "
        f"refused.",
        "",
        "CLEARING ACTION, and it is the OPERATOR'S:",
        "  if state=NO_CARD      dispatch daily-picks first; H1 samples the "
        "card that writes, and it has not been written",
        "  if NO_FIXTURE_IN_BAND nothing to do today - the card carries no "
        "fixture 4-6 hours out. Re-select the pinned leagues by BAND density, "
        "not by card size.",
        "  either way              H1 spends nothing while this holds, so the "
        "cost of waiting is days of the window, not credits.",
    ]
    if run_url:
        lines.append(f"Logs: {run_url}")
    return chr(10).join(lines)


def project_from_proxy(session) -> dict:
    """The GO/NO-GO, run against the proxy week. Spends nothing.

    Wired into `main --project` so the verdict is a COMMAND rather than a
    figure someone once computed: "no number without provenance" is not
    satisfied by a number I cannot re-run.
    """
    from sqlalchemy import text

    from scripts.h1_collection_check import TARGET_N
    from src.data.odds_quota import (CREDITS_PER_REQUEST,
                                     H1_RESERVATION_FROM,
                                     H1_RESERVATION_UNTIL,
                                     H1_RESERVED_CREDITS)

    days = (H1_RESERVATION_UNTIL - H1_RESERVATION_FROM).days
    a, b = PROXY_WEEK
    rows = [(r.d, r.league, r.ko.hour * 60 + r.ko.minute)
            for r in session.execute(text(
                "SELECT match_date::date d, league, match_date::time ko "
                "FROM matches WHERE match_date >= :a AND match_date < :b "
                "AND training_exclusion_reason IS NULL"), {"a": a, "b": b})]
    if not rows:
        return {"state": STATE_UNEVALUATED,
                "why": f"the proxy week {a}..{b} holds no match row, so the "
                       f"projection could not be computed -- this is NOT n=0"}
    per_day = leagues_per_day(H1_RESERVED_CREDITS, CREDITS_PER_REQUEST, days)
    sweep = {}
    for T in range(6 * 60, 15 * 60 + 1, 30):
        sweep[T] = project_band_density(rows, T, per_day)
    # RANKED ON (dates, n) AND IN THAT ORDER. Spread is a CONSTRAINT, not a
    # tie-break: rho = 0.42 clusters on fixture, and a big n on two dates
    # makes DATE the cluster and voids the sizing that produced 33-39.
    best_T = max(sweep, key=lambda T: (sweep[T]["dates"], sweep[T]["n"]))
    r = sweep[best_T]
    return {"state": STATE_READY if (r["n"] >= 33 and r["dates"] >= 5)
            else STATE_NO_FIXTURE_IN_BAND,
            "verdict": "GO" if (r["n"] >= 33 and r["dates"] >= 5) else "NO-GO",
            "proxy": f"{a}..{b}", "provenance": PROXY_PROVENANCE,
            "best_start_min": best_T, "n": r["n"], "dates": r["dates"],
            "days": days, "leagues_per_day": per_day, "target_n": TARGET_N,
            "floor": 33, "by_date": r["by_date"],
            "sweep": {f"{T // 60:02d}:{T % 60:02d}":
                      (v["n"], v["dates"]) for T, v in sweep.items()}}


def main() -> int:
    """Report the precondition. Spends nothing, writes nothing."""
    import re
    for path in (".env", ".env.local"):
        try:
            for line in open(path, encoding="utf-8"):
                m = re.match(r"\s*([A-Z0-9_]+)\s*=\s*(.*)", line)
                if m and not _os.environ.get(m.group(1)):
                    _os.environ[m.group(1)] = (
                        m.group(2).strip().strip('"').strip("'"))
        except FileNotFoundError:
            continue
    if "--project" in _sys_argv():
        from src.data.database import get_db
        with get_db().get_session() as _s:
            r = project_from_proxy(_s)
        print("H1 GO/NO-GO from the labelled proxy")
        print(f"  {r.get('provenance', '')}")
        if r["state"] == STATE_UNEVALUATED:
            print(f"  UNEVALUATED: {r['why']}")
            return 1
        print(f"  band (T+{BAND_LO_MIN}, T+{BAND_HI_MIN}]  "
              f"leagues/day {r['leagues_per_day']}  days {r['days']}")
        for t, (n, d) in r["sweep"].items():
            print(f"    T={t}  n={n:3d}  on {d} date(s)")
        print(f"  BEST T={r['best_start_min'] // 60:02d}:"
              f"{r['best_start_min'] % 60:02d}  n={r['n']}  "
              f"spread {r['dates']} date(s)")
        print(f"  VERDICT: {r['verdict']}  (floor {r['floor']}, "
              f"TARGET_N {r['target_n']})")
        return 0
    leagues = [l for l in (_os.environ.get("H1_LEAGUES") or "").split(",") if l]
    if not leagues:
        leagues = ["england/championship", "europe/champions-league",
                   "europe/europa-league",
                   "europe/europa-conference-league"]
    from sqlalchemy import text

    from src.data.database import get_db
    db = get_db()
    with db.get_session() as s:
        p = evaluate(s, leagues)
        print(format_precondition(p))
        print(f"  COLLECT? {'YES' if p.can_collect else 'NO'} - a collection "
              f"round started against a failing precondition spends credits "
              f"to learn what this check establishes for free.")

        # ── AND THE SILENT ZERO, WIRED. ────────────────────────────────────
        #
        # The alarm existed only in its own tests when it was written, which
        # is PNC-1 in the week the detector for PNC-1 was built - caught by
        # that detector, in `scripts/pnc_scan.py --list`, before this shipped.
        # So it has a consumer: this entry point, which a person runs and the
        # H1 workflow can call before collecting.
        rows: List[ZeroDay] = []
        today = _dt.date.today()
        for back in range(6, -1, -1):
            d = today - _dt.timedelta(days=back)
            nxt = d + _dt.timedelta(days=1)
            fx = s.execute(text(
                "SELECT count(*) FROM matches WHERE match_date >= :a "
                "AND match_date < :b AND is_fixture"),
                {"a": d, "b": nxt}).scalar() or 0
            traj = s.execute(text(
                "SELECT count(*) FROM ("
                "  SELECT match_id FROM odds_snapshots "
                "  WHERE observed_at >= :a AND observed_at < :b "
                "  GROUP BY match_id HAVING count(DISTINCT observed_at) >= 3"
                ") t"), {"a": d, "b": nxt}).scalar() or 0
            state = STATE_READY if traj else (
                STATE_NO_CARD if fx == 0 else STATE_NO_FIXTURE_IN_BAND)
            rows.append(ZeroDay(d, int(fx), int(traj), state,
                                p.reason if d == today else
                                "historical; the reason was not captured"))

    streak = zero_collection_streak(rows)
    if streak:
        print()
        print(format_zero_alarm(streak, target_n=39, floor_n=33,
                                collected_n=sum(r.trajectories for r in rows),
                                credits_spent=6, credits_reserved=168,
                                run_url=_os.environ.get("H1_RUN_URL", "")))
    else:
        print("h1_zero_collection[0 consecutive days with a card and no "
              "trajectories]")
    return 0        # reporting only; never gates

# ── THE GUARD MOVED TO THE END. 2026-10-28. ──────────────────────────
#
# The block below was appended AFTER `if __name__ == "__main__"`, so
# `raise SystemExit(main())` executed before these definitions existed
# and `--project` died on `NameError: PROXY_WEEK`. Module-level code
# runs top to bottom; an entry-point guard in the MIDDLE of a file
# makes everything after it invisible to its own main().


# ══════════ GO/NO-GO FROM A LABELLED PROXY — PART C, 2026-10-28 ═════════════
#
# THE FORWARD FIXTURE HORIZON IS ZERO DAYS. `measured` 2026-10-28: no match
# row exists beyond today, so band density for 2026-11-01 -> 11-08 CANNOT be
# measured from the scraper. The same calendar position in the previous season
# can: 2025-11-01..11-09 holds 291 matches over 23 leagues and 8 dates.
#
#     IT IS A PROXY, IT IS LABELLED ONE, AND IT IS FOR GO/NO-GO ONLY.
#
# It answers exactly one question - is there enough card inside the band, on
# enough separate dates, to reach n=33 under the reservation - and it does NOT
# choose leagues. The selection is a RULE EVALUATED ON THE DAY from the live
# card, because the fixtures it would rank do not exist yet.
#
# THE PROXY'S OWN ERROR, stated rather than implied:
#
#   * THE FIXTURE LIST DIFFERS. Promotions, relegations, cup rounds and
#     rescheduling mean no league's 2026 card is last season's card.
#   * INTERNATIONAL WINDOWS MOVE. November carries a FIFA window; if any of
#     11-01..11-08 falls inside it the club card thins sharply, and the proxy
#     week (which did not) would overstate n. This is the largest single risk
#     and it is checkable on the day by the precondition, not now.
#   * COMPETITION PHASE DIFFERS. European group stages fall on different
#     midweeks, and a midweek round is the densest thing in the band.
#
# So the projection below is a FLOOR-CLEARING ARGUMENT, not a plan. If the
# live card on 11-01 is thinner than the proxy, the precondition reports
# `NO_FIXTURE_IN_BAND` that morning and the daily projection carries it to the
# 11-04 rate decision.

#: `measured` 2026-10-28 from the proxy week. Rounds x leagues under the
#: reservation: 168 credits / 2 = 84 requests over 8 days = 10.5/day, and at
#: H1_ROUNDS rounds that prices 3 leagues a day.
#: ── THE FIFA WINDOW, CHECKED 2026-10-28 AND NOT DEFERRED ─────────────────
#:
#: It was the proxy's largest stated risk, it is PUBLISHED EXTERNALLY, and a
#: published fact is checked when it is knowable rather than when it binds.
#: Not inferred from our own fixture table, which has a ZERO-DAY forward
#: horizon and could not answer.
#:
#:   men's window, November 2026   9-17 November 2026
#:       source: en.wikipedia.org/wiki/FIFA_International_Match_Calendar,
#:       fetched 2026-10-28 ("9-17 November 2026", 2 matches); corroborated
#:       by beinsports.com and the CONCACAF Nations League League A
#:       quarter-finals, dated 9-17 November 2026.
#:
#:   men's window, November 2025   10-18 November 2025
#:       source: search consensus 2026-10-28 across fan/aggregator sites, so
#:       LOWER CONFIDENCE than the 2026 figure and labelled as such. It is
#:       used only to establish the proxy's comparability, not any projection.
#:
#: THE VERDICT ON THE EIGHT DATES: 2026-11-01..11-08 is entirely OUTSIDE the
#: window, which opens 11-09. The margin is ONE DAY, and 11-09 is also the
#: reservation's exclusive end date -- the collection sits exactly in the
#: club-football gap before the break.
#:
#: THAT IS LUCK AND IS RECORDED AS LUCK. The window was derived from the
#: credit reset plus the pre-registered eight days; nobody checked the FIFA
#: calendar when choosing it. One day later on either count and the last
#: collection days would have fallen into the break.
#:
#: AND IT STRENGTHENS THE PROXY RATHER THAN MERELY CLEARING THE RISK. The
#: proxy week 2025-11-01..11-08 sits in the SAME POSITION RELATIVE TO THE
#: BREAK -- the eight club days immediately before a November international
#: window (2025's opened 11-10). The two weeks are aligned on this axis, not
#: just on the calendar.
#:
#: WHAT IS STILL OPEN: the window's edge effects. Clubs release players before
#: it opens, so the final pre-break weekend can carry a thinner or
#: rescheduled card in some leagues. The precondition checks the live card on
#: the day; this check buys the ability to act in advance, and does not
#: replace it.
#: ── THE START TIME, DERIVED FROM BOTH ENDS. 2026-10-28. ─────────────────
#:
#: T = 14:00 UTC. Rounds fire at 14:00 / 16:00 / 18:00 and the band is
#: (18:00, 20:00].
#:
#: LATE END - the card must EXIST before T. `measured` 2026-10-28 over the
#: NEW-CRON regime only (`0 1` weekday / `30 0` weekend, runs 10-23..10-28,
#: n=6): lateness mean 281.6 max 325.5 min; duration mean 41.2 max 65.2 min.
#:
#:     weekday  card-ready  max 07:30   CLEARANCE +389 min
#:              card-ready  m+3sd 08:02 CLEARANCE +358 min
#:     weekend  card-ready  max 07:00   CLEARANCE +419 min
#:              card-ready  m+3sd 07:32 CLEARANCE +388 min
#:
#: The duration term is run END, not time-to-picks, so every figure above is
#: CONSERVATIVE: the card exists before the run finishes. Regimes are not
#: pooled - the pre-10-22 runs sat under three different crons and mixing them
#: gave a 128-870 minute spread that supported nothing.
#:
#: EARLY END - the band must land on REAL kickoffs. A fixture kicking off
#: before the LAST round yields no trajectory, so the binding quantity is the
#: gap between the last round (18:00) and the earliest in-band kickoff.
#: `measured` on the proxy week: earliest in-band kickoff 19:00, latest 20:00,
#: 81 in-band fixtures.
#:
#:     CLEARANCE from the last round  +60 min
#:
#: SO THE EARLY END BINDS, AT +60 MINUTES, and the late end has 358 to spare.
#: A fixture would have to be rescheduled an hour EARLIER to be lost. Moving T
#: later does not buy early-end margin, because the band moves with it.
H1_START_MIN_UTC = 14 * 60
H1_CLEARANCE_LATE_MIN = 358      #: weekday, mean+3sd, the tighter statistic
H1_CLEARANCE_EARLY_MIN = 60      #: last round -> earliest in-band kickoff
H1_START_PROVENANCE = (
    "measured 2026-10-28: late end from the new-cron regime n=6 "
    "(lateness mean 281.6 max 325.5; duration mean 41.2 max 65.2, run END so "
    "conservative); early end from the proxy week's earliest in-band kickoff "
    "at 19:00 against a last round at 18:00")

#: ── OVERLAP: "NONE" IS NOT ACHIEVABLE, SO THE COST IS MEASURED INSTEAD ───
#:
#: `measured` 2026-10-28 over the new-cron regime, occupancy against
#: 14:00-18:00: `paper-trading-report` ran 14:02-16:49 (n=5), `closing-lines`
#: 00:31-23:50 (n=22), `ci-audit` 00:32-23:52 (n=40), `reminders` 05:16-19:06
#: (n=13). Scheduled `daily-picks` ends by 07:05, but a DISPATCHED one ended
#: at 13:57 - three minutes before T.
#:
#: GitHub fires this repo's crons 3-5 HOURS LATE, so no choice of T can
#: guarantee non-overlap, and every workflow declares its own concurrency
#: group while NOTHING GUARDS THE SET. So the question is what overlap costs:
#:
#:   CREDIT CONTENTION - SOLVED, by the two-sided allocation shipped
#:       2026-10-28. H1 cannot take normal pricing's share and normal pricing
#:       cannot take H1's, in either order, with nothing stranded.
#:
#:   OBSERVATION CONTAMINATION - BOUNDED, and the grain is the whole answer.
#:       At FIXTURE-DAY grain the in-band data looked alarming: 67-103
#:       distinct instants per fixture-day. At SERIES grain - the unit
#:       `qualifying_triple` actually sees - all 1,939 in-band series-days
#:       carry EXACTLY ONE instant and ZERO carry three. An existing writer
#:       therefore cannot manufacture a trajectory.
#:
#:       THE RESIDUAL, in the inflating direction: if H1 completes only 2 of
#:       3 rounds, one foreign instant could supply the third and the triple
#:       would count. Narrow, measured at one foreign instant per series-day,
#:       and recorded rather than closed.
#:
#:   RUNNER CONTENTION - not a correctness question; GitHub runs jobs in
#:       parallel.
H1_OVERLAP_PROVENANCE = (
    "measured 2026-10-28; credit contention closed by the allocation, "
    "observation contamination bounded at series grain (0 of 1,939 in-band "
    "series-days carry 3 instants), residual is the 2-of-3 completion path")

FIFA_WINDOW_NOV_2026 = ("2026-11-09", "2026-11-17")
FIFA_WINDOW_NOV_2025 = ("2025-11-10", "2025-11-18")
FIFA_WINDOW_SOURCE = (
    "en.wikipedia.org/wiki/FIFA_International_Match_Calendar, fetched "
    "2026-10-28; corroborated by beinsports.com. The 2025 dates are search "
    "consensus only and carry lower confidence.")

PROXY_WEEK = ("2025-11-01", "2025-11-09")
PROXY_PROVENANCE = (
    "measured 2026-10-28 over 291 matches / 23 leagues / 8 dates in "
    "2025-11-01..2025-11-09, the same calendar position one season earlier; "
    "a PROXY for go/no-go only, never a plan")


def leagues_per_day(reserved_credits: int, credits_per_request: int,
                    days: int, rounds: int = H1_ROUNDS) -> int:
    """How many leagues one day may price, under the reservation.

    Every round re-prices the SAME pinned leagues -- a trajectory is three
    observations of one fixture -- so the cost of a day is
    `rounds * leagues * credits_per_request`, not `leagues * credits`.
    """
    if days <= 0 or rounds <= 0 or credits_per_request <= 0:
        return 0
    return int(((reserved_credits // credits_per_request) / days) // rounds)


def project_band_density(rows, start_minute: int, per_day: int,
                         lo: int = BAND_LO_MIN, hi: int = BAND_HI_MIN) -> dict:
    """`{'n', 'dates', 'by_date'}` for a job starting at `start_minute`.

    `rows` are `(date, league, kickoff_minute)`. THE UNIT IS THE FIXTURE: `n`
    counts fixtures inside the band in the `per_day` densest leagues, summed
    over the dates. A league request prices every fixture in that league at
    one instant, which is why ranking by fixtures-in-band is the rule.

    SPREAD IS REPORTED BESIDE `n` AND IS NOT OPTIONAL. rho = 0.42 clusters on
    FIXTURE; if every fixture comes from one date then date becomes the
    cluster and the sizing that produced 33-39 stops applying. A projection
    with a large `n` on two dates is a worse plan than a smaller one on eight.
    """
    by_date: dict = {}
    for d, lg, km in rows:
        if start_minute + lo < km <= start_minute + hi:
            by_date.setdefault(d, {})
            by_date[d][lg] = by_date[d].get(lg, 0) + 1
    out, dates = 0, 0
    detail = {}
    for d, per_lg in sorted(by_date.items()):
        top = sorted(per_lg.items(), key=lambda kv: -kv[1])[:per_day]
        got = sum(n for _, n in top)
        if got:
            dates += 1
            out += got
            detail[str(d)] = {"n": got, "leagues": [lg for lg, _ in top]}
    return {"n": out, "dates": dates, "by_date": detail}

if __name__ == "__main__":
    raise SystemExit(main())
