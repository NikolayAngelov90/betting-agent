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
STATE_OK = "OK"
STATE_NO_CARD = "NO_CARD"
STATE_NO_FIXTURE_IN_BAND = "NO_FIXTURE_IN_BAND"
STATE_UNEVALUATED = "UNEVALUATED"

PRECONDITION_STATES = (STATE_OK, STATE_NO_CARD, STATE_NO_FIXTURE_IN_BAND,
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
        return self.state == STATE_OK


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
    elif band_pinned == 0:
        state, why = STATE_NO_FIXTURE_IN_BAND, (
            f"NO FIXTURE IN BAND: {as_fix} fixture(s) on the card, {band_n} "
            f"anywhere in ({band[0]},{band[1]}] minutes, {band_pinned} of "
            f"those in the {len(lg)} pinned league(s). A fixture nearer than "
            f"{band[0]} min cannot carry {H1_ROUNDS} instants at "
            f"{H1_GAP_MINUTES}-minute gaps, so counting the WINDOW overstates "
            f"what is collectable.")
    else:
        state, why = STATE_OK, (
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
            state = STATE_OK if traj else (
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


if __name__ == "__main__":
    raise SystemExit(main())
