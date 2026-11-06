"""The cron move of 2026-10-22, and what it cost — `s5.19`.

weekday `0 0` -> `0 1`, weekend `0 0` -> `30 0`.

WHY, AND IT IS NOT "LATER IS BETTER". Niki asked for a ~11:00 Sofia start. The
early start he reported IS the bimodality of the scheduler delay: the cron was
sized for a SLOW cluster and a FAST one has been running since 2026-10-19.

THE SLOW CLUSTER GOVERNS, because nothing explains the gap and nothing
licenses assuming the fast one persists. The optimistic cluster is reported
here and relied on nowhere.

`0 2` WAS REJECTED, and that refusal is the rule this file pins hardest: a
configuration the project's own instrument calls a breach is not shipped.
"""

import datetime as dt

import pytest

import scripts.ci_audit as ci
import scripts.schedule_deadline as sd

SUMMER = dt.date(2026, 10, 24)      # UTC+3
WINTER = dt.date(2026, 10, 26)      # UTC+2

#: The delay clusters, `measured` 2026-10-22 under the RETIRED `0 0` cron.
#: Recorded because the move EMPTIES the live series -- membership keys on the
#: run's own `headSha` carrying the cron in force -- so this derivation cannot
#: be re-run from live data until the new crons have fired. See
#: `test_the_SERIES_RESET_is_recorded_as_a_COST`.
CLUSTERS = {
    ("weekday", "slow"): {"n": 6, "mean": 276.3, "sd": 18.8, "max": 304.9,
                          "mean_plus_3sd": 332.6},
    ("weekday", "fast"): {"n": 4, "mean": 135.6, "sd": 6.9, "max": 144.1,
                          "mean_plus_3sd": 156.2},
    ("weekend", "all"): {"n": 6, "mean": 253.2, "sd": 29.7, "max": 293.7,
                         "mean_plus_3sd": 342.3},
}

EXPECTED_CRON = {"weekday": 60, "weekend": 30}


def hm(m):
    m = int(round(m))
    return f"{m // 60:02d}:{m % 60:02d}"


def local(m, day):
    return hm(m + sd.operator_offset_min(day))


# ── the cron itself ───────────────────────────────────────────────────────

def test_the_CRONS_ARE_WHAT_WAS_DERIVED():
    crons = ci.current_crons_by_daytype()
    assert crons == EXPECTED_CRON, crons


def test_the_CRON_is_the_LATEST_that_clears_60_on_BOTH_pairings():
    """Re-derived rather than quoted. The margin takes the WORSE of (max,
    picks_max) and (mean+3sd, picks_3sd), so both must clear."""
    import math
    for dtype in ("weekday", "weekend"):
        c = CLUSTERS.get((dtype, "slow")) or CLUSTERS[(dtype, "all")]
        floor = sd.observed_earliest_min(dtype, SUMMER)
        cap = min(
            floor - sd.MIN_ACCEPTABLE_MARGIN_MIN - c["max"]
            - ci.FULL_CARD_PICKS_MAX,
            floor - sd.MIN_ACCEPTABLE_MARGIN_MIN - c["mean_plus_3sd"]
            - ci.FULL_CARD_PICKS_3SD)
        assert EXPECTED_CRON[dtype] == int(math.floor(cap / 15) * 15), (
            dtype, cap)
        # and the NEXT boundary up must FAIL, or it is not the latest
        assert floor - (EXPECTED_CRON[dtype] + 15 + c["mean_plus_3sd"]
                        + ci.FULL_CARD_PICKS_3SD) < \
            sd.MIN_ACCEPTABLE_MARGIN_MIN, (
            f"{dtype}: a cron 15 minutes later also clears, so this is not "
            f"the latest")


def test_SUMMER_BINDS_because_one_UTC_cron_serves_both_seasons():
    """The localised floor is an hour earlier in summer, so summer sets the
    cron and winter inherits the extra hour. Deriving from winter would ship
    a cron that breaches for seven months of the year."""
    for dtype in ("weekday", "weekend"):
        assert sd.observed_earliest_min(dtype, SUMMER) < \
            sd.observed_earliest_min(dtype, WINTER)


# ── THE FOUR START TIMES, both clocks, per cluster ───────────────────────

#: `simulated` from the measured clusters: start = cron + delay. Both clocks,
#: and per CLUSTER because an average across a bimodal series describes
#: neither half.
#: CORRECTED while writing this file: the first table was computed at the
#: REJECTED `0 2` / `15 1` and its UTC column was an hour and a quarter early.
#: The shipped crons are `0 1` / `30 0`.
EXPECTED_STARTS = {
    ("weekday", "slow", "mean_plus_3sd"): ("06:33", "09:33", "08:33"),
    ("weekday", "slow", "mean"): ("05:36", "08:36", "07:36"),
    ("weekday", "fast", "mean_plus_3sd"): ("03:36", "06:36", "05:36"),
    ("weekday", "fast", "mean"): ("03:16", "06:16", "05:16"),
    ("weekend", "all", "mean_plus_3sd"): ("06:12", "09:12", "08:12"),
    ("weekend", "all", "mean"): ("04:43", "07:43", "06:43"),
}


@pytest.mark.parametrize("key,expect", sorted(EXPECTED_STARTS.items()))
def test_the_START_TIME_under_each_cluster_in_BOTH_CLOCKS(key, expect):
    dtype, cl, stat = key
    utc, loc_s, loc_w = expect
    start = EXPECTED_CRON[dtype] + CLUSTERS[(dtype, cl)][stat]
    assert hm(start) == utc, (key, hm(start))
    assert local(start, SUMMER) == loc_s, (key, local(start, SUMMER))
    assert local(start, WINTER) == loc_w, (key, local(start, WINTER))


def test_the_ANSWER_TO_ELEVEN_is_PLUS_ONE_HOUR_not_four():
    """Stated as the shift rather than as a target, because the brief asked
    for the number and not for steering to it.

    A single UTC cron cannot deliver a fixed LOCAL start while the delay is
    bimodal: the slow case lands at 09:33 Sofia and the fast case at 07:16.
    """
    assert EXPECTED_CRON["weekday"] - 0 == 60
    assert EXPECTED_CRON["weekend"] - 0 == 30
    slow = EXPECTED_CRON["weekday"] + CLUSTERS[("weekday", "slow")]["mean_plus_3sd"]
    fast = EXPECTED_CRON["weekday"] + CLUSTERS[("weekday", "fast")]["mean"]
    assert local(slow, SUMMER) == "09:33"
    assert local(fast, SUMMER) == "06:16"
    assert local(slow, SUMMER) != "11:00", (
        "if this ever equals 11:00 the derivation changed; re-read the entry "
        "rather than assuming the request was met")


# ── THE MARGINS, at both floors ──────────────────────────────────────────

#: `simulated` from the measured clusters at the localised floors.
EXPECTED_MARGINS = {
    ("weekday", "summer"): (117.0, 70.4),
    ("weekend", "summer"): (128.2, 60.7),
    ("weekday", "winter"): (177.0, 130.4),
    ("weekend", "winter"): (188.2, 120.7),
}


@pytest.mark.parametrize("key,expect", sorted(EXPECTED_MARGINS.items()))
def test_the_MARGIN_at_BOTH_FLOORS_clears_the_registered_minimum(key, expect):
    dtype, season = key
    day = SUMMER if season == "summer" else WINTER
    c = CLUSTERS.get((dtype, "slow")) or CLUSTERS[(dtype, "all")]
    floor = sd.observed_earliest_min(dtype, day)
    m_max = floor - (EXPECTED_CRON[dtype] + c["max"] + ci.FULL_CARD_PICKS_MAX)
    m_3sd = floor - (EXPECTED_CRON[dtype] + c["mean_plus_3sd"]
                     + ci.FULL_CARD_PICKS_3SD)
    assert m_max == pytest.approx(expect[0], abs=0.1), (key, m_max)
    assert m_3sd == pytest.approx(expect[1], abs=0.1), (key, m_3sd)
    assert min(m_max, m_3sd) >= sd.MIN_ACCEPTABLE_MARGIN_MIN, (
        f"{key} binds at {min(m_max, m_3sd):.1f}, below the registered "
        f"{sd.MIN_ACCEPTABLE_MARGIN_MIN}")


def test_the_MOVE_SPENDS_REAL_MARGIN_and_is_NOT_a_repair():
    """CORRECTED while writing this file, and the correction changes what this
    change IS.

    The first version asserted "the move is a repair, not a preference",
    on the strength of the old weekday margin reading +20.7 — below even the
    30-minute alarm. That figure is the POOLED one, and the pooled mean+3sd of
    a bimodal series measures between-cluster distance: it is the artefact
    this project recorded on 2026-10-22, not a bound.

    UNDER THE SLOW CLUSTER — the population that actually governs — the OLD
    cron had **+130.4**. It cleared the registered 60 with 70 minutes to
    spare. So:

        the move is a PREFERENCE, paid for with 60 minutes of real margin.

    That is a legitimate trade and it is Niki's to make. It is recorded here
    rather than dressed as a repair, because a change sold as a fix does not
    get re-examined and a change sold as a trade does.
    """
    c = CLUSTERS[("weekday", "slow")]
    floor = sd.observed_earliest_min("weekday", SUMMER)

    def binding(cron):
        return min(floor - (cron + c["max"] + ci.FULL_CARD_PICKS_MAX),
                   floor - (cron + c["mean_plus_3sd"]
                            + ci.FULL_CARD_PICKS_3SD))

    old, new_ = binding(0), binding(EXPECTED_CRON["weekday"])
    assert old == pytest.approx(130.4, abs=0.1), old
    assert new_ == pytest.approx(70.4, abs=0.1), new_
    assert old >= sd.MIN_ACCEPTABLE_MARGIN_MIN, (
        "the premise of this test is that the OLD cron already cleared; if it "
        "did not, the move was a repair after all and the ledger is wrong")
    assert new_ >= sd.MIN_ACCEPTABLE_MARGIN_MIN
    assert old - new_ == pytest.approx(60.0, abs=0.1), (
        "the move no longer costs exactly the hour it buys; re-derive")


def test_the_POOLED_reading_is_the_ARTEFACT_that_made_it_look_like_a_repair():
    """+20.7 pooled against +130.4 on the slow cluster, same cron, same day.

    Kept as its own test because the gap between those two numbers is the
    whole reason the per-cluster report exists.
    """
    floor = sd.observed_earliest_min("weekday", SUMMER)
    pooled = floor - (0 + 442.3 + ci.FULL_CARD_PICKS_3SD)
    assert pooled == pytest.approx(20.7, abs=0.3), pooled
    assert pooled < ci.MARGIN_ALARM_MIN
    slow = floor - (0 + CLUSTERS[("weekday", "slow")]["mean_plus_3sd"]
                    + ci.FULL_CARD_PICKS_3SD)
    assert slow == pytest.approx(130.4, abs=0.1)
    assert slow - pooled == pytest.approx(109.7, abs=0.5), (
        "the pooled-versus-slow gap moved; it is the size of the artefact")


# ── `0 2` REJECTED, and the rule behind the refusal ──────────────────────

#: The brief's own derivation, from the PARALLEL picks bound.
PARALLEL_PICKS_BOUND = 55.1          # simulated n=72; MEASURED n=1 at 35.1
REJECTED_CRON = {"weekday": 120, "weekend": 75}


def test_the_REJECTED_CRON_is_what_the_PARALLEL_bound_would_allow():
    """It is not a strawman: `0 2` genuinely clears 60 at the parallel bound,
    and it would put the slow case at 10:33 Sofia — nearer what was asked."""
    floor = sd.observed_earliest_min("weekday", SUMMER)
    c = CLUSTERS[("weekday", "slow")]
    assert floor - (REJECTED_CRON["weekday"] + c["mean_plus_3sd"]
                    + PARALLEL_PICKS_BOUND) >= sd.MIN_ACCEPTABLE_MARGIN_MIN
    start = REJECTED_CRON["weekday"] + c["mean_plus_3sd"]
    assert local(start, SUMMER) == "10:33"


def test_the_REJECTED_CRON_is_a_BREACH_by_the_projects_OWN_instrument():
    """THE REASON IT IS NOT SHIPPED. At `0 2` the margin computed with the
    picks constants this project still RETAINS reads +10.4 — under the
    30-minute alarm, never mind the 60-minute minimum. Those constants are
    retained precisely because the parallel bound is n=1.
    """
    floor = sd.observed_earliest_min("weekday", SUMMER)
    c = CLUSTERS[("weekday", "slow")]
    retained = floor - (REJECTED_CRON["weekday"] + c["mean_plus_3sd"]
                        + ci.FULL_CARD_PICKS_3SD)
    assert retained == pytest.approx(10.4, abs=0.1), retained
    assert retained < ci.MARGIN_ALARM_MIN, (
        "the rejection's premise is gone: `0 2` no longer breaches, so "
        "re-open the decision rather than leaving it refused")
    assert ci.FULL_CARD_PICKS_3SD == 107.0, (
        "the retained picks constant moved; the rejection was argued against "
        "107.0 and must be re-argued")


@pytest.mark.live
def test_the_PARALLEL_bound_carries_its_n_which_is_ONE():
    """The weakest input in the whole derivation, stated rather than buried."""
    s = ci.parallel_regime_picks_series()
    # ── IT FIRES AT n=10, NOT AT n=2. ────────────────────────────────────
    #
    # The first version asserted `n in (None, 0, 1)` and failed the next day
    # at n=2 — a reminder reporting that one more run had happened, which is
    # not a decision. The decision is due when the series can actually be
    # RE-DERIVED, which is `MIN_N_TO_REDERIVE_PICKS_BOUND`. Same over-eager
    # threshold as the series-reset reminder, made twice in two days.
    # ── THE WORK IS DONE AND THE DECISION IS NOT MINE. 2026-11-06. ────────
    #
    # This reminder is LEFT RED on purpose. It fired correctly at n=12, the
    # re-derivation was performed, measured, and then REVERTED, and the
    # figures are recorded in `ci_audit.PREPARED_PICKS_BOUND` so the red is
    # actionable rather than a request to start over.
    #
    # WHY IT WAS REVERTED. Substituting the bound is arithmetic on every
    # margin (+47.75 on the max pairing), and it was checked against
    # `schedule_deadline.py`, which references neither constant. That check
    # was too narrow: the constants also feed the CRON DERIVATION, and the
    # latest 15-minute boundary clearing the registered 60 moves 60 -> 105 on
    # weekdays and 30 -> 60 at the weekend. TEN assertions in this file encode
    # the ARGUMENT for the authorised `s5.19` move -- that it spends real
    # margin, that the pooled reading was an artefact, that the rejected cron
    # was a breach.
    #
    #     Re-deriving a bound is a MEASUREMENT.
    #     Rewriting the recorded argument for an AUTHORISED decision is not.
    #
    # So it waits, and it waits VISIBLY: `reminders` red means a decision is
    # due, and one is.
    n = s.get("n")
    prepared = ci.PREPARED_PICKS_BOUND
    assert n is None or n < ci.MIN_N_TO_REDERIVE_PICKS_BOUND, (
        f"the parallel series has reached n={n} of "
        f"{ci.MIN_N_TO_REDERIVE_PICKS_BOUND}. THE RE-DERIVATION IS ALREADY "
        f"MEASURED AND PREPARED -- `ci_audit.PREPARED_PICKS_BOUND`: max "
        f"{prepared['max']} (was {prepared['replaces']['max']}), mean+3sd "
        f"{prepared['mean_plus_3sd']} (was "
        f"{prepared['replaces']['mean_plus_3sd']}), n={prepared['n']} "
        f"measured {prepared['measured']}. Applying it adds "
        f"{prepared['margin_delta']['max_pairing']}m to every max-pairing "
        f"margin and opens "
        f"{prepared['cron_headroom_it_opens']['weekday']}m of weekday cron "
        f"headroom. BLOCKED ON: {prepared['blocked_on']}. This red clears "
        f"when that decision is taken, not by a measurement.")


# ── PART C: the series reset, as a COST ──────────────────────────────────

#: The 11-04 condition's input is the weekday DELAY series, and the move
#: empties it. `simulated` 2026-10-22: weekday scheduled runs arrive ~5/week,
#: so n>=10 under the new cron takes ~2.0 weeks from the first run on
#: 2026-10-23 -> 2026-11-06.
SECOND_STEP_WAS = dt.date(2026, 11, 4)
SECOND_STEP_NOW = dt.date(2026, 11, 6)


@pytest.mark.live
def test_the_SERIES_RESET_is_recorded_as_a_COST_with_the_new_date():
    """A series reset is a cost of the change that caused it. The second step
    slips two days, and it is attributed here rather than discovered later."""
    assert SECOND_STEP_NOW > SECOND_STEP_WAS
    assert (SECOND_STEP_NOW - SECOND_STEP_WAS).days == 2
    # the delay series IS empty under the new cron, which is the reset
    by_type, crons = ci.collect_lag_series_by_daytype()
    assert crons == EXPECTED_CRON
    # ── IT FIRES WHEN THE CONDITION BECOMES EVALUABLE, NOT TOMORROW. ──────
    #
    # The first draft asserted the series was still EMPTY, which would have
    # failed on the first scheduled run under the new cron -- i.e. a reminder
    # on 2026-10-23, reporting something already known and recorded. This is
    # a `live` test in the reminders selection, so its failure IS a due
    # decision; the decision is due when n reaches the 10 the second-step
    # condition needs, which is the projected 2026-11-06.
    # ── THE CONDITION WAS EVALUATED ON 2026-11-06 AND IT FAILED. ──────────
    #
    # The reminder above fired as designed when n reached 10, and it offered
    # two clearing actions: take the second step, or RECORD THAT THE SLOW
    # CLUSTER STILL GOVERNS. The second one is a measurement, so it was
    # taken rather than deferred.
    #
    # `measured` 2026-11-06, grain=RUN, n=11 weekday lags under `0 1`:
    #   272.3 272.9 287.3 325.5 336.0 336.4 357.4 363.8 381.8 388.8 389.8
    #
    #   criterion              required    measured   verdict
    #   no point above            200.0      389.75   NOT MET  (by 189.75)
    #   gap ratio below             8.0       91.76   NOT MET  (11x over)
    #   mean+3sd at most          156.2      463.77   NOT MET  (3.0x over)
    #
    # FAILED ON ALL THREE, and not marginally. The slow cluster governs and
    # the second step is NOT taken. That is recorded here rather than in a
    # comment elsewhere, because the assertion below is what stops the
    # reminder firing again every day for a decision already taken.
    #
    # WHAT WOULD RE-OPEN IT: the series becoming fast enough to meet the
    # condition. The assertion is therefore on the CONDITION, not on n -- so
    # it fires again if the lateness genuinely improves, which is the only
    # event that should re-open the question.
    SECOND_STEP_EVALUATED = {
        "on": "2026-11-06", "n": 11, "verdict": "NOT MET on all three",
        "max": 389.75, "gap_ratio": 91.76, "mean_plus_3sd": 463.77,
    }
    import statistics
    lag = [l for _d, l in (by_type.get("weekday") or [])]
    n = len(lag)
    assert n >= SECOND_STEP_CONDITION["n_min"], (
        f"the weekday series fell to n={n}, below the "
        f"{SECOND_STEP_CONDITION['n_min']} the condition needs. A series that "
        f"SHRANK is a different event from one that never filled -- say which")
    worst = max(lag)
    m3 = statistics.mean(lag) + 3 * statistics.pstdev(lag)
    assert (worst > SECOND_STEP_CONDITION["no_point_above"]
            or m3 > SECOND_STEP_CONDITION["mean_plus_3sd_at_most"]), (
        f"the second-step condition now PASSES on the level criteria "
        f"(max {worst:.2f} vs {SECOND_STEP_CONDITION['no_point_above']}, "
        f"mean+3sd {m3:.2f} vs "
        f"{SECOND_STEP_CONDITION['mean_plus_3sd_at_most']}). The lateness "
        f"has genuinely improved since it was evaluated at "
        f"{SECOND_STEP_EVALUATED['on']} "
        f"(max {SECOND_STEP_EVALUATED['max']}, mean+3sd "
        f"{SECOND_STEP_EVALUATED['mean_plus_3sd']}): RE-EVALUATE the second "
        f"step, which is now a live option rather than a closed one.")


def test_BOTH_SERIES_are_RETAINED_and_LABELLED_BY_CRON_never_pooled():
    """The `390a4be` rule. A lag is comparable only within one schedule, and
    pooling the `0 0` and `0 1` series would reproduce the error that chose
    `0 4` from another cron's spread."""
    assert CLUSTERS[("weekday", "slow")]["n"] == 6
    assert CLUSTERS[("weekday", "fast")]["n"] == 4
    src = __import__("pathlib").Path("tests/test_cron_moved.py").read_text(
        encoding="utf-8")
    assert "under the RETIRED `0 0` cron" in src, (
        "the retained clusters no longer say WHICH cron they were measured "
        "under, so they could be pooled with the new series by accident")


#: The condition is UNCHANGED by the reset; only its date moved.
SECOND_STEP_CONDITION = {
    "n_min": 10,
    "membership": "by headSha under the new cron, never pooled",
    "no_point_above": 200.0,
    "gap_ratio_below": 8.0,
    "mean_plus_3sd_at_most": 156.2,
}


def test_the_SECOND_STEP_CONDITION_is_UNCHANGED_and_still_FALSIFIABLE():
    c = SECOND_STEP_CONDITION
    assert c["n_min"] == 10
    assert c["no_point_above"] == 200.0
    # 200 sits between the two clusters with clearance on each side
    assert CLUSTERS[("weekday", "fast")]["max"] < 200.0 < \
        CLUSTERS[("weekday", "slow")]["mean"] - 50
    assert c["gap_ratio_below"] == ci.LAG_CLUSTER_GAP_RATIO
    assert c["mean_plus_3sd_at_most"] == \
        CLUSTERS[("weekday", "fast")]["mean_plus_3sd"]


def test_KNOWN_WRONG_the_OLD_cluster_FAILS_the_second_step_condition():
    """The condition must be capable of refusing. Driven on the slow cluster,
    which is the population it exists to exclude."""
    slow = [255.9, 256.8, 271.7, 282.7, 285.8, 304.9]
    assert any(v > SECOND_STEP_CONDITION["no_point_above"] for v in slow)
    mixed = slow + [128.4, 132.0, 137.7, 144.1]
    _s, _f = ci.split_lag_clusters(mixed)
    assert _f is not None, "the mixed series no longer splits, so clause 3 "\
        "cannot refuse it"


def test_KNOWN_WRONG_a_PURE_FAST_series_PASSES_every_clause():
    """The other direction: the condition must be satisfiable, or it is a
    refusal dressed as a test."""
    fast = [128.4, 132.0, 137.7, 144.1, 130.0, 135.0, 139.0, 141.0, 133.0,
            136.0]
    assert len(fast) >= SECOND_STEP_CONDITION["n_min"]
    assert all(v <= SECOND_STEP_CONDITION["no_point_above"] for v in fast)
    _s, _f = ci.split_lag_clusters(fast)
    assert _f is None, "a unimodal fast series split, so clause 3 misfires"
    mean = sum(fast) / len(fast)
    sdv = (sum((x - mean) ** 2 for x in fast) / (len(fast) - 1)) ** 0.5
    assert mean + 3 * sdv <= SECOND_STEP_CONDITION["mean_plus_3sd_at_most"]


# ── PART D: two things the correction changed that outlive it ───────────

def test_the_WEEKEND_is_the_EARLIER_FLOOR_on_BOTH_sides_PERMANENTLY():
    """12:00 local against 12:30. Not seasonal — the relation holds whichever
    offset applies, because both anchors shift together."""
    for day in (SUMMER, WINTER):
        assert sd.observed_earliest_min("weekend", day) < \
            sd.observed_earliest_min("weekday", day), day
    assert ci.SCHEDULE_DEADLINE_MIN == min(
        ci.DEADLINE_MIN_BY_DAYTYPE.values())


def test_the_BREACH_CLEARED_BY_A_MEASUREMENT_CORRECTION_not_the_pipeline():
    """+20.7 -> +80.7 on 2026-10-26, from the localisation alone.

    RECORDED SO NOBODY READS IT AS THE PARALLELISATION'S DOING. Nothing in the
    pipeline got faster: the weekday floor moved 09:30 -> 10:30 UTC because
    the anchor is a LOCAL time and the clocks went back. The cron move is a
    separate, later change with its own margin.
    """
    pooled_3sd = 442.3           # the series the old margin used
    before = sd.RETIRED_OBSERVED_EARLIEST_UTC_MIN["weekday"] - (
        0 + pooled_3sd + ci.FULL_CARD_PICKS_3SD)
    after = sd.observed_earliest_min("weekday", WINTER) - (
        0 + pooled_3sd + ci.FULL_CARD_PICKS_3SD)
    assert before == pytest.approx(20.7, abs=0.3)
    assert after == pytest.approx(80.7, abs=0.3)
    assert after - before == pytest.approx(60.0, abs=0.1), (
        "the 60 minutes are the DST correction and nothing else; if this "
        "changed, the attribution in the ledger is wrong")
