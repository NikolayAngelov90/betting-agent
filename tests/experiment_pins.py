"""The cohort pins, stated once.

Six hardcoded `"s5.2"` literals across five test files was one fact duplicated
six ways: the next bump edits five and misses one, and the miss is silent
because a passing test looks the same as a correct one.

**These values are stated independently of `src/` on purpose.** A pin that
imports the value it is pinning can never fail — it would be a vacuous guard
built into the very mechanism whose job is to fail when a cohort changes. The
literals below must be edited by a human, deliberately, as the act of
authorising a cohort break. That edit is the authorisation.

When you bump these, you are asserting: the change was intended, its effect on
the fingerprint is fully attributable to the edits you made, and picks before
and after are not one cohort.

History:
    s5.2  Stage 8   over/under cross pairs, post-Claude correlation re-check,
                    normalized dedup key
    s5.3  Stage 13  Part B team-identity gate (name + country) and Part C
                    one-pick-per-match with a single ranking definition;
                    also covers a training-data correction, which the
                    fingerprint itself does not cover
    s5.9  2026-08-31  the per-match cap keys on FIXTURE identity, not ROW
                    identity. s5.3 established one-pick-per-match; two rows
                    for one fixture defeated it silently, measured on
                    2026-08-30 (Deportivo v Valencia, two picks). 835 pairs
                    match the provable branch; exactly 2 second-picks would
                    have been refused out of 1,458.
    s5.11 2026-09-11  the merge's survivors were invisible to the duplicate
                    check. flashscore._get_or_create_team scanned
                    filter_by(league=<scraped>) and every s5.10 survivor carries
                    league IS NULL, so same_team_strict was never called against
                    it. 26 of 129 merged rows were re-created within a day.
                    Scan now covers league == scraped OR league IS NULL.
    s5.10 2026-09-10  Stage 22 team identity repair. 2 wrong provider ids
                    cleared, 42 rows absorbed into 40 provable components,
                    86 unresolved rows merged into evidenced twins;
                    teams 1561 -> 1433. Fixtures whose both participants
                    resolve: 79.86% -> 89.18% since 2026-08-01, 83.38% ->
                    94.26% over 365 days. SELECTION-AFFECTING: it changes
                    which fixtures resolve and therefore which are picked,
                    and Elo/Poisson key on team_id, so a club whose history
                    was split now trains as one club.
    s5.12 2026-09-13  Stage 23. ONE resolution function. Three creation paths
                    with three matching regimes produced three blind spots and
                    44 resurrections in three days; resolve_team() replaces
                    them, with a former-name lookup as step 2 and a test that
                    fails on a fourth construction site. The AF-id gate's
                    country check was applied to the name path on the same day.
                    MEASURED 09-16: new team rows 14/day -> 0, and the
                    duplicate count held flat for three days.
    s5.13 2026-09-16  TWO selection-affecting changes.
                    (a) A CREDIT READING NOW HAS AN AGE. The period check could
                    not see a figure from this month describing a state six days
                    gone: the file said 154 remaining while the provider said
                    100 and the ledger said 0 spendable. Stale-and-too-high is
                    permission to spend that does not exist, and the file only
                    updates when a run spends — so staleness correlates with the
                    refusal it must not fail open on. It changes whether the
                    odds fetch is skipped, hence which prices exist to pick from.
                    (b) STAGE 24's MERGE. 32 shared-provider-id components
                    merged, 32 rows removed, 150 references re-pointed, teams
                    1534 -> 1502. Selection-affecting because step 1 was
                    `.first()` over an UNORDERED provider-id match held by two
                    rows — which row a fixture attached to was a coin flip, and
                    Elo/Poisson key on team_id.
                    DEL-3 ships in the same revision and is NOT
                    selection-affecting: it changes how a report is DELIVERED,
                    never which picks it contains.

    s5.14 2026-09-17  filter_generation() returns None, not the string
                    "unknown", when the exclusion predicate's source cannot be
                    read — and both consumers (the Parquet history mirror and
                    the ML pickles) refuse on unknown BEFORE comparing. A
                    CONSTANT COMPARES EQUAL TO ITSELF, so every cache stamped
                    "unknown" validated against every other one: the digest
                    stopped discriminating at exactly the moment validity became
                    uncomputable and reported that as validity. `None == None`
                    is also True, hence the refusal precedes the equality.
                    Selection-affecting by the same reading as s5.13's
                    credit-age change: it decides whether a possibly
                    contaminated cache reaches the models.
                    ALSO: the amend-while-empty rule is RETIRED. It evaluates a
                    mutable count — s5.13 held 0 picks at the amend and 16
                    within the hour, so ee60cd labels two configurations. Always
                    bump.

    s5.15 2026-10-01  THE CLOCK MOVED: `daily-picks.yml`'s cron 03:00 -> 00:00
                    UTC. Nothing in TRACKED_KEYS saw it, so without this bump the
                    two pricing regimes would pool under `00febf` — which 109
                    picks already carry.
                    Selection-affecting by the s5.2 precedent: the cron decides
                    WHEN prices are taken. Taken price moves (pricing completed
                    08:35-10:08, now 05:14-07:56) and lead time roughly TRIPLES
                    (-4min..1h29m -> 2h08m..4h50m). The discovered-fixture
                    population does NOT move: the `now + 1 day` cutoff lands
                    04:11-06:53 on D+1 instead of 07:11-09:53, and the earliest
                    D+1 kickoff is 10:15, so both admit exactly day D.
                    The CLV cost is UNMEASURABLE, not absent — closing-line
                    coverage is still 0%.
                    NOT a model change: no learner, feature, threshold or
                    selection rule moved. The cohort boundary is about prices.

    s5.16 2026-10-10  THE 1X2 ASSIGNMENT FIX AND ITS EXCLUSIONS, as ONE event.
                    `_save_game_odds` tested the HOME name FIRST with a fuzzy
                    predicate whose token rule accepts substring containment, so
                    the AWAY outcome satisfied the home branch and -- `odds`
                    being UNIQUE on (match, bookmaker, market, selection) --
                    OVERWROTE it. The stored Home price became the away team's.
                    Measured: 486 keys / 28 matches with Home present and Away
                    ABSENT, `missing_HOME` exactly 0. The API-Football control
                    population is 49/49, SYMMETRIC; an asymmetric absence rules
                    out a symmetric cause.
                    PREDICTION-AFFECTING BY THE MARKING, not only by the fix:
                    24 of the 28 are settled and leave the fitting set
                    (41,311 -> 41,287, 0.058%), so the models now read different
                    rows. 13 picks carry `void_corrupt_features`.
                    ONE BUMP: the fix stops new inverted prices and the
                    exclusion stops learning from the old ones. A boundary
                    between "fixed but still learning from it" and "fixed and
                    not" describes no configuration anyone ran.
                    WHY NOW, having been recorded as no-bump earlier the same
                    day: the deferral existed to protect H1's sized window, and
                    H1 has zero observations with a blocked mechanism. There is
                    no window left to protect.
                    AND THIS IS THE ARGUMENT invariant 6b ASKS FOR. Its own
                    docstring says a selection change "has to argue with a
                    test"; the argument is above, and the pin moves with it
                    rather than the test being relaxed.

    s5.17 2026-10-10  THE CRON IS SPLIT BY DAY OF WEEK.
                    `0 0 * * *` -> `0 0 * * 6,0` + `0 4 * * 1-5`.
                    Selection-affecting by the s5.2/s5.15 precedent: the cron
                    decides WHEN prices are taken.
                    WHY A SPLIT AND NOT THE FLAT 1h30 ASKED FOR: the deadline
                    is the earliest kickoff and it differs by day - weekend
                    10:15 UTC, weekday 12:00 UTC, both from an INDEPENDENT
                    reference (Flashscore pages, offset to UTC verified against
                    the production matches table). That reference does NOT
                    reproduce the 14:30 weekday figure on file; it is 2h30
                    stricter, and the stricter value is used.
                    Minimum margin STATED BEFORE COMPUTING: 60 min on mean+3sd.
                    `0 4` clears at +104/+76; `0 5` fails at +44/+16.
                    Weekday lead time 6h44m -> 2h44m (-240 min); weekend
                    UNCHANGED at 4h59m. Discovered-fixture population does NOT
                    move - the binding site is a FRIDAY run whose D+1 Saturday
                    earliest kickoff is 10:30 against a ~08:36 cutoff, 114 min.
                    All six date-derived sites re-enumerated; none resolves to a
                    different calendar date.
                    RULE 8 EXEMPTS INVARIANT 6b here because rule 7 orders the
                    bump; the argument is this entry and the test is NOT relaxed.

    s5.19 2026-10-22  THE DEADLINE IS LOCALISED AND BOTH CRONS MOVE.
                    weekday `0 0` -> `0 1`, weekend `0 0` -> `30 0`.
                    Selection-affecting by the s5.2/s5.15/s5.17 precedent: the
                    cron decides WHEN prices are taken. The deadline
                    localisation alone is NOT selection-affecting and shipped
                    separately at s5.18 with the fingerprint unchanged.
                    THE DEADLINE WAS A UTC CONSTANT MEASURED IN SUMMER. A
                    kickoff is scheduled in a LOCAL clock, so its UTC minute
                    moves twice a year. Re-derived as the minimum LOCAL
                    kickoff over the record (30 configured leagues, phantoms
                    excluded): weekday 12:30 local n=718, weekend 12:00 local
                    n=375. The retired weekend 09:45 UTC was the minimum of
                    the SUMMER SUBSET only, so localising TIGHTENS that floor
                    45 min in summer and loosens it 15 in winter; the weekday
                    gains 60 in winter and nothing in summer.
                    THE LAG SERIES IS BIMODAL UNDER ONE UNCHANGED CRON, and
                    that bimodality IS the early start Niki reported: weekday
                    SLOW n=6 mean 276.3 max 304.9 mean+3sd 332.6, fast n=4
                    mean 135.6 max 144.1 mean+3sd 156.2 (largest-gap /
                    median-gap 17.33; the weekend's is 2.61 and does not
                    split, so its whole n=6 series governs).
                    THE SLOW CLUSTER GOVERNS. Each cron is the latest
                    15-minute boundary clearing the registered 60 on BOTH
                    pairings the margin takes the worse of, at the SUMMER
                    floor: weekday cap(max 304.9, picks 88.1)=117.0 and
                    cap(3sd 332.6, 107.0)=70.4 -> `0 1`; weekend cap(max
                    293.7, 88.1)=98.2 and cap(3sd 342.3, 107.0)=30.7 ->
                    `30 0`.
                    `0 2` WAS REJECTED. The parallel picks bound (55.1,
                    simulated n=72; MEASURED n=1 at 35.1) allows it and a
                    10:33 Sofia slow-case start, but it leaves the margin at
                    +10.4 under the picks constants this project still
                    retains - a breach by its own instrument - and those
                    constants are retained BECAUSE the parallel bound is n=1.
                    A configuration the instrument calls a breach is not
                    shipped.
                    LEAD TIME, per day type and per CLUSTER because an average
                    across a bimodal series describes neither (Sofia): weekday
                    SLOW 3sd 08:33 -> 09:33, SLOW mean 07:36 -> 08:36, fast
                    mean 05:16 -> 06:16 (+60 min); weekend 3sd 08:42 ->
                    09:12, mean 07:13 -> 07:43 (+30 min).
                    DISCOVERED-FIXTURE POPULATION DOES NOT MOVE. `max_days_
                    ahead` is a rolling `now + 1 day`, so the cutoff moves
                    LATER with the run; the binding site is a FRIDAY run whose
                    D+1 Saturday earliest kickoff is 10:30 UTC, and all six
                    date-derived sites were re-enumerated against both new
                    crons with the Friday->Saturday clearance still above the
                    registered hour. Zero D+1 fixtures admitted under either
                    cron.
                    AND THE LAG SERIES RESETS TO n=0, which is a COST of this
                    move: membership keys on the run's own headSha carrying
                    the cron in force, so the margin reports NOT COMPUTED
                    until the new crons fire and the 11-04 condition's input
                    starts again from empty.
                    RULE 8 EXEMPTS INVARIANT 6b AND THE FOUR REVISION PINS
                    here because rule 7 orders the bump; the argument is this
                    entry and no test is relaxed.
"""

#: Must equal src.models.model_version.CODE_REVISION.
#:
#: MOVED s5.18 -> s5.19 on 2026-10-22, with the argument in the history above.
#: Invariant 6b pins this to a literal precisely so a bump has to argue; the
#: argument is the s5.19 entry and the test is unchanged.
CODE_REVISION_PIN = "s5.19"

#: Must equal model_version(config.example.yaml).
FROZEN_MODEL_VERSION = "stage5_baseline_20260807.cdec29"

#: The previous cohort, kept so a reader can see what moved and when.
#: NOTE: ee60cd labels TWO configurations — 16 picks made by 098a368 on 09-16
#: and 12 by 889f8ba on 09-17. That is the defect the always-bump rule closes.
PREVIOUS_CODE_REVISION = "s5.17"
PREVIOUS_MODEL_VERSION = "stage5_baseline_20260807.818026"


#: How many completed-match queries are exempt from the training-exclusion
#: filter. Pinned for the same reason CODE_REVISION is: an exemption marker
#: travels with the code, which beats a central list that drifts — but the
#: failure mode inverts. Someone silences the guard by pasting the comment.
#:
#: Shape alone cannot catch that. A count can: adding a seventh exemption
#: requires editing this literal, which is a deliberate act a reviewer sees.
#: Unobserved growth is where this stage kept finding defects.
TRAINING_EXCLUSION_EXEMPTIONS = 6

#: Evidence-gate exemptions: get_stats, the cold-streak alert, and
#: _reset_stale_ml_calibration. Pinned for the same reason as the
#: training-exclusion count.
EVIDENCE_GATE_EXEMPTIONS = 4


from datetime import date as _date


# ───────── THE H1 RESERVATION GUARD, ACKNOWLEDGED — NOT EXTENDED, 2026-10-18
#
# `test_the_H1_RESERVATION_WINDOW_EXPIRES` has fired every day since
# 2026-10-18. Its instruction is correct: DELETE the reservation rather than
# move the date. Its estimate of the work is not — MEASURED 2026-10-18, the
# deletion is:
#
#     87 references   42 x H1_RESERVED_CREDITS, 29 x h1_reservation, 4+2 imports
#     31 assertions   on the stop, the ceiling collapse and the attribution
#     9 test files    plus two H1 scripts that fail on import without it
#
# The deletion was performed, verified in `src/` (allocate -> pool 350,
# h1_reserved 0, normal_ceiling 350, window None) and REVERTED, because
# rewriting 31 live assertions in the same pass that restored a six-day outage
# is how the seventh card gets lost.
#
#     A guard that fires every day until a stage runs is an alarm that always
#     fires, and this project has paid for that six times.
#
# So the condition is ACKNOWLEDGED: recorded, dated, and alarming on CHANGE.
#
# ── AND THE ACKNOWLEDGEMENT IS NOW MOOT, NOT EXPIRED. 2026-10-28. ──────────
#
# "THE DATE IN THE GUARD DOES NOT MOVE" was written above, and the date has
# moved: the re-run is authorised and the window is 2026-11-01..2026-11-09.
# The argument is in the guard's own docstring — the first attempt is CLOSED
# at n=4, so what starts on 11-01 is a NEW allocation with its own
# pre-registration, which is precisely the carve-out the instruction allowed.
#
# WHAT THAT DOES TO THIS ACKNOWLEDGEMENT: it covered an INERT window, and from
# 11-01 the window is LIVE. Its branch in the guard is unreachable while the
# window is open, which is correct rather than a bypass — there is no longer a
# standing condition to acknowledge. The 87/31/9 deletion stays DEFERRED past
# the window, because those 31 assertions encode live facts about the stop and
# the attribution that the re-run depends on.
#
# The guard still evaluates its own predicate; it now fires on 2026-11-09.
ACKNOWLEDGED_H1_RESERVATION_GUARD = {
    "figure": "MOOT from 2026-11-01: the window moved to 2026-11-01..11-09 "
              "and is LIVE, so there is no inert window to acknowledge. Was: "
              "h1_reservation() == 0 and the window 2026-10-09..2026-10-17 "
              "is closed; the symbols are still present in odds_quota.py",
    "why_not_retired": "the guard's instruction is to DELETE the machinery, "
                       "and the machinery is 87 references / 31 assertions / "
                       "9 test files wide. Deleting the GUARD while leaving "
                       "the machinery would remove the alarm and keep the "
                       "defect - the SUP-1 shape the guard exists to prevent.",
    "sized_cost": {"references": 87, "assertions": 31, "test_files": 9,
                   "measured": "2026-10-18"},
    "clearing_action": "run the reservation-deletion stage: remove "
                       "H1_RESERVED_CREDITS, H1_RESERVATION_FROM, "
                       "H1_RESERVATION_UNTIL, h1_reservation and the "
                       "reservation term in reserved_from_me (NOT the `for_h1` "
                       "attribution flag, which keeps H1's spend measurable), "
                       "then rewrite the 31 assertions. Cost: one stage with "
                       "its own full-suite verification. It is the only thing "
                       "that clears this.",
    "review": "2026-11-01",
    "review_basis": "the TheOddsAPI monthly reset, which is also the earliest "
                    "date an H1 re-run can start - so the deletion and the "
                    "re-run decision are taken together rather than twice",
}

#: Deliberately a DATE, so the acknowledgement expires the way
#: `EMPTY_CARD_UNTIL` does. An acknowledgement with no expiry is a suppression.
H1_RESERVATION_ACK_UNTIL = _date(2026, 11, 1)
