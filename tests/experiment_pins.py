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

    s5.20 2026-11-07  ONE KEY, AND IT OPENS A BAND THAT WAS UNREACHABLE.
                    (Derived and pre-registered 2026-11-06; the bump
                    lands 11-07, so s5.19's last day is 2026-11-06.)
                    `betting.min_odds` 1.50 -> 1.14. Selection-affecting by
                    definition: it admits a price class the gate had refused
                    since the experiment began, so rule 7 orders a new cohort.

                    WHY THE BAND WAS UNREACHABLE AND NOT MERELY EMPTY. The FLB
                    test reported `[0.70,0.80)` and `[0.80,0.90)` as `n=0,
                    INSUFFICIENT` and attributed it to "our avg taken odds of
                    ~1.6 put the mass in the middle" -- a SAMPLING reading
                    that more data would fix. Measured 2026-11-06: 1/1.50 =
                    66.7% is a CEILING on implied probability and the band
                    starts at 70%, so 0 of 1,639 settled picks sat in it and
                    none COULD. It was a registered gate, not a thin sample.

                    THE BOUND IS DERIVED, AND 1/0.79 IS THE WRONG ARITHMETIC.
                    The bands are fixed on DE-VIGGED probability; `min_odds`
                    gates the RAW price; `raw = p_devig * (1 + margin)`. So a
                    leg at de-vigged 0.79 prices near 1.19, not 1.2658, and a
                    floor from 1/0.79 would open the band only partway -- the
                    measurement could not see the top of its own band.
                    Measured over 23,398 leg sets at taken prices: margin
                    median 6.66% / p95 8.17% / max 16.11%, implying floors
                    1.1720 / 1.1556 / 1.0765; the 5,314 legs actually in the
                    band run 1.1450..1.3700 raw. 1.14 is the two-decimal value
                    below every in-band leg measured and inside that range,
                    admitting 5,314 of 5,314 -- 100.00%.

                    A RAW FLOOR CANNOT EXPRESS A DE-VIGGED BAND, so the spill
                    is recorded rather than engineered away: of 9,334 newly
                    admitted legs, 56.9% are the target band, 31.6% sit
                    between the old ceiling and the band, 11.5% spill into
                    `[0.80,0.90)`. Each band is tested SEPARATELY, so that
                    POPULATES two further bands instead of confounding the
                    target -- and it is what `flb-preregistration.md` asked
                    for, since it calls a single band clearing weak evidence
                    and wants several bands moving together. Three of five
                    bands could never be tested before this.

                    MEASUREMENT, NOT PROFIT. `paper_trading_mode` stays
                    `true`; there has never been a live wager. The economic
                    bar is registered separately because measuring the
                    anomaly and collecting it are two questions: the band's
                    own cost at the taken price is 4.70pp, so the external
                    +3.74pp (n=134, Footiqo, one book, five leagues) nets
                    -0.96pp at current pricing. A positive measurement is
                    EXPECTED to fail that bar, and "the bias is real and we
                    cannot collect it" is a registered outcome.

                    THE n IS REGISTERED BEFORE THE FIRST PICK. n=829 in-band
                    picks for 80% power, one-sided alpha=0.05, at the band
                    midpoint; empirical power 82.8% over 4,000 bootstrap
                    draws; the pre-registration's own floor to test a band at
                    all is n=100. At the measured ceiling of 21.56
                    fixtures/day carrying an admissible in-band leg: n=100 in
                    5 days, n=829 in 38 -- BOTH CEILINGS, because that is the
                    FIXTURE rate and the EV, value and confidence gates are
                    untouched and will reject most candidates.

                    NOTHING ELSE MOVED. Not `max_odds`,
                    `max_picks_per_match`, `min_expected_value`,
                    `min_confidence`, the blend, the markets or the gates.
                    `betting.min_odds` is a TRACKED_KEY, so the fingerprint
                    moved by the mechanism that already exists rather than by
                    hand: cdec29 -> f20105 on the config change alone, then
                    -> 12f2c7 with the revision bump.
                    RULE 8 EXEMPTS INVARIANT 6b AND THE FOUR REVISION PINS
                    here because rule 7 orders the bump; the argument is this
                    entry and `docs/flb-band-preregistration.md` (committed
                    `fed8338` BEFORE this bump), and no test is relaxed.

    s5.21 2026-11-11  THE FLOOR IS RESTORED, AND THE ROUND TRIP IS RECORDED
                    WITH ITS COST. `betting.min_odds` 1.14 -> 1.50.

                    DERIVED FROM THE RECORD, NOT CHOSEN. All FIFTEEN cohorts
                    from 2026-08-11 to 2026-11-07 ran at exactly 1.50:
                    n=1,798 picks, per-cohort average taken odds
                    1.592..1.671, and the MINIMUM taken price in every one of
                    them is 1.50 to the cent -- so the floor was binding
                    throughout and the figure is a property of the deployed
                    record, not a preference asserted over it.

                    THE COST. `s5.20` is TWO DAYS: 2026-11-08..2026-11-09,
                    n=86, average taken odds 1.464 against 1.59..1.67 for
                    every cohort before it, minimum 1.14. Prices the operator
                    did not want published. RETAINED AND LABELLED -- it
                    settles normally, is NEVER pooled with s5.19 or s5.21,
                    and is far too small to support any conclusion.

                    AND THE BAND DID NOT NEED OPENING TO BE MEASURED. The
                    hypothesis is about MARKET CALIBRATION; its unit is a LEG
                    with a de-vigged probability and a known result, and
                    5,314 in-band legs exist in the odds record at ANY floor.
                    Opening it was needed to ACT, not to measure, and acting
                    is blocked on an exchange whose depth is UNAVAILABLE. The
                    leg test arrives TODAY instead of 2026-12-15.

                    NOTHING ELSE MOVED, and `paper_trading_mode` stays true.
                    RULE 8 EXEMPTS INVARIANT 6b AND THE FOUR REVISION PINS
                    here because rule 7 orders the bump; the argument is this
                    entry.
"""

#: Must equal src.models.model_version.CODE_REVISION.
#:
#: MOVED s5.20 -> s5.21 on 2026-11-11: the floor is RESTORED to 1.50, the
#: value all fifteen prior cohorts ran at. A ROUND TRIP, and the s5.20 entry
#: above keeps what it cost -- 86 picks over two days at an average 1.464.
#: Invariant 6b pins this to a literal precisely so a bump has to argue; the
#: argument is the s5.21 entry, and no test is relaxed.
CODE_REVISION_PIN = "s5.21"

#: Must equal model_version(config.example.yaml).
#:
#: MOVED on 2026-11-11. Restoring the floor moved the fingerprint
#: `12f2c7 -> 6d17e7` on the config alone, and the `s5.21` bump moved it again
#: to `52678c`.
#:
#: NOTE THAT IT DOES **NOT** RETURN TO `cdec29`. Restoring a config value does
#: not restore a cohort: the fingerprint covers `CODE_REVISION` too, and that
#: is the point of always bumping. `s5.21` runs the same CONFIG as `s5.19` and
#: is a DIFFERENT COHORT, because the code between them is not the same code.
FROZEN_MODEL_VERSION = "stage5_baseline_20260807.52678c"

#: The previous cohort, kept so a reader can see what moved and when.
#: NOTE: ee60cd labels TWO configurations — 16 picks made by 098a368 on 09-16
#: and 12 by 889f8ba on 09-17. That is the defect the always-bump rule closes.
PREVIOUS_CODE_REVISION = "s5.20"
PREVIOUS_MODEL_VERSION = "stage5_baseline_20260807.12f2c7"


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
# ── RETIRED 2026-11-06. THE RESERVATION IS INERT AND NOTHING IS DEFERRED. ──
#
# `H1 WINDOW FAILED` on 2026-11-05 (day-5 rule, n=12 FIXTURE-grain against
# 18, NOT extended). The acknowledgement expired on 2026-11-01 and `reminders`
# was red by verdict for five days; its instruction was the deletion stage.
#
# WHAT WAS DONE, AND WHY IT IS NOT THE DELETION THE GUARD ASKED FOR.
#
# The successor contract the guard wanted is "h1_ceiling is 0, the window
# constants gone OR INERT, and the acknowledgement has nothing left to
# acknowledge". That is now true, by making the reservation inert rather than
# by removing it:
#
#   src/data/h1_window.STOPPED_ON = 2026-11-05  and is_collecting(day)
#   odds_quota.h1_reservation() returns 0 from 11-05
#   allocate()["h1_ceiling"] == 0; normal_ceiling 182 -> 350
#
# THE DATES DO NOT MOVE, because `contains()` defines the analysis population
# and `h1_analysis` clamps on it -- moving them would retroactively change a
# closed experiment. So the FUNDING question and the POPULATION question are
# now two functions, which is what they always were.
#
# RE-MEASURED 2026-11-06, as the retired guard's own instruction required
# before any re-acknowledgement: the sizing is NOT 87/31/9. It is
# 274 references / 61 assertions / 7 assertion-bearing test files, so the cost
# that justified deferring has grown 3.1x on references and 2.0x on
# assertions. That is an argument against deferring again, not for it.
#
# AND THE DELETION WOULD HAVE BEEN WRONG. Of the 61 assertions, 51 encode
# live facts a FUTURE attempt needs, measured by what each file protects:
#
#   17  the reservation ALLOCATES (it was one-sided; a for_h1 instance took
#       288 credits against a 168 reservation on 2026-10-23)
#   12  the enforced stop, and proof that each limb fires
#   10  credit attribution -- what keeps H1's spend measurable rather than
#       inferred, and what the guard's own clearing action told us to protect
#    6  the window has ONE definition and every consumer imports it; this
#       stage ADDED a consumer, so the rule binds harder, not less
#    3  terminal states reachable from the caller as wired
#    3  the dead ceiling
#
# Deleting that machinery would remove the guards that stopped an overdraw and
# keep nothing. The 10 remaining assertions are in
# `test_calendar_separation.py`, and only the acknowledgement's own retire.
H1_RESERVATION_SUCCESSOR = {
    "stopped_on": "2026-11-05",
    "stop_basis": ("the pre-registered day-5 rule: trajectories < 18 -> STOP "
                   "and report. n=12 at FIXTURE grain. NOT extended -- a "
                   "pre-registered stop that is extended is not a stop"),
    "contract": ("for ANY day >= h1_window.UNTIL, h1_reservation() is 0 and "
                 "allocate() h1_ceiling is 0 -- a property of the DEPLOYED "
                 "calendar design, true today and needing no code; so the "
                 "acknowledgement has nothing left to acknowledge"),
    "inert_via": ("the existing calendar: h1_reservation() is zero outside "
                  "FROM..UNTIL, so the machinery goes inert on 2026-11-09 "
                  "with no change. A stop-driven funding switch was built "
                  "and REVERTED the same day -- see h1_window.STOPPED_ON"),
    "accepted_cost": ("168 credits stay reserved 2026-11-05..11-08 for a "
                      "stopped experiment. Nothing collects, so nothing "
                      "spends them; the normal ceiling is 182 rather than "
                      "350 on four days, against a measured usage that has "
                      "never approached either. Cheaper than a funding "
                      "switch in a capture-path module, which flipped 13 "
                      "gate tests when tried"),
    "kept_for_a_future_attempt": (
        "the allocation and its ceiling, the enforced stop and its limbs, the "
        "for_h1 credit attribution, the one-source window rule, caller "
        "reachability"),
    "how_a_future_attempt_resumes": (
        "move FROM/UNTIL together in h1_window and clear STOPPED_ON; the "
        "enforcement machinery is already in place and needs no rebuild"),
    "sized_cost_when_deferred": {"references": 87, "assertions": 31,
                                 "test_files": 9, "measured": "2026-10-18"},
    "sized_cost_now": {"references": 274, "assertions": 61,
                       "test_files": 7, "measured": "2026-11-06"},
    "assertions_encoding_a_live_fact": 51,
    # NO REVIEW DATE, DELIBERATELY. This is not a deferral: there is no
    # standing condition left. A date here would be a reminder with nothing
    # to remind anyone of, which is the alarm-that-always-fires shape.
}
