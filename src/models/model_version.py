"""The model version identifier stamped on every prediction.

Stage 5, Phase 1. The model is now an experimental subject, frozen and observed
prospectively. That only works if every prediction records which configuration
produced it.

Why this is computed rather than hard-coded
-------------------------------------------
A literal string in a config file drifts. Across Stages 1-4 the blend weight
moved 0.40 -> 0.60 -> 0.80, the Poisson half-life 180 -> 540, rho -0.13 -> 0,
the de-vigging rule changed from single-book to gated cross-book consensus, and
six betting gates were switched off. Every one of those silently changed what
`predicted_probability` means, and nothing in the saved row recorded it — so a
pick from March and one from August were pooled in the same statistics as if
they came from the same system. That is the single biggest reason the Stage 1-4
analyses had to keep re-deriving their own cohorts.

So the version is a **label plus a fingerprint of the values that actually
change predictions**. Change any of them and the fingerprint changes on the next
prediction, without anyone remembering to bump a string.

Format
------
``stage5_baseline_20260807.a3f19c``
  │                │        └── 6-char BLAKE2s digest of the tracked settings
  │                └─────────── the date the baseline was frozen
  └──────────────────────────── the experiment label

The label and freeze date come from config so a future experiment can declare
itself; the fingerprint cannot be faked from config.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger()

#: Default experiment label + freeze date. Overridable via
#: models.experiment_label / models.experiment_frozen_at.
DEFAULT_LABEL = "stage5_baseline"
DEFAULT_FROZEN_AT = "20260807"

#: Config keys whose values change what a prediction MEANS. Deliberately a
#: closed list rather than "hash the whole config": scraping league lists,
#: Telegram tokens and logging levels churn constantly and would make every
#: prediction look like a new model.
TRACKED_KEYS: List[str] = [
    "models.bookmaker_blend_weight",
    "models.goals_ml_blend_weight",
    "models.extreme_confidence_ceiling",
    "models.dixon_coles_rho",
    "models.dc_rho_per_league",
    "models.strength_half_life_days",
    "models.shrinkage_sample_cap",
    "models.intl_goals_dampen",
    "models.poisson_use_xg",
    "models.poisson_xg_min_coverage",
    "models.probability_calibration_enabled",
    "models.bayesian_weight_half_life_days",
    "models.bayesian_prior_strength",
    "models.ensemble_weights",
    "betting.min_odds",
    "betting.max_odds",
    "betting.min_expected_value",
    "betting.min_confidence",
    "betting.min_ev_confidence_score",
    "betting.kelly_fraction",
    "betting.max_stake_percentage",
    "betting.excluded_markets",
    # Stage 13 Part C: one pick per match. Changing this changes
    # which picks exist, so it must split the cohort.
    "betting.max_picks_per_match",
    "betting.gates",
]

#: Bumped by hand only when the CODE path changes in a way config cannot express
#: — e.g. the Stage 4 switch from single-book de-vigging to gated cross-book
#: consensus. Without this, a pure-code change would leave the fingerprint
#: unmoved and two genuinely different models would share a version.
#:
#: History
#: -------
#: s5.1  Stage 5 freeze.
#: s5.2  Stage 8 (2026-08-10). SELECTION-affecting, not prediction-affecting:
#:       the model's probabilities are untouched, but the set of picks it
#:       persists changed, and a changed population of predictions is a
#:       different experiment. Three edits:
#:         1. `_CORRELATED_PAIRS` gained the six Over X.5 / Under Y.5 cross
#:            pairs — the table had every same-direction pair and no opposite
#:            ones, so that whole class passed the filter.
#:         2. The Claude KEEP/CHANGE review now re-checks correlation before
#:            switching a selection. It ran after `_filter_correlated_picks`
#:            and only guarded exact-selection duplicates, so a switch could
#:            land on a selection correlated with one already held — which is
#:            how all three correlated pairs in production were created.
#:         3. The in-memory duplicate key moved from (match_name, selection)
#:            to (match_id, market, selection), matching the DB unique index.
#: s5.3  Stage 13 (2026-08-23). SELECTION-affecting AND a training-data
#:       correction — the second of which this fingerprint does NOT cover, so
#:       read this entry before comparing anything across the boundary.
#:
#:       Config change (covered by the fingerprint):
#:         · `betting.max_picks_per_match: 1` — at most one pick per fixture,
#:           and it must be the best one. Three different orderings existed:
#:           picks were sorted by `_rank_key` (EV x confidence x agreement x
#:           contrarian), the per-match survivor was then chosen by CONFIDENCE
#:           ALONE, and the final order dropped the contrarian term. At a cap
#:           of 2 that was survivable; at a cap of 1 it decides which single
#:           pick represents the match. All three now use `_rank_key`.
#:
#:           MEASURED EFFECT, recorded because a cohort break with an unstated
#:           magnitude invites the assumption that it was large: the 2 -> 1 cap
#:           removes AT MOST 1.6% of picks — 3 of 181 matches carried a second
#:           pick at all. **The fingerprint break is therefore dominated by the
#:           ranking unification, not by the cap.** Anyone comparing across this
#:           boundary should expect the selection to differ on the ~1.6% the cap
#:           touches and on any match where the three old orderings disagreed,
#:           and should not attribute a large behavioural change to the cap.
#:
#:       Selection-affecting, not in the fingerprint's inputs:
#:         · Team-identity gate at API-Football team resolution. A row matched
#:           by AF id is now verified against the payload in hand — country
#:           first (unconditional: a club plays in exactly one domestic
#:           league), then a lexical-anchor name check. Fails closed: the
#:           fixture is skipped, the suspect row is neither renamed nor
#:           re-keyed. Correct by construction and UNVERIFIED IN PRODUCTION
#:           while the API-Football account is suspended (ledger OPS-1).
#:         · The KEEP/CHANGE decision prompt no longer contains statistics
#:           computed from paper picks (ledger EXP-1). Changing the prompt
#:           changes Claude's decisions, which changes which picks persist.
#:
#:       TRAINING-DATA CORRECTION — the dimension `model_version` cannot see:
#:         29 matches carry a participant whose row belongs to a different club
#:         and are marked `training_exclusion_reason = corrupt_team_identity`:
#:           Telstar/Maccabi Tel Aviv 2, SK Rapid/Rapid Bucuresti 10,
#:           St. Pauli/Pau FC 14, Levski Sofia 3.
#:         Picks 1148, 309 and 314 are marked
#:         `evidence_status = void_corrupt_features` — excluded from every
#:         learner and measurer, retained in the ROI record, because the wagers
#:         were real.
#:
#:         A future REPAIR that lifts an exclusion re-includes those matches in
#:         the fitting set. That is prediction-affecting and needs its own
#:         CODE_REVISION bump. It is not bookkeeping.
#:
#:       HOW THE REFIT ACTUALLY WORKS — an earlier claim in this stage was
#:       overturned and the corrected version is what follows.
#:         OVERTURNED: "Poisson and Elo need no artifact surgery because fit()
#:         replays from `self.ratings = {}` against the DATABASE."
#:         Half right. Both DO replay from an empty state — no rating or
#:         strength table was edited and none needed to be — but they replay
#:         from whatever `get_completed_matches` returns, and that is the
#:         Parquet mirror whenever one is warm. The database is the fallback.
#:         A stale mirror would have fed the excluded matches straight back
#:         into a fit that believed it had excluded them.
#:         CORRECTED: exclusion is sufficient only because BOTH caches of the
#:         excluded data are stamped with the filter's generation and refuse
#:         themselves on mismatch — the Parquet mirror and the ML pickles. Two
#:         mechanisms, not one. An exclusion is only real where every cached
#:         derivative of the excluded data is invalidated.
#:
#:         The first retrain is not forced by a flag, a deletion or a dispatch.
#:         The stamp is a property of the DEPLOYED CODE, so the run that has
#:         the filter is the run that refuses the artifact: the restored pickle
#:         has no stamp, `is_fitted` goes false, `trained_at` is CLEARED so the
#:         age check cannot call it fresh, and `--train` retrains. There is no
#:         ordering for anyone to get wrong later.
#:
#:       THIS COHORT OPENS INSIDE AN OUTAGE. API-Football has been suspended
#:       since 2026-08-19 10:10:28 UTC, so the first picks under s5.3 are made
#:       with no fixtures, odds, xG or injuries from that provider. If the
#:       account is restored mid-cohort, this fingerprint spans two materially
#:       different input regimes — ledger OPS-1 records the boundary so any
#:       analysis can split rather than pool. Low pick counts in the first days
#:       are the expected consequence of a one-pick cap on a card discovered
#:       without API-Football, not a defect.
#:
#:       Verification prompt written BEFORE deployment:
#:       docs/stage-13-s53-verification-prompt.md
#: s5.4  Stage 19 (2026-08-26). ONE break covering three prediction-affecting
#:       changes, deliberately folded together rather than taken as three.
#:       Authorised on the ground that Stage 16 answered the frozen
#:       experiment's question — MODEL CLV upper bound +0.107% against a
#:       +1.85% requirement — so there is no longer an experiment being
#:       protected by holding the cohort still.
#:
#:       1. 510 phantom matches EXCLUDED from fitting via
#:          `training_exclusion_reason = 'phantom_kickoff_now_stamp'`. Marked,
#:          never deleted.
#:
#:          MEASURED EFFECT, recorded so nobody later attributes a cohort
#:          difference to it — and recorded as DECAY WEIGHT, because the count
#:          understates it:
#:            · 503 of 39,290 fitting rows = 1.280% BY COUNT
#:            · 486.8 of 17,281.6 decay weight = 2.817% BY WEIGHT at H=540d
#:            · average weight 0.9677 vs 0.4330 for a real match — 2.23x
#:          A `now()`-stamped row is a real result inserted at MAXIMUM recency
#:          weight and in the wrong sequence, so it also perturbs Elo replay
#:          order, rolling form windows and H2H recency. Its harm is
#:          disproportionate to its count, which is the opposite of the usual
#:          argument for tolerating 1.3% of a fitting set.
#:
#:          Repair would have been better and was not cheaply available: the
#:          date failed to parse because the source gave something unusable,
#:          and recovering the true kickoff means matching teams and score
#:          against a source that could not see fixtures either.
#:
#:       2. `_parse_match_date` now FAILS CLOSED. It returned `datetime.now()`
#:          on parse failure — the fourth instance of "a default that makes a
#:          failure look like a success". Future rows are refused and logged
#:          with the raw unparsed text instead of invented.
#:
#:       3. Flashscore row/time selectors repaired for the 2026 redesign:
#:          `event__match--static` became `event__match--withRowLink`, and
#:          `.event__time` no longer exists — the kickoff moved into
#:          BUILD-HASHED CSS-module classes (`wcl-scores-simple-text-01_-OvnR`)
#:          that change on every deploy. The time is now read from the row's
#:          TEXT by shape, which is stable across deploys. This changes WHICH
#:          fixtures are discovered and therefore what the model sees.
#:
#:       ONE UPSTREAM CHANGE CAUSED 1 AND 3. The same redesign that killed
#:       fixture discovery on 2026-05-30 also made every results row's kickoff
#:       unparseable, which is what manufactured the phantoms. Two symptoms,
#:       88 days apart in visibility, one cause.
#: s5.5  Stage 19 item 2a (2026-08-27). ONE change, and the split from s5.4 was
#:       FORCED, not chosen: ac8bedb was already pushed to the public remote
#:       when this landed, and rewriting a pushed commit is the history rewrite
#:       this project has refused twice. It is materially costless — NO
#:       prediction was ever stamped s5.4 (`...0976b8` has zero rows in
#:       saved_picks), so the split separates an empty cohort from an empty one.
#:
#:       football-data.org's fixture filter no longer reads `status` to answer
#:       "has this been played". MEASURED 2026-08-26: a real LaLiga fixture
#:       arrived with status='2026-08-26 19:00:00Z' — a timestamp where an enum
#:       belongs — while a call minutes earlier returned 'TIMED'. The field
#:       FLAPS, and a bare `continue` removed the fixture with no log line.
#:
#:       `utcDate` now decides scheduled-versus-played, because it is
#:       verifiable rather than guessed. `status` stays authoritative only for
#:       what a date cannot express — POSTPONED, CANCELLED, SUSPENDED, AWARDED.
#:       An unrecognised status on a future kickoff ADMITS the fixture and logs
#:       a warning; when neither field can answer, it REFUSES and logs. Never a
#:       silent continue.
#:
#:       MEASURED EFFECT: over 11 days sampled 2026-08-27 (3 back, 7 ahead),
#:       102 matches in mapped competitions returned {FINISHED: 8, TIMED: 94}
#:       and the repaired filter admits exactly the same 94. The malformed
#:       status was TRANSIENT and had already resolved. **The discovery floor
#:       does not move.** This is insurance against a recurrence, not a gain,
#:       and it is recorded that way so nobody later credits it with fixtures.
#: ── THE AMEND-OR-BUMP RULE (Stage 19, 2026-08-27) ──────────────────────────
#:
#: CODE_REVISION exists to stop picks made under different configurations being
#: pooled. POOLING CANNOT HAPPEN IN A COHORT WITH NO MEMBERS.
#:
#:   While `saved_picks` holds ZERO rows at the current fingerprint, a further
#:   prediction- or selection-affecting change AMENDS this revision's entry
#:   instead of bumping. If ANY pick carries it, BUMP.
#:
#: The guarantee is untouched: no two configurations ever share a cohort that
#: contains anything.
#:
#: VERIFY WITH A QUERY, NOT FROM MEMORY — `python -m scripts.cohort_status`
#: prints AMEND or BUMP against the live database. That requirement is what
#: makes this a refinement rather than a loophole, and it is the standard
#: applied to every other claim here.
#: ───────────────────────────────────────────────────────────────────────────
#:
#: s5.6  Stage 19 (2026-08-27). The fixture-league CIRCULARITY removed, so
#:       Flashscore is attempted for every configured league rather than only
#:       for leagues already known to have fixtures.
#:
#:       WHY THIS COULD NOT WAIT FOR THE RUN. MEASURED 2026-08-27, against
#:       production state rather than in principle: `_today_leagues` was EMPTY
#:       and `_important` held exactly {europe/champions-league,
#:       portugal/primeira-liga}, both from unsettled picks. `spain/laliga` was
#:       in NEITHER — on a day it has two fixtures. The pre-registered test
#:       would have returned UNTESTED: the instrument would not have run.
#:
#:       The set could only shrink, because membership required a fixture that
#:       only a fixture scrape could create: 12 leagues on 2026-08-10, 3 on
#:       2026-08-26, and spain/laliga absent by 08-27.
#:
#:       `_FIXTURES_BUDGET_S = 300` is now the only bound, which is what the
#:       results loop has always relied on — the two paths are symmetric
#:       instead of one being silently narrower.
#:
#:       THREE CONSECUTIVE REVISIONS CARRY ZERO PICKS — s5.4, s5.5 and s5.6 —
#:       because no daily-picks run has fired since s5.3 was stamped. Recorded
#:       so a future reader sees explained churn rather than thrash.
#:
#:       EACH SPLIT WAS FORCED, NOT CHOSEN:
#:         · s5.4 -> s5.5: ac8bedb was already pushed to the public remote, and
#:           rewriting a pushed commit is the history rewrite this project has
#:           refused twice.
#:         · s5.5 -> s5.6: ccadbfe likewise (verified by `git ls-remote`).
#:       Each was costless, because each predecessor cohort was empty.
#:
#:       THE RULE NOW PERMITS AMENDMENT WHILE EMPTY (see the heading above), so
#:       this sequence should not recur: a further change today amends s5.6
#:       rather than opening s5.7. These three are NOT collapsed retroactively —
#:       they are pushed, and history is not rewritten. Verified at commit time
#:       via `scripts/cohort_status.py`: 0 picks at `...60caed` of 1,338 total.
#: s5.7  Stage 20 (2026-08-27). ONE bump covering both selection-affecting
#:       changes, per the amend-or-bump rule: `cohort_status.py` reported
#:       s5.6 carrying 34 picks, so BUMP rather than amend.
#:
#:       1. IDENTITY GATE — one alias added, and it was the ONLY one missing.
#:          The gate fired 3 times on its first day with a working
#:          API-Football account. Classified against the provider
#:          (GET /teams?id=), not from recollection:
#:            · 604  Maccabi Tel Aviv (Israel, 1906) vs stored "Telstar"
#:              (Netherlands) — CORRECT refusal, left refused.
#:            · 531  Athletic Club (Spain, Bilbao, code BIL) vs "Ath Bilbao"
#:              — FALSE POSITIVE. It cost Barcelona vs Ath Bilbao, one of the
#:              two fixtures Stage 19 predicted; it survived only because
#:              Flashscore found it independently.
#:            · 3502 FC Iberia 1999 (Georgia, Tbilisi, 1999) vs "Saburtalo"
#:              (Tbilisi, 1999) — FALSE POSITIVE, a rename.
#:
#:          THE KNOWLEDGE FOR THE ATHLETIC CASE ALREADY EXISTED.
#:          `TEAM_NAME_ALIASES["Athletic Club"] = "Ath Bilbao"` was already in
#:          the tree and simply unreachable: it is consulted at step 2 of
#:          `_get_or_create_team_id`, while the gate refuses at step 0. So the
#:          fix is to canonicalise through that table BEFORE the anchor test,
#:          and only "FC Iberia 1999" was genuinely new.
#:
#:          The anchor rule itself is UNCHANGED — no ratio, no threshold, no
#:          widened country band. Knowledge, not tolerance.
#:
#:       2. FIXTURES WAIT 45s -> 20s (`FIXTURES_WAIT_S`), both call sites.
#:          MEASURED on run 33075828280: 285.7s of a 301.7s budget — 95% —
#:          went to leagues returning ZERO fixtures, and champions-league alone
#:          cost 90.2s against a 9.6s mean, from two 45s waits timing out. The
#:          page is not at fault: loaded directly it returns 144 rows in ~5s.
#:
#:       MEASURED EFFECT ON THE DISCOVERED-FIXTURE POPULATION, stated so a
#:       later reader does not attribute a cohort difference to guesswork:
#:         · the alias admits fixtures that were previously skipped whenever
#:           API-Football names Athletic Club or FC Iberia 1999 — 2 of 36
#:           fixtures on 2026-08-27 (5.6%), both otherwise recoverable only if
#:           another source happened to see them.
#:         · the timeout frees ~50-70s per run, reaching ~5-7 more leagues of
#:           the 7 that were never attempted. Whether that is sufficient is
#:           NOT asserted — it is the next run's measurement.
#: s5.8  Stage 21 (2026-08-30). ONE bump covering both parts.
#:       `cohort_status.py` reported s5.7 carrying 35 picks -> BUMP.
#:
#:       1. IDENTITY GATE — the Stage 20 regression removed. Canonicalisation
#:          now UNIONS raw and aliased anchors instead of REPLACING the name.
#:          Stage 20 replaced, and `TEAM_NAME_ALIASES["Standard Liege"] =
#:          "Standard"` then left {stan, standard} against {lieg, liege},
#:          refusing a pair whose RAW forms share "liege". Union can only ADD
#:          anchors, so no previously-passing pair can be refused.
#:
#:          Selection-affecting: it admits fixtures the gate was skipping.
#:          MEASURED — 4 refusals over 08-27..08-29, of which ONE was this
#:          false positive (a Jupiler Pro League fixture), so ~25% of the
#:          gate's refusals were wrong.
#:
#:       2. CRON 09:37 -> 03:00 UTC. Selection-affecting twice over: it changes
#:          WHEN prices are taken and WHICH fixtures are inside the window.
#:
#:          MEASURED EFFECT ON PICK LEAD TIME, recorded so no later reader
#:          attributes a cohort difference to the clock:
#:            · on-time runs 2026-08-14..08-25 : 4.4 - 8.8h mean lead
#:            · late run 2026-08-29            : 2.1h median (lateness
#:              COMPRESSES lead — it does not extend it)
#:            · projected at 03:00             : 7.1h at the p10 kickoff,
#:              10.7h at the median, 15.4h at p90
#:
#:          MEASURED EFFECT ON THE DISCOVERED-FIXTURE POPULATION: at ZERO delay
#:          the two crons see nearly the same card, because the earliest
#:          kickoff is 10:04 UTC and both start before it. The change is
#:          almost entirely in DELAY TOLERANCE — 0h at 09:37, 6h45m at 03:00 —
#:          and in the longer lead above. A reader comparing cohorts should
#:          expect lead-time distributions to differ and fixture COUNTS on
#:          undelayed days to be similar.
#:
#:       WHAT NEITHER FIXES: GitHub's scheduler. A 03:00 cron delayed 11h lands
#:       at 14:00 and still misses a Saturday afternoon card. OPS-3 stays open.
#:
#: s5.9  THE PER-MATCH CAP NOW KEYS ON FIXTURE IDENTITY, NOT ROW IDENTITY.
#:
#:       `max_picks_per_match` is a guarantee about a FIXTURE; `match_id`
#:       identifies a ROW. Two rows for one real fixture were two groups and
#:       therefore two independent pick slots, and the cap could not see it.
#:
#:       THE VIOLATION THAT FORCED THIS — measured, not hypothetical:
#:         2026-08-30  Deportivo La Coruña v Valencia, spain/laliga 17:30
#:           row 50920 (API-Football) -> Double Chance X2 @1.515, settled loss
#:           row 50927 (Flashscore)   -> Under 2.5 @1.56, EV -0.1924
#:         One fixture, two picks, both s5.7. The correlation filter also keys
#:         on the match, so the positively-correlated pair was never compared —
#:         the lower-EV member is exactly what it exists to drop.
#:       A second case, 2026-08-14 Sporting CP v Guimarães, carried THREE picks
#:       across two rows; it predates `bef66ca` (2026-08-23) when the cap was
#:       still 2, so only the 08-30 case violates s5.3 proper.
#:
#:       NOT A BETTER MATCHER, DELIBERATELY. `Vitória SC` and `Guimaraes` are
#:       one club sharing zero tokens — a residual already documented as
#:       unreachable by any lexical test. Aliases close the two known pairs and
#:       not the next two. This asks whether two rows are the same FIXTURE and
#:       answers from provider identity, so it holds when the matcher fails.
#:
#:       SELECTION-AFFECTING, and here is the size of it. MEASURED against the
#:       whole database 2026-08-31:
#:         · 835 pairs match the provable branch (same league + same kickoff
#:           minute + >=1 club sharing a provider id)
#:         · of those, exactly 2 would have had a second pick refused, out of
#:           1,458 saved picks — both are the violations above
#:         · the heuristic branch (no shared id, both stored name pairs
#:           similar) fires on 89 pairs, of which ZERO are provably different
#:           fixtures under any available decider: not team provider id, not
#:           `matches.apifootball_id`, not `flashscore_id`. Measured
#:           false-positive rate 0 of 89, bounded [0.0%, 40.4%] only if every
#:           pair where NEITHER row carries a provider fixture id is also
#:           wrong, which inspection contradicts.
#:
#:       A reader comparing s5.8 to s5.9 should expect NO systematic difference
#:       in pick volume — 2 refusals in 1,458 is 0.14% — and should not
#:       attribute any cohort difference to this change.
#:
#:       THE POPULATION IS A FLOOR. 750 duplicate pairs under a strict
#:       identical-kickoff test; the one confirmed violation sits OUTSIDE it,
#:       in the weak set, because identical-kickoff bought precision and lost
#:       every pair whose sources disagree on kickoff time. A wider ±26h
#:       membership test implicates 4,329 rows. The true population is between
#:       and is NOT established.
#:
#:       WHAT s5.9 ENFORCES, AND WHAT IT CANNOT REACH — qualified 2026-09-09,
#:       because this entry was written as though the per-fixture cap were
#:       enforced outright, and it is not.
#:
#:       Of 61,329 co-scheduled pairs (same league, identical kickoff minute):
#:         · 844 are reachable by BRANCH 1 (a shared resolved provider club id)
#:         · 89 by BRANCH 2 (both stored name pairs similar)
#:         · 202 fall in BRANCH 3a — exactly ONE side matches — and s5.9
#:           CANNOT SEE THEM. Both of that class's double-picked members turned
#:           out to be real violations.
#:         · 60,188 match on neither side; that is the ordinary matchday, not a
#:           blind spot.
#:
#:       So the guarantee holds for 933 pairs and has a 202-pair residual that
#:       was UNBOUNDED until 2026-09-09. 187 of those 202 (92.6%) involve a row
#:       carrying NO provider identity at all — an identity-COVERAGE gap, not a
#:       naming one — and only 15 are a pure naming residual.
#:
#:       Two violations have occurred: 2026-08-30 Deportivo v Valencia (two
#:       correlated markets) and 2026-09-08 NEC v Nijmegen (the SAME selection
#:       twice, 1X2 Home Win at 1.62 and 1.60). The second happened UNDER s5.9.
#:       Both second picks are now disposition='consolidated'.
#:
#:       This is a MECHANISM WITH A STATED BOUND, not a promise — the same
#:       treatment the identity gate's own residual received.
#:
#:       WHY NOW rather than later: every CLV interval resamples FIXTURES, so
#:       duplicated rows inflate the cluster count and narrow every interval.
#:       That separation was temporal — captures ran 08-14..08-27 and the live
#:       duplicates are dated 08-28 onward — so it would not have survived the
#:       next capture window unaided.
#:
#:       AND THE VERDICT IS NOW STABLE IN BOTH DIRECTIONS, which is the
#:       strongest argument for having shipped before the window rather than
#:       after it:
#:
#:       RETROSPECTIVELY. Re-measured 2026-09-02 under the predicate this
#:       revision actually ships (>=1 shared provider club id, or both stored
#:       name pairs similar), ONE of the 48 MODEL observations sits on a
#:       duplicate-pair row: obs 173 on match 49496, the 2026-08-14
#:       Sporting CP v Guimarães fixture. Its twin 49520 carries 2 observations
#:       and ZERO captures, and that fixture's closing window has passed — so
#:       those observations can never be captured. The cluster count is
#:       therefore fixed at 48 observations over 48 distinct fixtures, and
#:       `deff = 1.00` cannot move retrospectively.
#:       (An earlier check reported 0 rather than 1; it required BOTH sides to
#:       match, which is narrower than the shipped rule. The conclusion held,
#:       the instrument did not.)
#:
#:       PROSPECTIVELY. A duplicated fixture now yields ONE pick, so it yields
#:       one observation. The second pick is refused before it can become a
#:       second cluster member.
#:
#:       So the guarantee protecting the per-match cap is the same guarantee
#:       protecting every confidence interval in this project. They are not two
#:       benefits; they are one mechanism read from two ends.
#:
#: s5.10 TEAM IDENTITY REPAIR — the residual removed at its source, not routed
#:       around. Stage 22, applied 2026-09-10.
#:
#:       `teams.apifootball_team_id` was incompletely and sometimes WRONGLY
#:       populated, and every mechanism keyed on it inherited the gap — s5.9's
#:       branch 1 above most of all. Four symptoms carried as four separate
#:       deferrals for three weeks; one subject.
#:
#:       THE MEASURED EFFECT ON THE DISCOVERED-FIXTURE POPULATION, which is the
#:       number a later reader needs to attribute a cohort difference to this
#:       revision rather than guess at it. Fixtures whose BOTH participants
#:       resolve to a provider id — the API-Football route, and s5.9 branch 1's
#:       precondition:
#:
#:         since 2026-08-01   1236/1544  80.05%  ->  1378/1544  89.25%  (+142)
#:         last 365 days      8108/9717  83.44%  ->  9160/9717  94.27%  (+1052)
#:
#:       WHAT WAS DONE. 2 wrong provider ids cleared, 45 rows absorbed into
#:       provable shared-id components, 84 unresolved rows merged into
#:       evidenced twins. teams 1577 -> 1448; unresolved 178 -> 94; rows
#:       sharing a provider id: 0. No match, pick or reference row was
#:       destroyed; every reference was repointed and verified to zero before
#:       any row was deleted, and zero dangling foreign keys remain.
#:
#:       THE VERIFICATION ITSELF HAD TO BE REWRITTEN, and the reason belongs in
#:       the record. Its first form asserted every table count was UNCHANGED.
#:       It aborted on the live database — players +31, injuries +50,
#:       injury_observations +100 — because a scraper was writing CONCURRENTLY
#:       between the before and after snapshots. The abort was correct; the
#:       check was not. It asserted the whole database was static, which is a
#:       property of neither this repair nor production. Replaced with
#:       invariants about what the operation actually does: every merged row
#:       gone with zero surviving references, no UNRELATED team row vanished,
#:       and no count DECREASED. A repoint cannot lose a row; it can only fail
#:       to move one.
#:
#:       WHY THIS IS SELECTION-AFFECTING, and therefore why it is a revision
#:       rather than a cleanup. Merging changes which fixtures resolve,
#:       therefore which are priced, therefore which are picked. AND Elo and
#:       Poisson both key on `team_id` and rebuild from `matches` on every fit,
#:       so a club whose history was split across two rows now trains as ONE
#:       club. A reader comparing s5.9 to s5.10 should expect a real difference
#:       in both the fixture set and the ratings, in the direction of more
#:       history per club.
#:
#:       THE SPEC'S OPERATION ORDER WAS WRONG AND WAS CORRECTED. It ordered the
#:       merges before the wrong-id clearance. But the twin-finder anchors on
#:       provider ids, and the clearance exists precisely because two ids are
#:       wrong — so running merges first let a known-wrong id anchor a merge.
#:       It did: with row 411 (`Rakow`) still holding af=350, the evidence
#:       proposed `Cracovia -> Rakow`, two distinct Polish clubs. Reordered to
#:       clearance-first, that proposal does not arise at all.
#:
#:         A repair that consumes the field a later step is about to fix must
#:         run AFTER that step, not before it.
#:
#:       HOW THE MERGES WERE DECIDED, AND THE TWO VETOES THAT WERE REJECTED.
#:       Evidence is a SHARED FIXTURE — s5.9 branch 1 run in reverse, no name
#:       consulted — falling back to `same_team_strict` only where that is
#:       silent. But shared-fixture evidence IS NOT PROOF, because `matches`
#:       itself carries mis-resolved rows: 14 `france/ligue-2` rows place
#:       `St. Pauli` in fixtures belonging to `Pau FC`, written by API-Football
#:       while the genuine rows came from Flashscore, because `_tok_match`'s
#:       prefix rule makes "pauli".startswith("pau") true. The evidence
#:       faithfully reported the consequence and proposed fusing two clubs.
#:       A comparison is only as good as the resolution state of its inputs.
#:
#:       Two vetoes were measured and REJECTED. Primary domestic league rejects
#:       `Wrexham AFC`/`Wrexham` and `Celtic FC`/`Celtic`, whose unresolved rows
#:       appear only in European ties, and caught 1 of 3 known-bad merges.
#:       `team_names_similar` MISSES the worst case — it returns True for
#:       "Pau FC"/"St. Pauli", by the very prefix rule that created the
#:       corruption — while rejecting nine correct merges. Both reason about
#:       names. The rule adopted does not:
#:
#:         A CLUB CANNOT PLAY TWO DIFFERENT FIXTURES AT THE SAME TIME, AND
#:         CANNOT PLAY ITSELF.
#:
#:       It disqualified 3 of 3 known-bad merges and one further case
#:       (`Sport Lisboa e Benfica`/`Benfica`, one collision), and rejected
#:       nothing else. It can only ever REFUSE a merge, never create one.
#:       Refusing is free; a wrong merge fuses two clubs' histories.
#:
#:       WHAT IT DOES NOT CLOSE. 94 rows stay unresolved — no defensible twin,
#:       and leaving them is cheaper than a wrong merge. Two were refused as
#:       AMBIGUOUS: `Sporting Clube de Braga` and `Sporting Clube de Portugal`
#:       each drew two candidates (`Sporting CP` and `Braga`), which is the
#:       Sporting CP naming residual the spec ruled must stay SEPARATE and
#:       UNTAKEN — a threshold is tolerance, an alias is knowledge, and the two
#:       must not be bundled or the cohort break becomes unattributable.
#:       Branch 3's residual does not go to zero: this removes the CONDITION
#:       for most of it, and does not prove the class empty.
#:
#:       MACCABI TEL AVIV WAS NOT CREATED. Clearing row 124's wrong id unblocks
#:       creation; it does not perform it. Nothing creates the row until
#:       Maccabi next appears in a fetched fixture, which depends on European
#:       participation and is not in this system's control. Stated separately
#:       so a reader does not assume the clearance completed the creation.
#:
#:       A FIFTH SYMPTOM WAS FOUND AND IS NOT FIXED HERE: match rows assigned to
#:       the wrong team row by name-first matching at ingestion (the Pau/
#:       St. Pauli class, 14 rows on one club). The merge is guarded against it
#:       and does not propagate it, but the rows remain mis-assigned and the
#:       `_tok_match` prefix rule that creates them is unchanged. Recorded as
#:       its own item rather than folded into this revision.
#:
#:       AMENDED 2026-09-10 (cohort empty, 0 picks stamped — amend-while-empty,
#:       not a bump). THE PICK-TIME ODDS PATH NOW CLAIMS FROM THE CREDIT LEDGER.
#:
#:       `TheOddsScraper.update()` — the path daily-picks uses to price the
#:       card — passed no quota, so it claimed nothing, could not be declined,
#:       and never reconciled. Measured 2026-09-01..09-10 it spent 204 of the
#:       346 credits the provider actually charged: 59% of consumption was
#:       invisible to the mechanism built to bound it, and the ledger diverged
#:       from the provider on 7 of 13 spending runs.
#:
#:       WHY IT IS PREDICTION-AFFECTING AT ALL: the ledger can now decline a
#:       pick-time odds request, and a declined league is priced from stale or
#:       missing odds, which changes what the bookmaker blend reads. The window
#:       where that bites is the 50-credit safety margin — the ledger refuses at
#:       450 while the provider would still serve to 500.
#:
#:       MEASURED EFFECT OVER THE LAST TEN DAYS: ZERO. Replaying both consumers
#:       against the 450 limit, peak ledger usage was 342. No request was
#:       declined, and none that the provider would have honoured. The change is
#:       behaviour-neutral on the observed window and only acts at the boundary
#:       it exists to defend.
#:
#:       The per-run ceiling is DISABLED on this path (max_credits_per_run=0).
#:       The 24-credit default is sized for the imminent-refresh job; inheriting
#:       it here would decline roughly half of every day's leagues, because this
#:       path routinely wants 20-23 (40-46 credits). That would be a volume
#:       change wearing an accounting fix's clothes.
#:
#:       ALSO IN THIS AMENDMENT, because one edit closes both: the 429 branch
#:       now reads the response headers BEFORE returning. It used to return
#:       first, so on the one response that says "you are out of credits" the
#:       pipeline learned nothing and `reconcile()` went permanently blind at
#:       the moment its number mattered. And 429 is now split by
#:       `x-requests-remaining`: 0 is EXHAUSTION (logged CRITICAL), above zero
#:       is RATE LIMITING (logged WARNING). All six 429s in this project's
#:       history had credits in hand — 402, 306, 276, 67, 31, 362 — and all six
#:       were reported as "quota exhausted". The alarm for the real event had
#:       been spent on a different one.
#:
#:       THE PER-RUN CEILING IS THE SECOND LINE OF DEFENCE AND IT IS NOW OFF
#:       ON THIS PATH. `max_credits_per_run=0` means the monthly ledger is the
#:       ONLY guard on the consumer responsible for 59% of spend. A runaway
#:       single run is unlikely because the request count is bounded by the
#:       number of leagues with fixtures today — but that bound is INCIDENTAL,
#:       set by the football calendar rather than by this code. Observed maximum
#:       23 leagues / 46 credits (2026-09-05). Stated so a later reader does not
#:       read the 0 as an oversight.
#:
#:       STALE-PERIOD GATE, added 2026-09-11 in the same amendment.
#:       `_load_persisted_credits()` read `remaining` and never read `updated`,
#:       though `_persist_credits` has always written both. August closed at 15
#:       and cleared the <=10 hard skip by five credits; September is projected
#:       to exhaust around 09-14, so 1 October's first run would have skipped
#:       the odds fetch entirely on a figure describing a finished month, with a
#:       full 500-credit tier unused. A figure from a previous billing period is
#:       NOT a low reading — it is NO reading, and the answer to no reading is to
#:       PROBE (/v4/sports is free), not to skip. Same three-states-collapsed-
#:       into-two as `[]` vs `None` and 429-with-credits vs 429-with-zero: the
#:       third instance in this module.
#:
#:       This one CAN change a prediction, in the direction of making more: a
#:       run that would have skipped the odds fetch now performs it. Recorded
#:       under s5.10 because the cohort is still empty.
#:
#:       Neither sub-change alters a prediction on its own; both alter what the
#:       pipeline KNOWS about its own budget, and the first can alter which
#:       odds exist when a pick is priced. Recorded here rather than in a
#:       separate revision because the cohort was empty and the two are one
#:       edit.
CODE_REVISION = "s5.10"


def _stable(value: Any) -> Any:
    """Normalise a config value so equal settings always hash equally."""
    if isinstance(value, dict):
        return {str(k): _stable(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple, set)):
        items = [_stable(v) for v in value]
        # excluded_markets / gates are order-insensitive sets in meaning.
        try:
            return sorted(items, key=lambda x: json.dumps(x, sort_keys=True))
        except TypeError:
            return items
    if isinstance(value, float) and value.is_integer():
        # 0.8 and 0.80 must hash the same; so must 1 and 1.0.
        return float(value)
    return value


def fingerprint_inputs(config) -> Dict[str, Any]:
    """The exact settings that feed the fingerprint, for logging and debugging."""
    out: Dict[str, Any] = {"__code__": CODE_REVISION}
    for key in TRACKED_KEYS:
        try:
            out[key] = _stable(config.get(key, None))
        except Exception:
            out[key] = None
    return out


def fingerprint(config) -> str:
    """6-char digest of the prediction-affecting configuration."""
    payload = json.dumps(fingerprint_inputs(config), sort_keys=True,
                         separators=(",", ":"), default=str)
    return hashlib.blake2s(payload.encode(), digest_size=3).hexdigest()


def model_version(config) -> str:
    """The identifier to stamp on a prediction.

    Never raises: a prediction must not fail because versioning failed. On error
    it returns a clearly-marked unknown value rather than a plausible-looking
    wrong one, because a wrong version silently pools incomparable cohorts.
    """
    try:
        label = config.get("models.experiment_label", DEFAULT_LABEL) or DEFAULT_LABEL
        frozen = config.get("models.experiment_frozen_at", DEFAULT_FROZEN_AT) or DEFAULT_FROZEN_AT
        return f"{label}_{frozen}.{fingerprint(config)}"
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"model_version could not be computed ({e}) — stamping 'unknown'")
        return "unknown"


def describe(config) -> str:
    """Human-readable breakdown for --stats and the Stage 5 report."""
    lines = [f"model_version = {model_version(config)}", "tracked settings:"]
    for key, value in fingerprint_inputs(config).items():
        lines.append(f"    {key:<45} {value}")
    return "\n".join(lines)
