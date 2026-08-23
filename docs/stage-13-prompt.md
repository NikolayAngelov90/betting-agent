# Stage 13 — Daily Operations Audit, Fixture-Identity Defect, One-Pick-Per-Match

Production is running commit `4557a23` on `main`. Stage 12 (`docs/stage-12-production-audit-prompt.md`) was a read-only evidence gate. Stage 13 is different: **you have full implementation authority** inside the boundaries declared in §0.

This stage has three parts and they are ordered deliberately:

* **Part A** — verbatim inspection of every CI run not yet audited, and a permanent mechanism so "not yet audited" has a definition tomorrow.
* **Part B** — one confirmed production defect: a pick was published for the wrong fixture.
* **Part C** — one user-mandated policy change: at most **one** pick per match, and it must be the best one.

Do Part A first and do not let Parts B and C contaminate it. Part A is a survey — you are looking for the defects you have *not* been told about. If you go hunting for the CSKA Sofia bug first, you will read the logs for that and stop reading.

---

## 0. Boundaries

**In scope — you may change:**

* CI-log auditing tooling, a run ledger, a repeatable daily audit procedure
* fixture/team identity resolution and cross-source name matching
* the per-match pick cap and the ranking rule that decides which pick survives it
* the Claude review path where it consumes multiple picks on one match
* tests, config, documentation, migrations required by the above

**Out of scope — do not touch:**

* the predictive core. No new models, no retuning of `ensemble_weights`, `bookmaker_blend_weight`, `dixon_coles_rho`, `strength_half_life_days`, calibration, or the Bayesian weight learner. Not as a "small improvement", not as a side effect.
* `betting.min_expected_value`, `min_confidence`, `min_ev_confidence_score`, `kelly_fraction`, `min_odds`, `max_odds`, `excluded_markets`, `gates`. These are the frozen experiment's parameters.
* `paper_trading_mode`. It stays `true`.
* the 8 disabled `edge` gates. They stay off.
* historical `saved_picks`, `pick_observations`, `model_version`, `is_paper`. No backfill, no rewrite, no deletion of settled or pending history.

**No new statistical claims.** Nothing in this stage may be justified by settled outcomes. If you find yourself writing "this market won 61% of 32 picks", stop — that is exactly the noise-fitting the 2026-08-07 audit condemned.

---

# PART A — Daily CI audit

## A1. Establish the audit ledger

There is currently no record of which CI runs have been looked at, so "runs not yet inspected" is undefined. Create it.

Add `docs/ci-audit-ledger.md`, committed to the repo, one row per run per workflow:

| run_id | workflow | started (UTC) | conclusion | steps failed | audited on | verdict | notes |
| ------ | -------- | ------------- | ---------- | ------------ | ---------- | ------- | ----- |

`verdict` is one of `CLEAN`, `DEGRADED`, `BROKEN`.

* `CLEAN` — every step did what it was supposed to do.
* `DEGRADED` — the run completed but something silently did less than intended (a scraper returned nothing, the Claude review no-opped, a quota claim was refused, a step swallowed an exception).
* `BROKEN` — a step failed or produced wrong output.

For this first pass, audit **every run of all three workflows started on or after 2026-08-11 00:00 UTC** (the Stage 12 deployment boundary), plus any earlier run you find evidence for in Part B. Do not audit selectively.

## A2. Read the logs verbatim

Use the GitHub CLI: `gh run list --workflow <file> --limit N` and `gh run view <run_id> --log`. Save each full log under `ci_logs/run_<run_id>/` following the existing layout so it is re-readable without re-fetching.

Read **the whole log of every step**, not the tail, not only the failed steps, not only lines matching `ERROR`.

**`conclusion: success` is not evidence.** Every core step in `daily-picks.yml` runs under `continue-on-error: true`. A run is green whenever the *runner* survived, which is not the same as the pipeline having worked. This is the single most likely place for a silent regression to be sitting right now.

For each run report, per step:

* what the step was supposed to produce
* what it actually produced (counts, not adjectives)
* every exception, traceback, retry, timeout, `WARNING`, `PICK_REJECTED`, and swallowed-exception `debug` line
* every place a count is zero or implausible

Pay specific attention to:

| Area | What to establish from the log |
| --- | --- |
| `--update` | fixtures scraped per source, odds rows written, teams created, teams *newly* created (a new team row mid-season is a fixture-identity smell), injuries fetched |
| Flashscore / Camoufox | whether the browser actually loaded pages or silently returned empty league pages |
| `--train` | whether models retrained or were skipped, and on how many rows |
| `--picks` | candidates analysed, `PICK_REJECTED` reasons and counts, picks saved, `pick_observations` written |
| Claude review | backend used, per-match decisions, KEEP vs CHANGE counts, Pro session-limit short-circuits, `Pick review failed for match` warnings, prose-vs-decision divergence warnings |
| Telegram | messages actually delivered vs constructed |
| `--settle` | picks settled, voided, stuck-pick sweeps |
| closing-lines workflow | credits claimed/spent, observations considered/resolved/missing/late/invalid |
| paper-trading report | MODEL and FINAL counts, whether it ran at all |
| failure-alert step | whether it fired, and whether it *should* have fired and did not |

## A3. Classify everything you find

Produce one table of every anomaly found in Part A:

| # | Run | Step | Symptom (quoted from log) | Root cause | Severity | Fix now / defer |
| - | --- | ---- | ------------------------- | ---------- | -------- | --------------- |

Severity: `CRITICAL` (wrong picks published, wrong data persisted, money-relevant), `HIGH` (a pipeline stage silently did nothing), `MEDIUM` (degraded coverage), `LOW` (noise).

Fix `CRITICAL` and `HIGH` in this stage. List `MEDIUM`/`LOW` with a proposed fix and leave them.

## A4. Make it repeatable

The purpose of this stage is not one audit — it is a daily one. Produce:

1. `.claude/commands/daily-ci-audit.md` — a slash command that performs A2/A3 for every run not already in the ledger, appends to the ledger, and prints a short verdict. Follow the style of the existing `.claude/commands/review-daily-picks.md`.
2. A hardening of the CI failure alert so that `DEGRADED` is detectable without a human reading logs. At minimum, the daily workflow must fail loudly (Telegram) when any of these is true:
   * `--picks` saved 0 picks while fixtures were available
   * a scraper returned 0 rows for all leagues
   * the Claude review reviewed 0 of N matches
   * `pick_observations` written ≠ 2 × picks saved
   * the Odds API ledger refused a claim

   Implement it as an explicit post-run assertion step, not as a grep over the log text.

---

# PART B — Wrong fixture published (CRITICAL)

## B1. The observed defect

In the CI run of **2026-08-13**, a pick was generated and published to Telegram for **`CSKA Sofia vs Telstar`**. That fixture does not exist. The real match was **`CSKA Sofia vs Maccabi Tel Aviv`**.

This is the most serious class of bug this system can have. A prediction attached to the wrong opponent is not a bad prediction — it is a prediction about nothing. Everything downstream is corrupt: the features were computed against the wrong team, the odds were matched to the wrong market, the `pick_observation` records a price that belongs to a different game, the closing line will be resolved against a different game, and settlement will grade it against a result that was never in question.

## B2. Establish the facts before theorising

From the production database (read-only) and the 2026-08-13 logs, determine:

* the `match_id`, `home_team_id`, `away_team_id`, `league`, `match_date` of the row the pick was attached to
* which source created that row, and when
* whether a *separate* correct row for `CSKA Sofia vs Maccabi Tel Aviv` also exists
* whether `Telstar` and `Maccabi Tel Aviv` are distinct `teams` rows, and when each was created
* which source supplied the odds attached to the pick, and under which team names
* whether the wrong name entered at scrape time, at team resolution, at odds matching, or only at *display* time (the pick may have been right and the rendered string wrong — these have completely different fixes)
* how many other picks, past or pending, sit on fixtures with the same defect signature

Report the counts. Do not stop at the one match Niki happened to notice.

## B3. Hypotheses to test explicitly

Test each, state the evidence, state the verdict. Do not accept the first one that fits.

1. **Fuzzy cross-source name collision.** `src/scrapers/theodds_scraper.py` matches team names with `difflib.SequenceMatcher` at ratio ≥ 0.75 (lines ~279, ~293), and `footballdataorg_scraper.py` at ≥ 0.80 after suffix stripping. Compute the actual ratios for `Telstar` against `Maccabi Tel Aviv` and its normalised/tokenised variants under each of those code paths. A threshold that admits this pair admits a whole class of them.
2. **Stale fixture from an unresolved draw.** UEFA qualifying fixtures exist as placeholders or prior-round entries before the opponent is known. Check whether the row predates the draw and was never corrected.
3. **API-Football fuzzy link.** `apifootball_scraper.py` (~line 823) links to an existing match "by league + date + fuzzy team name". Check whether that linked a CSKA fixture to the wrong opponent row.
4. **Team-row aliasing.** Two different clubs collapsed into one `teams` row, or one club split across two.
5. **Display-only defect.** The stored ids were correct and only the rendered `match` string was wrong.

## B4. Fix

The fix must be structural, not a special case for these two clubs.

Requirements:

* **Cross-source team matching must not resolve two clubs to each other on string similarity alone when they are in different competitions or countries.** Similarity may *propose* a match; something verifiable must *confirm* it (an id mapping, a curated alias, a league/country agreement, or a fixture-date-and-opponent agreement). If nothing confirms it, the correct behaviour is to **refuse the match and log it**, not to guess. An unmatched fixture costs one missing pick; a wrongly matched fixture costs a corrupt record.
* **A fixture whose identity is not fully resolved must not be analysed.** Add an explicit pre-analysis validity check: both teams resolved to real rows, both consistent with the league, kickoff in the future, odds matched to the same two teams the fixture names. Fail closed — no pick.
* **Log every refusal** with both candidate names and the reason, so the daily audit in Part A can see how often it fires.
* Add curated aliases only as a supplement, never as the mechanism.

Then re-run the resolution over the current fixture set and report how many fixtures the new check refuses and why. If it refuses a large fraction, the check is wrong — say so rather than shipping it.

## B5. Cleanup of affected records

For picks already created on defective fixtures:

* do **not** silently delete or rewrite them
* identify them precisely and report the list
* propose the correct disposition (void with a recorded reason vs leave and exclude from CLV) and explain which one preserves the experiment's integrity
* implement only after stating the choice explicitly in the report

`is_paper` and `model_version` remain write-once regardless.

---

# PART C — One pick per match, and it must be the best one

## C1. What is required

At most **one** saved pick per match, ever, across reruns and shards. Not two. The one that survives must be the single best selection the system can make for that fixture.

## C2. Why (state this correctly in the code comment)

This is a **policy decision by the operator**, not a finding derived from settled outcomes. Write it that way. It is not justified by ROI on any market and must not be defended with one.

The operational reason is real and belongs in the comment: with two picks on a match, the Claude review's relationship to that match becomes ambiguous. `finalize_picks_with_claude` groups by `match_id` but `_apply_decision` binds the verdict to `picks[0]` only; `_sync_recs_from_db` then zips recs to DB rows by EV order, which can re-map a decision onto the wrong pick; and the CHANGE path contains a consolidation branch that *deletes* the primary pick when another pick on the same match already holds the target selection. That branch exists because switching a second pick onto the first's selection would duplicate the bet. One pick per match removes the entire ambiguity class rather than patching it.

## C3. Current behaviour you must change

* `max_picks_per_match` is a **hardcoded Python default of `2`** — `src/agent/betting_agent.py` line ~1239 (analysis entry point) and line ~1718 (`finalize_picks`). It is **not** a config key. It must become one.
* The per-match cap sorts each match's group by `(-confidence, selection)` and keeps the top N — but the portfolio ranking is `EV × confidence × agreement_bonus × contrarian_bonus`. **These are different orders.** With a cap of 2 the discrepancy rarely mattered. With a cap of 1 it decides every match. "The best possible pick" must mean one thing, and it must be the same thing the portfolio ranking means.

## C4. Implementation requirements

1. Add `betting.max_picks_per_match: 1` to `config/config.example.yaml` with a comment stating it is an operator policy. Remove the hardcoded default; read it from config at both call sites. Keep a sane fallback if the key is absent, and make the fallback `1`.
2. **Unify the ordering.** The per-match cap must select using the same score as `_rank_key` — `EV × confidence × agreement_bonus × contrarian_bonus`, with `(match_id, market, selection)` as the deterministic tiebreak. Extract that score into one named function used by both the global sort and the per-match cap so they can never diverge again. Report how many of the last 30 days' matches would have kept a *different* pick under the unified rule than under the old confidence-only rule.
3. **Forced picks obey the cap too.** The WC (`wc_pick_every_match`) and club forced-pick paths (`club_pick_min_coverage`, `club_pick_min_blend`, `club_pick_min_ev`) must not be able to add a second pick to a match that already has one.
4. **Rerun safety.** The existing pre-population from already-saved picks means `slots = max(0, 1 - already)` → a rerun of `--picks --force` adds nothing to a match that already has a pick. Confirm this holds and that it does not instead *replace* or delete the existing pick. Add a test.
5. **The Claude review path.** With one pick per match: `_apply_decision`'s `picks[1:]` consolidation branch becomes unreachable in normal operation. Do not delete it — keep it as a defensive invariant, but make it *assert and log loudly* if it is ever entered, because entering it means the cap leaked. `_sync_recs_from_db`'s zip-by-EV becomes a one-to-one map; simplify it so a mis-zip is structurally impossible.
6. **Duplicate guard.** The in-memory dedup key `(match_id, market, selection)` and the DB unique index `(match_id, selection, pick_date)` are not the same key. Verify a CHANGE cannot produce a row that passes one and violates the other, and that a CHANGE onto a selection already held is impossible now that only one pick per match exists. Add a test for the CHANGE path specifically.
7. **Correlation filter.** `_filter_correlated_picks` operates on picks within a match. With one pick per match it can no longer fire for same-match pairs. Do not remove it — verify it, note in the code that its same-match branch is now unreachable by construction, and confirm no test depends on the two-pick case in a way that now passes vacuously.

## C5. Cohort consequence — do not skip this

This change alters **which picks the system persists**. That is precisely the situation `CODE_REVISION` exists for. `src/models/model_version.py` records the Stage 8 precedent verbatim: *"SELECTION-affecting, not prediction-affecting: the model's probabilities are untouched, but the set of picks it persists changed, and a changed population of predictions is a different experiment."*

Therefore:

* add `betting.max_picks_per_match` to `TRACKED_KEYS`
* bump `CODE_REVISION` to `s5.3` and document it in the history block with the same rigour as the `s5.2` entry
* the new `model_version` starts a **new cohort**. Existing picks keep their old version. Nothing is restamped.
* the paper-trading report must not pool the cohorts. State plainly in the Stage 13 report what this costs: the 500-closing-line checkpoint counter for the new cohort **starts from zero**, and the previously accumulated MODEL/FINAL observations belong to the old cohort. If that is unacceptable, say so and stop — do not quietly merge them.
* update the README where it states "Max 2 picks per match" and the "18.9% of fixtures carry two picks / 31.8% of all picks" cluster statistic, which becomes historical rather than current. Do not delete the historical figure — label it.

## C6. What the evidence bar does and does not require here

`scripts/run_baseline.py` paired bootstrap is required for changes that **claim a predictive improvement**. Part C claims none — it is a policy constraint. Do not attempt to justify it with a bootstrap and do not revert it if a bootstrap is inconclusive.

Do, however, report descriptively: over the last 30 days, how many picks would not have been made, which markets they were in, and how many matches would have kept a different selection. Report it as an observed consequence of the policy, not as evidence for it.

---

## D. Tests

Every fix in Parts B and C needs a test that fails before it and passes after.

Required minimum:

| Test | Asserts |
| --- | --- |
| fixture identity | a fixture whose two teams do not resolve consistently is refused, not analysed |
| name matching | a similarity-only match across different competitions is refused; a confirmed match is accepted |
| the specific pair | `Telstar` and `Maccabi Tel Aviv` never resolve to each other under any code path |
| per-match cap | at most 1 pick per match, single run |
| per-match cap, rerun | a second `--picks --force` adds no pick and deletes none |
| per-match cap, sharded | the union across shards still yields 1 |
| best-pick rule | the surviving pick is the top-ranked one by the unified score, not the highest-confidence one, when they differ |
| forced picks | WC and club forced paths cannot exceed the cap |
| review CHANGE | a CHANGE on a single-pick match produces exactly one row, one MODEL observation, one FINAL observation, no duplicate, no delete |
| review consolidation branch | if entered, it logs loudly (assert the log, not the behaviour) |
| cohort | `max_picks_per_match` is in `TRACKED_KEYS`; changing it changes `model_version`; `CODE_REVISION == s5.3` |
| config identity | `config.yaml` and `config.example.yaml` still agree on every tracked key |

`pytest -q` must be green — the suite is currently 626 tests. Report the new count. A test that now passes vacuously is a regression; find them.

---

## E. Deliverables

1. `docs/ci-audit-ledger.md` — populated for every run since 2026-08-11.
2. `ci_logs/run_<id>/` — full saved logs for each audited run.
3. `.claude/commands/daily-ci-audit.md` — the repeatable procedure.
4. `docs/stage13-daily-operations-2026-08-14.md` — the stage report, in the style of the existing stage reports: what was measured, what was found, what was changed, what was deliberately not changed, and the corrections where an earlier hypothesis was overturned.
5. Code, config, tests and README updates per Parts A, B, C.
6. One final evidence table:

| Area | Result | Evidence | Status |
| --- | --- | --- | --- |
| CI runs audited (count) | | | |
| Anomalies found / fixed / deferred | | | |
| Silent-failure alerting | | | PASS/FAIL |
| Wrong-fixture root cause identified | | | PASS/FAIL |
| Fixture identity check in place | | | PASS/FAIL |
| Affected historical picks identified | | | PASS/FAIL |
| One pick per match enforced | | | PASS/FAIL |
| Best-pick rule unified with ranking | | | PASS/FAIL |
| Forced picks obey the cap | | | PASS/FAIL |
| Review CHANGE cannot duplicate | | | PASS/FAIL |
| `CODE_REVISION` bumped, cohort documented | | | PASS/FAIL |
| Paper mode still ON | | | PASS/FAIL |
| Predictive core untouched | | | PASS/FAIL |
| Tests green | | | PASS/FAIL |

---

## F. Hard rules

1. **Report before you commit.** Land Part A's findings as a report first. If Part A turns up something more severe than the CSKA defect, stop and say so — Parts B and C wait.
2. **No fix without a reproduction.** If you cannot demonstrate the defect, you have not found it. Say "not reproduced" rather than fixing a plausible-looking line.
3. **Fail closed, never guess.** Anywhere this stage adds a decision about identity, the uncertain branch must refuse and log, not approximate.
4. **No scope drift into the model.** If a change starts to look like it improves predictions, that is the signal that it is out of scope.
5. **No history rewriting.** Nothing restamps `model_version` or `is_paper`, nothing backfills `pick_observations`, nothing edits settled results.
6. **One commit per part**, with a message that says what changed and why, in the repo's existing style.
7. If any invariant in `tests/test_experiment_invariants.py` fails: **STOP, do not fix, report** — the failure, the affected rows, the root cause, the severity, the minimal fix, whether a migration is needed, whether the frozen identity changes, and whether production should be paused.

When everything is done and green, declare:

`STAGE 13 — DAILY OPERATIONS AUDIT COMPLETE`

If Part A alone justifies stopping, declare:

`STAGE 13 — HALTED AT PART A` and explain why.
