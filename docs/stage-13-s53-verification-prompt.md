# Stage 13.1 — s5.3 Production Verification & Evidence Gate

**Read-only. Gather evidence. Fix nothing.**

Written *before* `s5.3` was deployed, deliberately. A verification prompt drafted
after a deployment tends to check what was built rather than what was claimed —
so this one is written from the claims, while they are still claims.

The first run under `s5.3` changes almost everything at once: a new fingerprint,
the history mirror discarded and rebuilt, the ML pickles refused and retrained,
the exclusion filter live at 14 sites, the identity gate live at three, one pick
per match, a new ranking definition, and two newly marked populations. Each
section below states **what was claimed** and **what would disprove it**.

---

## Frozen identity for this cohort

```
CODE_REVISION        s5.3
model_version        stage5_baseline_20260807.098437
previous cohort      s5.2 / stage5_baseline_20260807.485823
migration            008_evidence_and_training_exclusion (applied 2026-08-23)
```

Anything that disagrees with these two strings is a finding, not a rounding
error.

---

## 0. The context this cohort opens in — read first

**This cohort opens inside the OPS-1 outage window.** API-Football has been
suspended since 2026-08-19 10:10:28 UTC, so `s5.3`'s first picks are produced
with no fixtures, no odds, no xG and no injuries from that provider.

Two consequences to hold throughout:

1. **Low pick counts are expected, not a defect.** One pick per match, on a card
   discovered without API-Football. Do not read a thin day as a failure of Part
   C. Report the count; do not alarm on it.
2. **If the account is restored mid-cohort, `…098437` spans two materially
   different input regimes.** Record the restoration timestamp in OPS-1 the
   moment it happens, so this cohort can be split rather than pooled. This is
   the same configuration-not-data blindness OPS-1 exists to record.

---

## 1. The fingerprint on new picks

**Claimed:** every pick saved after deployment carries
`stage5_baseline_20260807.098437`, and no pick carries a version that exists in
neither cohort.

```sql
SELECT model_version, count(*), min(pick_date), max(pick_date)
FROM saved_picks WHERE pick_date >= '<deploy date>'
GROUP BY 1 ORDER BY 2 DESC;
```

**Disproves it:** any `…485823` after deployment (the bump did not reach
production), or a third value (something else moved).

## 2. Exactly one pick per fixture

**Claimed:** `betting.max_picks_per_match: 1`, and the survivor is the highest
ranked by `_rank_key` — not the highest-confidence one.

```sql
SELECT match_id, count(*) AS picks
FROM saved_picks WHERE pick_date >= '<deploy date>'
GROUP BY 1 HAVING count(*) > 1;
```

**Disproves it:** any row returned.

Then, separately: in the run log, find a match where `PICK_REJECTED
reason=same_fixture_limit` fired and confirm the **surviving** pick has the
higher `EV × confidence × agreement × contrarian`, not merely the higher
confidence. A cap that keeps the wrong pick is the failure this part was for.

## 3. The mirror was discarded and rebuilt

**Claimed:** every pre-existing mirror carries no `filter_generation` and is
refused; the rebuilt one carries the current stamp.

Expect in the first run's log:

```
history mirror was built under exclusion filter None, current is '<gen>'
— discarding it and reading from the database
```

**Disproves it:** the line is absent *and* the mirror metadata already shows the
current generation — that would mean a stale Parquet was accepted. Read
`data/history_mirror/*.json` and confirm `filter_generation` matches
`src.data.history_mirror.filter_generation()`.

## 4. The retrain actually fired

**Claimed:** the restored pickle is refused on its missing stamp, `is_fitted`
goes false, `trained_at` is cleared, and `--train` retrains — *without* anyone
forcing it, and **not** satisfiable by `ml_retrain_days: 3`.

Expect:

```
ML artifact was trained under exclusion filter None, current is '<gen>'
— discarding it and forcing a retrain
```
followed by an actual training run in the same job.

**Disproves it:** the warning appears but no training follows (the refusal did
not reach the staleness path), or no warning appears at all (the artifact
carried a stamp it should not have had).

Confirm the new pickle's `training_filter_generation` equals the mirror's. If
the two caches disagree, one was invalidated and the other was not — which is
the whole finding this mechanism came from.

## 5. The 29 are absent from the fitting set

**Claimed:** 29 matches marked `corrupt_team_identity`, excluded at 14 sites,
exempted at 6 with a named category.

```sql
SELECT training_exclusion_reason, count(*)
FROM matches WHERE training_exclusion_reason IS NOT NULL GROUP BY 1;
```

Expect exactly 29 / `corrupt_team_identity`.

Then the harder half — **the exclusion is only real where the caches
invalidate**. Confirm the rebuilt mirror's row count equals the DB count of
completed, non-excluded matches. A mirror holding 29 more rows than the
database's filtered count means the sync ran without the predicate.

**Disproves it:** the counts differ, or any of the 29 appears in the mirror.

## 6. `_valid_evidence()` gates what it should and nothing more

**Claimed:** three picks (1148, 309, 314) carry
`evidence_status = 'void_corrupt_features'`; they are excluded from the five
learning and measurement sites and from both CLV series; they remain in
`get_stats()` and in the cold-streak alert, because the wagers were real.

```sql
SELECT id, is_paper, result, disposition, evidence_status
FROM saved_picks WHERE evidence_status IS NOT NULL;
```

**Disproves it, in either direction:**

* the settled-record ROI/win-rate **changed** — they were wrongly excluded from
  `get_stats()`; the bets happened
* they appear in a Bayesian weight update or an EV-threshold recalibration —
  they were not excluded where they should be

The second is the one that matters and the harder one to see. Check
`data/models/bayesian_weights.json` and `ev_threshold.json` modification times
against the first post-deploy `--settle`, and confirm the learner's input count
excludes them.

## 7. The identity gate's skip count

**Claimed:** the gate refuses on country contradiction or total name
disagreement, fails closed by skipping the fixture, and reports the count at end
of run.

Expect either no line (nothing refused) or:

```
API-Football update complete (N requests used, M fixture(s) SKIPPED on
team-identity mismatch)
```

**Note:** while OPS-1 is open the gate cannot be exercised at all — no fixtures
arrive. A zero here is **not** evidence the gate works; it is evidence nothing
was tested. Say so plainly rather than recording a pass. The gate is correct by
construction and unverified in production until API-Football returns.

**Disproves it:** any `TEAM IDENTITY MISMATCH` error followed by a fixture
created for that pair anyway.

## 8. Nothing else moved

**Claimed:** cohort-neutral work stayed cohort-neutral.

* `pick_observations` written == 2 × picks saved
* no pick has `disposition` set by a second review pass
* the Odds API ledger still reads per-account, unchanged by rotation
* the standing `/v4/sports` key check passes (and, when the budget allows a real
  request, that the deployed secret is confirmed live)

---

## 9. Final evidence table

One row per section: claim, what was measured, verdict
(`CONFIRMED` / `DISPROVED` / `UNTESTABLE — reason`).

`UNTESTABLE` is a first-class verdict here and will be the honest answer for
several sections while OPS-1 is open. Recording a pass for something that could
not run is the failure mode this whole stage was about.

## 10. Hard rules

1. **Read-only.** No code, config, schema, migration, model parameter, workflow
   or production data changes. No commits, no pushes, no deploys.
2. **Do not trigger a workflow manually.** Wait for the scheduled run.
3. **Do not spend Odds API or API-Football credits to manufacture evidence.**
4. **Do not create synthetic picks, observations, matches or team rows.**
5. `conclusion: success` is not evidence. Read the logs.
6. A zero is only an anomaly relative to what was asked for — and equally, a
   zero is only *evidence* if something was actually attempted.
7. If an invariant in `tests/test_experiment_invariants.py` fails: **stop,
   report, fix nothing.**

## Final requirement

State plainly, in one paragraph, whether the s5.3 break did what it claimed —
and list every claim that could not be tested because the account was still
suspended. That list is the input to the next stage, not a footnote.
