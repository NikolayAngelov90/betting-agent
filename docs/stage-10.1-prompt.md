# Stage 10.1 — Pre-Deployment Integrity Audit & Stage 5–10 Reconciliation

> **Note (Stage 10.2, added later):** this prompt quotes `stage5_baseline_20260807.d1b522`, computed from the local gitignored config. The deployed identity is `stage5_baseline_20260807.485823`. The request text below is unchanged.


You have completed Stage 10 and correctly stopped before commit/push/deploy.

**Do NOT commit, push, merge, deploy, modify production data, or consume Odds API credits during this stage.**

The purpose of this stage is to perform a **final pre-deployment audit of the entire Stage 5 → Stage 10 chain** and determine whether it is actually safe to move the frozen paper-trading experiment into production.

This is an integrity gate, not a feature-development stage.

---

## 1. Current known state

Stage 10 reports:

* `pick_observations` migration 006 was applied successfully to production.
* `pick_observations` currently contains 0 rows.
* 605 tests pass.
* `CODE_REVISION = s5.2`
* `model_version = stage5_baseline_20260807.d1b522`
* Paper trading is configured ON locally.
* No production rows were modified by Stage 10.
* No Odds API credits were consumed.
* `origin/main = 6f0e42c`
* Stage 5 commit `c0d3b7f` exists only on the local branch and was never pushed.
* Production cron is still running the pre-Stage-5 pipeline.
* Today's newly generated production picks have:

  * `model_version = NULL`
  * `is_paper = false`

Therefore the experiment is **not yet collecting paper observations**, despite the local implementation being ready.

---

# 2. First task — reconstruct the actual commit/dependency graph

Do not assume the previous stage descriptions are correct.

Inspect Git history and source and determine exactly:

1. Which commits contain Stage 5.
2. Which commits contain Stage 6.
3. Which commits contain Stage 7.
4. Which commits contain Stage 8.
5. Which commits contain Stage 9.
6. Which commits contain Stage 10.
7. Whether Stage 6–10 depend directly or indirectly on every previous stage.
8. Whether any Stage 5–10 changes were accidentally made on top of one another without being committed.
9. Whether `origin/main` contains any partial pieces of these stages.
10. Whether there are unrelated local changes that must NOT be included.

Produce a dependency table:

| Stage | Commit | Parent | Present in origin/main? | Required for experiment? |
| ----- | ------ | ------ | ----------------------- | ------------------------ |

Do not invent commit IDs. Read them from Git.

---

# 3. Verify the production/local schema boundary

The Stage 10 migration was applied to production even though the application code is not deployed.

Verify:

* `pick_observations` exists in production.
* Its exact columns match the intended design.
* PK exists.
* FK exists with `ON DELETE CASCADE`.
* `UNIQUE (pick_id, attribution)` exists.
* CHECK constraint exists.
* indexes exist.
* there are 0 rows.
* no existing table was altered unexpectedly.

Then inspect the ORM model and verify that the production schema and ORM model are exactly compatible.

Important:

**Do not create another migration.**
**Do not modify the migration unless a real incompatibility is discovered.**

If there is an incompatibility, STOP and report it.

---

# 4. Audit the complete data lifecycle

Trace one future paper pick from creation through settlement and closing capture.

The intended lifecycle is:

```text
model generates pick
        ↓
pick passes selection/correlation gates
        ↓
SavedPick is inserted
        ↓
model observation written
final observation written
        ↓
Claude KEEP / CHANGE
        ↓
if CHANGE:
    final observation updated
    model observation remains immutable
        ↓
paper/live isolation
        ↓
settlement
        ↓
closing capture
        ↓
MODEL CLV
FINAL CLV
paired CLV delta
        ↓
paper trading report
```

Trace this through actual code.

For every transition identify:

* function
* file
* transaction boundary
* database table
* fields written
* fields that must remain immutable
* whether paper/live isolation is enforced
* whether failure can create partial state

Pay particular attention to:

* `_save_picks`
* `_write_pick_observations`
* `_apply_decision`
* `_update_final_observation`
* `capture_closing_lines.py`
* `resolve_close`
* `paper_trading_report.py`

Do not merely read tests. Verify the production execution path.

---

# 5. Critical transaction audit

Stage 10 claims `_write_pick_observations()` uses:

```python
session.begin_nested()
```

Verify whether this actually gives the desired atomicity.

We need this exact invariant:

> A saved pick must never exist without its required observation rows, and an observation write failure must never silently leave an incomplete experiment record.

Test both:

### Case A — normal pick

Expected:

```text
SavedPick inserted
model observation inserted
final observation inserted
transaction commits
```

### Case B — observation write failure

Determine whether the actual transaction produces:

```text
SavedPick NOT persisted
observations NOT persisted
```

or whether it can produce:

```text
SavedPick persisted
one/both observations missing
```

If partial persistence is possible, STOP.

Do not "fix" it silently.

Report the exact transaction semantics and recommend the smallest safe correction.

---

# 6. Critical CHANGE audit

Reproduce this exact scenario using deterministic fixtures:

```text
Model:
    market = over_under
    selection = Over 2.5 Goals
    odds = 1.85

Claude CHANGE:
    market = 1X2
    selection = Home Win
    odds = 2.10
```

Verify after CHANGE:

### saved_picks

```text
selection = Home Win
market = 1X2
odds = 2.10
```

### pick_observations

MODEL:

```text
attribution = model
market = over_under
selection = Over 2.5 Goals
taken_odds = 1.85
taken_at = original pick created_at
```

FINAL:

```text
attribution = final
market = 1X2
selection = Home Win
taken_odds = 2.10
taken_at = original pick created_at
```

The MODEL observation must never be changed.

Also verify that `taken_at` does NOT move during CHANGE.

---

# 7. Critical KEEP audit

Reproduce:

```text
Model:
    Over 2.5 @ 1.85

Claude:
    KEEP
```

Expected:

* MODEL observation exists.
* FINAL observation exists.
* Both point to the same market/selection.
* Both have the same taken price.
* Both have the same causal timestamp.
* Closing capture performs only one underlying odds resolution.
* Report counts the observation correctly for both attribution series.
* Fixture count remains one, not two.

---

# 8. Critical closing-capture audit

Trace a CHANGE pick through closing capture.

Verify that the odds request contains BOTH required markets:

```text
model market
final market
```

The MODEL market must not disappear merely because `SavedPick.market`
was changed by Claude.

Verify:

* each attribution resolves its own selection;
* same-market/same-selection KEEP case resolves once;
* CHANGE case can resolve two different observations;
* no final close can be substituted for a missing model close;
* no model close can be substituted for a missing final close;
* same-snapshot rule applies independently;
* stale prices remain rejected;
* invalid bookmaker/market structure/overround rules remain intact;
* no extra API call occurs for the shared KEEP observation if the resolver can reuse it.

Do not consume live API credits. Use mocks or deterministic fixtures.

---

# 9. Paper/live isolation audit across Stage 5–10

Search the entire repository for every database read that can influence:

* ROI
* EV threshold tuning
* ensemble weights
* calibration
* rolling backtest
* market breakdowns
* drift checks
* model training
* probability calibration
* report statistics

Verify that paper picks cannot contaminate live model/evaluation state except where explicitly intended.

Create a final table:

| Path | Paper allowed? | Live-only filter present? | Verified by test? |
| ---- | -------------: | ------------------------: | ----------------: |

Pay particular attention to the Stage 8 discovery:

```text
ProbabilityCalibrator.fit_from_db()
```

Verify that the fix is actually on the same code path that production will execute.

---

# 10. Version/fingerprint audit

Verify the distinction between:

### Model identity

```text
stage5_baseline_20260807.d1b522
```

and:

### Code revision

```text
s5.2
```

Confirm:

* Stage 8 selection-affecting changes caused the version bump.
* Stage 9 evaluation-only changes did NOT cause another bump.
* Stage 10 storage/evaluation changes do NOT change the frozen model identity.
* no prediction path reads a Stage 10 artifact and silently changes predictions.
* the version is actually persisted by the deployed Stage 5–10 code.
* future paper picks will receive the expected `model_version`.

Do not change the version.

---

# 11. Cron / deployment reality check

Inspect the GitHub Actions workflows and production deployment mechanism.

Determine exactly:

1. Which branch is deployed.
2. Which commit production currently executes.
3. Which workflow generates daily picks.
4. Which workflow captures closing lines.
5. Which workflow runs the paper report.
6. Whether the Stage 5–10 code will all be deployed together.
7. Whether there is a migration/code ordering problem.
8. Whether migration 006 already being live creates any compatibility concern with the old application.
9. Whether the first post-deployment cron can safely run with paper mode ON.

Important:

The desired deployment order should be safe even though migration 006 is already present.

Do NOT deploy anything now.

---

# 12. First-run production safety

Before recommending deployment, determine what happens on the **first run after Stage 5–10 is deployed**.

We need to know:

* will existing historical picks be touched?
* will existing live picks be reclassified as paper?
* will old picks receive observations?
* will the system attempt historical backfill?
* will the first run call Odds API unnecessarily?
* will the quota ledger claim credits before discovering there are no eligible markets?
* will existing picks be re-settled?
* will closing capture process old rows unexpectedly?
* will `pick_observations` remain empty until genuinely new Stage 5+ picks are created?

The expected behavior is:

> Historical data remains untouched. New paper picks start a new prospective collection period.

If the actual behavior differs, STOP and report it.

---

# 13. Migration safety

Inspect:

```text
migrations/006_pick_observations.sql
migrations/006_pick_observations.rollback.sql
```

Verify:

* migration is additive;
* no existing column is changed;
* no historical row is rewritten;
* rollback cannot silently destroy the only copy of model taken prices;
* rollback warning is accurate;
* migration is idempotency-safe or its expected execution semantics are clear;
* migration numbering does not conflict with existing migrations.

Do not execute rollback.

---

# 14. Test the exact experiment acceptance contract

Add tests ONLY if a real missing invariant is discovered.

Do not add tests merely to increase the number.

The minimum acceptance contract is:

### A. Frozen model

```text
model_version unchanged
CODE_REVISION unchanged
```

### B. New pick

```text
new paper pick
→ model observation
→ final observation
```

### C. KEEP

```text
one underlying close
two attributions
one fixture
```

### D. CHANGE

```text
two independent markets
two taken prices
two possible closes
one fixture
```

### E. Same snapshot

```text
own pricing row cannot become closing line
```

### F. Missing close

```text
unavailable/missing ≠ zero CLV
```

### G. Paper isolation

```text
paper never enters live ROI/calibration/tuning
```

### H. Historical integrity

```text
existing 1,070+ production picks untouched
```

### I. Quota

```text
no test consumes real Odds API credits
```

---

# 15. Search specifically for hidden regressions

Perform repository-wide searches for:

```text
SavedPick.odds
closing_odds
model_selection
model_market
model_probability
review_action
is_paper
model_version
CODE_REVISION
pick_observations
closing_capture_status
api_budget
theoddsapi
```

For each occurrence determine whether Stage 10 introduced a semantic conflict.

Do not assume a passing test suite catches this.

---

# 16. Production data verification

Read-only query production after the audit.

Report:

```text
saved_picks total
saved_picks settled
saved_picks paper
saved_picks with model_version
pick_observations total
closing_odds populated
pending closing captures
max created_at
max settled_at
odds rows
api budget / provider usage if available without spending credits
```

Separate:

### Before this audit

from

### After this audit

If production changed because scheduled jobs ran naturally, distinguish that from changes made by this audit.

Do not modify anything.

---

# 17. Final readiness gate

At the end, classify the system as exactly one of:

```text
READY FOR DEPLOYMENT
```

or

```text
NOT READY — BLOCKER
```

or

```text
READY WITH NON-BLOCKING RISKS
```

Do not use "ready" merely because tests pass.

The experiment is READY only if:

1. Stage 5–10 dependency chain is complete.
2. Production schema is compatible.
3. Transaction semantics cannot create incomplete observations.
4. KEEP attribution is correct.
5. CHANGE attribution is correct.
6. Both markets can be captured independently.
7. Same-snapshot rule remains intact.
8. Paper/live isolation remains intact.
9. Model identity remains frozen.
10. First production run is safe.
11. Deployment workflow is understood.
12. Historical data will remain untouched.
13. Quota protection remains intact.

---

# 18. IMPORTANT STOPPING RULE

If you find any issue affecting:

* data meaning,
* attribution,
* transaction integrity,
* paper/live isolation,
* model identity,
* historical integrity,
* migration safety,
* deployment ordering,
* or first-run behavior,

**STOP. Do not fix it automatically.**

Report:

1. exact defect;
2. evidence;
3. affected rows/code path;
4. severity;
5. minimal safe fix;
6. whether it requires a migration;
7. whether it changes model selection/prediction/evaluation;
8. whether a version bump is required.

Only fix issues that are clearly mechanical, evaluation-only, and explicitly safe under the frozen-model contract.

---

# 19. Output format

Return a concise but evidence-heavy Stage 10.1 report:

## Stage 10.1 — Pre-Deployment Integrity Audit

### Executive verdict

### Git / dependency chain

### Production schema

### Pick → observation lifecycle

### Transaction integrity

### KEEP scenario

### CHANGE scenario

### Closing capture

### Paper/live isolation

### Model/version integrity

### Deployment/cron reality

### First-run behavior

### Migration safety

### Hidden regression search

### Production read-only snapshot

### Tests

### Remaining risks

### Final decision

If the final decision is NOT READY, do not provide generic advice.

Give me the **exact next action/prompt required to remove the blocker**.

Remember:

**No commit.
No push.
No deploy.
No production writes except inspection.
No Odds API credits.
Do not guess.
Do not silently repair blockers.**
