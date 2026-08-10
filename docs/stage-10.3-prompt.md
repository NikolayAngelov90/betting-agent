# Stage 10.3 — Atomic Pick + Observation Persistence

Stage 10.2 is complete and approved.

The frozen model identity is now reconciled to the **deployed configuration**:

* `model_version = stage5_baseline_20260807.485823`
* `CODE_REVISION = s5.2`
* `config/config.example.yaml` is authoritative for production
* `config/config.yaml` now matches it
* `TRACKED_KEYS` are identical
* 614 tests passed
* No production data was modified
* No source/model/prediction logic was changed
* Nothing has been committed, pushed, merged, or deployed

Do **only Stage 10.3** now.

---

## Objective

Fix the remaining Stage 10.1 blocker:

> `_write_pick_observations()` currently swallows observation-write failures inside `session.begin_nested()`. This allows `saved_picks` to commit without its required `pick_observations`, creating an incomplete attribution record.

The required invariant is:

> **A newly created pick must never commit without both its required `model` and `final` observation rows. If observation persistence fails, the entire pick-save transaction must fail and roll back.**

This is a persistence-integrity fix only.

---

## Required changes

### 1. Preflight the observation table

At the beginning of the `--picks` save operation, before any pick is inserted:

* Verify that `pick_observations` exists.
* If it does not exist, abort the entire pick-save operation immediately.
* Do not insert any `saved_picks`.
* Produce a clear actionable error explaining that migration 006 must be applied.

This must happen **once per `--picks` run**, not once per pick.

Do not perform an API call or any unrelated work before this preflight that could consume quota unnecessarily.

---

### 2. Remove the observation-error swallow

Currently `_write_pick_observations()` uses a nested transaction/savepoint and catches observation failures.

Change this behavior.

If either observation cannot be written:

* propagate the exception;
* do not silently log-and-continue;
* allow the outer `_save_picks` transaction to roll back;
* ensure no partially persisted `saved_picks` remain.

Do not use a fallback that creates a pick without observations.

Do not convert the failure into `unavailable`.

`unavailable` is an attribution/reporting state for genuinely unmeasurable historical data. It must **not** be used to hide a persistence failure for a newly created pick.

---

## 3. Preserve transaction semantics

Be careful with SQLAlchemy transaction handling.

The intended structure is:

```text
begin outer transaction
    ↓
preflight pick_observations
    ↓
insert pick
    ↓
insert model observation
    ↓
insert final observation
    ↓
next pick
    ↓
commit only if the entire operation succeeds
```

If any observation insert fails:

```text
exception
    ↓
outer transaction rollback
    ↓
NO partially persisted picks
NO orphaned observations
```

Do not call `session.rollback()` from inside a nested operation if that would interfere with the caller's transaction lifecycle.

Keep the implementation compatible with the existing unit-of-work/session architecture.

---

## 4. Preserve all Stage 10 behavior

Do not change:

* `model` / `final` attribution semantics
* `taken_odds`
* `taken_at`
* CHANGE handling
* KEEP handling
* `model_market`
* model selection snapshot
* final selection
* closing-line capture
* CLV calculations
* same-snapshot rule
* cluster bootstrap
* paired comparison
* checkpoint separation
* paper/live isolation
* quota logic
* Odds API behavior
* model parameters
* selection gates
* `CODE_REVISION`
* `model_version`

The model remains frozen:

```text
stage5_baseline_20260807.485823
```

Do not introduce a new model version.

Do not modify `config/config.example.yaml`.

Do not create another migration.

---

## 5. Required tests

Add focused tests for the exact failures found in Stage 10.1.

### Test A — table missing

Simulate `pick_observations` being unavailable.

Expected:

```text
saved_picks inserted = 0
pick_observations inserted = 0
operation fails loudly
```

The error should clearly identify the missing migration/table.

---

### Test B — observation constraint failure

Force one observation insert to fail.

Expected:

```text
saved_picks inserted = 0
pick_observations inserted = 0
transaction rolled back
exception propagated
```

There must be no partial pick.

---

### Test C — heterogeneous batch failure

Use a batch containing at least two picks.

Make observation persistence fail for the second pick.

Expected:

```text
pick 1 = rolled back
pick 2 = rolled back
observations for pick 1 = rolled back
observations for pick 2 = rolled back
```

This specifically protects against the Stage 10.1 B3 failure mode.

---

### Test D — normal success

Verify the normal path still produces:

```text
1 saved_pick
1 model observation
1 final observation
```

For a KEEP/unchanged selection, preserve the existing one-underlying-observation semantics used by the closing capture.

For a CHANGE, preserve two independently attributable observations.

---

### Test E — migration preflight occurs once

Verify that the table-existence check is performed once per `--picks` run rather than once per pick.

---

## 6. Test the actual invariant

Do not only test that an exception was raised.

Assert the database state after the failed transaction.

The strongest invariant is:

```text
failed pick-save transaction
    => saved_picks delta = 0
    => pick_observations delta = 0
```

For a multi-pick batch:

```text
failed batch
    => saved_picks delta = 0
    => pick_observations delta = 0
```

---

## 7. Run the complete test suite

Run:

```bash
pytest -q
```

Report:

* previous count
* new count
* number of new tests
* failures
* regressions

Do not weaken or delete existing tests to make the suite pass.

---

## 8. Production verification

After implementation, perform **read-only** production verification.

Confirm:

```text
saved_picks count
settled count
paper count
model_version count
pick_observations count
closing_odds count
odds row count
api_budget state
```

Expected:

* no production picks modified
* no production picks inserted
* no observations inserted
* no closing odds written
* no quota consumed
* no model/version changes

If production changed naturally because scheduled jobs ran during the session, distinguish those changes explicitly from Stage 10.3 changes.

Do not manually "restore" naturally occurring production changes.

---

## 9. Git safety

Do not:

* commit
* push
* merge
* deploy
* reset unrelated changes
* modify Stage 10.2's configuration decision

The working tree must remain uncommitted.

---

## 10. Final audit

At the end, report:

### Implementation

Which files changed and exactly what changed.

### Transaction invariant

Demonstrate that observation failure cannot leave a persisted pick.

### Tests

Full suite result.

### Production

Read-only state and confirmation of zero Stage 10.3 writes.

### Model integrity

Confirm:

```text
model_version = stage5_baseline_20260807.485823
CODE_REVISION = s5.2
```

### Decision

Return exactly one of:

```text
READY FOR FINAL DEPLOYMENT GATE
```

or

```text
NOT READY
```

If anything remains uncertain, do not guess. Identify the exact blocker and stop there.

---

## Important boundary

This stage is **not** the deployment stage.

Do not create or modify:

* Git commits
* GitHub branches
* pull requests
* GitHub Actions workflows
* cron schedules
* deployment configuration

Those belong to the next final deployment gate after Stage 10.3 passes.
