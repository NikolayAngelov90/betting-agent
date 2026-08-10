# Stage 10 — Approve and Implement Dual CLV Observation Storage

> **Note (Stage 10.2, added later):** this prompt quotes `stage5_baseline_20260807.d1b522`, computed from the local gitignored config. The deployed identity is `stage5_baseline_20260807.485823`. The request text below is unchanged.


## Context

Stage 9 is complete and has correctly stopped at the schema boundary.

The experiment must measure two distinct attribution series:

1. **MODEL** — the frozen Stage 5 model's original selection.
2. **FINAL** — the final persisted selection after the Claude KEEP/CHANGE review.

Stage 9 established that the existing schema cannot correctly support this because:

* On a Claude `CHANGE`, `SavedPick.odds` is overwritten with the new selection's price.
* The original model selection's taken price is therefore destroyed.
* `saved_picks` contains only one set of `closing_*` columns.
* A changed pick can require two different closing observations in two different markets.
* Historical model odds cannot be safely reconstructed from `pre_claude_ev` and `model_probability`, especially for DNB.
* Historical rows must therefore NOT be backfilled or reconstructed.

The Stage 9 recommendation is approved.

---

# Objective

Implement the additive `pick_observations` storage layer so that prospective paper trading can correctly record:

* the frozen model's taken selection and taken odds;
* the final selection and taken odds;
* independent closing observations for each attribution;
* correct CLV attribution without ever substituting one series' price for the other.

Do **not** change the frozen model.

Do **not** optimize the model.

Do **not** change thresholds, gates, features, probabilities, EV calculations, ensemble weights, or selection logic.

This stage is strictly about **observation storage and CLV attribution infrastructure**.

---

# Hard Rules

## 1. DO NOT modify historical data

Absolutely no backfill.

Do not:

* reconstruct historical model odds;
* derive historical odds from EV;
* modify existing `saved_picks` rows;
* modify historical `closing_*` values;
* create synthetic observations;
* infer missing `model_selection`;
* infer whether historical selections changed.

Historical rows remain historical.

For historical rows, preserve the existing truthful states such as:

```text
no_model_snapshot
model_taken_price_not_recorded
```

Do not attempt to improve historical coverage artificially.

---

## 2. Frozen model remains frozen

These MUST remain unchanged:

```text
model_version = stage5_baseline_20260807.d1b522
CODE_REVISION = s5.2
```

Stage 10 is an evaluation/storage change.

It must NOT create another model-version bump.

Do not modify:

* model parameters;
* prediction algorithms;
* features;
* thresholds;
* gates;
* calibration;
* ensemble weights;
* Elo/Poisson/Dixon-Coles logic;
* Odds API market selection;
* refresh scheduling.

---

# 3. Create the additive table

Create a migration for:

```sql
CREATE TABLE pick_observations (
    id                   SERIAL PRIMARY KEY,
    pick_id              INTEGER NOT NULL
                         REFERENCES saved_picks(id)
                         ON DELETE CASCADE,

    attribution          VARCHAR(8) NOT NULL,
    market               VARCHAR(50) NOT NULL,
    selection            VARCHAR(100) NOT NULL,

    taken_odds           DOUBLE PRECISION NOT NULL,
    taken_at             TIMESTAMP NOT NULL,

    closing_odds         DOUBLE PRECISION,
    closing_captured_at  TIMESTAMP,
    closing_status       VARCHAR(16) NOT NULL DEFAULT 'pending',
    closing_book_count   INTEGER,
    closing_fair_prob    DOUBLE PRECISION,

    UNIQUE (pick_id, attribution)
);
```

Allowed attribution values are ONLY:

```text
model
final
```

Prefer enforcing this at the database level if the project's existing migration conventions support a CHECK constraint.

For example:

```sql
CHECK (attribution IN ('model', 'final'))
```

Do not introduce unnecessary fields.

Do not alter the meaning of existing `saved_picks` columns.

---

# 4. Migration safety

Before applying the migration:

1. Inspect the project's existing migration conventions.
2. Follow the existing style.
3. Include a rollback/down migration if the project convention supports it.
4. Verify foreign-key behaviour.
5. Verify the uniqueness constraint.
6. Verify no existing production rows are modified by the migration itself.

The migration must be **additive**.

Do not:

* rename existing columns;
* alter existing `saved_picks` columns;
* drop existing columns;
* rewrite historical records;
* backfill observations.

After migration, explicitly verify:

```text
saved_picks row count unchanged
settled picks unchanged
closing data unchanged
paper/live state unchanged
model_version unchanged
CODE_REVISION unchanged
```

---

# 5. Critical timing requirement

This is the most important implementation rule.

The MODEL observation must be written **before Claude can modify the pick**.

The sequence must be:

```text
Frozen model generates pick
        ↓
Capture MODEL market/selection/odds/probability
        ↓
Persist MODEL observation
        ↓
Claude KEEP/CHANGE review
        ↓
Persist FINAL observation
        ↓
Closing-line capture
```

Never do:

```text
Frozen model
    ↓
Claude CHANGE
    ↓
try to reconstruct model odds
```

That is exactly the defect Stage 9 identified.

---

# 6. MODEL observation

Before the Claude review can overwrite `SavedPick.odds`, record:

```text
attribution = model
market      = model_market
selection   = model_selection
taken_odds  = original model odds
taken_at    = pick creation timestamp
```

The value must represent the actual price at which the frozen model's selection was taken.

Do not calculate or reconstruct it later.

If the model selection cannot be recorded for a genuinely new prospective pick, fail explicitly rather than silently substituting the final selection.

---

# 7. FINAL observation

After the final selection is known, create:

```text
attribution = final
market      = market
selection   = selection
taken_odds  = final odds
taken_at    = pick creation timestamp
```

The final observation must represent the actual persisted final selection.

---

# 8. KEEP scenario

Example:

```text
MODEL:
Over 2.5 Goals @ 1.85

Claude:
KEEP

FINAL:
Over 2.5 Goals @ 1.85
```

The table should contain:

```text
pick_id | attribution | selection       | taken_odds
--------|-------------|-----------------|-----------
123     | model       | Over 2.5 Goals  | 1.85
123     | final       | Over 2.5 Goals  | 1.85
```

There are two attribution records, but they represent the same underlying observation.

Closing capture MUST NOT consume two Odds API observations for this.

One real closing observation can populate both attribution records.

---

# 9. CHANGE scenario — mandatory end-to-end test

This test is mandatory.

Example:

```text
MODEL:
Over 2.5 Goals @ 1.85

Claude:
CHANGE

FINAL:
Home Win @ 2.10
```

The database MUST contain:

```text
pick_id | attribution | market        | selection       | taken_odds
--------|-------------|---------------|-----------------|-----------
123     | model       | totals        | Over 2.5 Goals  | 1.85
123     | final       | h2h           | Home Win        | 2.10
```

The MODEL price must remain `1.85` even though `saved_picks.odds` may now contain `2.10`.

The final price must remain `2.10`.

Neither value may be reconstructed later.

---

# 10. Closing capture

Extend closing capture so that it can resolve both attribution series.

For an unchanged pick:

```text
MODEL → Over 2.5 @ 1.85
FINAL → Over 2.5 @ 1.85
```

Resolve one actual closing observation and attribute it to both records.

For a changed pick:

```text
MODEL → Over 2.5
FINAL → Home Win
```

resolve the two selections independently.

The model and final markets may differ.

A closing price for FINAL must NEVER be reused as MODEL closing price.

A closing price for MODEL must NEVER be reused as FINAL closing price unless they are genuinely the same selection/market and therefore represent the same observation.

---

# 11. Odds API quota protection

This is critical.

Do NOT double API requests for unchanged picks.

For:

```text
MODEL == FINAL
```

one underlying Odds API observation must serve both attribution records.

For:

```text
MODEL != FINAL
```

only request the additional market/selection when actually required by the existing refresh/capture architecture.

Preserve the existing:

```text
400-credit monthly budget
50-credit safety margin
24-credit per-run ceiling
claim-before-spend ledger
provider reconciliation
```

Do not weaken quota protection.

Do not make unnecessary live API calls during development.

Use mocks/fixtures for tests.

---

# 12. CLV validity rules

All Stage 8 CLV rules remain mandatory.

Do not weaken any of them:

* correct match;
* correct market;
* correct selection;
* valid decimal odds;
* valid timestamp;
* observation before kickoff;
* closing observation inside the configured window;
* valid bookmaker;
* market structure validation;
* overround validation;
* corrupt-source collision protection;
* `missing`;
* `late`;
* `invalid`;
* same-snapshot rule.

The same-snapshot rule remains:

```text
closing observation timestamp MUST be strictly after taken_at
```

A closing observation at exactly the taken timestamp is invalid.

Never allow a pick's own pricing observation to masquerade as its closing line.

---

# 13. No historical backfill

After the migration, DO NOT automatically create 2 × 1,070 historical observations.

Do not populate:

```text
pick_observations
```

from existing historical `saved_picks`.

The table should begin collecting valid observations prospectively.

Historical data stays untouched.

---

# 14. Existing reporting

Stage 9 already implemented:

* model series;
* final series;
* paired subset;
* review-action breakdown;
* coverage cross-tab;
* separate MODEL/FINAL checkpoints;
* cluster bootstrap;
* effective-n reporting.

Repoint these reports to `pick_observations` where appropriate.

Do not redesign the statistical methodology.

Do not revert to i.i.d. bootstrap.

Do not pool MODEL and FINAL as independent observations.

Keep fixture-level clustering.

---

# 15. Paired CLV

When both MODEL and FINAL have valid closes:

```text
delta = final_CLV - model_CLV
```

Report this as:

> observed difference in CLV

NOT:

> causal effect of Claude

Do not make causal claims from this experiment.

---

# 16. Checkpoints

Keep separate counters:

```text
MODEL:
valid closing-line observations
independent fixtures
effective n
design effect

FINAL:
valid closing-line observations
independent fixtures
effective n
design effect
```

Do not let a FINAL observation advance the MODEL checkpoint.

Do not let a MODEL observation advance the FINAL checkpoint.

For unchanged picks, one physical observation may count toward both attribution series, but it remains one independent fixture.

---

# 17. Database integrity

Add tests for:

### Exact identity

```text
UNIQUE (pick_id, attribution)
```

### Allowed attribution

Only:

```text
model
final
```

### Timing

MODEL observation must be persisted before Claude CHANGE.

### CHANGE preservation

Original model odds remain available after final selection changes.

### KEEP

MODEL and FINAL share the same observation when appropriate.

### CHANGE

MODEL and FINAL remain independent observations.

### No historical backfill

Migration must not populate historical observations.

### Foreign key

Deleting a pick should cascade according to the migration design.

---

# 18. Mandatory regression test

Create an end-to-end test that reproduces the exact Stage 9 blocker:

```text
1. Frozen model produces:
   market = totals
   selection = Over 2.5 Goals
   odds = 1.85

2. MODEL observation is written.

3. Claude changes selection to:
   market = h2h
   selection = Home Win
   odds = 2.10

4. FINAL observation is written.

5. Assert:
   MODEL taken_odds == 1.85
   MODEL selection == "Over 2.5 Goals"
   FINAL taken_odds == 2.10
   FINAL selection == "Home Win"

6. Assert saved_picks may contain final odds,
   but MODEL observation remains 1.85.

7. Simulate closing capture:
   MODEL closing = independently resolved
   FINAL closing = independently resolved

8. Assert:
   MODEL CLV uses 1.85 and MODEL closing
   FINAL CLV uses 2.10 and FINAL closing

9. Assert no price substitution occurs.

10. Assert no second request is made when MODEL == FINAL.
```

This is the **acceptance test for Stage 10**.

---

# 19. Production safety

Before declaring completion, report:

```text
Production rows modified:
Production rows inserted:
Historical rows modified:
Historical observations created:
Odds API credits consumed:
Migration applied:
Paper trading:
Model version:
CODE_REVISION:
Tests:
```

Expected during implementation:

```text
Historical rows modified: 0
Historical observations created: 0
Model version: unchanged
CODE_REVISION: unchanged
```

Do not run unnecessary live API calls.

If the migration is applied to production, explicitly verify all production counts before and after.

---

# 20. STOP conditions

STOP immediately and report instead of guessing if:

* the original model odds cannot be captured before Claude review;
* the existing save flow makes timing ambiguous;
* the migration would require modifying `saved_picks` semantics;
* historical backfill appears necessary;
* the existing Odds API quota mechanism cannot safely support dual attribution;
* a closing observation cannot be unambiguously assigned to MODEL vs FINAL;
* a test requires relaxing the Stage 8 same-snapshot rule;
* a model parameter or prediction path would need to change;
* `CODE_REVISION` appears necessary;
* a production data modification is required beyond the approved additive migration.

Do NOT solve any of these by approximation.

---

# 21. Final acceptance criteria

Stage 10 is complete only if ALL are true:

* [ ] `pick_observations` migration created.
* [ ] Migration follows project conventions.
* [ ] No historical rows modified.
* [ ] No historical observations backfilled.
* [ ] MODEL taken odds captured before Claude review.
* [ ] FINAL taken odds captured after final selection.
* [ ] KEEP creates two attribution records but only one underlying close.
* [ ] CHANGE preserves two different taken prices.
* [ ] MODEL and FINAL closing lines are independently attributable.
* [ ] Same-snapshot rule remains strict.
* [ ] Existing quota protection remains intact.
* [ ] No unnecessary Odds API credits spent.
* [ ] Cluster bootstrap remains unchanged.
* [ ] MODEL/FINAL checkpoints remain separate.
* [ ] Paired CLV remains non-causal.
* [ ] DNB cannot be reconstructed from EV/probability.
* [ ] Mandatory CHANGE end-to-end test passes.
* [ ] All existing tests pass.
* [ ] New tests cover migration and observation lifecycle.
* [ ] `model_version` remains `stage5_baseline_20260807.d1b522`.
* [ ] `CODE_REVISION` remains `s5.2`.
* [ ] No model logic changed.
* [ ] Production safety report provided.

---

## Final instruction

Implement Stage 10 exactly within the boundaries above.

Do not commit, push, merge, or deploy automatically.

Do not proceed to another stage.

When finished, provide a concise report containing:

1. Files changed.
2. Migration details.
3. Test count.
4. Production impact.
5. Historical-data impact.
6. Odds API credits consumed.
7. Confirmation that MODEL taken odds survive Claude CHANGE.
8. Confirmation that MODEL and FINAL closing lines are independently attributable.
9. Confirmation that `model_version` and `CODE_REVISION` are unchanged.
10. Any remaining blocker.

If any acceptance criterion cannot be proven, mark:

**NOT READY**

and stop.
