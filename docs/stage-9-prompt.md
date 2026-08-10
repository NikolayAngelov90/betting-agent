# Stage 9 — Dual CLV Attribution: Frozen Model vs Final Selection

> **Note (Stage 10.2, added later):** this prompt quotes `stage5_baseline_20260807.d1b522`, computed from the local gitignored config. The deployed identity is `stage5_baseline_20260807.485823`. The request text below is unchanged.


## Mission

Continue the paper-trading / prospective-CLV experiment from **Stage 8**.

Stage 8 is complete and uncommitted. It established:

* paper trading enabled
* frozen Stage 5 model
* `model_version = stage5_baseline_20260807.d1b522`
* `CODE_REVISION = s5.2`
* correlation integrity fixed
* Claude CHANGE correlation bypass fixed
* normalized duplicate identity fixed
* paper/live isolation fixed
* same-snapshot CLV defect fixed
* cluster bootstrap / effective-n reporting added
* 566 tests passing
* zero production rows modified
* zero Odds API credits consumed
* nothing committed, pushed, merged, or deployed

Stage 8's only blocker is **experimental attribution**:

> `SavedPick.selection` can represent a Claude-reviewed CHANGE rather than the frozen model's original selection. If CLV captures only `SavedPick.selection`, the experiment measures "model + Claude review", not the frozen Stage 5 model.

Stage 8 explicitly recommends **Option 1**:

> Capture closing lines for both the frozen model selection and the final persisted selection, and report two CLV series.

Implement that option.

---

# IMPORTANT OPERATING RULES

## 1. Read the existing Stage 8 work first

Before modifying anything, inspect:

* `docs/stage8-experimental-integrity-2026-08-10.md`
* `docs/stage7-operational-paper-trading-2026-08-10.md`
* `docs/stage7-model-observations-2026-08-10.md`
* the Stage 7/8 prompts
* current `capture_closing_lines.py`
* current `paper_trading_report.py`
* `betting_agent.py`
* `match_briefing.py`
* `SavedPick` model/schema
* existing CLV tests
* existing `model_selection`, `model_market`, `model_probability` fields
* current `model_version` / `CODE_REVISION` implementation

Do not assume Stage 8's report is complete implementation documentation. Verify the actual source.

Do not undo or weaken any Stage 7 or Stage 8 integrity fixes.

---

# 2. Core experiment definition

The experiment now has **two explicitly named attribution series**.

### Series A — Frozen Model CLV

The original prediction produced by the frozen Stage 5 model:

* `model_market`
* `model_selection`
* `model_probability`

This answers:

> Did the frozen Stage 5 model identify selections whose prices moved in the expected direction before closing?

### Series B — Final Selection CLV

The selection actually persisted after the optional Claude review:

* `market`
* `selection`
* `probability`

This answers:

> Did the final model + Claude selection move in the expected direction before closing?

These are different experimental questions and MUST NOT be silently merged.

---

# 3. Attribution rules

Implement explicit attribution rather than duplicating ambiguous fields.

Every CLV observation must identify:

```text
attribution = model
```

or

```text
attribution = final
```

Do not use vague names such as:

* `original`
* `saved`
* `current`
* `pick`
* `selection2`

Use terminology that makes the experimental meaning obvious.

If the existing schema can support both observations without a migration, prefer that.

If a schema migration is genuinely required:

1. STOP before creating it.
2. Report exactly why the existing schema cannot represent the two observations.
3. Do not create or apply a migration without explicit approval.

Prefer an additive application-level representation if one can preserve full auditability without changing production schema.

---

# 4. Preserve the Stage 8 identity rule

The Stage 8 same-snapshot rule remains mandatory.

For BOTH attribution series:

A closing observation MUST:

* match the correct fixture
* match the correct market
* match the correct selection
* have valid decimal odds
* have a valid timestamp
* be before kickoff
* be inside the configured closing window
* come from a valid bookmaker/source
* satisfy the existing market-structure / overround rules
* not collide with corrupt source data
* be observed **strictly after the pick was created**
* use the existing `missing` / `late` / `invalid` semantics

Do NOT weaken this rule to increase coverage.

The model selection and final selection may have different markets. Each must independently satisfy the CLV validity rules.

---

# 5. Critical timestamp requirement

The two attribution series must not accidentally use different temporal semantics.

The pick creation timestamp is the common causal boundary.

For each attribution:

```text
not_before = pick.created_at
```

and the selected closing observation must satisfy:

```text
odds.timestamp > pick.created_at
```

The existing Stage 8 exclusive comparison must remain intact.

Do not use:

```text
>=
```

Do not reset `created_at` for the model series.

Do not fabricate a separate model timestamp unless the existing data explicitly contains one.

The model and final selection represent two selections made during the same pick lifecycle; the closing observation is what differs.

---

# 6. Handle identical model/final selections efficiently

If:

```text
model_market == market
AND
model_selection == selection
```

then there is only one underlying market observation.

Do NOT make two logically duplicate capture operations.

Instead:

* capture the close once
* record attribution for both series
* preserve independent statistics/counters

This is important for:

* API quota
* correctness
* avoiding duplicate rows
* avoiding artificial observation counts

If the existing data model requires one row per attribution, both rows may reference the same underlying closing observation, but they must remain distinguishable as `model` vs `final`.

---

# 7. Handle Claude CHANGE correctly

If:

```text
model_selection != selection
```

then both must be evaluated independently.

Example:

```text
model_selection = Away Over 0.5
selection       = Over 2.5 Goals
```

The experiment must NOT assume that the final selection's CLV represents the model.

Instead:

```text
model CLV → model_selection
final CLV → selection
```

If one has a valid close and the other does not:

```text
model   = valid
final   = missing
```

or vice versa.

Do NOT drop the valid series merely because the other cannot be measured.

Do NOT substitute one selection's closing line for the other.

---

# 8. Historical compatibility

The database already contains historical rows with:

* `model_market`
* `model_selection`
* `model_probability`
* `market`
* `selection`
* `probability`
* `review_action`

Use these snapshots where available.

Do not reconstruct historical model selections from current model code.

Do not rerun the model against historical fixtures.

Do not overwrite historical selections.

If a historical row has no model snapshot, mark the model attribution as:

```text
unavailable
```

rather than guessing.

The final selection may still be measured independently.

---

# 9. No model change

This stage is **evaluation-only**.

Do NOT change:

* model parameters
* model features
* thresholds
* gates
* Elo
* Poisson
* Dixon-Coles
* half-life
* blend weights
* Bayesian learner
* correlation policy
* selection policy
* Claude review behaviour
* Odds API market selection
* Odds API refresh frequency
* paper/live isolation
* model probability
* prediction generation

Do NOT change:

```text
CODE_REVISION
```

Do NOT create a new model version.

The current version must remain:

```text
stage5_baseline_20260807.d1b522
```

The experiment is measuring the existing frozen selection population, not creating a new one.

---

# 10. Statistical reporting

Extend the Stage 8 cluster-aware reporting so that it reports separately:

## Frozen Model

```text
model:
    valid closing lines
    independent fixtures
    effective n
    design effect
    CLV mean
    CLV median
    CLV CI
```

## Final Selection

```text
final:
    valid closing lines
    independent fixtures
    effective n
    design effect
    CLV mean
    CLV median
    CLV CI
```

## Paired subset

Where both model and final selections have valid CLV for the same fixture/pick:

report:

```text
paired observations
model CLV mean
final CLV mean
final - model CLV
```

The paired comparison is particularly important because it isolates the incremental effect of Claude's review among cases where both selections can actually be measured.

Do NOT present the paired difference as an independent CLV sample.

Use fixture-level clustering consistently with Stage 8.

---

# 11. Review-action breakdown

The report must distinguish at minimum:

```text
review_action = none
review_action = KEEP
review_action = CHANGE
```

For CHANGE specifically, report:

```text
model CLV
final CLV
delta CLV = final - model
```

This is the key diagnostic for the Claude review.

Do not infer causality from the delta.

Use wording such as:

> "observed difference in CLV"

not:

> "Claude improved CLV"

unless the statistical evidence actually supports that conclusion.

---

# 12. Coverage reporting

The report must make coverage transparent.

Report separately:

```text
model selections eligible for CLV
model valid closes
model missing
model late
model invalid

final selections eligible for CLV
final valid closes
final missing
final late
final invalid
```

Also report:

```text
same selection
changed selection
model-only measurable
final-only measurable
both measurable
neither measurable
```

Do not hide reduced coverage.

Do not count missing observations as zero CLV.

Do not treat unavailable model snapshots as failed model predictions.

---

# 13. Checkpoint integrity

The existing checkpoint definition remains unchanged:

```text
100 / 200 / 500 valid closing-line picks
```

But now checkpoints must clearly identify attribution:

```text
MODEL:
100 valid closing-line picks
84 independent fixtures
76 worst-case effective n

FINAL:
100 valid closing-line picks
...
```

Never combine model and final observations into a single checkpoint.

If a single underlying closing observation represents both attribution series, it still counts as:

* one model valid CLV
* one final valid CLV

but MUST NOT be described as two independent fixtures.

---

# 14. Quota protection

This stage must not spend unnecessary Odds API credits.

Before touching live API calls:

* inspect current quota
* determine whether live calls are actually required
* prefer existing captured odds and deterministic mocks for implementation validation

If live capture testing is necessary:

* use the existing quota guard
* respect the existing 400 monthly budget
* respect the existing 24-credit per-run ceiling
* do not bypass the ledger
* do not use the project's production key for exploratory testing if an isolated key/test path already exists

At the end report:

```text
Odds API credits consumed by Stage 9: X
```

If X > 0, explain exactly why each call was necessary.

---

# 15. Production safety

This stage must be **read-only against production data during development and validation**.

You may:

* SELECT
* inspect
* calculate
* run tests
* run deterministic simulations

You must NOT:

* insert picks
* update picks
* delete picks
* settle picks
* write closing odds
* modify quota state
* alter production configuration
* apply migrations
* change paper/live state

The only exception is if an already-existing operational workflow is explicitly required for the final implementation. If so, STOP before executing it and report what would be changed.

---

# 16. Tests

Add focused tests for the dual-attribution contract.

Minimum coverage:

### Test 1

Model and final selections identical:

* one underlying close
* both attribution series valid
* no duplicate capture

### Test 2

Model and final selections differ:

* model gets model close
* final gets final close
* neither substitutes for the other

### Test 3

Model close exists, final close missing:

* model remains valid
* final is missing

### Test 4

Final close exists, model close missing:

* final remains valid
* model is missing

### Test 5

Both use the same odds row:

* attribution remains separate
* underlying observation is not double-counted as two independent fixtures

### Test 6

Same-snapshot rule:

* odds timestamp == created_at → rejected
* odds timestamp < created_at → rejected
* odds timestamp > created_at → accepted

### Test 7

Historical row without model snapshot:

* model attribution = unavailable
* final attribution still measurable

### Test 8

CHANGE attribution:

* model CLV uses `model_selection`
* final CLV uses `selection`

### Test 9

Paired comparison:

* difference = final CLV − model CLV
* fixture clustering preserved

### Test 10

Checkpoint separation:

* model and final counters cannot contaminate one another

### Test 11

Paper/live isolation remains intact.

### Test 12

Stage 8 invariants remain intact.

All existing tests must continue passing.

Target:

```text
0 regressions
```

Report the exact before/after test count.

---

# 17. Historical read-only validation

Before declaring Stage 9 complete, run a deterministic/read-only analysis against the existing production dataset.

Answer:

1. How many historical picks have both model and final snapshots?
2. How many have identical selections?
3. How many are Claude KEEP/CHANGE/none?
4. How many model selections are CLV-eligible?
5. How many final selections are CLV-eligible?
6. How many historical observations can currently produce a valid close under the Stage 8 rules?
7. How many would be model-only measurable?
8. How many would be final-only measurable?
9. How many would be paired?
10. Does the historical dataset reveal any additional attribution ambiguity?

Do not fabricate historical closing lines.

Do not treat today's zero production closing lines as a reason to weaken the validation rules.

---

# 18. Important distinction: historical CLV vs prospective CLV

Be explicit in the report:

The historical database currently has:

```text
closing_odds = 0
```

Therefore the historical analysis can validate:

* attribution availability
* selection differences
* eligibility
* schema integrity
* coverage potential
* statistical grouping

It cannot manufacture prospective CLV results.

Do not report simulated historical CLV as real CLV.

---

# 19. Documentation

Create:

```text
docs/stage9-dual-clv-attribution-2026-08-10.md
```

The report must contain:

## Executive Summary

What changed and why.

## Experiment Definition

Frozen model vs final selection.

## Implementation

Files/functions changed.

## Attribution Rules

How each series maps to database fields.

## CLV Integrity

Confirmation that Stage 8 rules remain intact.

## Statistical Method

Cluster bootstrap and paired comparison.

## Coverage

Model vs final coverage.

## Historical Validation

Read-only findings.

## Tests

Before/after counts and new tests.

## Quota

Credits consumed.

## Production Safety

Rows/config/migrations affected.

## Decision

State exactly one of:

```text
READY FOR PAPER TRADING
```

or

```text
NOT READY
```

If not ready, list only concrete blockers.

---

# 20. STOPPING RULE

This is critical.

Do NOT invent another design decision.

If implementation reveals an ambiguity that materially changes what the experiment measures, STOP and report:

1. the exact ambiguity
2. the relevant existing code
3. the competing interpretations
4. the smallest decision required

Do not silently choose.

Do not commit.

Do not push.

Do not deploy.

---

# 21. Final state required

At the end of Stage 9:

```text
Working tree:
    changes present
    NOTHING COMMITTED

Branch:
    unchanged

Production:
    unchanged

Paper trading:
    remains ON

Model:
    frozen
    stage5_baseline_20260807.d1b522

CODE_REVISION:
    remains s5.2

Tests:
    all passing

Odds API:
    no unnecessary credits consumed

Decision:
    READY FOR PAPER TRADING
    or NOT READY with concrete blocker
```

The purpose of Stage 9 is **not to improve the model**.

The purpose is to make sure that when the first 100 valid closing lines arrive, we can answer two separate questions honestly:

1. **Did the frozen Stage 5 model generate positive/negative CLV?**
2. **Did the final Claude-reviewed selection behave differently from the frozen model selection?**

Do not optimize either series against the historical data.

Do not change the experiment after seeing the results.

Stop before commit and wait for review.
