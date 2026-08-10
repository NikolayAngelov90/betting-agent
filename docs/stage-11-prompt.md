# Stage 11 — Final Deployment Gate: Stage 5–10 Review, Commit, Push & Deploy

You are now at the **final deployment gate** for the frozen CLV experiment.

Stages 5–10 have been implemented and audited. Stage 10.2 resolved the frozen model identity, and Stage 10.3 fixed atomic pick + observation persistence.

**Do NOT start by committing or pushing.**
First perform a complete final audit. Only if every gate below passes may you commit, push and prepare deployment.

---

## 1. Current authoritative experiment identity

The authoritative production configuration is:

```text
model_version = stage5_baseline_20260807.485823
CODE_REVISION = s5.2
```

The authoritative configuration is:

```text
config/config.example.yaml
```

`config/config.yaml` must match all tracked configuration keys but remains gitignored/local.

Do NOT change:

* model parameters
* thresholds
* gates
* excluded markets
* feature engineering
* prediction logic
* ensemble weights
* calibration logic
* CODE_REVISION
* model_version

This is a deployment/infrastructure gate, NOT a model-development stage.

---

# 2. Full diff audit against origin/main

Run a complete audit of:

```bash
git status
git diff
git diff --stat
git diff origin/main...HEAD
git ls-files --others --exclude-standard
git log --oneline --decorate --graph --all -30
```

Determine exactly what belongs to Stages 5–10.

The current known state is:

```text
origin/main = 6f0e42c
local Stage 5 commit = c0d3b7f
Stages 6–10 = currently uncommitted working-tree changes
```

Do NOT assume this is still true. Verify it.

Produce an exact inventory:

```text
Stage 5:
Stage 6:
Stage 7:
Stage 8:
Stage 9:
Stage 10:
Stage 10.2:
Stage 10.3:
Unrelated changes:
Untracked files:
```

If anything is unrelated to the experiment, STOP and report it.

---

# 3. Verify the dependency chain

The intended history must become reviewable.

Verify that Stage 5 is based on `origin/main`.

Then determine the correct logical commit boundaries for:

```text
Stage 5
Stage 6
Stage 7
Stage 8
Stage 9
Stage 10
```

Do not artificially split commits if doing so would create broken intermediate states.

However, wherever practical, each stage must be independently reviewable.

Before committing, show:

```text
Stage → files → purpose → dependencies
```

and explain any stage that must be combined with another because of dependency constraints.

Do NOT rewrite or squash existing commits unless necessary and explicitly justified.

---

# 4. Inspect every untracked file

Pay particular attention to:

```text
closing-lines.yml
```

and:

```text
scripts/capture_closing_lines.py
scripts/paper_trading_report.py
migrations/006_pick_observations.sql
migrations/006_pick_observations.rollback.sql
```

For every untracked file answer:

1. Is it required for the experiment?
2. Is it safe?
3. Is it complete?
4. Does it belong in Git?
5. Does it contain secrets, local paths, credentials or environment-specific values?

If `closing-lines.yml` is required, it MUST be included.

---

# 5. CI/CD audit

Inspect every workflow under:

```text
.github/workflows/
```

Verify the actual production path from GitHub Actions.

Specifically verify:

### Daily picks

Confirm:

```text
checkout/update
→ settle
→ train
→ generate picks
→ settle
→ report
```

and confirm which branch/workflow actually executes.

### Closing lines

There must be a scheduled workflow that actually executes:

```text
scripts/capture_closing_lines.py
```

Verify:

* schedule
* timezone assumptions
* secrets
* environment variables
* configuration source
* database connection
* Odds API key
* quota/ledger integration
* failure behavior
* concurrency protection
* whether it can accidentally run twice

Do not merely check that the YAML exists.

Follow the command chain into the actual Python code.

---

# 6. Paper trading report must actually run

Stage 10.1 found that no workflow currently runs:

```text
paper_trading_report
```

This must be resolved before deployment.

Determine the intended schedule.

The experiment needs:

```text
pick generation
→ closing-line capture
→ settlement/reporting
```

Verify that the report runs only after the required data is available.

The report must:

* include MODEL and FINAL separately
* preserve the paired subset
* preserve review-action breakdown
* preserve coverage
* preserve separate checkpoint counters
* never substitute FINAL price for MODEL price
* never count unavailable as zero CLV

Do not modify statistical methodology.

---

# 7. Migration 006 audit

Verify:

```text
migrations/006_pick_observations.sql
migrations/006_pick_observations.rollback.sql
```

Confirm:

```text
PRIMARY KEY
FOREIGN KEY → saved_picks(id) ON DELETE CASCADE
UNIQUE (pick_id, attribution)
CHECK attribution IN ('model','final')
indexes
```

Confirm the ORM exactly matches the production schema.

Confirm:

```text
taken_odds
taken_at
closing_odds
closing_captured_at
closing_status
closing_book_count
closing_fair_prob
```

have the intended semantics.

DO NOT modify the production schema during this gate.

Migration 006 is already applied in production.

---

# 8. Atomic persistence audit

Verify Stage 10.3's invariant:

> A pick and its model/final observations are atomic.

Required behavior:

### Missing table

```text
abort before any pick write
```

### Observation constraint failure

```text
entire batch rolls back
```

### Second pick fails

```text
first pick also rolls back
```

### Normal operation

```text
KEEP   → one underlying observation, two attributions
CHANGE → two independent observations
```

Verify tests cover all four.

Do not reintroduce a savepoint or swallowed exception.

---

# 9. Frozen model audit

Search the entire repository for:

```text
stage5_baseline_20260807.d1b522
stage5_baseline_20260807.485823
CODE_REVISION
model_version
```

The old `d1b522` identity must not remain as an active/pinned identity.

Historical documentation may mention it only if clearly marked as superseded/corrected.

There must be exactly one current frozen production identity:

```text
stage5_baseline_20260807.485823
```

Verify that CI and local config produce the same fingerprint.

---

# 10. No model drift audit

Compare Stage 5 baseline behavior with current code.

Specifically inspect whether Stages 6–10 changed:

* prediction formulas
* model features
* calibration
* ensemble weights
* selection thresholds
* EV calculations
* market filtering
* Claude decision inputs
* market eligibility
* bookmaker selection
* correlation policy

Expected result:

```text
No model/prediction behavior changes.
```

If any are found, STOP.

---

# 11. CLV integrity audit

Verify all Stage 8/9/10 invariants remain intact.

Especially:

### Same-snapshot rule

The closing observation must not precede the pick's causal boundary.

Verify exact comparison semantics in source.

### MODEL

Uses:

```text
model_market
model_selection
model taken_odds
taken_at
```

### FINAL

Uses:

```text
final market
final selection
final taken_odds
taken_at
```

### CHANGE

Must produce two independent observations.

### KEEP

Must produce one underlying close and two attributions.

### Never

Do not:

* reconstruct model odds from EV
* substitute final odds for model odds
* fabricate timestamps
* relax closing-window rules
* pool MODEL and FINAL into one independent sample
* count unavailable as zero CLV

---

# 12. Odds API quota audit

Verify that the deployment cannot accidentally exceed the intended budget.

Known policy:

```text
monthly budget = 400
safety margin = 50
per-run ceiling = 24
claim-before-spend ledger
```

Audit:

* closing-line workflow frequency
* refresh window
* leagues queried
* markets queried
* regions queried
* credit calculation
* ledger claim
* failure/retry behavior
* duplicate workflow execution

Calculate worst-case monthly credit consumption.

Show the calculation explicitly.

The experiment must remain inside the quota policy.

Do NOT spend live Odds API credits during this audit.

---

# 13. Paper/live isolation audit

Verify all previously identified isolation points.

Search for every path that:

* trains models
* calibrates probabilities
* tunes thresholds
* calculates ROI
* performs backtests
* performs settlement
* calculates CLV

Confirm paper picks cannot influence model training/tuning.

Confirm live-only filters remain intact.

Confirm:

```text
is_paper
model_version
```

are assigned only at pick creation and cannot be silently changed later.

---

# 14. Database production snapshot

Run read-only verification.

Record:

```text
saved_picks
settled picks
paper picks
model_version distribution
pick_observations
closing_odds
pending/late
odds row count
api_budget
```

Do not modify any production data.

Expected from the last audit:

```text
saved_picks = 1,074
settled = 1,070
paper = 0
model_version = 0
pick_observations = 0
closing_odds = 0
```

If production has naturally changed since the previous audit, explain the delta rather than treating it as an error.

---

# 15. Test suite

Run the complete test suite.

Required:

```text
0 failures
0 unexpected skips
```

Report:

```text
total tests
new tests
regressions
```

Also run any project-specific lint/type/static checks that already exist.

Do not introduce unrelated formatting changes.

---

# 16. Deployment workflow design

Before committing, establish the exact deployment sequence.

It should be conceptually:

```text
commit Stage 5–10 changes
        ↓
push branch
        ↓
open/review PR if repository policy requires
        ↓
merge to main
        ↓
GitHub Actions deploys/runs main
        ↓
production uses Stage 5.2 frozen identity
        ↓
new picks create pick_observations
        ↓
closing capture populates MODEL + FINAL
        ↓
paper trading report calculates both series
```

Do not claim deployment happened unless it actually did.

---

# 17. Commit plan

Only after ALL audits pass, create clean commits.

Prefer:

```text
Stage 5 — <actual stage title>
Stage 6 — <actual stage title>
Stage 7 — <actual stage title>
Stage 8 — <actual stage title>
Stage 9 — Dual CLV Attribution
Stage 10 — Pick Observations
```

If a stage cannot safely be separated, explain why and combine only the minimum necessary changes.

Do NOT create meaningless commits just to satisfy numbering.

Before committing show the proposed commit list.

---

# 18. Commit safety

Before each commit:

```bash
git diff --check
git status
```

Confirm no secrets or local-only files are included.

Do not commit:

```text
.env
credentials
API keys
local database files
IDE state
temporary files
```

The gitignored:

```text
config/config.yaml
```

must remain uncommitted.

---

# 19. Push

Only after the complete audit passes:

1. Commit the verified Stage 5–10 changes.
2. Push the branch.
3. Verify remote state.
4. Verify CI.

Do NOT force push.

Do NOT rewrite public history.

If the branch is behind remote, stop and resolve safely.

---

# 20. Deployment

Do not deploy until:

* all tests pass
* CI passes
* migration 006 is already present
* `closing-lines.yml` is included
* paper trading report workflow is included
* frozen model identity is `stage5_baseline_20260807.485823`
* no model behavior changed
* quota policy passes
* paper/live isolation passes

Then deploy through the repository's existing deployment mechanism.

Do not manually mutate production data to "make it work."

---

# 21. Post-deployment smoke test

After deployment, verify:

### Pick creation

A newly generated pick has:

```text
is_paper = true
model_version = stage5_baseline_20260807.485823
```

and creates:

```text
pick_observations:
  model
  final
```

### KEEP

Verify:

```text
model observation
final observation
same underlying market/selection/price
```

### CHANGE

When a real CHANGE occurs, verify:

```text
model observation != final observation
model.taken_odds remains original
final.taken_odds reflects final selection
```

### Closing capture

Verify the capture process resolves the correct markets for both observations.

Do not deliberately consume extra Odds API credits just for testing.

Use existing scheduled execution or deterministic/test data where possible.

---

# 22. Final report

At the end provide a concise deployment report containing:

```text
FINAL VERDICT:
READY / NOT READY

Current commit:
Branch:
Remote commit:

Model:
CODE_REVISION:

Tests:
CI:

Migration 006:
Closing workflow:
Paper report workflow:

Quota:
Paper/live isolation:

Production snapshot:

Commits created:
Push completed:
Deployment completed:

Post-deployment smoke test:

Remaining risks:
```

## Critical stop conditions

STOP immediately and do NOT commit/push/deploy if:

1. Any model behavior changed.
2. Frozen model identity is inconsistent.
3. Any Stage 5–10 change cannot be accounted for.
4. An observation can still be silently lost.
5. MODEL CLV can use FINAL odds.
6. Closing capture cannot resolve both markets independently.
7. Paper picks can influence model training/tuning.
8. Quota can be exceeded under the scheduled workflow.
9. `closing-lines.yml` is missing or invalid.
10. `paper_trading_report` is not actually scheduled.
11. Tests fail.
12. Secrets would be committed.
13. Production data would need manual modification.
14. The deployment path is not the actual production path.

**Do not "fix" a blocker silently. Report it and stop.**

The goal is not merely to get green tests.

The goal is:

> **A reproducible, reviewable, deployable frozen CLV experiment where MODEL and FINAL attribution are collected from the moment of pick creation and can never be silently conflated.**
