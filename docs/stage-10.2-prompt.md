# Stage 10.2 — Freeze and Reconcile the Experimental Model Identity

## Objective

Resolve **Blocker 2** from Stage 10.1.

Do not work on the transaction blocker yet.

Do not commit, push, merge, deploy, or modify production data.

Do not consume Odds API credits.

The purpose of this stage is to establish, unambiguously and permanently, **which exact configuration is the frozen experimental subject** before any prospective paper-trading observations are collected.

---

## Known finding

Stage 10.1 discovered:

```text
local config/config.yaml
    model_version = stage5_baseline_20260807.d1b522

CI config/config.example.yaml
    model_version = stage5_baseline_20260807.485823
```

The difference is caused by:

```yaml
betting:
  excluded_markets:
```

Local:

```text
[btts_no, under_1.5]
```

CI/example:

```text
[btts_no, over_3.5, under_1.5]
```

CI currently does:

```text
cp config/config.example.yaml config/config.yaml
```

Therefore production will use `config.example.yaml`, not the local ignored `config/config.yaml`.

Production evidence also shows:

* `Over 3.5` picks stop at 2026-06-15.
* `Under 1.5` never appears.
* production has never actually executed the local `d1b522` configuration.

This means the previous Stage 5–10 references to:

```text
stage5_baseline_20260807.d1b522
```

cannot currently be treated as the production frozen model identity.

---

# 1. First: independently verify the finding

Before changing anything, inspect:

* `config/config.yaml`
* `config/config.example.yaml`
* model-version/fingerprint implementation
* `TRACKED_KEYS`
* `CODE_REVISION`
* all Stage 5–10 tests that pin the model version
* CI workflow that creates `config.yaml`
* any other workflow/script that constructs production configuration

Do not assume the Stage 10.1 report is correct.

Produce:

```text
LOCAL CONFIG HASH:
EXAMPLE CONFIG HASH:
PRODUCTION CONFIG SOURCE:
TRACKED_KEYS:
CURRENT LOCAL MODEL VERSION:
CURRENT CI MODEL VERSION:
```

Also verify whether `excluded_markets` is definitely part of the model fingerprint and selection behavior.

---

# 2. Determine the correct experimental subject

There are two possible choices.

## Option A — Example/production configuration is authoritative

Accept:

```text
excluded_markets =
[btts_no, over_3.5, under_1.5]
```

Then:

* production configuration remains authoritative;
* recompute the frozen model identity;
* update the model-version expectation;
* update tests that pin the identity;
* update Stage 5–10 documentation that incorrectly calls `d1b522` the production frozen model;
* explicitly record that `over_3.5` is excluded from the frozen experimental subject.

## Option B — Local configuration is authoritative

Change `config/config.example.yaml` so it matches the local configuration:

```text
[btts_no, under_1.5]
```

Then retain:

```text
stage5_baseline_20260807.d1b522
```

However, do NOT choose Option B merely because it preserves the existing version.

You must establish from repository history and the Stage 5 experiment definition whether `over_3.5` was intentionally excluded as part of the frozen model.

If `over_3.5` was historically excluded intentionally, do not re-enable it merely to make hashes match.

If the example's exclusion is an accidental configuration drift and the intended frozen model clearly excludes only `btts_no` and `under_1.5`, document the evidence.

---

# 3. IMPORTANT — Do not optimize the decision

This is NOT an opportunity to improve the model.

Do not:

* add markets;
* remove markets for performance reasons;
* change thresholds;
* change probabilities;
* change gates;
* change features;
* change weights;
* change bookmaker logic;
* change EV;
* change confidence;
* change Claude behavior;
* change ML;
* change Elo;
* change Poisson;
* change calibration;
* change any selection rule except the configuration reconciliation itself.

The only permitted model-affecting change is making the **authoritative experimental configuration explicit and identical across local and CI**.

---

# 4. Historical evidence

Inspect Git history around:

```text
origin/main = 6f0e42c
Stage 5 = c0d3b7f
```

Search when `over_3.5` was added to `config.example.yaml`.

Determine:

1. Which commit introduced it.
2. Whether the local `config.yaml` was ever committed.
3. Whether Stage 5's experiment documentation explicitly names the excluded markets.
4. Whether `over_3.5` was intentionally excluded or is configuration drift.
5. Whether production behavior before Stage 5 confirms the intended setting.

Do not infer intent from the current hash alone.

Report the evidence.

---

# 5. Choose and implement exactly one authoritative configuration

After the historical inspection, choose the configuration supported by evidence.

Then make:

```text
config/config.example.yaml
config/config.yaml
```

semantically identical for all **TRACKED_KEYS**.

Remember:

* `config/config.yaml` may be gitignored.
* The important production artifact is `config.example.yaml`.
* Do not add secrets.
* Do not commit real credentials.
* Do not alter unrelated untracked local configuration.

If local `config.yaml` is ignored and cannot be part of the repository change, verify equivalence programmatically instead of trying to force-add it.

---

# 6. Model identity

After reconciliation:

1. Compute the actual fingerprint.
2. Record the resulting `model_version`.
3. Verify `CODE_REVISION` remains:

```text
s5.2
```

unless the chosen configuration change logically requires otherwise.

Do NOT bump `CODE_REVISION` simply because the configuration was reconciled.

The model version must describe the actual frozen configuration.

---

# 7. Add a configuration consistency invariant

Add a regression test that ensures:

```text
config/config.example.yaml
```

and the production configuration template are identical for every `TRACKED_KEY`.

The test must fail if:

* `excluded_markets` diverges;
* any other tracked model parameter diverges;
* the CI template produces a different model fingerprint from the expected frozen configuration.

Do not create a brittle test that merely asserts a hardcoded version string without checking the underlying configuration.

Prefer testing:

```text
fingerprint(example_config) == fingerprint(authoritative_frozen_config)
```

and separately pin the resulting model version with a rationale.

---

# 8. Audit every place that reads model_version

Search repository-wide for:

```text
model_version
CODE_REVISION
TRACKED_KEYS
excluded_markets
```

Verify:

* pick persistence uses the reconciled version;
* reports filter/use the correct version;
* paper/live isolation does not accidentally exclude new paper rows;
* tests use the new authoritative identity;
* no historical rows are rewritten;
* no existing production pick is re-versioned.

---

# 9. Production safety

Production inspection only.

Do NOT:

* update saved picks;
* update model_version on historical rows;
* backfill observations;
* run pick generation;
* run settlement;
* run closing capture;
* consume Odds API credits.

The only acceptable production activity is read-only verification.

Confirm:

```text
historical picks unchanged
pick_observations = 0
closing observations unchanged
paper picks = 0
```

If scheduled production jobs naturally modify data during the session, distinguish those changes from this stage.

---

# 10. Tests

Run the complete test suite.

Expected:

```text
previous baseline: 605
```

If tests fail because they correctly pin the old model identity, update only those tests affected by the authoritative configuration decision.

Do not weaken tests.

Do not delete tests merely because they fail.

Do not change unrelated behavior to make tests pass.

---

# 11. Version semantics

Explicitly classify the configuration reconciliation:

### Prediction-affecting?

Yes, if `excluded_markets` changes the set of possible selections.

### Selection-affecting?

Yes.

### Evaluation-only?

No.

Therefore, if the authoritative configuration actually changes relative to the previously frozen subject, explain whether the model identity must change.

Do not automatically bump `CODE_REVISION`.

Distinguish:

```text
model_version
```

from:

```text
CODE_REVISION
```

and explain why each does or does not change.

---

# 12. STOPPING RULE

STOP immediately if any of the following is unclear:

* whether `over_3.5` was intentionally excluded;
* which configuration Stage 5 actually defined as frozen;
* whether the fingerprint includes the relevant configuration;
* whether changing the example changes model behavior;
* whether the resulting model identity can be reproduced deterministically.

Do not guess.

Report the ambiguity and the exact evidence needed to resolve it.

---

# 13. Required output

Return:

# Stage 10.2 — Frozen Model Identity Reconciliation

## 1. Finding verification

## 2. Git/history evidence

## 3. Authoritative configuration decision

Explicitly state:

```text
OPTION A
```

or

```text
OPTION B
```

and why.

## 4. Configuration diff

Show only the relevant tracked differences.

## 5. Resulting model identity

```text
model_version:
CODE_REVISION:
```

## 6. Prediction/selection/evaluation classification

## 7. Tests

```text
before:
after:
new/changed tests:
```

## 8. Production verification

Confirm no production data was modified.

## 9. Files changed

List every changed file.

## 10. Remaining risks

Only real risks.

## 11. Final decision

Use exactly one:

```text
READY FOR STAGE 10.3
```

or

```text
NOT READY — BLOCKER
```

If READY, do NOT start Stage 10.3 automatically.

Do not commit.

Do not push.

Do not deploy.

Do not create the transaction fix in this stage.

---

## Non-negotiable rules

**NO COMMIT.**

**NO PUSH.**

**NO DEPLOY.**

**NO PRODUCTION WRITES.**

**NO ODDS API CREDITS.**

**NO MODEL OPTIMIZATION.**

**NO SILENT VERSION CHANGES.**

**DO NOT GUESS THE FROZEN MODEL.**

The objective is to know exactly what model we are about to measure before the first prospective paper-trading observation is collected.
