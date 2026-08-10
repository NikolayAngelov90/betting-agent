# Stage 10.2 — Frozen Model Identity Reconciliation

**Date:** 2026-08-10
**Decision:** **OPTION A** — the deployed `config/config.example.yaml` is authoritative
**Frozen identity:** `stage5_baseline_20260807.485823` · `CODE_REVISION = s5.2`
**Tests:** 614 passed (was 605)
**Production:** unchanged, read-only. Odds API credits: 0
**Status:** uncommitted, not pushed, not deployed

---

## 1. Finding verification

Verified independently of the Stage 10.1 report.

```
LOCAL CONFIG HASH        : sha256:69fd379a02cb6f36   (config/config.yaml, file digest)
EXAMPLE CONFIG HASH      : sha256:e385912bba27a846   (config/config.example.yaml)
PRODUCTION CONFIG SOURCE : config/config.example.yaml
                           (daily-picks.yml:120 and closing-lines.yml both run
                            `cp config/config.example.yaml config/config.yaml`)
TRACKED_KEYS             : 23
CURRENT LOCAL MODEL VERSION : stage5_baseline_20260807.d1b522
CURRENT CI MODEL VERSION    : stage5_baseline_20260807.485823
```

**Exactly one** of the 23 tracked keys differed:

```
betting.excluded_markets
    local = ['btts_no', 'under_1.5']
    CI    = ['btts_no', 'over_3.5', 'under_1.5']
```

All 22 others were byte-identical, including `bookmaker_blend_weight = 0.8`,
`min_expected_value = 0.05`, `min_confidence = 0.55`, the ensemble weights and
all four gates (`False`).

**`excluded_markets` is genuinely part of the model, not just the hash.**
`"betting.excluded_markets"` is in `TRACKED_KEYS`, and a probe confirms that
changing it moves the fingerprint. It is enforced at **three** selection sites in
`value_calculator.py` — `find_value_bets` (124), `find_best_bet` (405) and
`build_selection_pick` (566). The third matters: it also constrains the menu
Claude's CHANGE may switch to.

---

## 2. Git / history evidence

**Was `config/config.yaml` ever committed?** No. `git log --all -- config/config.yaml`
returns nothing; it is in `.gitignore`. It has never existed in any commit, on
any branch, and has therefore never been deployed. It is one machine's scratch
file.

**Which commit introduced `over_3.5`?** `bc7eacc`, 2026-06-16:

> `feat: WC prediction quality — market-trust on thin data, drop losing markets, youth filter`
>
> "…Exclude **over_3.5** globally (settled: 38% win, -14% ROI / 32 picks); unders
> & btts_no already excluded."

The diff adds one line, with the reason inline:

```yaml
- over_3.5    # proven loser in settled data: 38% win, -14% ROI over 32 picks
```

**The earlier exclusions** came from `7be908d` (2026-03-09,
`feat: exclude Under markets from picks`), which introduced the
`excluded_markets` key with `under_1.5`, `under_2.5`, `under_3.5`.

**Production behaviour corroborates.** Read-only:

```
Over 3.5   33 picks   2026-02-28 → 2026-06-15
Under 1.5   0 picks   (never appears)
```

Over 3.5 picks stop **the day before** `bc7eacc` landed. The exclusion took
effect in production immediately and has held since.

**Verdict: intentional exclusion, documented in both the commit body and the
config, and confirmed by production data. Not drift.** The drift is on the other
side — the local file is missing a line that the repository has carried since
June.

---

## 3. Authoritative configuration decision

```
OPTION A
```

`config/config.example.yaml` is the frozen experimental subject.

Three reasons, in order of weight:

1. **It is what production runs.** CI overwrites `config.yaml` with it. Any
   other choice would define a "frozen model" nothing executes — which is the
   defect being repaired, not a repair of it.
2. **The local file has no authority.** Never committed, never deployed,
   gitignored. Treating it as a specification was the original error.
3. **`over_3.5` was excluded deliberately** and §2 of the brief is explicit:
   *"If `over_3.5` was historically excluded intentionally, do not re-enable it
   merely to make hashes match."* It was, so it stays.

Option B would have required re-enabling a market that a documented decision
removed, purely to preserve a version string. That is choosing the answer and
working backwards.

**One observation, recorded and deliberately not acted on.** The exclusion rests
on 32 settled picks — the same species of small-sample, outcome-derived rule that
Stage 3 rejected when it disabled all six edge gates, and that
`settled-pick-segments-are-noise` warns about. Revisiting it may well be correct
*later*, as a considered model change with its own identity. It is out of scope
here: §3 forbids adding or removing markets, and doing it inside an identity
reconciliation would smuggle a model change into a bookkeeping fix.

---

## 4. Configuration diff

Only the tracked difference, now closed:

```diff
  betting:
    excluded_markets:
    - under_1.5
    - btts_no
+   - over_3.5
```

Applied to `config/config.yaml` (local, gitignored) so it matches the deployed
example. `config/config.example.yaml` — the production artifact — is
**unchanged**; it was already correct.

After reconciliation:

```
tracked differences : NONE
local  model_version: stage5_baseline_20260807.485823
CI     model_version: stage5_baseline_20260807.485823
identical           : True
```

---

## 5. Resulting model identity

```
model_version : stage5_baseline_20260807.485823
CODE_REVISION : s5.2
```

For the record, the deployed configuration's identity at each code revision:

| Code revision | Stages | Deployed identity | Previously (mis)reported |
|---|---|---|---|
| `s5.1` | 5–7 | `stage5_baseline_20260807.326fcf` | `…ac04cc` |
| `s5.2` | 8–10 | `stage5_baseline_20260807.485823` | `…d1b522` |

`ac04cc` and `d1b522` were computed from the local file and describe a
configuration that has never run anywhere.

---

## 6. Prediction / selection / evaluation classification

The distinction that matters: **the frozen subject did not change. The record of
it did.**

| Question | Answer |
|---|---|
| Did production's configuration change? | **No.** `config.example.yaml` is untouched. |
| Did any prediction path change? | No. |
| Did the selection space change? | No — production has excluded `over_3.5` since June. |
| Did the *local* config change? | Yes, to match production. Local dev was running a different model. |
| Is this a version bump? | **No — a correction.** `485823` is what production would always have stamped. |
| `CODE_REVISION`? | **Unchanged, `s5.2`.** No code changed. §6 is explicit that reconciliation alone must not bump it. |

Had this gone the other way — Option B, editing the example — it *would* have
been a genuine selection change requiring a new identity. It did not.

**No historical row was re-versioned.** All 1,074 production picks carry
`model_version = NULL` (they predate Stage 5 deployment); nothing rewrites them.

---

## 7. Tests

```
before : 605
after  : 614
```

**New — `tests/test_config_identity.py` (9 tests):**

| Test | Guards |
|---|---|
| `test_ci_builds_its_config_from_the_example` | that a workflow still `cp`s the example — if that stops, the whole premise moves |
| `test_local_config_is_not_tracked_and_carries_no_authority` | `config/config.yaml` stays gitignored |
| `test_example_and_local_agree_on_every_tracked_key` | the invariant; names the differing keys on failure |
| `test_example_and_local_produce_one_fingerprint` | same property, stated directly |
| `test_the_deployed_config_produces_the_frozen_model_version` | pins `485823` — **computed from the deployed file**, so it fails when the configuration moves, not when a constant is edited |
| `test_code_revision_is_unchanged_by_the_reconciliation` | `s5.2` |
| `test_over_35_is_excluded_in_the_deployed_config` | the exact excluded set; removing `over_3.5` must be a deliberate model change |
| `test_excluded_markets_is_part_of_the_fingerprint` | probes that changing it moves the hash |
| `test_excluded_markets_actually_constrains_selection` | ≥3 enforcement sites in `value_calculator` |

The two config-comparison tests `skip` (not fail) when no local `config.yaml`
exists — a fresh clone or CI checkout legitimately has none, and there is nothing
to diverge.

**Changed (2):** `test_dual_clv_attribution.py::test_12b_...` and
`test_pick_observations.py::test_model_version_and_code_revision_unchanged` both
read `config/config.yaml` and pinned `d1b522`. **That was the root cause in test
form** — they asserted against the gitignored file, so they would have passed on
one machine and failed everywhere else. Both now read `config.example.yaml` and
import `FROZEN_MODEL_VERSION` from the new module, so there is one constant.

No test was weakened or deleted.

---

## 8. Production verification

Read-only. Identical before and after this stage:

```
saved_picks       1,074      settled          1,070
paper                 0      model_version         0
pick_observations     0      closing_odds          0
max created_at    2026-08-10 11:09:57   (the 09:37 cron, not this stage)
max settled_at    2026-08-10 11:06:33   (same)
odds rows       333,023
theoddsapi ledger  no row  →  0 Odds API credits consumed
```

No pick updated, no `model_version` written, no observation created, no
settlement, no capture, no pick generation, no migration.

---

## 9. Files changed

| File | Change |
|---|---|
| `config/config.yaml` | **local, gitignored** — added `over_3.5` to match the deployed example |
| `tests/test_config_identity.py` | new — 9 tests + `FROZEN_MODEL_VERSION` |
| `tests/test_dual_clv_attribution.py` | repointed to the deployed config |
| `tests/test_pick_observations.py` | repointed to the deployed config |
| `docs/stage5-paper-trading-2026-08-07.md` | correction banner |
| `docs/stage6-odds-api-optimization-2026-08-10.md` | correction banner |
| `docs/stage7-model-observations-2026-08-10.md` | correction banner |
| `docs/stage7-operational-paper-trading-2026-08-10.md` | correction banner |
| `docs/stage8-experimental-integrity-2026-08-10.md` | correction banner |
| `docs/stage9-dual-clv-attribution-2026-08-10.md` | correction banner |
| `docs/stage10.2-model-identity-2026-08-10.md` | this report |

`config/config.example.yaml` — the production artifact — is **not** in this list.
No source file changed. No migration.

The stage reports keep their original text and carry a banner rather than being
rewritten: what was believed at the time is part of the record, and silently
editing six documents to match a later correction would erase the very drift this
stage exists to document.

---

## 10. Remaining risks

1. **`config/config.yaml` is unversioned and will drift again.** The new test
   catches it, but only when someone runs the suite locally. The structural fix
   is to stop having a second file at all — e.g. have local runs read the example
   directly. Out of scope here.
2. **The `over_3.5` exclusion rests on 32 picks.** Preserved as-is, correctly,
   but it is a small-sample rule inside a frozen experiment. If it is ever
   revisited it must be a deliberate model change with a new identity — not a
   config edit.
3. **Blocker 1 from Stage 10.1 is untouched**, as instructed: a pick can still
   commit without its observations.

---

## 11. Final decision

```
READY FOR STAGE 10.3
```

The frozen experimental subject is now unambiguous, identical across local and
CI, computed from the file production executes, pinned by a test that checks the
configuration rather than a string, and documented with the historical evidence
for why `over_3.5` is excluded.

Stage 10.3 (the transaction-atomicity blocker) is **not** started. Nothing
committed, pushed or deployed.
