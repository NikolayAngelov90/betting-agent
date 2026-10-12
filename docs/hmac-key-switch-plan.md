# The HMAC Key Switch — Plan Only. Nothing Here Is Applied.

**Written 2026-10-12. Executed after 2026-10-17**, when H1's collection window
closes. No secret is generated, configured, rotated or read by this document.
No key value, digest or preimage appears in it.

The guarantee this restores and why it never held is in the ledger entry of
2026-10-12. This file is the ordering, the rollback per step, and the cost of
getting the order wrong.

---

## WHAT A RUN ACTUALLY DOES — measured, not assumed

`measured` 2026-10-12 from the local audit log cache (`ci_logs/`, 807 cached
runs) and from the workflow and module source:

| question | answer |
| --- | --- |
| restore or retrain? | **Restore, usually.** `daily-picks` restores `data/models/` from cache `betting-models-<run#>` with `restore-keys: betting-models-`, then runs `--train`, which retrains **only if `trained_at` is older than `ml_retrain_days: 3`**. 239 of 807 cached runs performed a load; 172 performed a save. |
| what happens on a verification failure? | **Refuse and continue.** `_safe_load` raises `IntegrityError`; both handlers (`ml_models.py:591`, `:915`) log and `return`, leaving `is_fitted` False. `ensemble.py:174` gates the ML term on `is_fitted`, so the ensemble omits it. **Picks are still produced, from Poisson + Elo + the bookmaker blend.** This is code, not a docstring claim. |
| does it crash? | **No** — not on a signature mismatch. The one place it *would* crash is the module-level `raise RuntimeError` when **neither** `MODEL_HMAC_KEY` nor the opt-in is set: that fires at **import**, outside every handler. See the cost table. |
| cached pairs now | **8** entries, `betting-models-317` … `-324`, ~6.94 MiB each |
| which runs read them | run *N* restores the newest `betting-models-*` entry, so the **next** scheduled run reads whatever the last one saved |
| can a forced retrain fit the budget? | **Yes, 96% of the time.** Retrain spans `measured` n=173: min 1.8, **median 4.5**, mean 10.5, max 71.9 minutes. The step's `timeout-minutes: 50` was exceeded by **7 of 173 (4.0%)**, and the step is `continue-on-error: true` with an alert on `outcome == 'failure'`. |

### So what does a failed switch cost?

**A slower run and one ML-free pick set — not a card.** Three independent
reasons, each measured above: the failure is caught rather than fatal; the
ensemble degrades instead of stopping; and the retrain that repairs it fits the
step budget in 96% of observed cases, with the overflow already non-fatal.

**The single exception is step 3 out of order**, and it is the whole reason this
document exists.

---

## THE SEQUENCE

Each step has a checkpoint that must pass before the next, and a rollback.

### Step 1 — generate the secret and configure it. Nothing else.

Add `MODEL_HMAC_KEY` as a repository secret and pass it in the `env:` of
**every workflow that imports `src.models.ml_models`** — `measured`
2026-10-12, that is all six, and only `daily-picks.yml:180` passes it today.
**Leave every `BETTING_AGENT_ALLOW_DEFAULT_HMAC: '1'` exactly where it is.**

* **Checkpoint:** one `workflow_dispatch` of `daily-picks`. Expect
  `MODEL SIGNATURE MISSING`: no — and **`MODEL INTEGRITY FAILURE`: yes, once**,
  because the restored pair is still signed with the old key. Part A's alarm
  must fire. If it does not, the wire is broken and the switch stops here.
* **Rollback:** delete the secret. The next run falls back to the default key
  and verifies the old artefacts again.
* **Cost if skipped or late:** see step 2.

> **Why the opt-in stays:** with the secret present, `MODEL_HMAC_KEY` is
> non-empty and the opt-in branch is never reached. It is harmless here and it
> is the only thing standing between step 3 and an import-time crash.

### Step 2 — retire the old cache so no run meets an old-key pair

Either delete the `betting-models-*` entries, or let one `--train` re-sign and
save a new entry under the current run number. Deleting is cleaner: eight
entries, ~56 MiB, and `restore-keys` will simply miss.

* **Checkpoint:** the following run logs **no** `MODEL INTEGRITY FAILURE`, and
  `Models saved to data/models/ml_models.pkl` appears — i.e. it retrained from
  cold and signed with the new key.
* **Rollback:** none needed, and none possible — a deleted cache is not
  recoverable. That is acceptable because losing it costs one cold retrain,
  which the workflow header already states is safe.
* **Cost of doing this BEFORE step 1:** the run retrains and re-signs with the
  **default** key, so the cache is refreshed into exactly the state being
  escaped. One wasted retrain, median 4.5 min. **No card.**

### Step 3 — remove the local-dev opt-in from the production workflows

Delete `BETTING_AGENT_ALLOW_DEFAULT_HMAC: '1'` from `daily-picks.yml:181` and
from the other four workflows that set it, so the module-level `RuntimeError`
can fire where it was meant to.

* **Checkpoint:** a dispatch completes normally. Then, separately, confirm the
  guard is live by asserting in a test — not in production — that import with
  neither variable raises.
* **Rollback:** restore the line. It is one line in each workflow and it
  re-enables the default-key path immediately.

> ### THIS IS THE STEP THAT COSTS A CARD IF IT IS TAKEN FIRST
>
> The refusal is a **module-level `raise RuntimeError`**, not a load-time
> `IntegrityError`. It fires on `import src.models.ml_models`, which
> `src.agent.betting_agent` does unconditionally — so **every** step of
> `daily-picks` (`--update`, `--picks`, `--settle`, `--train`, `--report`)
> dies at import with no picks, no settlement and no report. None of the
> `except IntegrityError` handlers is reached, because the exception happens
> before any of that code exists.
>
> Measured by driving it 2026-10-12: with both variables unset,
> `import src.models.ml_models` raises `RuntimeError` immediately.
>
> **A test gate has already cost one card permanently and nearly cost a
> second. This would cost one per day until someone noticed.**

* **Cost of step 3 before step 1:** a lost card every run. **The only
  card-costing error in this sequence.**
* **Cost of step 3 before step 2:** none beyond step 1's — the integrity
  failure is still caught and still degrades to Poisson + Elo.

### Step 4 — make a missing `.sig` fail closed

Only after Part A's measured unsigned-load count has been **zero over a
window**. It is zero today (0 in 1,400 load events across 239 runs,
`measured` 2026-10-12), which is the baseline, not yet the window.

* **It must raise `IntegrityError` inside the existing `try` in `_safe_load`**,
  so both handlers catch it and the run degrades. Raising anything else, or
  raising outside the try, converts a degradation into a crash.
* **Checkpoint:** a test proving an unsigned pickle now refuses **and** that
  the agent still produces picks.
* **Rollback:** restore the warning-and-load branch.
* **Cost of taking step 4 before the count is measured at zero:** every run
  whose cache carries a pickle without its `.sig` loses ML silently — and
  before Part A, *silently* was literal: nothing read the marker.

---

## THE ORDERING, AS A TABLE

| order | consequence |
| --- | --- |
| 1 → 2 → 3 → 4 | intended. One integrity alarm at step 1's checkpoint, one cold retrain at step 2. |
| **3 first** | **every step of every run dies at import. A lost card per run.** |
| 2 before 1 | one wasted retrain; the cache is re-signed with the key being escaped. |
| 1, 3, no 2 | one ML-free pick set plus one alarm, self-repairing at the next `--train`. |
| 4 before the window | ML silently lost on any run whose cache lacks a `.sig`. |

---

## THE EXPOSURE BOUND IS A PROVIDER PROPERTY, NOT OURS

The reason today's default key is a low-severity exposure rather than a high one
is that writing a `betting-models-*` cache entry requires repository write
access: all six workflows trigger on `schedule` and `workflow_dispatch` only —
no `push`, no `pull_request`, no `pull_request_target` (`measured` 2026-10-12) —
and GitHub scopes caches so that a branch cannot write a cache the default
branch reads.

> **That scoping is a property of the provider.** This project does not
> implement it, cannot test it, and would not be notified if it changed. If
> GitHub's cache isolation changes, the bound is gone and the severity of the
> default key moves with it — while nothing in this repository would look
> different.
>
> Recorded so the bound is never cited as a guarantee we hold. The same
> reasoning retires it: once step 3 is done, the exposure no longer depends on
> a provider behaviour at all.
