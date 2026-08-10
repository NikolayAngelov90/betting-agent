# Stage 9 — Dual CLV Attribution: Frozen Model vs Final Selection

> **⚠️ CORRECTED BY STAGE 10.2 (2026-08-10).** The model identity quoted below
> was computed from the local, gitignored `config/config.yaml`, which CI
> overwrites and production has never executed. The deployed configuration is
> `config/config.example.yaml`. Correct identities:
> `stage5_baseline_20260807.326fcf` (CODE_REVISION `s5.1`, Stages 5-7) and
> `stage5_baseline_20260807.485823` (CODE_REVISION `s5.2`, Stages 8-10).
> Everything else in this report stands. See
> `docs/stage10.2-model-identity-2026-08-10.md`.

**Date:** 2026-08-10
**Model:** frozen. `model_version = stage5_baseline_20260807.d1b522`, `CODE_REVISION = s5.2` — **both unchanged**
**Tests:** 589 passed (Stage 8 baseline: 566)
**Production:** unchanged. **Odds API credits consumed: 0**
**Status:** uncommitted, not pushed, not deployed

---

## Executive Summary

Stage 8 recommended Option 1: capture closing lines for both the frozen model's
selection and the final persisted selection, and report two CLV series. Stage 9
implements that — as far as the existing schema allows — and stops at a hard
boundary that §3 of the brief requires me to stop at rather than cross.

**What is built and working:** the attribution model itself. `model` and `final`
are now first-class, named series. Every pick resolves to two `SeriesSpec`s that
say what each series bets on, at what price, and — when a series cannot be
measured — exactly why, with distinct reason codes. The report is split into two
series with independent cluster-aware statistics, a paired subset, a
review-action breakdown, a full coverage cross-tab and **separate checkpoint
counters**. 23 new tests pin the contract.

**What is blocked, and why it is a stop rather than a workaround:** a Claude
CHANGE overwrites `SavedPick.odds` with the new selection's price. The model
selection's own taken price is then gone — and CLV is `taken / closing − 1`, so
without a taken price there is no model CLV to compute. Separately, the schema
has exactly one set of `closing_*` columns, so even with the price there is
nowhere to put a second closing observation.

Both gaps sit precisely on the rows where the two series would say something
different. Measured on production: of the 71 picks carrying a model snapshot, 49
are identical (one observation, two attributions — already handled) and **22 are
genuine changes, none of which can produce a model CLV.**

The blocker is one additive table. **I have not created or applied it.**

The most important thing Stage 9 ships is arguably not the reporting but the
guard: on a changed selection the model series returns `unavailable`, never the
final selection's price. Silent substitution there would have produced a clean,
well-clustered, entirely wrong answer to the experiment's headline question.

---

## Experiment Definition

| | Series A | Series B |
|---|---|---|
| **Name** | `model` | `final` |
| **Fields** | `model_market`, `model_selection`, `model_probability` | `market`, `selection`, `predicted_probability` |
| **Taken price** | `odds` *only when the review kept the pick* | `odds` |
| **Question** | Did the **frozen Stage 5 model** identify selections whose prices moved in the expected direction before closing? | Did the **final model + Claude** selection move that way? |

These are never merged. Every number in the report is labelled `model` or
`final`, and the checkpoint section prints two independent counters.

---

## Implementation

### New — `src/evaluation/attribution.py`

The pure, schema-independent core. No I/O, no closing-price logic; it answers
"what is each series betting on, and can it be measured at all?"

* `MODEL` / `FINAL` — the attribution vocabulary. No `original`, `saved`,
  `current`, `pick` or `selection2`.
* `SeriesSpec` — `attribution`, `market`, `selection`, `taken_odds`,
  `unavailable_reason`, `.measurable`.
* `resolve(pick) -> (model_spec, final_spec)` — the mapping rules below.
* `shares_one_observation(m, f)` — true when the review kept the pick, so one
  close serves both (§6).
* `coverage_class(m, f)` — the §12 cross-tab bucket.
* `selection_changed(pick)` — **tri-state**: `True` / `False` / `None`. A
  missing snapshot is not evidence the selection was unchanged.

### Changed — `scripts/paper_trading_report.py`

* `_Pick` and `load_picks` now carry `model_market` (the model series resolves
  its close in the **model's** market, which differs after a CHANGE).
* `section_attribution_coverage` — new. The full §12 table.
* `section_clv` — rewritten into Series A / Series B blocks, each with valid
  closing lines, independent fixtures, effective n, design effect, CLV mean,
  median and cluster-bootstrap CI.
* `_section_paired` — new. §10's paired subset.
* `_section_by_review_action` — new. §11's `none` / `KEEP` / `CHANGE` table with
  the model / final / delta columns.
* `section_checkpoints` — split into `MODEL:` and `FINAL:` counters.

### Not changed

`capture_closing_lines.py` is untouched. It resolves `SavedPick.selection`, i.e.
the final series, and where the review kept the pick that same close is the
model's close too — the report attributes it to both without a second capture,
a second row or a second credit. Extending capture to resolve a *second*
selection is exactly the blocked work; half-building it against a schema that
cannot store the result would be worse than not building it.

---

## Attribution Rules

| Condition | `model` series | `final` series |
|---|---|---|
| `model_selection` / `model_market` absent | `unavailable(no_model_snapshot)` | measurable |
| `model_* == market/selection` | measurable, `taken_odds = odds` | measurable, same price — **one observation** |
| `model_* != market/selection` (a CHANGE) | `unavailable(model_taken_price_not_recorded)` | measurable |
| no usable decimal price on the row | `unavailable(no_taken_price)` | `unavailable(no_taken_price)` |
| no market/selection at all | — | `unavailable(no_selection)` |

Four distinct reason codes, never pooled into one "missing" bucket, because they
call for different responses: `no_model_snapshot` is a historical record gap,
`model_taken_price_not_recorded` is the live blocker, and the other two are data
faults.

**`unavailable` is not a failed prediction and is never counted as zero CLV.**

---

## CLV Integrity — Stage 8 rules intact

Nothing was weakened. Verified by test and by source:

* correct fixture / market / selection, valid decimal odds, valid timestamp,
  before kickoff, inside the closing window, valid bookmaker, market-structure
  and overround rules, no corrupt-source collision — unchanged
* `missing` / `late` / `invalid` semantics — unchanged
* **the same-snapshot rule** — unchanged and now pinned by a parametrised
  boundary test at −5 min / 0 / +5 min relative to `created_at`, plus a source
  assertion that the comparison is `ts <= not_before` and not `>=`
* `created_at` is **not** reset for the model series and no separate model
  timestamp is fabricated. Both series share the one causal boundary the data
  actually contains: the pick's creation time.
* `CODE_REVISION` and `model_version` unchanged, asserted by test.

Each series must independently satisfy every rule; the model and final markets
may differ, and a valid close for one is never reused for the other.

---

## Statistical Method

Fixture-level clustering, consistent with Stage 8, applied per series:

* **Per series** — cluster bootstrap over that series' own fixtures. A shared
  observation contributes one model CLV and one final CLV but **one fixture** to
  each series' cluster set. Pooling the two series would report 2n picks on n
  fixtures and a design effect of 2.0 — a test asserts exactly this does not
  happen.
* **Paired subset** — only picks where both series produced a valid close. The
  statistic is the within-pick delta `final − model`, bootstrapped with the same
  fixture clustering. It is reported as an *observed difference in CLV*, never
  as an independent CLV sample and never as a causal claim about the review.
* **Review-action breakdown** — `none` / `KEEP` / `CHANGE` with model CLV, final
  CLV and delta. On CHANGE rows a model CLV of `n/a` means the model selection's
  own close was not obtainable; the final selection's close is never substituted,
  and the report says so in the output itself.

---

## Coverage

Measured on all 1,070 production picks, read-only:

```
picks considered        : 1070
same selection          :   49
changed selection       :   22
snapshot unavailable    :  999   (cannot say whether the selection changed)

measurability (pick record only, before any closing price):
    both_measurable_same_selection      49    4.6%
    both_measurable                      0    0.0%
    model_only_measurable                0    0.0%
    final_only_measurable             1021   95.4%
    neither_measurable                   0    0.0%

model series unavailable because:
    no_model_snapshot                  999   (predates the snapshot columns)
    model_taken_price_not_recorded      22   (CHANGE overwrote the taken price)
```

`both_measurable = 0` is the blocker stated as a number: **there is not one
production pick where the two series are different bets and both can be
measured.**

---

## Historical Validation (read-only)

All figures from `SELECT` only. **No closing line was fabricated.**

| # | Question | Answer |
|---|---|---|
| 1 | Picks with both model and final snapshots | **71** of 1,070 |
| 2 | Identical selections | **49** |
| 3 | `none` / `KEEP` / `CHANGE` | 840 / 134 / **96** |
| 4 | Model selections in CLV-eligible markets | **32** |
| 5 | Final selections in CLV-eligible markets | **803** |
| 6 | Historical observations that can produce a valid close today | **0** — `closing_odds` is NULL on every row |
| 7 | Model-only measurable | **0** |
| 8 | Final-only measurable | **1,021** |
| 9 | Paired (both measurable) | **0**; ceiling 49 once closes exist |
| 10 | Additional attribution ambiguity? | **Yes — see below** |

### Q10 — the ambiguity found

**`review_action = 'CHANGE'` does not reliably mean the row's selection
changed.** Cross-tabulating action against the snapshot:

| `review_action` | snapshot missing | identical | differs |
|---|---:|---:|---:|
| `NULL` | 837 | 3 | 0 |
| `KEEP` | 90 | 44 | 0 |
| `CHANGE` | 72 | **2** | 22 |

Two `CHANGE` rows have `model_selection == selection`. The cause is the
consolidation branch in `_apply_decision`: when Claude switches onto a selection
another pick on that match already holds, the *surviving* row is stamped
`review_action = "CHANGE"` and the primary is deleted. That row's own selection
never changed.

Grouping the CLV series by `review_action` alone would therefore misclassify
those rows. The implementation does not: `selection_changed()` and `resolve()`
both key on `model_selection != selection`, and `review_action` is used only for
the descriptive §11 breakdown. Recorded here so the distinction is deliberate
rather than accidental.

---

## Historical CLV vs Prospective CLV

The database has `closing_odds` on **0 of 1,070** rows. The historical analysis
above therefore validates *attribution availability, selection differences,
eligibility, schema integrity, coverage potential and statistical grouping* —
and nothing else.

**No historical CLV is reported, simulated or implied anywhere in this stage.**
Zero production closing lines is not a reason to relax any validity rule, and
none was relaxed.

---

## Tests

**566 → 589. Zero regressions.** 23 new tests in
`tests/test_dual_clv_attribution.py`:

| § | Test | Covers |
|---|---|---|
| 1 | identical selections | one shared observation; capture runs once, creates no second row |
| 2 | differing selections | **no price substitution in either direction**, unit and end-to-end |
| 3 | model measurable, final not | neither series drags the other down |
| 4 | final measurable, model not | the 999-row historical case |
| 5 | one odds row, two attributions | two counters, **one** fixture; pooling would give deff 2.0 |
| 6 | same-snapshot rule | −5 / 0 / +5 min boundary; source assertion that the bound is exclusive |
| 7 | no model snapshot | `unavailable`, not "failed"; `selection_changed` returns `None` |
| 8 | CHANGE attribution | each series maps to its own fields; KEEP is not a change |
| 9 | paired comparison | delta is `final − model`; fixture clustering preserved; wording is non-causal |
| 10 | checkpoint separation | series named separately; a changed selection's close cannot advance MODEL |
| 11 | paper/live isolation | still holds |
| 12 | Stage 8 integrity | correlation policy intact; `CODE_REVISION` and `model_version` unmoved |
| — | the blocker | pinned as a contract: the model price must be `None`, never reconstructed by inverting stored EV |

That last test deserves a note. `pre_claude_ev` and `model_probability` are both
recorded, and `odds = (ev + 1) / p` inverts the EV formula — so it is tempting to
recover the model's taken price arithmetically. It is unsound: `_market_ev`
scales Draw No Bet by `P(decisive)`, which is not stored, so a DNB pick would
silently yield a wrong price. DNB is an enabled market with 0 picks so far — the
trap is armed but has not fired. The test exists so a future "fix" cannot spring
it.

---

## Quota

**Odds API credits consumed by Stage 9: 0.**

No live API call was made. All validation used existing database rows,
deterministic fixtures and temp SQLite. The quota policy is untouched: 400
monthly budget, 50 safety margin, 24 per-run ceiling, claim-before-spend ledger.
The `api_budget` table still holds only the four `api-football` rows — no
`theoddsapi` row exists, confirming nothing in this stage wrote to it.

---

## Production Safety

Read-only throughout. `SELECT` only.

* picks inserted / updated / deleted / settled: **0**
* closing odds written: **0**
* quota state modified: **no**
* configuration altered: **no**
* migrations created or applied: **no**
* paper/live state changed: **no** (paper trading remains **ON**)
* secrets or keys added: **no**

---

## BLOCKER — the migration I did not create

§3 requires me to stop before creating a migration and report exactly why the
existing schema cannot represent the two observations. Two independent reasons:

### 1. The model selection's taken price is destroyed

`_apply_decision` assigns `primary.odds = float(new.odds)`. `model_market`,
`model_selection` and `model_probability` are deliberately preserved — but not
the price. CLV is `taken / closing − 1`; a probability is not a price.

It cannot be recovered afterwards: `ix_odds_match_bookie_market` is UNIQUE on
`(match_id, bookmaker, market_type, selection)`, so the odds table holds exactly
one row per book and every refresh overwrites it. **There is no price history to
look back into.** And the EV inversion is unsound for DNB, as above.

### 2. There is only one set of closing columns

`saved_picks` has a single `closing_odds`, `closing_odds_captured_at`,
`closing_capture_status`, `closing_bookmaker_count`, `closing_fair_probability`.
A CHANGE row needs **two** closing observations in **two different markets**. No
free column exists, and no existing column can carry it without abusing its
meaning.

### Recommended shape — one additive table, `saved_picks` untouched

```sql
CREATE TABLE pick_observations (
    id                   SERIAL PRIMARY KEY,
    pick_id              INTEGER NOT NULL REFERENCES saved_picks(id) ON DELETE CASCADE,
    attribution          VARCHAR(8)  NOT NULL,     -- 'model' | 'final'
    market               VARCHAR(50) NOT NULL,
    selection            VARCHAR(100) NOT NULL,
    taken_odds           DOUBLE PRECISION NOT NULL,   -- written at pick time
    taken_at             TIMESTAMP NOT NULL,          -- = saved_picks.created_at
    closing_odds         DOUBLE PRECISION,
    closing_captured_at  TIMESTAMP,
    closing_status       VARCHAR(16) NOT NULL DEFAULT 'pending',
    closing_book_count   INTEGER,
    closing_fair_prob    DOUBLE PRECISION,
    UNIQUE (pick_id, attribution)
);
```

Why this shape:

* **Additive.** No column added to, or altered on, `saved_picks`. Existing
  closing columns keep working; nothing already written changes meaning.
* **Solves both gaps at once.** `taken_odds` is written at pick-save time for
  both series, *before* the review can overwrite anything — which is the only
  moment the model's price exists.
* **`UNIQUE (pick_id, attribution)`** enforces §3's requirement that the two
  observations stay distinguishable and §6's that neither is duplicated.
* **Matches §6 exactly**: on an unchanged pick both rows carry the same market,
  selection and price, so capture resolves one close and writes it to both — one
  API observation, two attributions, still one fixture.
* Historical rows get no backfill. They keep `no_model_snapshot` /
  `model_taken_price_not_recorded`, which is the truth about them.

Work required after approval: the migration + rollback, a write in `_save_picks`,
capture resolving both attributions (needing the model's market in
`needed_markets`), and repointing the report at the new table. The report's
statistics, coverage table, paired comparison and checkpoint split are already
built and need no change.

**Nothing here is created or applied. It needs explicit approval.**

---

## Decision

```
NOT READY
```

**Concrete blockers — both are the same migration:**

1. **The model selection's taken price is not recorded on a Claude CHANGE.**
   Without it, model CLV is uncomputable for exactly the 22 production rows —
   and ~22% of future picks — where the two series differ. §11's CHANGE delta,
   the key diagnostic for the review, cannot be produced.
2. **No storage exists for a second closing observation.** `saved_picks` has one
   set of closing columns; a CHANGE row needs two, in two different markets.

Everything else Stage 9 asked for is built, tested and read-only-validated. The
`final` series is fully operational and will collect valid closing lines the
moment the cron produces them. The `model` series is operational for unchanged
picks — 49 of the 71 snapshotted rows, and the majority of future picks — and
correctly reports `unavailable` everywhere else instead of quietly borrowing the
final series' numbers.

Approve the table and Stage 9 closes; the remaining work is mechanical and the
statistics are already in place.
