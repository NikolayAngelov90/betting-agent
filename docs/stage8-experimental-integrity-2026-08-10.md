# Stage 8 — Experimental Integrity

> **⚠️ CORRECTED BY STAGE 10.2 (2026-08-10).** The model identity quoted below
> was computed from the local, gitignored `config/config.yaml`, which CI
> overwrites and production has never executed. The deployed configuration is
> `config/config.example.yaml`. Correct identities:
> `stage5_baseline_20260807.326fcf` (CODE_REVISION `s5.1`, Stages 5-7) and
> `stage5_baseline_20260807.485823` (CODE_REVISION `s5.2`, Stages 8-10).
> Everything else in this report stands. See
> `docs/stage10.2-model-identity-2026-08-10.md`.

**Date:** 2026-08-10
**Model parameters:** unchanged. **`CODE_REVISION`: `s5.1` → `s5.2`.**
**`model_version`: `stage5_baseline_20260807.ac04cc` → `stage5_baseline_20260807.d1b522`**
**Status:** uncommitted, not pushed, not deployed
**Tests:** 566 passed (Stage 7 baseline: 540)

---

## 1. Stage 7 baseline

Paper trading enabled; model frozen; 540 tests passing; no production rows
touched; no credits spent; closing capture using observed-price timestamps;
stale prices rejected; `is_paper` enforced through the live ROI/calibration
paths; provider quota reconciled.

Stage 7 closed with one open item: *"a correlation-filter gap related to
duplicate/correlated selections; fixing it requires a `CODE_REVISION` bump;
cheap now because the experiment has zero valid closing lines."*

**Stage 7 named the wrong mechanism.** The gap is real, but it is not primarily
the missing table entries Stage 7 identified. Reproducing it from source
changed the diagnosis, and the corrected finding is below.

---

## 2. The exact correlation gap

There are **two** independent defects, and the second is the one that actually
fired in production.

### 2a. `_CORRELATED_PAIRS` had no over↔under cross pairs

[betting_agent.py:4083](../src/agent/betting_agent.py#L4083) declared every
same-direction ladder — `Over 1.5/Over 2.5`, `Under 2.5/Under 3.5` — and the
1X2 ↔ Double Chance / DNB overlaps, but **no `Over X.5` / `Under Y.5` pair at
all**. Any such pair with `Y > X` passed untouched. This is what Stage 7 found.

### 2b. The correlation filter never runs again after Claude rewrites the pick

`_filter_correlated_picks` runs inside `get_daily_picks`. The Claude KEEP/CHANGE
review runs **after** it — after the picks are already persisted — and a CHANGE
**overwrites `market` and `selection` on the saved row**
([match_briefing.py:662](../src/reporting/match_briefing.py#L662)).

The only guard the CHANGE path had was exact-selection equality:

```python
for other in picks[1:]:
    if other.selection == new.selection:   # exact duplicate only
```

So a switch could land on any selection correlated — but not identical — to one
already held on that match, and nothing would ever look again.

---

## 3. Reproduction

Read-only production query of the three multi-pick fixtures from the last 30
days, with `review_action` and the pre-Claude snapshot (`model_selection`):

| Fixture | Model's picks | Claude's action | Result |
|---|---|---|---|
| 49308 Estrela–Sporting | `Home Over 0.5` + `Under 3.5 Goals` | CHANGE `Home Over 0.5` → **`Over 2.5 Goals`** | `Over 2.5` + `Under 3.5` |
| 48965 Levski–Kairat | `Away Over 0.5` + `Over 2.5 Goals` | CHANGE `Away Over 0.5` → **`Home Win`** | `Home Win` + `Over 2.5` |
| 48842 SJK–HJK | `Under 3.5 Goals` + `Double Chance 1X` | CHANGE `Under 3.5` → **`Over 2.5 Goals`** | `Over 2.5` + `DC 1X` |

**All three were manufactured by the review, not by the model.** In every case
the model's own output was uncorrelated and would have passed the filter
unchanged.

The decisive case is 48965: `("Home Win", "Over 2.5 Goals")` **is already in
`_CORRELATED_PAIRS`** and has been for months. It was persisted anyway. That
single row proves adding entries to the table cannot fix this, and it is why
Stage 7's proposed four-line patch would not have prevented any of the three.

Also confirmed, in the other direction: a 60-day check for the same team pairing
under different `match_id`s returned **empty** — the duplicate-fixture class is
not occurring.

---

## 4. Why it matters

Two picks on one fixture are not two independent observations. Correlated picks
on one fixture are worse than that: they are close to one observation counted
twice, and they enter the evidence at full weight.

Concretely, for a fixture emitting `Home Win` + `Over 2.5`:

| Statistic | Effect |
|---|---|
| CLV mean | pick-weighted — the fixture counts double |
| CLV bootstrap CI | **too narrow** — i.i.d. resampling assumes independence |
| ROI / Brier / log-loss | same double-weighting |
| Checkpoint counter | **inflated** — 2 toward 100/200/500 for ~1 fixture of evidence |
| Market-level aggregation | one fixture influences two market buckets |

The direction is the dangerous one: every distortion makes the evidence look
*stronger* than it is.

Measured on 180 days of production picks:

```
900 fixtures carried 1,070 picks
170 fixtures (18.9%) carried two
340 picks = 31.8% of all picks share a fixture with another pick
```

---

## 5. Affected code paths

| Path | Role | Defect |
|---|---|---|
| `finalize_picks` dedup key | exact-duplicate gate | keyed on the display name, not identity |
| `_filter_correlated_picks` | correlation gate | table missing the over↔under class |
| `MatchBriefingService._apply_decision` | Claude CHANGE | no correlation re-check at all |
| `paper_trading_report._boot` | all CIs | i.i.d. bootstrap over clustered data |
| `section_checkpoints` | 100/200/500 | counts picks with no independence measure |
| `ProbabilityCalibrator.fit_from_db` | prediction-applied artifact | no `is_paper` filter |

---

## 6. Policy chosen (Phase 2)

Explicit, minimal, and evidence-based — no invented correlation matrix.

| Class | Definition | Policy | Enforced at |
|---|---|---|---|
| **A. Exact duplicate** | same `(match_id, market, selection)` | never two picks | in-memory key + DB unique index `(match_id, selection, pick_date)` |
| **B. Same-market duplicate** | same fixture, same market, same selection | collapses into A | same |
| **C. Mutually exclusive** | `Home Win` + `Away Win` | *not* filtered — see note | — |
| **D. Overlapping** | `DC 1X` + `DC X2`, 1X2 ↔ DC/DNB | one pick only | `_CORRELATED_PAIRS` |
| **E. Cross-market correlation** | `Home Win` + `Over 2.5`, `BTTS Yes` + `Over 2.5`, ladder rungs (both directions) | one pick only | `_CORRELATED_PAIRS` |

**Note on class C.** Mutually exclusive selections on one fixture never arise:
the per-match cap keeps the top 2 by confidence, and the model cannot rate both
`Home Win` and `Away Win` above threshold on the same match. No production
instance exists in 180 days. Adding a rule for a case that cannot occur is the
"arbitrary matrix to reduce pick volume" Phase 2 warns against, so it is left
declared-but-unenforced rather than coded speculatively.

**Genuinely different markets are preserved.** `Over 2.5` + `Double Chance 1X`
on one fixture remains two picks — they are correlated in the loose sense that
everything on one match is, but not in the declared sense, and collapsing them
would discard real information. This is handled statistically (§8) rather than
by deletion.

---

## 7. Implementation

### 7a. Six cross pairs added

```python
("Over 1.5 Goals", "Under 2.5 Goals"),   ("Over 1.5 Goals", "Under 3.5 Goals"),
("Over 1.5 Goals", "Under 4.5 Goals"),   ("Over 2.5 Goals", "Under 3.5 Goals"),
("Over 2.5 Goals", "Under 4.5 Goals"),   ("Over 3.5 Goals", "Under 4.5 Goals"),
```

### 7b. One predicate, two call sites

`FootballBettingAgent.selections_are_correlated(a, b)` is now the single
authority. The pre-persist filter and the Claude CHANGE path both call it, so
the two gates cannot drift apart again.

### 7c. The CHANGE path re-checks correlation

```python
for other in picks[1:]:
    if self.agent.selections_are_correlated(other.selection, new.selection):
        primary.review_action = "KEEP"
        primary.review_reason = (
            f"CHANGE to {new.selection} rejected: correlated with "
            f"{other.selection} already held on this match")
        ...
        return True
```

**The switch is rejected; the other pick is not deleted.** The other pick is the
frozen model's own output. Dropping it to make room for a review-chosen
selection would let the review delete the evidence the experiment exists to
collect. Falling back to KEEP mirrors the existing "no real odds" branch.

### 7d. Normalized duplicate identity

`(rec.match, rec.selection)` → `(rec.match_id, rec.market, rec.selection)`.
The old key was a rendered string: two fixture rows for one game render
identically and would collapse a legitimate pick, while one fixture whose team
name changed between shards renders differently and would slip a duplicate
through. It now matches what the DB index means.

### 7e. Auditable rejection reasons

Every drop emits one greppable line:

```
PICK_REJECTED reason=duplicate_exact       match_id=… market=… selection=…
PICK_REJECTED reason=same_fixture_limit    … conf=… cap=2 already_saved=…
PICK_REJECTED reason=correlated_selection  … kept=… kept_ev=…
PICK_REJECTED reason=correlated_selection stage=claude_change … correlated_with=…
```

The Claude-stage rejection is also written durably to `review_reason` on the
row, so it survives log rotation.

---

## 8. Statistical implications (Phases 3 and 5)

### The statistical unit

Not `pick`, and not `fixture` either. The right answer is **pick as the unit of
observation, fixture as the unit of resampling** — a cluster bootstrap.

Collapsing to fixture level would discard genuinely different markets, which
Phase 5 explicitly warns against. Keeping the i.i.d. bootstrap would treat
correlated picks as independent. The cluster bootstrap does neither: every pick
contributes its own value, and fixtures are resampled with replacement so the
standard error reflects the real independence structure.

### Measured effect

On the 1,048 settled picks (878 fixtures):

```
design effect (rho = 1)  1.324      worst-case effective n = 791 of 1,048
mean ROI                 -4.03%
i.i.d.  95% CI           [-9.92%, +2.12%]   width 12.04%
cluster 95% CI           [-10.19%, +2.32%]  width 12.51%
CI widening              +3.94%
```

Two numbers, deliberately reported together. The CI widens by only ~4% because
the intra-fixture correlation of *outcomes* is well below 1 — a `Home Win` and
an `Over 2.5` on one match often disagree. The effective-n figure is the
worst-case bound (ρ = 1) and is much starker: 1,048 picks buy at most 791
independent observations.

For CLV specifically, ρ should be **higher** than for outcomes: both prices on a
fixture respond to the same information flow, so the closing movement is largely
common. Expect the CLV CI to widen more than 4% once real data exists.

### Checkpoint counting

The 100/200/500 **definition is unchanged** — valid closing lines, counted in
picks, exactly as before. What is new is that the counter prints the independence
structure beside it:

```
valid closing-line picks: 100
independent fixtures    : 84
worst-case effective n  : 76  (design effect 1.32)
```

and flags any checkpoint reached on pick count while effective n is below it as
`provisional`. Reading "100" as 100 independent observations is the specific
mistake this exists to prevent.

### Other statistics

Log-loss, Brier and win rate are means over picks and inherit the same
clustering. Their point estimates are unbiased; only their uncertainty was
overstated in precision. The bootstrap CIs on ROI and per-action breakdowns now
cluster; Brier and log-loss are reported without CIs, so nothing there overstates.

---

## 9. Model-version implications (Phase 6)

Every Stage 8 change, classified before touching the constant:

| Change | Prediction? | Selection? | Evaluation? | Version |
|---|---|---|---|---|
| Six over↔under pairs | no | **yes** | — | bump |
| Correlation re-check in CHANGE | no | **yes** | — | bump |
| Normalized dedup key | no | **yes** | — | bump |
| `selections_are_correlated` extraction | no | no (identical behaviour) | — | — |
| `PICK_REJECTED` logging | no | no | — | — |
| Same-snapshot closing rule | no | no | **yes** | none |
| Cluster bootstrap, effective n | no | no | **yes** | none |
| `fit_from_db` paper filter | *latent* | no | — | see below |

Three selection-affecting changes → **`CODE_REVISION` `s5.1` → `s5.2`**,
verified to move the identifier:

```
stage5_baseline_20260807.ac04cc  →  stage5_baseline_20260807.d1b522
```

The label and freeze date stay: the Stage 5 *model* is unchanged. Only the
population of predictions it persists has changed, which is what the fingerprint
segment is for.

The calibration-fit filter is classified as prediction-affecting-but-latent: the
artifact it writes is only applied when `models.probability_calibration_enabled`
is true, and that flag is `false`. No prediction has ever used a contaminated
map, so no version boundary is needed for it — but it would have become one
silently the moment the flag was flipped.

**No test pinned the old version string**, which is how a selection change could
previously have shipped without a bump. Invariants 6/6b/7 now cover it.

---

## 10. Paper/live isolation (Phase 7)

| Path | Can paper enter? | Expected? | Action |
|---|---|---|---|
| ROI (`get_stats`) | no | — | Stage 7 `_live_only()` |
| `rolling_backtest` | no | — | Stage 7 |
| EV threshold tuning | no | — | Stage 7 |
| Cold-streak market breakdown | no | — | Stage 7 |
| Ensemble weight learning | no | — | Stage 7 |
| Pick-outcome calibration | no | — | Stage 7 |
| Calibration drift check | no | — | Stage 7 |
| **Probability calibration fit** | **yes → no** | **no** | **fixed this stage** |
| ML model training | no | — | trains on match results, not picks |
| Gate learning | no | — | all gates disabled; registry is static |
| Claude review stats | yes | **yes** | KEEP/CHANGE rates are experiment output |
| Settlement | yes | **yes** | paper picks must settle |
| Idempotency guard | yes | **yes** | paper picks are today's picks |
| Injury fetch priority | yes | **yes** | operational ordering only |
| Fresh-DB `ml=0.0` reset | yes | harmless | one-way reset gated on `count == 0`; 1,048 live settled picks make it unreachable. Documented, not changed. |
| Closing capture / odds refresh | yes | **yes** | paper picks are the point |
| Paper trading report | yes | **yes** | it *is* the experiment report |

**The leak found:** `ProbabilityCalibrator.fit_from_db()` queried all settled
picks with no `is_paper` filter, then `save()`d the map to disk. Stage 7
filtered the drift check logged two lines below it but not the fit itself.
Dormant today only because the config flag is off — which makes it a reason to
fix, not a reason it was safe.

---

## 11. CLV integrity (Phase 8)

All Stage 5/7 validity rules verified intact and **not weakened**: match, market,
selection, decimal odds, timestamp, before kickoff, inside the window, valid
bookmaker, market structure, overround, no corrupt-source collision, and
`missing`/`late`/`invalid` recorded explicitly.

**One defect found — the same-snapshot rule.**

Stage 7's fix was a *time* rule: reject prices observed before
`kickoff − 180 min`. That does not catch an *identity* problem. A pick taken 90
minutes before kickoff is priced from an odds row that is itself inside the
closing window, so the time rule happily returns **that very row** as the
closing price. CLV would then be `taken / closing − 1` = exactly 0.00% — not a
measurement, an echo of our own price, and indistinguishable in the data from
genuine closing-line parity.

Fix: a closing observation must come from odds observed **strictly after** the
pick was created.

```python
if r.created_at is not None:
    not_before = max(not_before, r.created_at) if not_before else r.created_at
```

with the `not_before` comparison made exclusive (`ts <= not_before` → drop).
`Odds.timestamp` is refreshed on every upsert, so a book re-quoted at the same
number still counts — that is a genuine unchanged close. Only a row nobody
looked at again is excluded.

Five existing Stage 5/7 tests failed against this rule because their fixtures
let `created_at` default to "now" while seeding older odds — an order that cannot
happen in production. The fixtures were corrected to model the real sequence
(pick taken → market re-observed → capture), not the rule relaxed.

Duplicate capture, late capture, missing prices, multi-book consensus, fair
probability and overlapping-market handling were re-verified unchanged.

---

## 12. Quota impact (Phase 9)

**Odds API credits consumed by Stage 8: 0.** No live API call was made. All
work used existing database data, mocks and deterministic fixtures.

The quota policy is unchanged: 400-credit monthly budget, 50-credit safety
margin, 24-credit per-run ceiling, claim-before-spend ledger reconciled against
the provider's counter. `api_budget` still contains only the four
`api-football` rows — no `theoddsapi` row was created, confirming the read-only
status commands do not write.

---

## 13. Production safety (Phase 10)

Verified read-only at the end of the stage:

```
picks_total             1070      (unchanged)
picks_settled           1048      (unchanged)
picks_paper                0
closing_odds set           0
capture_status pending  1070
model_version set          0
max settled_at          2026-08-09 10:34:05     (predates this session)
max created_at          2026-08-09 10:45:19     (predates this session)
api_budget rows            4      (api-football only)
```

No test rows, no picks modified, no settlement data modified, no closing odds
modified, no destructive SQL, no new migration, no secrets, no API keys, no
import-time `load_dotenv()` (all four scripts load inside `_load_env()` called
from `main()`, with AST regression tests). `.env` and `config/config.yaml`
remain untracked.

---

## 14. Tests added

27 new tests in `tests/test_experiment_invariants.py`, one per invariant plus
boundary cases:

| Invariant | Tests |
|---|---|
| 1 — no double persistence | unique index holds; in-memory key is the normalized identity |
| 2 — correlated ≠ independent | over/under pair filtered; 6 declared pairs symmetric; uncorrelated pairs preserved; **CHANGE path re-checks and rejects rather than deletes** |
| 3 — paper cannot fit the model | calibrator refuses a 200-row all-paper sample; structural check that 5 learning paths carry `_live_only()` |
| 4 — stale snapshot ≠ close | 10-hour-old price rejected |
| 5 — same snapshot ≠ close | the pick's own pricing row rejected; a re-observed price still counts |
| 6 — prediction change moves version | `CODE_REVISION` feeds the fingerprint; `s5.2` pinned with rationale |
| 7 — evaluation change does not | `TRACKED_KEYS` is config-only, no reporting keys |
| 8 — correct statistical unit | checkpoints expose effective n; cluster bootstrap wider than i.i.d. on correlated pairs; singleton clusters behave like i.i.d.; checkpoint definition unaltered |
| 9 — no historical contamination | unversioned rows excluded by `model_version` scoping |
| 10 — paper never reaches live ROI | +100% live vs all-losing paper; paper mode pinned on |

Plus 5 corrected fixtures in `tests/test_closing_capture.py`.

---

## 15. Before / after test count

```
Stage 7 :  540 passed
Stage 8 :  566 passed        (+27 invariants, −1 placeholder)
```

Zero failures, zero skips. Runtime 127 s.

---

## 16. Remaining risks

### 16a. STOP — Claude CHANGE bypasses the value gates. Design decision required.

This is the Phase-13 stopping rule. **Not implemented; recommendation only.**

`build_selection_pick` is documented as *"Bypasses value/confidence gates"* — it
is deliberate. But it means Claude can persist a pick the frozen model's own
`min_expected_value` gate would have refused. Measured over 90 days:

| `review_action` | n | mean EV | negative EV | EV < −5% | worst |
|---|---:|---:|---:|---:|---:|
| (none) | 209 | +13.23% | 23 | 12 | −19.7% |
| KEEP | 134 | +1.49% | 59 | 23 | −24.3% |
| **CHANGE** | **96** | **−6.84%** | **70 (73%)** | **53** | **−33.2%** |

**21.9% of picks are Claude-selected, and 73% of those carry negative expected
value at the price taken.**

This collides with Phase-1 objective 1 — *"every paper pick is genuinely
generated by the frozen model"* — and with the experiment's own question:

> Does the frozen Stage 5 model generate information that moves prices in the
> right direction before the market closes?

Closing capture reads `SavedPick.selection`, which is Claude's final pick. As it
stands, **CLV would measure the review, not the model.**

**Options:**

1. **Measure the model, keep Claude's picks as a second series.** `model_market`
   / `model_selection` / `model_probability` already snapshot the pre-Claude
   pick (Stage 4, commit `4957e0a`). Capture closing lines for *both* and report
   two CLV series. Cost: closing capture must resolve two selections per row;
   some model selections are in markets with no obtainable close.
   **Recommended** — it answers the stated question and still measures the
   review's added value, which is the reason those columns exist.
2. **Disable the Claude review for the duration of the experiment**
   (`briefings.finalize_picks: false`). Cleanest measurement of the frozen
   model, and free. But it changes what the system does day to day, and the
   review's KEEP/CHANGE record so far (CHANGE ROI +9.5% vs KEEP −12.0%, both
   with CIs spanning zero on n=86/122) is not evidence it should be removed.
3. **Apply the EV gate to CHANGE picks.** Narrowest change, keeps the review,
   removes the negative-EV tail. But it silently alters what the review is
   allowed to do, and the review was built to override the gate on purpose.
4. **Accept and document.** The experiment then measures "model + Claude" as one
   system. Defensible, but it cannot answer the frozen-model question, and §12
   of the Stage 7 plan is written around that question.

**What changes depending on the decision:** option 1 needs a capture change and
a report section, no model-version bump (evaluation-only). Options 2 and 3 are
selection-affecting and need `s5.3`. Option 4 needs a one-line scope change in
the experiment definition. **All four are cheap today and expensive after
observations accumulate**, which is the same timing argument that made this
stage worth doing now.

### 16b. Bounded, non-blocking

* **~36% of picks are structurally outside CLV measurement** (team goals, BTTS,
  double chance — the pre-kickoff refresh covers only `h2h`/`totals`). Carried
  forward from Stage 7 §7b, unchanged.
* **The same-snapshot rule will reduce coverage** below Stage 7's 88% estimate,
  by exactly the fraction of picks whose books are never re-quoted between the
  pick and kickoff. That number is unknown until data exists. It is a coverage
  cost, not a bias — the rejected rows were never evidence.
* **Mutually exclusive selections (class C) are unenforced.** Cannot occur under
  the current cap; would need a rule if `max_picks_per_match` ever rises.
* **ρ is not yet measurable.** The effective-n figure is a worst-case bound
  until enough clustered CLV observations exist to estimate the real
  intra-fixture correlation.

---

## 17. Is the experiment ready to continue collecting data?

Yes for correlation integrity, isolation, CLV integrity and quota protection —
all four are now correct and test-covered.

But §16a is a live question about **what the collected data will mean**, and it
is a design decision, not a correctness fix. Collecting 100 observations before
answering it risks producing a clean, well-measured CLV series for the wrong
subject.

---

## Appendix — what did not change

No new ML algorithm. No Elo, Poisson, half-life, Dixon-Coles rho or blend-weight
tuning. No features added or removed. No EV or confidence thresholds changed. No
gates re-enabled. Nothing optimized against the 1,070 historical picks. No new
data source, no Understat, no change to the Odds API market selection or request
frequency. The CLV formula is untouched — the change was to *which observations
qualify*, not how CLV is computed. Checkpoint definitions unaltered. Paper
trading remains on; real money remains off.

The model's probabilities are bit-identical. What changed is which of its picks
survive to be persisted, and how honestly the resulting evidence is counted.
