# Stage 18 — Stop Discarding the Data

Stage 17 is closed: `STAGE 17 — NO SIGNAL, ROUTE IDENTIFIED`. Three of its four hypotheses were untestable — not disproved, untestable — because this system fetches price and injury data, uses it once, and overwrites it. Price momentum, the most-documented effect in this field, cannot be tested here at all: `odds` is unique on `(match_id, bookmaker, market_type, selection)` and overwritten in place, so there are exactly two observations per key and never a third.

This stage builds the substrate that makes those questions answerable. It does not answer them.

**Expectations, stated up front so nobody mistakes this for optimism.** Stage 17's own counterweight applies: the one effect that could be measured was real, replicated, and came to less than half of break-even. Expect what lies down this route to be small, real and insufficient too. This stage is justified on three grounds that do not depend on finding a signal:

1. the system already pays for this data and throws it away — retaining it costs **zero API credits**
2. the API-Football two-way trap is a live data defect that would poison any movement study built on the current table
3. it closes the last open question, so the project ends with everything tested rather than with one thing unknown

---

## 0. Scope

**In scope:**

* timestamped price observations instead of overwrite-in-place
* an explicit first-seen timestamp for opening prices
* retained injury history
* fixing the API-Football two-way / draw-excluded contamination in the `odds` table
* storage sizing and retention, because this changes the growth profile of a free-tier database

**Out of scope — record and do not pursue:**

* the review-path findings from the 2026-08-25 audit: 35 picks with no review decision on 08-15/08-16, and the two CHANGE decisions recorded but never applied on 08-14 and 08-17
* the June injury-fetch collapse
* the unexplained thin cards on 08-23 and 08-25
* MASK-1's three mechanism tests, invariants 2 and 3, MASK-3, L6a, L6b, the June briefing line
* the predictive core. No models, weights, blend, calibration, thresholds. No new features.
* `paper_trading_mode` stays `true`.

**Cohort discipline.** Storing more data must not change any prediction. If a change to what is stored alters what a model reads, stop and report — that is a cohort event and it does not belong here.

---

## PART A — Size it before you build it

This changes a free-tier database's growth profile. Establish the numbers first.

### A1. What the current table costs

`odds` currently holds ~317,657 rows under overwrite semantics. Report its actual size on disk, the row growth rate per month, and how the 400-day pruning interacts with the preservation rule for pick-bearing matches.

### A2. What snapshots would cost

Snapshot volume is bounded by **refresh volume, not by fixture count** — the pre-kickoff refresh touches only leagues with an imminent fixture and a pending pick, and pick-time pricing touches only leagues with analysable fixtures. Derive the growth from the measured request volume in the ledger, not from the number of matches.

Report:

* rows per month under snapshot semantics, with the derivation shown
* projected size at 6 and 12 months
* what fraction of Supabase's free-tier storage that consumes
* the egress consequence — Stage 3 cut egress ~97% by column projection, and a taller table changes what those queries read. If any existing query would now scan materially more, name it.

### A3. Decide retention deliberately

Full history is not required to test momentum. What is required is **enough observations per key to see a trajectory** — three or more points, spaced meaningfully.

Propose a retention policy and justify it against the research question rather than against instinct: how many observations per key, over what window, at what spacing, are needed to test whether movement from t0→t1 predicts movement from t1→t2. Then keep that and prune the rest.

If the honest answer is that full retention fits comfortably in the free tier, say so and keep it simple.

---

## PART B — The API-Football two-way contamination

This is a live defect and it must be fixed **before** any history accumulates, or the accumulated history inherits it.

### B1. What is established

Stage 17 measured it and it splits cleanly by provider:

| source | median movement | shortened : drifted |
| --- | --- | --- |
| API-Football (10 books) | **+37.3%** | 2,647 : 62 |
| The Odds API (26 books) | −0.61% | 661 : 757 |

The documented two-way / draw-excluded trap overwrites genuine 1X2 opening prices with shorter two-way prices, manufacturing a fake 37% shortening. It has been gated for the CLV series since Stage 13 and was **never excluded from the `odds` table**.

### B2. What to establish before fixing

* how many rows in the current table are affected, and over what period
* whether the contamination is detectable per row after the fact — an overround check should identify a two-way price written into a three-way slot
* whether any currently-pending observation or saved pick was priced from a contaminated row

### B3. The fix

Refuse at the write, not at the read. A two-way price must not be written into a three-way market slot at all — the same fail-closed-on-contradiction shape as the Stage 13 identity gate, and the same principle: the gate that already exists for the CLV series should exist once, at the source, rather than being re-implemented per consumer.

**That is THE HABIT again and this is the fourth data-layer instance.** Stage 15 found two market taxonomies, Stage 16 found `odds.selection` carrying both `Home` and `Home Win`, the audit found `disposition` mistaken for `review_action`. Consolidate rather than adding a second check.

### B4. Historical rows

Do not delete them. Mark them, the way the 29 contaminated matches were marked — an explicit reason column or equivalent, so a movement study can exclude them and a future repair can reverse it. Deletion destroys the evidence of the defect and this project has decided that question twice already.

State whether marking them is prediction-affecting. If any model or feature currently reads those rows, it is, and that stops this stage.

---

## PART C — Timestamped observations

### C1. What to store

For each price observation: the key that exists today, plus **when it was observed**, plus enough to reconstruct a trajectory. Design it so that the existing unique constraint's purpose — one current price per `(match, book, market, selection)` — is preserved for every consumer that needs "the current price", while history accumulates behind it.

The same-snapshot rule that CLV depends on becomes simpler under this design, not harder. Confirm that explicitly: a closing observation must come from a price observed strictly after `taken_at`, and with real timestamps that stops being an inference about row identity.

### C2. The opening timestamp

`opening_odds` is frozen at first sight by this system, which is not the market's open, and there is no record of when first sight occurred. Stage 17 established that `created_at` cannot proxy for it: **53.5% of match rows were created after their own kickoff**, mean +14 days, because they are backfill stamps.

Store an explicit first-seen timestamp. Without it, "how long before kickoff was this price taken" remains unanswerable, and that is H4.

### C3. Injury history

34 injury rows exist in the entire database, all dated 2026-08-17/18, while the audit shows runs fetching 128–198 injuries in March–May. The system fetches injuries daily and retains only current status.

Retain the history: what was known, and when it was known. The "when" is the whole point — an injury that moves a line moves it when the news arrives, and a current-status snapshot cannot distinguish a two-week-old absence from this morning's news.

Note that Stage 14 established injuries reach only the Claude review prompt and never the model. Retaining history changes no prediction, which keeps this cohort-neutral. Confirm that rather than assuming it.

---

## PART D — Prove it works before you trust it

Stage 15's rule applies directly: **a lever's null result must be distinguishable from its success.** L2 shipped inert and would have measured as "implemented, 0 saved" — indistinguishable from working.

So before declaring:

* verify snapshots actually accumulate — after one real day, report observations per key with their timestamps, and confirm at least one key has three
* verify the first-seen timestamp is populated on new rows and distinguishable from `created_at`
* verify injury history accumulates rather than overwriting
* verify the two-way refusal fires, by replaying real historical rows that should have been refused, and that it refuses nothing legitimate — the same 20-of-20 replay standard the identity gate met
* verify that no existing query reads more than it did, or name what does and by how much

A storage change that silently does nothing looks exactly like one that works.

---

## E. What this stage does not do

It does not test H1, H3 or H4. There is no history yet to test them on.

State in the report **when** the substrate will hold enough data to test each — derived from the retention design and the measured observation rate, not estimated. That date is the trigger for Stage 19, and Stage 19 is a research stage, not an engineering one.

---

## F. Hard rules

1. **Report at each part boundary.** A, B, C, D are four reports.
2. **No number without provenance** — `measured`, `simulated`, `assumed` — with a date.
3. **A single anomalous result is evidence about the measurement first.**
4. **One definition, and a guard.** Fourth data-layer instance of THE HABIT is in this stage's scope; do not add a fifth.
5. **Cohort-neutral or stop.** If storing more changes what any model reads, that is a cohort event and it halts this stage for a decision.
6. **No history rewriting.** Contaminated rows are marked, never deleted.
7. **Nothing from the deferred list.** If something surfaces, it goes in the ledger and is not pursued.
8. If any invariant in `tests/test_experiment_invariants.py` fails: **STOP, do not fix, report.** Invariants 2 and 3 are known defective; the declaration must not cite the invariant count.

Declare:

`STAGE 18 — SUBSTRATE BUILT` with the date each of H1, H3 and H4 becomes testable,

or `STAGE 18 — SUBSTRATE NOT AFFORDABLE` if Part A shows the free tier cannot hold what the research requires — in which case say what it would take, and let that be the honest end of the route.
