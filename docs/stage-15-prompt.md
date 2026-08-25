# Stage 15 — The Coverage/Credit Frontier

Stage 14 is closed: `STAGE 14 — INSTRUMENT REBUILT; D1 PREMISE FALSIFIED`. Production runs `s5.3` / `stage5_baseline_20260807.098437`, paper mode on, API-Football suspended (OPS-1 open).

Stage 14 established that the experiment **is** measuring — 57 valid CLV pairs, both series with intervals, 63% of what the refresh can price — and that the binding constraint is no longer code. It is credits.

This stage answers one question and does almost nothing else:

> **What does each additional valid closing observation cost, and which gains are cheapest?**

Every lever available — an earlier first run, wider markets, a different schedule, removing wasted spend — has a price in credits and a yield in observations. Nobody has priced any of them. Until they are priced, any change to the capture schedule is a guess dressed as an improvement, and this project has a documented history of those.

---

## 0. Scope

**In scope:** measurement, arithmetic, and one narrow class of implementation (see Part D).

**Out of scope:**

* the predictive core — no models, weights, blend, calibration, thresholds
* any change that trades credits for coverage. Those are priced here and **decided by Niki**, not implemented here.
* ST14-1 matcher consolidation, the injected-statistic question, MASK-1's three mechanism tests, invariants 2 and 3, MASK-3, the June briefing line, the 334-run backlog. All remain closed.
* `paper_trading_mode` stays `true`. The 8 `edge` gates stay off.

**Cohort discipline.** Nothing in this stage should move `model_version` or `CODE_REVISION`. `odds_api.*` is outside `TRACKED_KEYS` — verified in Stage 14. If any change turns out to be prediction- or selection-affecting, stop and report.

---

## 1. Open with the instrument you built

Run `.claude/commands/daily-ci-audit.md` over every run since Stage 14's last audited entry. Append verdicts to the ledger.

This is the first stage that begins with the audit rather than performing one by hand. If it finds something material, report it before continuing — but do not pursue it unless it changes this stage's arithmetic.

Confirm while you are there: is API-Football still suspended? OPS-1's boundary needs its closing timestamp the moment it is restored, and §2's ranking half and §7's identity gate unblock together when it is.

---

## PART A — Establish the arithmetic, then state the assumptions

No changes. Numbers only.

### A1. What does the 500 checkpoint actually require?

The README says real money stays off until **500 valid closing lines** exist and mean CLV is reliably positive. It also says MODEL and FINAL are counted separately and never merged.

So state precisely, from the code rather than the prose: **is the checkpoint 500 per series, or 500 combined?** The answer changes the completion date by a factor of two and nobody has written it down.

If the code does not answer it unambiguously, say so — an undefined decision threshold is a finding, not a detail.

### A2. The current rate is not the observed rate

The 57 observations came from 2026-08-14 → 08-22, a window **before** `s5.3` reduced the per-match cap from 2 picks to 1. Fewer picks means fewer observations, so the historical rate overstates the forward rate by whatever fraction of picks the cap removes.

Establish the forward rate properly:

* observations per day under `s5.3`, from `s5.3` runs only
* if that sample is too small to be meaningful — it will be — say so, and give the historical rate adjusted by the measured effect of the cap, with the adjustment stated as an assumption rather than folded silently into a number

Also note the confound: the `s5.3` window opened inside OPS-1, so its cards are thin for a reason unrelated to the cap. Separate the two effects or declare that you cannot.

### A3. Project the completion date, three ways

Given A1 and A2, and the credit ceiling:

| scenario | assumption | observations/month | 500 reached |
| --- | --- | --- | --- |
| status quo | 450 budget, current coverage, September exhausts ~27th | | |
| no credit ceiling | coverage unchanged, capture runs every scheduled window | | |
| full coverage | every priceable pick captured, no `late`, no exhaustion | | |

The gap between rows 1 and 2 is what the credit constraint costs. The gap between 2 and 3 is what coverage costs. Those two numbers are the whole point of this stage.

State every assumption as a labelled assumption. A projection whose inputs are not separable from its output is not checkable.

---

## PART B — Price each lever

### B0. First, establish whether the simulator can be trusted

`scripts/simulate_odds_quota.py` is the natural tool for this and Stage 14 just discredited its output: the README's `212 credits/month` came from it and reality is ~450.

So before pricing anything with it: **replay August through the simulator and compare against the ledger's actual 349 credits over 24 days.** Report the discrepancy and its cause.

* if it reconciles, the simulator is usable and say why the README's figure was wrong anyway (different rules, different period, different pick population)
* if it does not, **do not price the levers with it.** Price them from the ledger and the logs directly, and record the simulator as unusable until repaired.

This is the positive-control discipline applied to a tool instead of a mutation. A simulator that cannot reproduce a known month cannot price an unknown one.

### B1. The levers

For each, report **credits/month cost**, **additional valid observations/month**, and **credits per additional observation**. Where a number cannot be measured, say so rather than estimating.

**Lever 1 — an earlier first run.** All 16 `late` observations kick off at 11:00 or 11:30 UTC; the first capture run is ~11:17 (`17 11,13,15,17,19,21,23`). An earlier window catches them. Cost depends entirely on how many leagues have an imminent fixture *and* a pending pick at that hour — which may be very few, making this nearly free, or many, making it expensive. Measure it; do not assume either.

**Lever 2 — remove wasted spend.** 17 runs returned `no_rows` while claiming credits, on competitions the provider does not cover (established: 62 of 65 missing in the 08-10 → 08-13 window were UEFA). A cache of uncovered competitions spends nothing and loses nothing. Price the saving. **This is the only lever that can have a negative cost**, and if it does it belongs in Part D.

It must expire rather than being permanent — provider coverage changes, and a permanent exclusion would silently outlive the gap that justified it.

**Lever 3 — widen the markets.** 78 picks sit structurally outside the `h2h` + `totals` refresh — Team Goals, BTTS, Double Chance. Credits are `requests × regions × markets`, so each added market multiplies the whole bill. Price it exactly, and price it per market rather than as a block: some of those 78 may concentrate in one market that is cheap to add.

**Lever 4 — schedule reallocation.** The ledger shows many runs doing nothing at all ("no pending picks kick off in the next 120 minutes"), which cost no credits. If windows can be moved rather than added, coverage may rise at zero marginal cost. Establish whether the current cron's shape matches the actual distribution of kickoff times across the season, or whether it was set once and never revisited.

### B2. Rank them

One table, sorted by credits per additional observation, with the levers that cannot be measured listed separately and honestly.

Then answer the question that matters: **which combination of levers fits inside 500 credits/month, and what completion date does it produce?**

---

## PART C — Recommend, do not implement

Produce a recommendation with the arithmetic attached, in a form Niki can decide from:

* the cheapest lever and what it costs
* the most expensive lever worth considering and why it might still be worth it
* what the free tier's 500 credits actually buy at the frontier
* whether the 500-observation checkpoint is reachable this year under any combination, and if not, what would have to change

If the honest answer is that the free tier cannot reach the checkpoint in a reasonable time, **say that plainly.** That is a legitimate finding and it belongs in front of Niki rather than being softened into a plan.

Do not implement anything from this part.

---

## PART D — The only implementation permitted

Levers with **strictly non-negative coverage impact and strictly negative credit cost** — pure waste removal. On current evidence that is Lever 2 and nothing else.

If it prices out that way, implement it:

* with an expiry, not a permanent exclusion
* with the refusal logged, so the daily audit can see how often it fires
* with a test that fails if the cache never expires

Anything that trades credits for coverage waits for Niki's decision.

---

## E. Hard rules

1. **Report at each part boundary.** A, B, C, D are four reports, not one.
2. **No estimate presented as a measurement.** Stage 14 ended by relabelling `212 credits/month` as SIMULATED. Do not create the next one. Every number in this stage carries its provenance: measured, simulated, or assumed.
3. **No fix without a reproduction**, and no lever priced without its inputs shown.
4. **A single anomalous result is evidence about the measurement first.** Three instances in Stage 14 earned this rule; it applies to your own tooling, including the simulator.
5. **One definition, and a guard.** Any new predicate removes a duplicate rather than adding a parallel path.
6. **No history rewriting**, no restamping, no backfill.
7. If any invariant in `tests/test_experiment_invariants.py` fails: **STOP, do not fix, report.** Note that invariants 2 and 3 are known defective — a pass from either means nothing, and the declaration must not cite the invariant count as evidence.

Declare when A, B and C are complete and D is either implemented or shown not to qualify:

`STAGE 15 — FRONTIER MEASURED`

If Part B0 establishes the simulator cannot reproduce August, and the levers cannot be priced from the ledger either:

`STAGE 15 — FRONTIER UNMEASURABLE` — with what would be needed to make it measurable.
