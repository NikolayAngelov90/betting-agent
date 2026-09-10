# UNADDRESSABLE BY SELF-OBSERVATION

**A first-class verdict, alongside `UNTESTABLE` and `FIX DEPLOYED — UNDEMONSTRATED`.**

> ## A system running a single fixed policy produces no variance along the axis that policy sets. Questions about the policy are therefore unanswerable from the policy's own output.

**Named 2026-09-10, after three questions in this project died the same death and
the third was still being treated as a sample-size problem.**

---

## The definition, and what separates it from `UNTESTABLE`

| verdict | what is missing | does more data help? | remedy |
| --- | --- | --- | --- |
| **`UNTESTABLE`** | **the quantity was never recorded** — H3's injury history (34 rows, two-day snapshot), H4's opening timestamp | no — but **retaining it from now on** does | a schema or retention change |
| **`UNADDRESSABLE BY SELF-OBSERVATION`** | **variance along the axis in question** — the data is abundant and sits at one setting | **no, and more data actively misleads**: it tightens an interval around a single point on a curve nobody can see | **a deliberate variation, which always has a price** |

**The trap is that this class LOOKS like an under-powered sample.** The n is
often large and growing. Every instance below was, at some point, filed as
"needs more data" — and in every case waiting would have produced more
observations *at the same setting* and never one observation at a different one.

> ### It is the difference between "we need more" and "we need something different."
>
> **This project has now spent effort on the first when the answer was the
> second, three times.**

**The tell, stated so the next instance is caught before the effort:**

> **If the population you are measuring was SELECTED by the policy you are
> asking about, the measurement cannot answer the question — the policy is in
> both the question and the sampling frame.**

---

## The instances

### 1. Stage 21 — the lead-time benefit  → open, remedy priced but not taken

**Claim:** *"a longer lead leaves more room for the price to move before the
close, which is exactly what CLV measures."*

**Why it cannot be observed:** H5's lead-time strata, measured 2026-09-10 —

| bucket | observations | **fixtures** |
| --- | --- | --- |
| <6h | 42 | **2** |
| 6–12h | 2,573 | **127** |
| >12h | 0 | **0** |

**127 of 129 fixtures sit in one bucket.** `0 3 * * *` fires at a fixed hour, so
the lead is near-constant by construction. **There is no contrast.** Waiting a
year produces thousands more fixtures, all at 6–12h.

**Disposition: (c) — left open, and the claim stays uncitable as a benefit.**
The delay-tolerance argument stands on its own and remains the reason to keep
03:00. Recorded in `docs/stage21-schedule-prediction.md`.

### 2. H5 Q1 — the picked-versus-unpicked control  → open, remedy priced at credits

**The registration asserted the control existed:** *"Snapshots are written for
every fixture in a refreshed league, not only picked ones, so the control
exists without new spend."*

**Why it cannot be observed:** `refresh_and_capture.py` passes
`require_pending_pick=True` by default, so **a fixture receives its second
pre-kickoff price BECAUSE it carries a pending pick.** Of 129 fixtures holding
the two-point observation H5 requires, **129 were picked.** Of the 33
never-picked fixtures holding any snapshot, **30 hold exactly one** — and one
observation is not a drift measurement.

> **The policy generates the population, so the population cannot control for
> the policy.** This is the purest instance of the class, and it is the one
> where the assumption was written into a pre-registration as a fact.

**Disposition: (a) available, not taken.** `--any-fixture` already flips the
gate. It competes with H1's purchase under the same TheOddsAPI credit ceiling.

### 3. The Saturday schedule-margin boundary  → CLOSED, by substitution

**The checkpoint:** does the 03:00 cron's margin hold when a run starts late
enough to threaten the earliest kickoff?

**Why it cannot be observed:** the boundary is only exercised by a start between
**09:40 and the earliest kickoff**. A 03:00 cron plus the observed 4–5h delay
**lands before 09:40 every time.** Observed starts never enter the window. Two
consecutive Saturdays recorded "MARGIN HELD" while testing nothing.

**Disposition: (b) — substituted.** Closed on the **delay distribution observed
since 2026-08-28**, with the regime named, rather than by staging a manual
trigger inside the window. **A manual trigger would have been arithmetic, not an
experiment**: it would have confirmed a subtraction already known, at the cost
of a real run. OPS-3 stays open for the tail.

### 4. H1 — momentum  → RECLASSIFIED here from `UNTESTABLE`

**Filed in Part B as** *"`UNTESTABLE` — needs t0→t1 to predict t1→t2; only two
observations per key exist."*

**That is the right verdict under the wrong heading.** The third observation is
not missing because nobody thought to store it — `odds_snapshots` is append-only
and would keep it. **It is missing because the refresh policy permits one
refresh per window.** Same cause as instances 1–3: the collection policy sets
the axis, so the axis has no variance.

**Disposition: (a), priced.** See below — and the sizing does **not** transfer
from H5.

---

## The three dispositions

**A verdict is only useful if it tells you what to do next. This one has exactly
three exits, and naming them is the point of the class.**

| | disposition | when it applies | cost |
| --- | --- | --- | --- |
| **(a)** | **buy the variation** — deliberately operate off-policy to create the contrast | the question bears on a decision worth the price | credits, and a policy that is knowingly suboptimal while the variation runs |
| **(b)** | **substitute an answerable question** that bears on the same decision | a different measurable quantity constrains the decision adequately | usually free; **requires stating what was substituted and why**, or it reads as a pass |
| **(c)** | **leave it open and stop citing the claim** | the decision does not turn on it | free, and the discipline is that the unmeasured claim may not be used as an argument in the meantime |

> **(c) is not "unresolved". It is a ruling: the claim is retired from
> circulation until someone pays for (a).** Stage 21's lead-time benefit sat in
> the rationale for a cron change for six days before it was marked; that is the
> failure this disposition prevents.

---

## What it implies for H1 — the purchase got cheaper and the sizing did not transfer

**σ fell from 9.39% (n=15) to 5.937% (n=129), so H5's re-derivation drops H1's
figure from n=137 to n=55.** That is a real reduction and it makes the purchase
cheaper.

> ### But n=55 was computed for a MEAN, and H1 is a PREDICTIVE RELATIONSHIP.

`n = 6.185 · σ² / δ²` sizes a test that a mean movement differs from zero by δ.
**H1 asks whether t0→t1 predicts t1→t2** — a correlation, not a location shift.
Its sizing is a different derivation entirely, and it is **acutely sensitive to
an effect size H1 has never specified**:

    n ≈ ((z_a + z_b) / z_r)² + 3,   z_r = ½·ln((1+ρ)/(1−ρ)),  z_a+z_b = 2.487

| ρ (assumed) | required n |
| --- | --- |
| 0.30 | **68** |
| 0.20 | **154** |
| 0.10 | **618** |

**Shown as sensitivity, not as a registration.** No ρ is adopted here; H1's own
registration must fix one **before** any purchase, exactly as H5's fixed its
bands. **Carrying n=55 into H1 would be importing a number derived for a
different estimator** — and the reason this project computes things twice.

**And the purchase is itself disposition (a):** three separated observations do
not exist because the window/interval policy permits one. **~100 credits buys
the variation**, not merely more of the same data. It competes directly with
H5 Q1's `--any-fixture` control for the same ceiling, and the credit gate's own
deadline is end of September 2026.

---

## Filed instances

| # | question | status | disposition |
| --- | --- | --- | --- |
| 1 | Stage 21 lead-time benefit | **UNADDRESSABLE BY SELF-OBSERVATION** | (c) open, claim uncitable |
| 2 | H5 Q1 picked-vs-unpicked control | **UNADDRESSABLE BY SELF-OBSERVATION** | (a) available, ~credits, not taken |
| 3 | Saturday schedule-margin boundary | **CLOSED** by substitution | (b) delay distribution, regime named |
| 4 | H1 momentum | **UNADDRESSABLE BY SELF-OBSERVATION** *(was `UNTESTABLE`)* | (a) ~100 credits, sizing to be re-derived |

*Named 2026-09-10. Add instances here rather than rediscovering the class.*
