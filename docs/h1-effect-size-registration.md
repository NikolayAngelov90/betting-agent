# H1 EFFECT SIZE — argued from economics, written BEFORE any variance was measured

## What "actionable momentum" means

H1 asks whether pre-kickoff price drift predicts further drift. The only way
that is worth anything is if acting on it gets a BETTER TAKEN PRICE than the
naive policy of taking the price when the pick is made.

So the effect is measured in the same unit as CLV: **percentage improvement in
the taken price**, per fixture.

## The threshold is set by what must be crossed, not by what is convenient

An edge is actionable only if it clears the cost of acting. That cost is the
bookmaker's margin — the overround — which is what Stage 16 already priced:

| threshold | meaning |
| --- | --- |
| **+1.85%** | best-line break-even. Below this, no action is profitable. |
| **+2%** | minimum decision-relevant effect |
| **+4%** | comfortable — would change behaviour without argument |

**These are inherited, not re-derived.** They are properties of the market this
project bets into, and momentum must clear the same bar CLV had to.

## The effect size to power for: +2%

**Below +1.85% the signal cannot pay for itself, so an experiment powered to
detect less than that would be measuring something unactionable precisely.**
+2% is the smallest effect worth the credits, and it is the figure Stage 16
used for the same reason.

## The test and its form

One-sided, α = 0.05, power = 80% — Stage 16's convention.

    n = (z_alpha + z_beta)^2 * sigma^2 / delta^2
      = (1.645 + 0.842)^2 * sigma^2 / delta^2
      = 6.185 * sigma^2 / delta^2

with delta = 2%.

**sigma is the fixture-level standard deviation of pre-kickoff price movement,
and it has NOT been looked at yet.** It is measurable from the 1,379 keys that
already carry two pre-kickoff observations.

## Registered before measuring

**If sigma is 3.3% — the value implied by Stage 16's own n=17 at delta=2% —
then n = 6.185 * 10.9 / 4 = 17.** Anything materially larger than that moves the
required n above 50 and the purchase must be resized or refused.

**Clustering: sigma must be computed at the FIXTURE level, not per key.** ~30
keys share a fixture and are not independent; using key-level spread would
understate n by the design effect, which is the error Stage 16 exists to
prevent.

*Written 2026-09-03, before the variance query was run.*

---

# SIGMA MEASURED — 2026-09-10. The required n falls from 137 to 55.

**σ was registered as "NOT looked at yet". It has now been measured as part of
H5's run, at fixture level as this registration required.**

| σ | source | n = 6.185 · σ² / δ², δ = 2% |
| --- | --- | --- |
| 3.3% | Stage 16's implied value, quoted above as the favourable case | 17 |
| **9.39%** | H5's registration, on **n=15** fixtures | **137** |
| **5.937%** | **measured 2026-09-10 on n=129 fixtures** | **55** |

**The purchase resizes rather than being refused.** This registration set the
rule in advance — *"anything materially larger than 3.3% moves the required n
above 50 and the purchase must be resized or refused."* **55 > 50, so the
resize condition holds**, but the figure H1 has been carrying (137) was inflated
by a σ taken from fifteen fixtures.

**The clustering requirement was honoured**: σ is the standard deviation of the
**fixture-level** median move (one value per fixture), not per key. Key-level
spread across ~30 keys per fixture is the understatement this registration
exists to prevent, and H5's Q2 measured the design effect on the same data at
**~11** — which is what that error would have cost.

*Measured 2026-09-10 during the H5 run. See `docs/h5-drift-preregistration.md`.*

## THE n=55 DOES NOT TRANSFER TO H1's ACTUAL QUESTION — 2026-09-10

**`n = 6.185 · sigma^2 / delta^2` sizes a test that a MEAN differs from zero by
delta. H1 asks whether t0->t1 PREDICTS t1->t2 — a correlation, not a location
shift.** The two are sized by different formulae, and H1's is acutely sensitive
to an effect size this registration has never fixed:

    n ~= ((z_a + z_b) / z_r)^2 + 3,   z_r = 0.5*ln((1+rho)/(1-rho)),  z_a+z_b = 2.487

| rho | required n |
| --- | --- |
| 0.30 | **68** |
| 0.20 | **154** |
| 0.10 | **618** |

**Shown as sensitivity, NOT as a registration. No rho is adopted here** — it
must be fixed in advance, exactly as H5 fixed its outcome bands, and before any
purchase. **Carrying 55 into H1 would import a number derived for a different
estimator.**

**And the purchase is a deliberate variation, not more data.** Three separated
observations do not exist because the window/interval policy permits one — H1 is
filed under **`UNADDRESSABLE BY SELF-OBSERVATION`**, disposition (a), ~100
credits. It competes with H5 Q1's `--any-fixture` control for the same ceiling.
See `docs/unaddressable-by-self-observation.md`.

---

# ρ DERIVED AND REGISTERED — 2026-09-16, before the purchase

**Argued from what must be cleared, exactly as δ = +2% was, and NOT fitted from
data that would then be the test's own.** The sensitivity table above stays as
sensitivity; this fixes the number.

## The policy has to be fixed first, because ρ is not a property of the market alone

A correlation is only worth detecting if ACTING on it pays, and what "acting"
means decides how much of ρ is captured. For a signal `z = (t0->t1 move) / σ`,
predicted `t1->t2 = ρ·σ·z`, and the expected captured gain is `ρ·σ·E[z|acted]`.

| policy | `E[z | acted]` | ρ needed for +2% | **n** |
| --- | --- | --- | --- |
| **two-sided timing on every pick** | `E|z| = √(2/π) = 0.798` | **0.422** | **33** |
| top-decile selection | 1.755 | 0.192 | 167 |

> **A five-fold swing in n from the policy alone.** Registering a ρ without
> fixing the policy would have been registering nothing.

## THE POLICY IS FORCED HERE, so ρ follows

**This pipeline does not choose fixtures by drift.** Picks are already selected
on EV; the drift signal can only change **WHEN the price is taken**, never
**WHICH** fixture is bet. Every pick therefore gets a timing decision and the
signal's sign chooses the direction — a negative signal means take it now, a
positive one means wait. **That is two-sided timing on every pick, and it is the
only policy available**, so `E[z|acted] = E|z| = 0.798`.

## REGISTERED

| | |
| --- | --- |
| σ (fixture-level, measured 2026-09-10, n=129) | **5.937%** |
| policy | **two-sided timing on every pick** |
| δ (actionable, inherited from the overround) | **+2%** |
| **ρ** | **0.42** |
| **required n** | **33** |

`z_r = ½·ln((1+ρ)/(1-ρ)) = 0.448`, `n = (2.487/z_r)² + 3 = 33`.

**Why 0.42 and not something smaller:** below it, acting on the signal returns
less than the overround, so the correlation is real and worthless — the same
argument that set δ. **An experiment powered for ρ = 0.20 would spend 153
observations to detect something it could not act on.**

## What this changes about the purchase

**n = 33, not 55 and not 137.** 55 was computed for a mean and does not
transfer; 137 came from a σ taken from fifteen fixtures.

> **33 < 50, so the registration's own resize-or-refuse condition does NOT
> fire.** The purchase gets smaller rather than being re-argued, and it is
> sized **before** the 10-01 reset rather than after.

**Sensitivity, stated so the number is not read as precision:** at the
break-even +1.85% the requirement is ρ = 0.391 and n = 39, so the whole
actionable band is **n = 33-39**. The estimate is a LOWER BOUND on ρ in one
further respect — it assumes acting captures the full predicted move, and any
slippage raises the ρ needed and lowers n.

## WHY n = 33 IS TRUSTWORTHY RATHER THAN CONVENIENT

**A smaller n derived after the fact is exactly what a motivated analysis
produces, so the reason the policy is what it is matters more than the number.**

> **The policy is a CONSTRAINT THE SYSTEM IMPOSES, not a choice made to shrink
> the figure.** The pipeline selects picks on EV. A drift signal cannot change
> *which* fixture is bet — that decision is already made by the time any price
> is observed — so the only thing it can change is *when* the price is taken.
> **Two-sided timing is not the policy that gives the best n; it is the only
> policy the architecture permits.**

Had the pipeline been able to select fixtures by drift, the honest registration
would have been ρ = 0.192 and **n = 167**, and the purchase would have been
refused under this document's own resize-or-refuse rule. **The number fell
because the system is more constrained than the general case, not because the
question was asked more gently.**

## THE FIGURE IS LOWER-BOUND-DRIVEN, AND THE BAND IS 33-39

`ρ·σ·E[z|acted]` **assumes acting captures the full predicted move.** Any
slippage — a price that has already moved, a market that closes, a stake that
cannot be placed at the quoted line — reduces the captured fraction, which
raises the ρ needed and lowers n further. **So 33 is the floor of the actionable
band and not a point estimate:**

| | δ | ρ | n |
| --- | --- | --- | --- |
| break-even | +1.85% | 0.391 | **39** |
| actionable | +2.00% | 0.422 | **33** |

**Register the band, quote 33-39, and do not quote 33 alone.**

## THE PURCHASE, NOW SIZED

`docs/unaddressable-by-self-observation.md` anchors the variation at **~100
credits for n = 50** — 2 credits per fixture at `CREDITS_PER_REQUEST = 2`.
Scaling at that anchor:

| n | credits |
| --- | --- |
| 33 | **66** |
| 39 | **78** |

> **66-78 credits, about two-thirds of the ~100 it replaces.** The anchor is
> itself approximate, so the band should be read as *roughly two-thirds*, not to
> the credit.

**The sizing no longer waits on anything. The purchase waits on 10-01 for the
budget** — the gate refuses every request until the quota resets, and
66-78 credits against a 400-credit monthly budget is affordable on any day after
it.

**Still `UNADDRESSABLE BY SELF-OBSERVATION`, disposition (a).** Three separated
observations do not exist because the window/interval policy permits one. This
sizes the purchase; it does not authorise it, and it still competes with H5 Q1's
`--any-fixture` control for the same ceiling.

*Registered 2026-09-16, before any credit is spent.*
