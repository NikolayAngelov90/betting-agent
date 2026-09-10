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
