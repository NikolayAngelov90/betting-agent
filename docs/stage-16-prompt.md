# Stage 16 — What Is the Checkpoint For?

Stage 15 is closed: `STAGE 15 — FRONTIER MEASURED`. The frontier is flat at ~0.32 MODEL observations per credit. No re-timing, re-intervalling or market expansion beats it. Only more money or a smaller target moves the March 2027 date.

**Niki has decided: the free tier only. No paid plan, now or later.** That removes one of the two levers permanently.

So this stage examines the other one, and it examines it as a question rather than as a wish:

> **What effect size is the 500-observation checkpoint powered to detect, and what effect size would actually justify putting real money on this system?**

If the second is larger than the first, the checkpoint is over-specified — it is calibrated to resolve a difference too small to act on, and the experiment is scheduled to spend seven months distinguishing outcomes that would produce the same decision either way.

No code changes. No config changes. No schedule changes. This stage produces an argument and a number.

---

## 0. Why this is a legitimate question and not moving the goalposts

Changing a decision threshold after seeing data is normally indefensible, and this project has spent two stages refusing exactly that kind of move. So the constraint that makes this legitimate must be honoured precisely:

**The required sample size may be derived from the variance. The effect size may not be derived from the observed mean.**

Variance is a nuisance parameter — using the observed spread to compute how many observations a given precision needs is ordinary sample-size arithmetic and is not affected by having seen the data. But the effect size — *how much CLV would make this worth real money* — must be argued from betting economics and from this system's own cost structure. It must not be chosen because it happens to sit inside or outside the current interval.

Write the effect-size argument **before** looking at how it compares to the observed interval, and say in the report that you did. If the argument is contaminated, the whole stage is worthless.

**And note what 500 already is.** It appears in the README as policy with no derivation attached. That makes it the fourth instance of the pattern this project keeps finding — `212 credits/month`, the `~97%` egress reduction, `256 credits/month, 96% coverage`, and now the checkpoint itself. Three were figures describing the system. This one decides whether real money is ever staked. Record it as such.

---

## PART A — What is 500 powered to detect?

Pure arithmetic from the current sample. No interpretation yet.

### A1. The precision the current data gives

From today's paper-trading report: MODEL n = 42, mean −0.509%, 95% CI [−1.4%, +0.3%], design effect 1.00.

Report the standard error implied by that interval, and state whether the CLV distribution is close enough to symmetric for the normal approximation to hold. If it is not, use the bootstrap the report already runs rather than a closed form.

### A2. Precision as a function of n

Project the half-width of the 95% interval at n = 100, 150, 200, 300, 500, 750.

Two things to handle explicitly rather than assume:

* **Design effect.** It is 1.00 today. Stage 13's `s5.3` capped picks at one per match, so fixtures should now carry one observation each and the clustering that motivated the cluster bootstrap is largely gone going forward. Confirm that from the data rather than from the reasoning — the README's `18.9% of fixtures carry two picks` is a pre-`s5.3` figure and should be labelled as historical.
* **Variance stability.** 42 observations from nine days of one season is a thin basis for a variance estimate. Report the uncertainty in the SE itself, or state plainly that the projection inherits it.

### A3. State it plainly

> At n = 500, this experiment can distinguish a mean CLV of X% from zero.

That single sentence is Part A's output.

---

## PART B — What effect size would justify real money?

Do this before comparing anything to Part A.

### B1. Derive it, do not assert it

The question is: **what sustained mean CLV would indicate an edge large enough to bet real money on, for this system, at its prices?**

Argue it from the system's own numbers and its own cost structure:

* the settled record — 1,074 picks, 51.676% win rate, −3.836% flat ROI — and what CLV would correspond to break-even at the odds actually taken
* the bookmaker margin actually paid, computed from the overround on the books this system uses
* the relationship between closing-line value and realised edge, stated as the assumption it is: CLV is a proxy, not a payoff, and the strength of that proxy is itself an assumption
* the Kelly staking in use (`kelly_fraction: 0.25`, `max_stake_percentage: 4.0`) — an edge too small to stake meaningfully is not an actionable edge regardless of significance

Label every input `measured`, `assumed`, or `from literature`. If a needed quantity cannot be obtained, say so and carry it as an explicit assumption with a range rather than a point.

### B2. Give a range, not a point

Produce a minimum decision-relevant effect size and, if the argument supports it, a comfortable one. State what each implies.

Do not look at the observed interval while doing this.

---

## PART C — Compare

Now, and only now, put A and B together.

1. **What n does B's effect size actually require?** If B says a decision needs CLV ≥ +1.5%, and A says n = 150 resolves that, then 500 is over-specified by a factor of three and the checkpoint arrives in November rather than March.
2. **What does the current sample already exclude?** The upper bound of the current interval is a fact. If B's minimum decision-relevant effect already sits outside it, then the question worth asking is no longer "when will we know" but "do we already know."
3. **What would the answer have to look like to change the decision?** Enumerate the outcomes: MODEL CLV clearly above B's threshold, clearly below it, or indistinguishable. State what each means for the paper-trading experiment.

Handle the third outcome honestly. **"The model does not beat the closing line, real money stays off permanently, and the experiment's remaining purpose is to say so with confidence" is a legitimate result of this stage.** The 2026-08-07 audit already reached that conclusion three independent ways on settled outcomes; CLV was the one test it had not run. If CLV agrees with it, that is the experiment succeeding, not failing.

---

## PART D — Recommend, change nothing

The checkpoint is policy. It lives in the README and drives the paper-trading report's counters. Changing it is Niki's decision, not yours.

Produce:

* a recommended checkpoint with its derivation attached, in the provenance format Stage 15 established — the number, its effect size, its power, its date
* what the current data already supports and what it does not
* the reasoning stated so that a reader in six months can check it rather than trust it

If the recommendation is to keep 500, say that too, with the argument.

---

## E. Downstream, explicitly deferred

**L6a is not in this stage.** A pick-time write resets the refresh interval clock, and a `taken_at` row can never satisfy the strictly-after rule, so that reset guarantees a missing observation. It is a genuine defect and it was priced at +34 credits/month — the most expensive lever measured, against a cap already binding at ~436/450.

Its affordability depends on this stage's answer. If the checkpoint falls, the credit pressure falls with it and L6a becomes affordable on correctness grounds alone. If the checkpoint holds, it does not.

Note also that **L4 is shipped and inert** until L6a lands — the 10:47 window is suppressed by the same interval reset. That is a lever whose null result is currently indistinguishable from its success, which is the exact condition Stage 15 generalised. Record it as a known-inert deployment so nobody later reads its zero as a measurement.

---

## F. Hard rules

1. **Part B before Part C.** If the effect-size argument is written after seeing how it compares, the stage is void. Say in the report which order you worked in.
2. **No number without provenance** — `measured`, `simulated`, `assumed`, or `from literature`, with a date.
3. **A single anomalous result is evidence about the measurement first.**
4. **No code, no config, no schedule changes.** This stage produces an argument.
5. **No history rewriting**, no restamping, no backfill.
6. If any invariant in `tests/test_experiment_invariants.py` fails: **STOP, do not fix, report.** Invariants 2 and 3 are known defective; a pass from either means nothing, and the declaration must not cite the invariant count.

Declare:

`STAGE 16 — CHECKPOINT DERIVED`

or, if the effect-size argument cannot be made rigorously from what is available:

`STAGE 16 — CHECKPOINT UNDERIVABLE` — with what would be needed to derive it, and an explicit statement that 500 therefore remains an unjustified number rather than a validated one.
