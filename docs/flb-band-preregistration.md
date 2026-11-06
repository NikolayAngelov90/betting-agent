# FLB Band Pre-Registration — opening `min_odds` to measure the 70–79% band

**Registered 2026-11-06, before the first pick of cohort `s5.20` exists.**
Nothing in this document was written after seeing a result from the opened
band, because the band has produced no picks at the time of writing: the
config change ships in the commit *after* this one, by design.

This supplements `docs/flb-preregistration.md`, which fixed the bands, the
de-vigging rule and the band-level decision rule before any FLB measurement.
Those definitions are **not re-opened here**. What is new is that the band is
now reachable at all.

---

## 1. WHY THIS EXISTS — the band was unreachable, not merely empty

The earlier FLB test reported the `[0.70, 0.80)` and `[0.80, 0.90)` bands as
**`n=0, INSUFFICIENT`** and predicted that before querying, attributing it to
*"our avg taken odds of ~1.6 put the mass in the middle"*. That reading makes
the emptiness a **sampling** property, which more data would fix.

Measured 2026-11-06: it is not sampling. `betting.min_odds: 1.50` is a
**registered gate** (`gate_registry.py`: *"removes selections priced below
betting.min_odds (1.50)"*, reason *"Skip very short prices — too little
margin for model error"*). A price floor is a ceiling on probability: 1/1.50 =
**66.7%**, and the band starts at 70%. **0 of 1,639 settled paper picks sit in
the band and none could.** No further collection and no change of venue
reaches it.

> **So this stage opens the band to MEASURE the anomaly in our own data. It is
> not an attempt to profit from it.** `paper_trading_mode` stays `true`. There
> has never been a live wager and this change does not create one.

---

## 2. THE DERIVED BOUND — and why `1/0.79` is the wrong arithmetic

`docs/flb-preregistration.md` fixes the bands on **de-vigged** probability:
`p_devig = p_implied / Σ p_implied`, proportional normalisation over the
market's own legs. But `min_odds` gates the **raw** price. The two differ by
the market's margin:

```
raw implied  =  p_devig × Σ p_implied  =  p_devig × (1 + margin)
floor        =  1 / (0.80 × (1 + margin))
```

So a leg whose **de-vigged** probability is 0.79 has a **raw** implied
probability near 0.84 and odds near **1.19** — not 1.2658. Deriving the floor
from `1/0.79 = 1.2658` would open the band only partway and the measurement
could not see the top of its own band.

**The margin is measured, not assumed** (2026-11-06, grain = LEG SET,
n = 23,398 leg sets at taken median prices):

| margin (relative) | value | floor it implies |
|---|---|---|
| median | 6.66% | 1.1720 |
| p95 | 8.17% | 1.1556 |
| max | 16.11% | 1.0765 |

**Cross-check, measured**: of the **5,314** legs that actually fall in the
de-vigged `[0.70, 0.80)` band in our record, the raw odds run
**1.1450 … 1.3700**, with a 1st percentile of 1.1650.

> ### THE BOUND: **`betting.min_odds: 1.50 → 1.14`**
>
> Derived, not chosen: 1.14 is the two-decimal value that sits **below the
> minimum raw odds of every in-band leg measured** (1.1450) and **inside the
> theoretical range** [1.0765 at the max margin, 1.1720 at the median]. It
> admits **5,314 of 5,314** in-band legs — **100.00%** as measured.

### A RAW FLOOR CANNOT EXPRESS A DE-VIGGED BAND, AND THAT IS REPORTED

At 1.14, **9,334** legs are newly admitted. By de-vigged band:

| de-vigged band | legs | share |
|---|---|---|
| **`[0.70, 0.80)` — the target** | **5,314** | **56.9%** |
| `< 0.70` (between the old 66.7% ceiling and the band) | 2,945 | 31.6% |
| `[0.80, 0.90)` — spill into the next band | 1,075 | 11.5% |

This is **not** a second experiment smuggled in. The pre-registration tests
each band **separately**, so the spill **populates** `[0.60, 0.70)` and
`[0.80, 0.90)` rather than confounding `[0.70, 0.80)`. And it is what that
document asked for: it states that *"a single band clearing is weak evidence"*
and that the real test is *"several bands moving together in the predicted
direction"*. Before this change, three of five bands could never be tested.

---

## 3. THE HYPOTHESIS — directional, one-sided

> **H-BAND.** Among picks whose **de-vigged** implied probability falls in
> `[0.70, 0.80)`, the observed win rate **exceeds** the de-vigged implied
> probability.

One-sided, because the alternative of interest is a **positive** gap: that is
the direction the favourite–longshot bias predicts and the direction the
external study reports. A negative gap is not a finding in favour of anything
and is reported as a refutation.

**The comparator, kept labelled and NOT re-derived:**

> **`+3.74pp at n=134` — external, Footiqo, 2025/26, ONE bookmaker (1xBet),
> five leagues, 70–79% band.** Their own limitations section says one season,
> five leagues, one book. It is a **reference point, not a prior**: nothing
> below is conditioned on it being true of our fixtures.

---

## 4. THE DECISION RULE, AND THE n AT WHICH IT FIRES

Grain = **PICK**, and at `betting.max_picks_per_match: 1` that is also one per
fixture, so picks in the band are independent at the fixture grain. This is
why the n below is analytic and the bootstrap is a **check** on it rather than
a replacement.

**Required n**, to distinguish `+3.74pp` from zero at **80% power**,
one-sided **α = 0.05**, at the band's midpoint rate 0.75:

```
n = p(1-p) · ((z_0.05 + z_0.80) / effect)²
  = 0.75 · 0.25 · ((1.6449 + 0.8416) / 0.0374)²
  = 829 in-band picks
```

Empirical power at n = 829, bootstrap over 4,000 simulated bands: **82.8%** —
which confirms the analytic n rather than replacing it.

| | |
|---|---|
| **n to test the band at all** (from `flb-preregistration.md`) | **100** picks over ≥ 50 fixtures |
| **n to detect `+3.74pp` at 80% power** | **829** picks |
| ceiling rate: fixtures/day carrying an admissible band leg (measured) | **21.56** |
| **earliest** n=100 | **5 days** → ~2026-11-11 |
| **earliest** n=829 | **38 days** → ~2026-12-14 |

> **BOTH DATES ARE CEILINGS AND WILL BE LATE.** 21.56/day is the rate at
> which *fixtures carry an admissible in-band leg*, not the rate at which
> picks are made. The EV, value, confidence, correlation and
> `max_picks_per_match` gates are **unchanged** and will reject most
> candidates. The realised rate is bounded above by this and is expected to be
> far below it. **The honest statement of the arrival date is: not before
> 2026-11-11 for a testable band, and not before 2026-12-14 for a powered
> one.**

**THE RULE.**

1. **n < 100 in the band** → report `INSUFFICIENT`. Not a result, not a
   direction, not a hint. No decision.
2. **n ≥ 100 and the one-sided 95% lower bound on the gap > 0** → the bias is
   **PRESENT in this band**. Reported as one band, which
   `flb-preregistration.md` already fixed as **weak evidence** on its own.
3. **n ≥ 100 and the interval spans zero** → `NOT PRESENT at this n`, with
   the interval's width stated so a reader can see whether that is a null or
   an absence of power.
4. **n ≥ 829 and the interval spans zero** → **REFUTED at the powered n.**
   This is the outcome that ends the line of work, and it is the reason the
   829 is registered now rather than after.
5. **Several bands moving together** in the predicted direction is the only
   outcome `flb-preregistration.md` accepts as support for the *hypothesis*
   rather than for one band. That document's multiple-comparison caution
   stands unchanged: five bands, one test each, no re-banding, no merging.

---

## 5. THE ECONOMIC BAR — stated separately, because it is a different question

> **Measuring the anomaly and being able to collect it are two questions. This
> stage is the first one only.**

A gap against a de-vigged price is **not money**. Even a band that clears rule
2 or 4 says nothing about profitability until it clears the bar
`flb-preregistration.md` already set:

> *"A band passes economically only when its lower bound exceeds the overround
> measured at the prices we actually took in that band."*

And the bar in the band is **known to be high before we start**, measured
2026-11-06: the band's own cost at the taken price is **4.70pp**, so the
external `+3.74pp` nets **−0.96pp** at our current pricing, and **+0.05pp** at
the cheapest bookmaker whose price is established takeable (`Pinnacle`,
50.9% coverage). **A positive measurement in rule 2 is therefore expected to
fail the economic bar**, and that combination — *the bias is real and we
cannot collect it* — is a legitimate, registered outcome, not a
disappointment to be re-analysed.

Anything that would turn a measured gap into money is a **separate stage**
with its own registration: a venue change, an exchange with established size,
or a lower-margin market. None of those is in scope here and none is assumed.

---

## 6. WHAT WOULD REFUTE H-BAND

* **the powered null**: n ≥ 829 and the one-sided interval spans zero (rule 4);
* **a negative gap** at n ≥ 100: the de-vigged price *over*states the win rate
  in the band, which is the opposite of the favourite–longshot direction;
* **bands moving in opposite directions**: if `[0.70, 0.80)` is positive while
  `[0.60, 0.70)` and `[0.80, 0.90)` are negative, that is noise with a
  favourable band selected by chance, and `flb-preregistration.md`'s
  multiple-comparison rule already refuses it;
* **the gap disappearing when the population is whole**: if an interim
  positive at n≈100–200 regresses as n approaches 829, the interim was the
  artefact. The n=829 reading governs and the interim is not re-interpreted.

**AND ONE FAILURE MODE THAT IS MINE, NOT THE HYPOTHESIS'S.** If the opened
band produces picks whose *selection* differs systematically from the rest of
the record — a different market mix, a different league mix — then a band
result is partly about that difference. The cohort entry for `s5.20` records
the market and league composition of the new picks against the old, measured,
so this is checkable rather than remembered.

---

## 7. WHAT THIS STAGE CHANGES, EXACTLY

**One key.** `betting.min_odds: 1.50 → 1.14`, in `config/config.example.yaml`
(which CI copies over `config/config.yaml`, so the example **is** the deployed
config) and in `config/config.yaml` to keep the two identical.

**Nothing else.** Not `max_odds`, not `max_picks_per_match`, not
`min_expected_value`, not `min_confidence`, not the blend weights, not the
markets, not the gates, not `paper_trading_mode`. `betting.min_odds` is a
`TRACKED_KEY`, so the change moves the fingerprint and the cohort bumps by the
mechanism that already exists — `CODE_REVISION` `s5.19 → s5.20` — rather than
by hand.

*Registered 2026-11-06. Derivation reproducible via
`python -m scripts.band_open`.*
