# H5 — PRE-KICKOFF DRIFT: pre-registration

**Registered 2026-09-03, before any of the tests below were run.**

**H5 is the first quantity measured in this project that sits on the right side
of the vig.** Every other candidate has come in below break-even — H2 at
**+0.705%** before it turned out to be the two-way trap, MODEL CLV at
**−0.587%**. The observed drift is **+3.92%**, against a **+1.85%** best-line
break-even and a **+2%** decision-relevant threshold.

**And it does not need what H1 needs.** H1 asks whether movement predicts
movement and therefore requires three separated observations. **Drift needs
only two — a taken price and a later price — which is exactly what the current
policy produces for free.** 1,470 two-point keys are already held and the count
grows daily at zero credit cost. **Nothing here requires a purchase.**

---

## WHAT HAS ALREADY BEEN SEEN — declared, so this registration is not false

**This was measured on 2026-09-03 while deriving H1's σ, before H5 was
formulated.** It is the observation that prompted H5 and it cannot be
un-seen:

| market | selection | n (keys) | mean | sd |
| --- | --- | --- | --- | --- |
| 1X2 | Home | 283 | **+3.69%** | 9.21% |
| 1X2 | Away | 283 | **−1.82%** | 10.53% |
| 1X2 | Draw | 283 | +1.68% | 3.78% |
| over_under | Over 2.5 | 144 | −0.86% | 4.52% |
| over_under | Under 2.5 | 144 | +1.83% | 4.76% |
| over_under | Over 3.5 | 25 | −2.10% | 1.78% |
| over_under | Under 3.5 | 25 | +2.26% | 1.69% |

Fixture-level, 1X2 Home, one price per fixture: **mean +3.92%, σ 9.39%, n=15
fixtures.**

**Everything below has NOT been looked at**: the picked-versus-unpicked control,
the taken-selection direction test, and every breakdown by league, price band
and lead time. **No query answering any registered question has been run.**

---

## THE THREE QUESTIONS, and what each outcome would mean

### Q1 — Is the drift REAL, or is it selection bias?

**The control the pick population makes available:** fixtures in the **same
league on the same day** that the system did **not** pick. Snapshots are written
for every fixture in a refreshed league, not only picked ones, so the control
exists without new spend.

| outcome | condition | meaning |
| --- | --- | --- |
| **MARKET EFFECT** | picked and unpicked Home both drift, and the gap between them is **< 1 percentage point** | the drift is a property of the market, not of this system's choices |
| **SELECTION BIAS** | picked Home drifts **≥ 2pp more** than unpicked Home | the system picks Homes whose price is about to lengthen — it is systematically on the wrong side, which is itself a finding |
| **MIXED** | gap between 1pp and 2pp | both present; report both components and do not claim either |

### Q2 — Is it HOME-specific, or does everything drift the way the system took it?

**The distinction decides whether this is a market effect or a pricing
artefact of WHEN this system takes prices.**

For every pick, measure the movement of **the selection actually taken**.

| outcome | condition | meaning |
| --- | --- | --- |
| **PRICING ARTEFACT** | the taken selection drifts out by **>+1.85%** across **all** market types, Home and Away and Over and Under alike | this system takes stale or generous prices that then correct. Nothing to do with Home; it is a statement about our timing, and the remedy is when we take prices, not what we back |
| **HOME-SPECIFIC** | 1X2 Home drifts out while at least one other taken selection drifts **in** | a directional market phenomenon |
| **NEITHER** | taken-selection drift < +1.85% overall | the +3.92% was a fixture-selection artefact of the 15 |

### Q3 — Does it hold, or concentrate?

Breakdowns, each reported whether or not it is flattering: **by league**, **by
price band** (odds < 2.0, 2.0–3.5, > 3.5), and **by lead time** (< 6h, 6–12h,
> 12h from pick to kickoff).

| outcome | condition |
| --- | --- |
| **BROAD** | drift > +1.85% in **≥ 60%** of strata holding ≥ 10 fixtures |
| **CONCENTRATED** | drift is carried by **< 3 leagues** or by a single price band |

**A concentrated effect is not disqualifying and is not tradable as a general
rule.** It becomes a narrower hypothesis about those strata, and the segment
warning applies: **`settled-pick-segments-are-noise` established that per-league
splits on small samples produce spurious structure** (ROI spread p=0.407). **Q3
is descriptive. No threshold will be fitted to whatever it shows.**

---

## SAMPLE SIZE AND STOPPING RULE

**Fixed now, so the analysis cannot be run repeatedly until it passes.**

* **Minimum n: 50 fixtures with a two-point 1X2 Home observation ≥30 minutes
  apart.** Currently **15**.
* **Analysis runs ONCE, when n ≥ 50 is reached.** Not before, and not again
  after a disappointing result.
* At σ = 9.39% and δ = 2%, **n=50 gives ~45% power** — under-powered for the
  decision-relevant effect and **adequate for the +3.92% actually observed**
  (n≈34 at δ=4%). **Stated in advance: a null at n=50 does NOT close the
  question**; it bounds the effect below ~4%.
* **Re-derive σ at the same time.** The H1 sample-size decision depends on it
  and it currently rests on 15 fixtures.

**Expected date: ~3 weeks (2026-09-24), at the observed accumulation rate.**

---

## WHY THIS IS WORTH REGISTERING RATHER THAN JUST RUNNING

**The direction is coherent with what is already known**, which is exactly when
a spurious result is most persuasive: if Home odds systematically lengthen
before kickoff, then backing Home early yields negative CLV — **consistent with
the measured MODEL CLV of −0.587%** on a portfolio that takes Home some of the
time.

> **A finding that explains an existing puzzle is the easiest kind to believe
> and the hardest to check.** That is the reason for fixing the outcomes now.

*Registered 2026-09-03. n=15 at registration; analysis at n≥50.*
