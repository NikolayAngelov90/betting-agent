# Findings Inventory — what this project holds, and what may be published

**Compiled 2026-11-12**, on measurements and licence fetches dated
**2026-11-11**. Nothing here is published. This document settles what
*could* be, with provenance, and it records one correction that removes the
headline finding of four days ago.

Every row carries: the claim, the population, n, the grain, the unit, the
method, the date, and a label — **ORIGINAL**, **REPLICATION**, or
**WARNING**. A figure a reader cannot reproduce is an assertion, and is
labelled as one.

---

## 0. THE CORRECTION THAT COMES FIRST

> ### The band calibration result was a de-vigging artefact.
>
> On 2026-11-11 this project reported **four consecutive probability bands
> "PRESENT"** — gaps of +1.14 / +1.64 / +1.43 / +2.38pp, every CI above zero —
> and called H-BAND supported. **Net of the artefact its own method produces,
> not one of those four survives.**

**THE ARTEFACT IS ANALYTIC, not an estimate.** `docs/flb-preregistration.md`
fixed **proportional de-vigging** in advance: `p_devig = p_implied / Σ
p_implied`. A bookmaker who adds margin in equal shares — `quoted_i = p_i +
M/n` — and is **perfectly calibrated** still produces

```
gap = M · (p − 1/n) / (1 + M)
```

because every leg is divided by the same `1 + M` while the margin was added as
`M/n`. **The null is not zero.** The gap is positive above `1/n` and negative
below it: the sign flips at **0.500** for a two-way set and **0.333** for a
three-way one.

Subtracting that term per leg, using **each leg set's own arity and own
margin**, and re-bootstrapping over fixtures:

| band | raw gap | artefact | **residual** | 95% CI on residual | verdict |
|---|---|---|---|---|---|
| 0.30–0.40 | −1.63 | −0.69 | **−0.94** | [−1.92, **−0.02**] | **OVERPRICED** |
| 0.40–0.50 | −0.35 | −0.17 | −0.18 | [−1.08, +0.73] | explained by the de-vig |
| 0.50–0.60 | +1.14 | +0.37 | +0.77 | [−0.18, +1.75] | explained by the de-vig |
| 0.60–0.70 | +1.64 | +1.03 | +0.61 | [−0.48, +1.73] | explained by the de-vig |
| 0.70–0.80 | +1.43 | **+1.62** | **−0.19** | [−1.13, +0.73] | explained by the de-vig |
| 0.80–0.90 | +2.38 | **+2.17** | +0.21 | [−0.96, +1.36] | explained by the de-vig |

**The 70–80 band's residual is NEGATIVE.** The artefact there (+1.62pp) is
larger than the gap that was reported as a finding (+1.43pp).

**ONE RESULT SURVIVES, and it is on the side we cannot take.** Legs at
30–40% de-vigged probability are **overpriced by 0.94pp**, CI [−1.92, −0.02] —
the *longshot* half of the classic favourite–longshot bias. Collecting it
means **laying** longshots, which needs an exchange, whose depth is
`UNAVAILABLE`.

**The literature said so before we measured it.** Whelan's *Estimating
Expected Loss Rates in Betting Markets* states that under this bias "an
equal-weighted average of loss rates across each of the available bets will be
higher than predicted by this widely recommended calculation" — i.e. the naive
overround treatment is biased. Shin and Power de-vigging exist precisely for
this. **We pre-registered the biased method and then read its artefact as a
result.** Pre-registration protected us from choosing the bands after the
fact; it did not protect us from choosing the wrong estimator before it.

**What would settle it:** re-run under **Shin** and **Power** de-vigging and
report all three side by side. Until then the calibration table is
**not publishable as evidence of mispricing.**

---

## 1. `INVENTORY` — ranked

Ranked by: *is it original*, *is n large enough to stand alone*, *can a reader
reproduce it from obtainable data*.

### Rank 1 — **WARNING**: a de-vigged gap is not evidence of mispricing

| | |
|---|---|
| **claim** | Measuring calibration against a proportionally de-vigged price manufactures a gap of `M(p−1/n)/(1+M)` even from a perfectly calibrated book; on 52,737 legs it accounts for the entire apparent effect |
| population | 37,079 completed matches, 30 configured leagues, 2023-01-01 → |
| n / grain | **52,737 legs** over ~10,700 fixtures; grain = LEG, bootstrap over FIXTURES |
| unit | probability points |
| method | cross-book median price per leg; proportional de-vig per leg set; artefact subtracted per leg from its own arity and margin; 4,000-draw fixture bootstrap |
| date | 2026-11-11 |
| label | **WARNING** — the arithmetic is known; the **demonstration at this scale is not, as far as one literature search shows** |
| why rank 1 | it is the only finding here that is both *useful to others* and *not undercut by its own method*. It is a negative result about method, which is the kind that survives |

### Rank 2 — **WARNING**: best lines are not takeable

| | |
|---|---|
| **claim** | **21.28%** of 1X2 best-line sets imply **under 100%**, rising to **73.6%** in the 2–24h quote-spread band — a maximum assembled across books *and times* manufactures a price nobody can take |
| population | the odds record, 1X2 leg sets with ≥2 books |
| n | 927–1,372 leg sets per book; 21.28% over the best-line population |
| unit | share of leg sets |
| method | per-leg maximum across books, then `Σ1/o` against 1.0; contrasted with single-book sets (0.00–0.54%) |
| date | 2026-11-06 |
| label | **WARNING**, plausibly **ORIGINAL in this quantified form** — academic work uses single-book closing prices, so the artefact does not arise there. **Originality UNVERIFIED** (no literature review done) |
| why rank 2 | anyone running a naive best-line backtest is measuring a price that did not exist. Reproducible from any multi-book odds feed |

### Rank 3 — **WARNING**: the API-Football two-way trap

| | |
|---|---|
| **claim** | API-Football's **"Home/Away"** market is the two-way *draw-excluded* market, not 1X2. Mapping it to 1X2 corrupted the implied probabilities of **2,486 matches** across **seven books including Pinnacle (26%)** |
| population | our own ingested odds |
| n | 2,486 matches |
| method | per-(match, book) overround outside `[1.005, 1.250]` identifies the two-way set in a three-way slot |
| date | 2026-08 → 2026-10 |
| label | **WARNING**, specific to one provider; directly useful to other users of it |
| why rank 3 | a reproducible, checkable data-quality trap with a stated detection rule. Narrow audience, high value within it |

### Rank 4 — **REPLICATION**: per-book margin structure

| | |
|---|---|
| **claim** | Across **29 books**, relative margin is **higher in the favourites band** for every bookmaker and **flat or lower** for both exchanges |
| n | 1,489–18,928 leg-set observations per book |
| unit | **relative margin** — see the correction below |
| date | 2026-11-06 |
| label | **REPLICATION.** The literature already reports that "long odds on longshots are being adjusted more aggressively downwards than short odds on favourites, while overall margins are being kept roughly at bookmaker-specific target levels" |

> **AND A UNIT CORRECTION.** The ledger's "+0.77 to +1.18pp band premium"
> mixes a real relative increase with a **units effect**: `pp` is
> `(1/o)·c/(1+c)`, so a *constant* relative margin yields more probability
> points at shorter odds. In **relative margin** the increase is **+0.13 to
> +0.59 points** (1xBet 8.07%→8.43%, Pinnacle 4.57%→5.16%, Bet365
> 6.57%→6.70%), and the exchanges move **−0.15 / −0.19**. The direction
> holds; the magnitude was overstated by the unit.

### Rank 5 — **REPLICATION**: a real anomaly is smaller than the cost of taking it

| | |
|---|---|
| **claim** | At every venue with an established price, the in-band cost exceeds any surviving gap; one cell (80–90% at an exchange) clears by +0.48pp before commission and needs a rate below **3.6%** |
| n | 29 books × 4 bands; 1,639 settled picks for the pick-level arithmetic |
| date | 2026-11-06 … 11-11 |
| label | **REPLICATION** — "the margin eats the edge" is the standard result. Our contribution is the *arithmetic laid out per venue with coverage beside it* |
| **status now** | **weakened by §0**: with the gaps explained by the de-vig, there is less to net against. The conclusion (not collectable) is *overdetermined*, not undermined |

### Rank 6 — **REPLICATION, n too small to stand alone**: the settled paper record

| | |
|---|---|
| **claim** | 1,798 settled paper picks across **fifteen cohorts** at `min_odds 1.50` show no edge over the price |
| n / grain | 1,798 picks; grain = PICK; ROI −5.68%, CI excluding zero |
| caveats | **paper only**, never a live wager; `is_paper` collinear with time; per-segment spread p=0.407; MDE +5.60% against a +1.85% break-even — so "cannot exclude an edge 3× larger", **not** "no edge" |
| label | **REPLICATION** at small n. Not independently interesting |

### Also held, and NOT publishable

* **the s5.20 cohort** — 86 picks over two days at a floor the operator
  rejected. Retained and labelled; too small for anything.
* **CLV coverage 13%** — `avg_clv` is the model's *self-declared* edge, not a
  market measurement. Internal warning.
* **`_find_best_odds` returns the cross-book MEDIAN**, not the best, despite
  its name and docstring. Internal; a naming warning only.
* **the exposure / structural-margin re-specification**, the cron and deadline
  work, the capture-gate design, the absence-class register — **operational**.
  Interesting as engineering practice, not as findings about football markets.

---

## 2. `REPRODUCIBLE` — per finding

| finding | reproducible by a reader? | from what |
|---|---|---|
| the de-vig artefact (rank 1) | **YES, fully** — the formula is analytic and the demonstration needs only odds + results | any multi-book odds history with results. **The Odds API** historical endpoint (paid) or **football-data.co.uk** (free CSVs, closing odds for many leagues) |
| best lines not takeable (rank 2) | **YES** | any feed with ≥2 books per leg. football-data.co.uk carries multiple books' closing odds free |
| the API-Football two-way trap (rank 3) | **YES**, for anyone with an API-Football key | API-Football, free tier |
| per-book margin structure (rank 4) | **PARTLY** — 29 books at our granularity needs a paid multi-book feed | The Odds API (paid historical) |
| cost vs anomaly (rank 5) | **PARTLY** — the in-band cost per venue needs the same feed | as above |
| the settled paper record (rank 6) | **NO — this is an ASSERTION** | our own `saved_picks`. No reader can check it. It is our private record of our own model |

**Which figures depend on data only we hold, and are therefore assertions
rather than reproducible results:** everything derived from `saved_picks` —
the paper ROI, the win rate, the cohort table, the s5.20 distribution, the
CLV series. Also the `pick_observations` quote-age figures. These are
**assertions**, and may be reported only as such.

**THE METHOD, STATED WELL ENOUGH TO RE-RUN:**

* **price per leg** — cross-book **median** of all books quoting that
  (fixture, market, selection); this is what the pricer takes.
* **de-vigging** — proportional, **per leg set**: `p_devig = (1/o) / Σ(1/o)`
  over that set's own legs. **Never a pooled or median margin** — a pooled
  margin misassigns the legs whose own set is wide, and the spread across
  books is 0.6–3.9pp.
* **bands** — `[0.30,0.40) … [0.80,0.90)`, half-open, fixed at the external
  study's boundaries before our data was read. No re-banding, no merging.
* **bootstrap** — 4,000 draws resampling **FIXTURES**, not legs: a 1X2 set is
  three legs settled by one scoreline.
* **exclusions** — `matches.training_exclusion_reason` in
  {`corrupt_1x2_assignment` (28), `corrupt_team_identity` (81),
  `phantom_kickoff_now_stamp` (503)}; leg sets whose overround falls outside
  the market's registered band (728 — this is what removes a two-way price
  from a three-way slot); selections the vocabulary cannot canonicalise
  (45,283, almost all `team_goals` Under/2.5+/3.5+ and non-half
  `over_under` lines, for which the deployed grader has no branch); void and
  ungradeable outcomes (764).
* **tie handling** — `extract_legs` returns prices only, so each price's
  selection must be recovered **by reconstructing the key the way
  `extract_legs` builds it**, never by matching the price back. `Over 2.5`
  and `Under 2.5` both at 1.90 are one number and grade **oppositely** with
  an identical de-vigged probability: the band would be right and the result
  wrong. Fixing this moved the 70–80 gap from **+1.66 to +1.43pp**.
* **the artefact subtraction** — per leg, `M(p−1/n)/(1+M)` from that leg
  set's own `M` and arity `n`.

**THE SCRIPTS THAT REGENERATE EACH TABLE** — the figure and its method ship
together:

| table | script |
|---|---|
| band calibration + artefact residual + in-band cost per venue | `python -m scripts.band_legs` |
| per-book margin, coverage, counterfactual, takeability | `python -m scripts.cost_by_book` |
| exchange commission, size, coverage consequence | `python -m scripts.exchange_terms` |
| per-market cost and required edge | `python -m scripts.market_mix` |
| the band floor derivation and spill | `python -m scripts.band_open` |
| schedule exposure by day type | `python -m scripts.margin_exposure` |

Each takes `--json`. None writes to the database.

---

## 3. `RIGHTS` — per provider, quoted, with URL and fetch date

**Nothing may be published until this section is satisfied. "Unclear" is not
permission.**

### The Odds API — **PERMISSIVE for derived statistics**

*fetched 2026-11-11, <https://the-odds-api.com/terms-and-conditions.html>*

> Permitted: **"Calculating and displaying values you derive from our data"**
>
> **"Displaying our data in a UI, website, or mobile app, including for
> commercial use"** … **"including commercial use, provided our data is not
> the primary product being sold or redistributed."**
>
> Prohibited: **"Do not resell, repackage, or redistribute our data as a
> standalone data product."** … **"We mainly prohibit reselling the data as a
> raw data feed, i.e. a competing product."**
>
> **"Attribution to The Odds API is not required, but is always
> appreciated."**

**Reading:** derived aggregate statistics are **expressly permitted**,
including commercially. Publishing the *raw odds rows* is not. Attribution
optional — **give it anyway.**

### football-data.org — **PERMISSIVE, ATTRIBUTION MANDATORY**

*fetched 2026-11-11, <https://www.football-data.org/about> (the T&C live on
this page; `/terms` and `/termsandconditions` both 404). Effective date
stated in the document: **"12.1 These Terms and Conditions become effective
on June 1st, 2018."***

> **"7. ATTRIBUTION — 7.1 You agree to include the following attribution to
> Football-Data in your app or website: 'Football data provided by the
> Football-Data.org API'. This attribution could be published in the footer,
> about us section or another visible location in your app or website."**
>
> **"9.1 After cancellation of the subscription to the Service, the Customer
> is not permitted to reference the football data (incl. …"**
>
> **"2.3 The API Key applies to a single Application"**

**Reading:** attribution is **required, with that exact wording**. There is
**no clause in the T&C restricting commercial use** — a "non-commercial only"
claim appears on their *blog*, not in the terms, and the terms are what bind.
**9.1 means the attribution and the data must come down if the subscription
ends.** The terms are **eight years old**; that is recorded, not resolved.

### API-Football — **RESTRICTIVE, and NOT DIRECTLY FETCHED**

*<https://www.api-football.com/terms> returns **HTTP 403** to both WebFetch
and curl on 2026-11-11. The clauses below were read from that official page
**via web search**, not by direct fetch. Labelled accordingly.*

> **"The data we provide is obtained from partners, it is prohibited to resell
> this data to third parties"**
>
> **"We do not provide a 'license' for the use and publication of the data
> provided by our services on applications, websites or any other products
> made by the user. Any license or permission to publish the data must be
> requested by the user from the competent authorities."**
>
> **"The use of our data for betting platforms, television broadcasting,
> fantasy sports platforms, or any mass media distribution may require
> additional licenses from the relevant rights holders."**

**Reading, strict:** API-Football **expressly declines to license
publication** and pushes the obligation onto the user to obtain permission
from rights holders. **Under the strictest reading, nothing derived from
API-Football data may be published** without clearing rights with the
leagues/federations.

**What would clarify it:** a direct fetch of the live page, and a written
answer from API-Football on whether *aggregate derived statistics* count as
"publication of the data". **Do not assume permission.**

### Flashscore — **PROHIBITIVE**

*fetched 2026-11-11, <https://www.flashscore.com/terms-of-use/>*

> **"2.2 You may not use the Site for any commercial purpose."**
>
> **2.10:** "You may not burden our server on which the Site is hosted with
> automated requests" … "by embedding, aggregating, scraping or recreating it
> without our express consent"
>
> **2.9:** "no extraction (copying) or utilization (making available to the
> public) of Database Content" … "or of a qualitatively or quantitatively
> substantial part thereof"
>
> **2.8:** "for direct or indirect economic gain"

**Reading:** Flashscore forbids commercial use, automated access without
consent, and making database content available to the public. **Nothing
derived from Flashscore may be published**, and our *scraping* of it for
results is already outside these terms.

### WHAT IS PUBLISHABLE UNDER THE STRICTEST READING

| finding | verdict |
|---|---|
| **the de-vig artefact (rank 1)** | **YES, but only if re-derived from a clean source.** The arithmetic is ours and needs no provider data at all; the 52,737-leg demonstration currently rests on API-Football + Flashscore and must be **re-run on football-data.co.uk free CSVs** (or The Odds API historical) before publication |
| **best lines not takeable (rank 2)** | **YES, re-run on The Odds API data** (derived values expressly permitted) or football-data.co.uk |
| **the API-Football trap (rank 3)** | **YES in substance** — it is a statement about an *API's semantics*, not a redistribution of its data. Publish the detection rule and the match count, not the rows |
| **per-book margin (rank 4)** | **YES on The Odds API data only.** Our 29-book table mixes API-Football in and would need rebuilding |
| **cost vs anomaly (rank 5)** | as rank 4 |
| **the settled paper record (rank 6)** | **YES** — it is our own record of our own model, and no provider's data is redistributed. But it is an **assertion**, not reproducible |

> **The binding constraint is not statistics, it is provenance.** Our strongest
> candidates are built on the two providers whose terms are most restrictive.
> Re-running them on **football-data.co.uk** (free, multi-book closing odds)
> and **The Odds API** (derived values permitted) is what makes them
> publishable — and the re-run is also the Shin/Power check §0 demands. **One
> piece of work settles both.**

---

## 4. `PIPELINE CONSEQUENCE` — stated, not decided

**Which stages feed the publishable findings** — the odds record and results:

| stage | share of run (n=3, 2026-10-20) |
|---|---|
| `--update` (fixtures + odds capture) | **24.7 / 24.1 / 32.9 min — 43–50%** |
| `--settle` + Flashscore results | 14.1 / 11.2 / 4.0 min (post-picks) |

**Which stages feed only pick generation:**

| stage | cost |
|---|---|
| picks step (model inference, value/EV gates, correlation filter) | 11.3 / 5.4 / 21.1 min |
| ·· **the Claude review inside it** | **4.9 / 2.0 / 9.5 min** + API tokens |
| ML retrain (when stale) | occasional |
| the pick-level CLV series | storage only |

**What stopping pick generation would save per day:** the picks step —
**5.4–21.1 runner minutes**, of which the **Claude review is 2.0–9.5 minutes
plus its token cost** — and the occasional ML retrain. **Provider calls would
not fall at all**: odds capture lives in `--update`, which is the stage that
*feeds the publishable product*.

> **So the saving is small and the expensive stage is the one worth keeping.**
> The odds record is the asset; pick generation is a thin, cheap layer on top
> of it. "Stop making picks to save money" is not the argument — the argument
> would be about what the system is *for*.

**AND THE TENSION, NAMED:**

> The client Telegram channel currently publishes **picks and the
> settlement**, and that was chosen **two days ago**. If the product becomes
> market calibration, picks are not an input to it — and the channel's
> content is exactly the output that would stop.
>
> **These two choices are in tension. The resolution is Niki's.** Nothing in
> this document decides it, and no channel or pipeline change is made here.

---

*Compiled 2026-11-12 on 2026-11-11 measurements. Nothing published.
`cohort_status.py` reads
`s5.21 / 52678c`, unmoved: this stage reads.*
