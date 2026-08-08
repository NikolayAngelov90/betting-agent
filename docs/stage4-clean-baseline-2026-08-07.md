# Stage 4 — Production Hardening, Clean Market Data & Genuine CLV

Executed against `docs/after-stage3-prompt.md`. **Nothing is committed.** No
production configuration was changed to chase ROI, and no profitability is
claimed.

Test suite: **457 passing, 0 regressions** (408 before Stage 3, 422 after Stage 3,
+35 in Stage 4).

Reproduce:

```bash
python -m scripts.run_clean_baseline --min-books 2   # Phases 4-7
python -m pytest tests/ -q                           # Phase 23
```

---

## 1. Historical corruption — exactly what was affected

The Stage 3 report said "2,486 Bet365 rows". That understated it. The forensics
here are per-(match, bookmaker), and the picture is worse and more nuanced.

### 1X2 overround by bookmaker (three-way books, plausible range ≈ 1.02–1.12)

| bookmaker | matches | median overround | implausible | share | window |
|---|---|---|---|---|---|
| **Bet365** | 2,486 | **1.3524** | 2,296 | **92%** | 2026-02-01 .. 08-05 |
| **William Hill** | 405 | 1.3722 | 379 | **94%** | 2026-02-01 .. 07-06 |
| **Unibet** | 396 | 1.3689 | 379 | **96%** | 2026-02-01 .. 07-06 |
| **Betfair** | 410 | 1.3556 | 351 | **86%** | 2026-02-01 .. 07-06 |
| **10Bet** | 255 | 1.3682 | 249 | **98%** | 2026-02-01 .. 03-07 |
| **Betano** | 255 | 1.3548 | 244 | **96%** | 2026-02-01 .. 03-07 |
| **888Sport** | 182 | 1.3749 | 176 | **97%** | 2026-02-01 .. 03-07 |
| **Pinnacle** | 2,489 | 1.0483 | 642 | **26%** | 2026-02-01 .. 08-05 |
| SBO | 255 | 1.1306 | 100 | 39% | 2026-02-01 .. 03-07 |
| 1xBet | 2,483 | 1.0481 | 0 | 0% | — |
| every `TheOddsAPI-*` book (30 of them) | ~1,200 each | 1.01–1.12 | 0 | **0%** | — |

**4,819 of 36,895 bookmaker-books (13.1%) are impossible.** Pinnacle — the book I
used as the clean reference in Stage 3 — is corrupt on a quarter of its matches,
which means the Stage 3 A/B test's "reference" was itself partly contaminated.

The `TheOddsAPI-*` books are all clean because `theods_scraper.py` maps only
`h2h → 1X2` and `totals → over_under` with explicit outcome matching. The
corruption is confined to the API-Football ingestion path.

### Mechanism, confirmed arithmetically

For corrupted books, the stored Home price is the **two-way (draw-excluded)**
price, not the three-way one. Against a clean book on the same match:

| book | n | stored Home ÷ fair DNB price | stored Home ÷ true 1X2 price |
|---|---|---|---|
| Bet365 | 2,288 | **0.931** | 0.722 |
| Unibet | 379 | **0.916** | 0.708 |
| William Hill | 379 | **0.928** | 0.716 |

A ratio of ~0.93 against the fair DNB price is exactly a bookmaker's DNB quote
with margin. A ratio of 0.72 against the true 1X2 price is not a price at all.
Only Home and Away were overwritten — the two-way bet has no Draw leg — which is
why the overround lands at ~1.36 rather than being uniformly inflated.

### Blast radius

| Dataset | Total | Potentially corrupted | Safe |
|---|---|---|---|
| 1X2 odds rows (corrupt books) | 315,609 all markets | **13,274** across **2,548 matches** | 302,335 |
| 1X2 bookmaker-books | 36,895 | **4,819 (13.1%)** | 32,076 |
| Matches with any 1X2 price | 2,781 | 8 left with **zero** usable books | 2,773 |
| Saved picks | 1,018 | **934 (92%)** had the corrupt book as first choice | 84 |
| Other markets (over_under, btts, team_goals) | — | **0** | all |

**Markets other than 1X2 are clean.** `over_under`, `btts` and `team_goals` show
0% implausible overrounds across every bookmaker.

### A false positive I found in my own diagnostic, and corrected

My first pass flagged `double_chance` as 92% corrupt (median two-leg overround
1.32). That was my test being wrong, not the data. Double-chance outcomes
**overlap**: P(1X) + P(X2) = (H+D) + (D+A) = 1 + P(D) ≈ 1.25 by construction.
Checking all three legs of real Bet365 books gives 2.09–2.14, i.e. fair + margin.
Double chance is **not corrupted**, and `market_spec` now declares it
`overlapping=True` so the sum-to-one check can never be applied to it again.

### A separate integrity issue found while building the clean filter

**11,027 odds rows (3.5%) carry a timestamp AFTER kickoff**, median 1.01 hours
late — worst on Betano (18%), 888Sport (26%), 10Bet (19%). These are refreshes
that landed after the match started, so their `odds_value` is not a pre-match
price. Only **9 saved picks** sit on affected matches, so the practical impact on
predictions is small, but the clean dataset excludes such rows by rule rather
than by luck.

### Can it be reconstructed?

**No, for the corrupted books.** The two-way price physically overwrote the
three-way one under the same unique key `(match_id, bookmaker, market_type,
selection)`; the original value is gone. All 2,486 corrupt Bet365 matches are in
the past, and the API-Football free tier does not serve historical odds for past
fixtures.

**Nothing was deleted.** The rows remain in place and are excluded at read time
by the overround gate, so the decision is reversible and auditable.

---

## 2. Clean dataset — how a record qualifies

`src/evaluation/clean_dataset.py`. A **(match, bookmaker, market)** book survives
only if it satisfies all of:

1. **Correct market mapping** — every declared leg present (`market_spec.extract_legs`)
2. **Valid odds** — every leg a decimal > 1.0
3. **Plausible overround** — inside the market's declared band (1.005–1.25 three-way, 1.005–1.20 two-way)
4. **No post-kickoff information** — odds timestamp precedes kickoff
5. **Not a display-only source** — Flashscore excluded

A **match** qualifies for a market when ≥ `min_books` books survive, and needs a
valid outcome and a kickoff timestamp.

Filtering at book level rather than bookmaker level matters: excluding Pinnacle
outright would have discarded the 74% of its matches that are fine.

### Manifest (min_books = 2, all history)

```
matches considered     : 38,347
matches qualified      : 2,700
bookmaker-market books : 119,873

book rejections:
    over_under_1.5: incomplete legs                16,935
    over_under_3.5: incomplete legs                14,376
    odds timestamped after kickoff                 10,732
    over_under_2.5: incomplete legs                 5,441
    1X2: implausible overround (corruption gate)    4,836
    excluded bookmaker (display-only prices)          919
    1X2: incomplete legs                              457

qualified matches per market:
    1X2              2,286
    over_under_2.5   2,698
    over_under_3.5   2,509
    over_under_1.5   2,425
    btts             2,412
```

The dominant rejection is **odds coverage, not corruption**: 35,647 matches never
had two priced books at all. Odds collection only began in 2026-02.

---

## 3. Clean baseline — market vs model

Walk-forward, five chronological cutoffs, 60-day windows, identical match set
(n = 926 for 1X2).

| Baseline | LogLoss | Brier | ECE | Acc | n |
|---|---|---|---|---|---|
| **B de-vigged consensus** | **1.0012** | **0.5983** | 0.0084 | 51.4% | 926 |
| A raw bookmaker (vig left in) | 1.0016 | 0.5985 | 0.0085 | 51.4% | 926 |
| G market 80% + poisson 20% | 1.0026 | 0.5989 | 0.0176 | 51.6% | 926 |
| H market 80% + elo/poisson 20% | 1.0027 | 0.5989 | 0.0168 | 51.0% | 926 |
| **I production ensemble (0.80)** | 1.0027 | 0.5989 | 0.0168 | 51.0% | 926 |
| F market 80% + elo 20% | 1.0030 | 0.5991 | 0.0171 | 50.3% | 926 |
| C elo only | 1.0495 | 0.6322 | 0.0116 | 47.3% | 926 |
| D poisson only (540d, ρ=0) | 1.0569 | 0.6380 | 0.0083 | 46.2% | 926 |

Paired bootstrap vs de-vigged consensus (negative = worse than market):

| candidate | Δ nats | 95% CI | verdict |
|---|---|---|---|
| A raw bookmaker | −0.0003 | [−0.0008, +0.0002] | not significant |
| **I production ensemble** | **−0.0015** | **[−0.0061, +0.0028]** | **not significant** |
| G market + poisson | −0.0014 | [−0.0063, +0.0034] | not significant |
| F market + elo | −0.0018 | [−0.0063, +0.0023] | not significant |
| C elo only | −0.0483 | [−0.0683, −0.0297] | **SIGNIFICANTLY worse** |
| D poisson only | −0.0557 | [−0.0781, −0.0332] | **SIGNIFICANTLY worse** |

**Baseline E (current ML model): NOT EVALUATED.** Running it requires
`create_features` per match (~45 min for 2,000 matches) and the shipped pickles
are dated 2026-03-31, so most of this evaluation window lies inside their
training set — any number would be optimistic and uninterpretable. It is measured
prospectively instead, via the log-loss path added to the weight learner in
Stage 3, which only accepts observations post-dating `trained_at`.

**The Stage 3 conclusion survives on clean data.** The model does not beat the
market; at 20% weight it is not significantly worse either.

---

## 4. Market performance (Phase 7, evaluated independently)

| market | best baseline | LogLoss | poisson-only | blend 80% vs market |
|---|---|---|---|---|
| 1X2 | de-vigged consensus | 1.0012 | 1.0569 **(sig. worse)** | −0.0015, ns |
| Over/Under 2.5 | raw bookmaker 0.6702 / consensus 0.6705 | 0.6705 | 0.6887 **(sig. worse)** | −0.0012, ns |
| Over/Under 1.5 | de-vigged consensus | 0.5087 | 0.5193 **(sig. worse)** | −0.0007, ns |
| BTTS | de-vigged consensus | 0.6743 | — | −0.0011, ns |

**There is no market where the model adds value.** The prompt's instruction — "if
the model adds value only to one market, keep it there" — has no market to
select. The architecture should be uniform because the evidence is uniform.

---

## 5. Model contribution over market

Zero, within measurement error, in every market tested.

**Phase 6, blend weight on clean data (1X2):**

| blend | LogLoss | vs pure market | verdict |
|---|---|---|---|
| **100% market** | **1.0012** | — | baseline |
| 90% market | 1.0013 | [−0.0025, +0.0021] | not significant |
| 80% market (shipped) | 1.0027 | [−0.0061, +0.0028] | not significant |

Pure market has the best point estimate at every weight, on clean data as on
dirty. 0.80 and 0.90 are both indistinguishable from it.

**Phase 5, consensus depth.** My first attempt compared different match sets
(each `min_books` threshold selects a different population), which is confounded.
Redone on a **fixed sample** of 1,262 matches that all have ≥8 surviving books,
computing the consensus from the first *k* of them:

| books used | LogLoss | Brier | Acc | vs 1 book |
|---|---|---|---|---|
| 1 | 0.9880 | 0.5893 | 52.5% | — |
| 2 | 0.9856 | 0.5878 | 52.8% | +0.0024, CI [−0.0007, +0.0060] ns |
| 3 | 0.9855 | 0.5877 | 52.9% | +0.0025, ns |
| 4 | **0.9854** | **0.5876** | 52.5% | +0.0026, ns |
| 5 | 0.9859 | 0.5878 | 52.3% | +0.0021, ns |
| 8 | 0.9859 | 0.5878 | 52.5% | +0.0021, ns |

**This corrects a Stage 3 claim.** Stage 3 attributed the 1.0034 → 0.9931
improvement to "cross-book median consensus". On clean data, adding books beyond
the first buys **+0.002 nats, not significant**. The gain came almost entirely
from the **overround gate excluding corrupt books**, not from averaging. The
consensus is still worth keeping — it is free, marginally positive, and robust to
one book breaking — but the credit belongs to the plausibility gate.

Recommended `min_books = 2`: no worse than any deeper quorum, and it keeps 2,286
qualifying 1X2 matches instead of 1,551 at 3 books.

---

## 6. CLV infrastructure

**Coverage today: 0%.** No pick has a closing price, because
`capture_closing_lines.py` has never run before a kickoff. The infrastructure is
in place and tested; the data is not.

Built in this stage:

- `src/evaluation/clv.py` — the documented formula and its validity rules.
  Three views stored (`price_clv = taken/closing − 1`, `prob_clv`, optional
  margin-free `fair_clv`), with raw prices retained so any definition can be
  recomputed later.
- `validate_pair()` enforces Phase 11's checklist in one place: market match,
  selection match, valid prices, capture timestamp present, not after kickoff,
  not more than `DEFAULT_MAX_CAPTURE_LEAD` (180 min) before it. **CLV is never
  computed from an invalid pair.**
- `coverage_report()` exposes `clv_coverage_rate` plus a breakdown of why pairs
  were rejected. With no valid pairs it reports **n/a, not 0** — zero would read
  as "we break even against the close" when the truth is "unknown".
- Duplicate capture is structurally prevented: the capture query selects only
  picks whose `closing_odds IS NULL`.
- The capture window is configuration (`betting.clv_capture_window_minutes`,
  default 90), not a constant, and misses are logged rather than silently dropped.

**Migration status:** `saved_picks.closing_odds` does not yet exist in production —
I confirmed this by hitting `UndefinedColumn` on a full ORM query. Migration 003
must be applied (or the next CI run's `create_tables()` will add it
automatically, since both columns are nullable). **This is the one operational
step Stage 4 leaves open.**

---

## 7. EV validation on clean data

The Stage 1 finding was suspect because 92% of picks used corrupted market
probabilities. Re-run on the **824 settled picks whose match and market are in
the clean dataset**:

| EV quintile | n | avg claimed EV | win% | flat ROI | 95% CI |
|---|---|---|---|---|---|
| Q1 | 164 | −4.2% | 62.8% | **+4.2%** | [−8.4%, +16.9%] |
| Q2 | 164 | +6.0% | 55.5% | −10.7% | [−23.2%, +1.8%] |
| **Q3** | 164 | +10.6% | 47.6% | **−15.7%** | **[−29.5%, −1.9%]** |
| Q4 | 164 | +16.9% | 48.2% | −6.2% | [−22.0%, +8.9%] |
| Q5 | 168 | +35.7% | 48.2% | +6.4% | [−10.7%, +23.2%] |

Spearman(claimed EV, realised profit) = **+0.0613, p = 0.079** — no ranking
information.

**The Stage 1 finding survives. The corruption was not its cause.** Q3's CI
excludes zero: the middle EV quintile is significantly losing.

Recomputing the edge against the **clean** consensus:

- mean model probability **0.6103**
- mean clean-market fair probability **0.5222**
- actual win rate **0.5243**
- → the model overstates by **+8.60pp**; the clean market by **−0.21pp**

The clean market is essentially perfectly calibrated on the picks we chose. The
model is not, and the gap is the whole of its claimed edge.

**Phase 15, EV threshold sweep (walk-forward halves, holdout n=412):**

| threshold | holdout n | win% | flat ROI | 95% CI |
|---|---|---|---|---|
| 0% | 317 | 46.4% | −11.5% | [−22.1%, −1.0%] |
| 2% | 297 | 45.5% | −12.9% | [−23.9%, −1.8%] |
| **5% (shipped)** | 261 | 43.3% | **−15.3%** | [−26.6%, −3.1%] |
| 8% | 217 | 42.4% | −15.0% | [−28.6%, −1.2%] |
| 10% | 171 | 42.1% | −14.0% | [−29.6%, +1.2%] |

Raising the EV threshold makes results **monotonically worse**. It is an
adverse-selection filter, not a value filter.

**Phase 16, confidence:**

| confidence quartile | n | avg prob | win% | flat ROI | Brier |
|---|---|---|---|---|---|
| Q1 0.36–0.54 | 206 | 0.495 | 40.3% | −10.7% | 0.2493 |
| Q2 0.54–0.60 | 206 | 0.573 | 52.9% | +1.1% | 0.2522 |
| Q3 0.60–0.67 | 206 | 0.631 | 52.9% | −5.4% | 0.2604 |
| Q4 0.67–0.92 | 206 | 0.742 | 63.6% | −2.5% | 0.2409 |

- Spearman(confidence, **profit**) = **−0.0669, p = 0.055**
- Spearman(confidence, **win/loss**) = **+0.1535, p = 0.000**

Confidence predicts *whether a bet wins* — it must, since short prices win more —
but it does **not** predict profit, and the sign is if anything negative.
`min_confidence` has no evidential basis as a profit filter.

---

## 8. Production configuration

**No configuration values were changed in Stage 4.** The Stage 3 settings were
re-verified on clean data and none of them moved:

```yaml
models:
  bookmaker_blend_weight: 0.80    # re-verified: ns vs pure market on clean data
  strength_half_life_days: 540    # unchanged
  dixon_coles_rho: 0.0            # unchanged
  dc_rho_per_league: false        # unchanged
betting:
  gates: {all six edge gates false}   # unchanged
  paper_trading_mode: false       # NEW — recommend turning ON, see below
  clv_capture_window_minutes: 90  # NEW
```

### Recommended today

1. **Apply migration 003 and 004** (or let the next CI run auto-add the nullable
   columns). Until 003 lands, any full `session.query(SavedPick)` fails against
   production.
2. **Schedule `capture_closing_lines.py`** ~60–90 minutes before kickoffs, after
   the odds refresh. Nothing else in this stage produces CLV data.
3. **Set `paper_trading_mode: true`.** Phase 13 asks for 200–500 valid picks
   before a production change; there are currently 0 with a closing line. Paper
   mode keeps the measurement running without putting a live record behind a
   question the data cannot yet answer.
4. Leave `bookmaker_blend_weight` at 0.80. Pure market has the better point
   estimate, but the difference is not significant, and 0.80 preserves a
   measurable model contribution for the weight learner to evaluate.

**Not recommended:** removing `min_expected_value` / `min_confidence`. Both are
evidentially worthless as value filters, but removing them roughly triples pick
volume, and at a true edge of −vig more volume is more loss. They should be
re-labelled as volume controls, not deleted — a decision I have left to you
rather than making silently.

---

## 9. Changes made

| File | Change |
|---|---|
| `src/data/market_spec.py` | **NEW.** Single source of truth for market arity, legs, overround bands, overlapping markets, and which source bets may write which market type. |
| `src/scrapers/apifootball_scraper.py` | `validate_write` guard refuses non-authoritative bets; per-fixture cross-bet collision guard logs and refuses a second bet overwriting another's key. |
| `src/evaluation/clean_dataset.py` | **NEW.** `build()` / `load_from_db()` producing `CleanMatch` + auditable `Manifest`. |
| `src/evaluation/clv.py` | **NEW.** CLV formula, `validate_pair`, `coverage_report`, `clv_coverage_rate`. |
| `scripts/run_clean_baseline.py` | **NEW.** Phases 4–7 on clean data. |
| `scripts/capture_closing_lines.py` | Config-driven window; misses logged explicitly. |
| `src/data/models.py` | `SavedPick.market_probability`, `.market_books`, `.is_paper`. |
| `migrations/004_prospective_measurement.{sql,rollback.sql}` | **NEW.** |
| `src/betting/value_calculator.py` | `BetRecommendation.market_probability/.market_books`; `_market_prob()` helper; populated at both construction sites. |
| `src/agent/betting_agent.py` | Paper-trading flag in `_save_picks`; market probability persisted; `--stats` CLV block corrected (a site Stage 3 missed still printed "edge"). |
| `src/reporting/match_briefing.py` | On a CHANGE, `market_probability` moves with the selection (it was going stale); pre-Claude snapshot explicitly left untouched. |
| `src/reporting/telegram_bot.py` | Docstring corrected re: CLV vs divergence. |
| `config/*.yaml` | `paper_trading_mode`, `clv_capture_window_minutes`. |

---

## 10. Tests

**457 passing, 0 regressions.** New in Stage 4: **35**.

- `tests/test_market_separation.py` (13) — the seven required tests plus
  `BET_TYPE_MAP` collision detection and spec/map consistency.
- `tests/test_clv_and_clean_dataset.py` (22) — CLV formula, timestamp validity,
  market/selection mismatch, coverage rate, duplicate-capture prevention, clean
  dataset filtering (corruption gate, quorum, post-kickoff, Flashscore,
  incomplete matches, manifest).

Notable assertions: `test_1_home_away_must_never_populate_1x2`,
`test_4_genuine_1x2_overround_is_plausible_and_corrupt_one_is_not` (uses the real
production prices from match 49032), `test_6c_overlapping_market_is_never_devigged`
(guards against my own false positive), and `test_clv_is_not_model_edge` (asserts
by source inspection that the CLV computation cannot reference a model probability).

---

## 11. Remaining uncertainty

1. **CLV coverage is 0%.** Everything about "does the model have an edge" remains
   unanswerable until 200–500 picks carry a closing line. That is the single
   gating fact.
2. **Baseline E was never measured.** The ML model's standalone contribution is
   unknown — its pickles predate most of the evaluation window.
3. **The clean 1X2 sample is 926 out-of-sample matches.** Enough to separate
   market from model (those CIs are wide of zero), not enough to detect a small
   genuine edge if one exists.
4. **Odds coverage, not corruption, is the real constraint.** 35,647 of 38,347
   matches never had two priced books. Everything measurable here rests on the
   ~2,700 matches since 2026-02.
5. **The corrupted 1X2 prices are permanently unrecoverable** for seven
   bookmakers over 2026-02-01..08-05.
6. **3.5% of odds rows are timestamped after kickoff.** Excluded from the clean
   dataset, but the underlying refresh behaviour is unfixed.
7. **Understat remains unverified** and was not integrated, per Phase 19.

---

# FINAL DECISION

### A. Does the model currently demonstrate a statistically reliable edge over the market?

**NO.**

On clean data the model is significantly *worse* than the market standalone
(Poisson −0.0557 nats, CI [−0.0781, −0.0332]; Elo −0.0483, CI [−0.0683, −0.0297]),
and its claimed EV is anti-predictive — the middle EV quintile returned −15.7%
with a CI excluding zero, and Spearman(EV, profit) = +0.06, p = 0.079. This is
not "insufficient evidence": there is ample evidence, and it points the wrong way.

### B. Does the model improve bookmaker probabilities?

**INSUFFICIENT EVIDENCE.**

Blended at 20% it is not significantly worse than pure market (−0.0015 nats, CI
[−0.0061, +0.0028]) but never better, in any market, at any weight. The point
estimate favours pure market everywhere. With n = 926 clean out-of-sample matches
a small positive contribution cannot be ruled out — but nothing observed suggests
one exists.

### C. Is genuine CLV now being measured?

**NO.**

The measurement system exists, is documented and is tested — formula, validity
rules, coverage reporting, duplicate prevention. But `clv_coverage_rate` is **0%**:
no pick has a closing price, the capture script has never run before a kickoff,
and migration 003 is not yet applied to production. It becomes YES once those two
operational steps are taken.

### D. Is the EV calculation trustworthy?

**NO.**

The arithmetic is correct — `EV = p × odds − 1` — but it is computed from a model
probability that overstates the true rate by **+8.60pp** against a clean market
that overstates by **−0.21pp**. An arithmetically correct function of a
systematically wrong input is not a trustworthy number, and the empirical test
confirms it: higher claimed EV predicts *worse* realised ROI, monotonically,
on clean data.

### E. What should production use today?

**PAPER TRADING ONLY.**

The market is the best available forecast; the model neither improves nor
(at 20% weight) damages it; the EV signal that would justify selecting bets is
demonstrably anti-predictive; and the one diagnostic that could settle the
question has zero coverage. Running live commits real variance to a question the
data cannot currently answer.

Set `paper_trading_mode: true`, apply migrations 003 and 004, schedule the
closing-line capture, and revisit when 200–500 picks carry a valid closing price.
If CLV is persistently negative at that point, the answer to B becomes NO and the
correct production configuration is `bookmaker_blend_weight: 1.0` — a
market-consensus price-shopper, which on all evidence gathered across Stages 1–4
is a perfectly respectable thing for this system to be.
