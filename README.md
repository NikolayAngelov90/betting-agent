# Football Betting Agent

An automated football prediction system combining statistical models, machine learning and multi-source data scraping. Each pick is reviewed by Claude (web research + a final KEEP/CHANGE ruling), recorded to PostgreSQL, and delivered to Telegram. Runs daily on GitHub Actions.

> ### ⚠️ PAPER TRADING ONLY — no real-money betting is justified by this system
>
> A full predictive audit (Stages 1–3, [`docs/predictive-audit-2026-08-07.md`](docs/predictive-audit-2026-08-07.md)) established, by three independent tests, that **the model adds no measurable information over the bookmaker price**. The strongest evidence: sweeping the bookmaker blend weight improves out-of-sample log-loss monotonically all the way to `w = 1.0` — i.e. the best version of this model is the one that ignores the model.
>
> The settled record agrees: **1,048 settled picks on 878 fixtures — 51.3% win rate, −4.03% flat ROI** (95% CI `[−10.2%, +2.3%]`, cluster-bootstrapped by fixture; measured 2026-08-10).
>
> The system is therefore now a **frozen, instrumented experiment** rather than a betting tool. It records what it would have bet and measures whether those prices move in the right direction before the market closes (CLV). Real money stays off the table until **500 valid closing lines** exist and mean CLV is reliably positive. At the current rate that is ~3.5 months away.
>
> `betting.paper_trading_mode: true` — every pick is stamped `is_paper`, excluded from the live record, and the Telegram message carries a *"DO NOT BET REAL MONEY"* banner.

**Frozen model identity:** `stage5_baseline_20260807.485823` (`CODE_REVISION = s5.2`)

## How It Works

```
Flashscore (Camoufox) + API-Football + football-data.org + The Odds API
                              ↓
        PostgreSQL — Supabase (matches, odds, injuries, stats)
                              ↓
             Feature Engineering (14 sections, 80+ features)
                              ↓
    Ensemble Prediction (Poisson + Elo + ML + 80% bookmaker blend)
         + Bayesian per-league/per-market adaptive weights
                              ↓
      Value Calculation (EV, Kelly fraction, confidence filter)
         + drawdown circuit breaker + correlation filter
                              ↓
   SavedPick  +  pick_observations{model, final}   ← attribution recorded here,
                              ↓                       BEFORE the review can move it
   Claude pick review — web research → KEEP / CHANGE
         (correlation re-checked; a correlated switch is rejected)
                              ↓
       Telegram picks summary (with the PAPER TRADING banner)
                              ↓
   Closing-line capture, 2-hourly — resolves MODEL and FINAL independently
                              ↓
       Settlement (90-min score for knockouts) → auto-learning
           (live picks only — paper outcomes never feed the model)
                              ↓
       Paper-trading report — two CLV series, cluster-aware, checkpointed
```

## The Experiment

This is the part that matters now. Everything above feeds it.

| Concept | What it means |
|---|---|
| **Frozen model** | `model_version` is a hash of 23 prediction-affecting config keys plus a hand-bumped `CODE_REVISION`. Every pick is stamped with it, so cohorts from different configurations can never be silently pooled. The authoritative config is [`config/config.example.yaml`](config/config.example.yaml) — the file CI deploys. |
| **Paper/live isolation** | Paper picks are filtered out of ROI, backtests, EV-threshold tuning, ensemble-weight learning and both calibrators. **The experiment cannot retrain the MODEL. Until s5.3 it could, and did, inform the REVIEW** — see the correction below. |
| **Dual CLV attribution** | Two named series. **MODEL** = the frozen model's own selection and price. **FINAL** = what was actually taken after Claude's review. They are different bets on ~22% of picks and are never conflated. |
| **`pick_observations`** | One row per `(pick, attribution)`, written at pick-save time — *before* the review can overwrite `SavedPick.odds`. Without it the model's taken price is unrecoverable: the odds table keeps one row per (match, book, market, selection) and every refresh overwrites it. |
| **Same-snapshot rule** | A closing observation must come from odds observed **strictly after** the pick was taken. Otherwise a pick's own pricing row is returned as its "close" and CLV reads exactly 0.00% — an echo, not a measurement. |
| **Honest coverage** | `missing` / `late` / `invalid` / `unavailable` are recorded explicitly. No price is ever invented, substituted between series, or counted as zero CLV. |
| **Cluster-aware statistics** | 18.9% of fixtures carry two picks (31.8% of all picks), so confidence intervals resample **fixtures**, not picks. Effective sample size is printed beside every checkpoint. |
| **Checkpoints** | 100 (data quality only — no model decision) → 200 (initial CLV signal) → 500 (decision-grade). MODEL and FINAL are counted separately and never merged. |

**Known limitation:** the pre-kickoff refresh covers `h2h` + `totals` only, so ~36% of picks (Team Goals, BTTS, Double Chance) are structurally outside CLV measurement. Accepted deliberately — widening the request would multiply the credit cost.

Audit trail: [`docs/`](docs/) carries one report per stage, including the corrections where a later stage overturned an earlier diagnosis.


> **Correction (Stage 13, s5.3).** The claim above previously read: *"Paper
> picks are filtered out of ROI, backtests, EV-threshold tuning,
> ensemble-weight learning and both calibrators (11 filter sites). The
> experiment cannot retrain its own subject."*
>
> That was true of the **model** path and false of the **review** path.
> `match_briefing._recent_selection_stats` and `_recent_review_stats` computed
> win rates from settled picks with **no paper filter at all** — `is_paper`
> appeared zero times in that file — and injected them into the KEEP/CHANGE
> decision prompt. Paper-pick outcomes were therefore informing the review that
> produces the FINAL series the experiment measures.
> `probability_calibration.fit_from_db` excluded paper picks correctly but not
> picks whose features described a different club.
>
> All three are gated as of s5.3, the predicates now live in one place
> (`src/data/pick_filters.py`) so the next module imports rather than
> reinvents, and `tests/test_valid_evidence_gate.py` enforces both filters
> across every module — recognising every spelling of "settled", because
> scoping it to one spelling is exactly what hid this for three audits.
>
> **The count "11 filter sites" is deliberately not replaced with another
> number, and should not be.** A count of the sites that call a predicate
> cannot see the sites that do not — that arithmetic is precisely what produced
> the false claim, and a fresher count would carry the same defect with more
> authority. What replaces it is a guard that scans for readers rather than
> callers. If you find yourself wanting to put a number back here, put a test
> there instead.
>
> **The cause has now appeared twice, which makes it a pattern rather than an
> incident.** `feature_engineer` hand-copied `is_fixture == False` because the
> shared projection was somewhere it did not import from; `match_briefing`
> hand-wrote its paper filter because the predicate lived inside
> `betting_agent`. Both times the copy drifted from the original, both times a
> hand enumeration could not see it, and both times the fix was to move the
> definition somewhere every caller can reach.

## AI Pick Review (Claude)

Every saved pick is reviewed by Claude inside the `--picks` step, **before** the Telegram summary is sent.

- **What it does:** server-side web search for form, head-to-head, injuries by name, what's at stake and current prices, then a machine-readable **`KEEP` / `CHANGE`** decision from the priced menu at odds ≥ 1.50. On a CHANGE the saved pick is rewritten and the FINAL observation moves with it — the MODEL observation never does.
- **Correlation is re-checked.** A switch that would land on a selection correlated with another pick on the same match is rejected and falls back to KEEP. Every correlated pair ever seen in production was created here, including one pair the filter table already declared.
- **It bypasses the value gates by design.** `build_selection_pick` does not apply `min_expected_value`; measured over 90 days, 73% of CHANGE picks carried negative EV at the price taken. This is why the MODEL series exists separately.
- **Backends** (`briefings.backend`): `claude_code` (headless CLI on a Claude Pro subscription, **$0 API cost**) or `anthropic_api` (paid fallback when the Pro session limit is hit).
- **Fails safe:** no auth or a failed call → the review no-ops and the model's own pick is sent unchanged.

## Models

| Model | Algorithm | Role |
|---|---|---|
| **Poisson (Dixon-Coles)** | Score matrix | Match outcome probabilities; time-decay (540d half-life), xG-enhanced when ≥35% of matches have xG. `rho = 0.0` — the low-score correction was tuned to zero on clean data |
| **Elo** | Rating system | Team strength, home advantage in Elo points, season regression |
| **ML Classifier** | LR + Random Forest (+ XGBoost) | 1X2 classifier on the 14 feature sections; isotonic calibration |
| **GoalsMLModel** | Binary XGBoost/RF/LR | Over/Under 2.5, blended at 25% alongside Poisson |
| **Bookmaker blend** | Implied probability extraction | **80%** bookmaker / 20% model on both goals and 1X2. Raised from 40% → 60% → 80% by successive out-of-sample sweeps; the sweep is monotone to 100% |
| **Bayesian weight learner** | Hedge / multiplicative weights | Per-league and per-market adaptive ensemble weights, decayed log-loss, 90-day half-life |
| **De-vigging** | Gated cross-book consensus | Per-(match, book) overround validation before a book contributes — added after a market-corruption incident affected 7 books including Pinnacle |

Ensemble weights: XGBoost 35%, Poisson 25%, RF 20%, Elo 20%.

## Features (14 sections)

| Section | Key Features |
|---|---|
| **Team form** | Rolling 10-game home/away/overall win rate, goals for/against (exponential decay) |
| **Poisson strengths + Elo** | Attack/defence strengths, Elo differential |
| **Head-to-head** | H2H win rate, average goals, recent H2H form |
| **League position** | Rank difference, relegation gap, title gap |
| **International competition** | CL/EL/ECL experience and quality differential |
| **xG-based** | xG for/against, differential, over/underperformance |
| **Extended stats** | Dangerous attacks, saves, offsides, free kicks (Flashscore) |
| **Referee** | Cards/fouls per match, over-2.5 rate, avg yellow/red (7 features) |
| **Momentum** | RSI and MACD per side + differential |
| **Bookmaker implied probs** | 1X2, over/under, BTTS, team goals |
| **Odds movement** | Opening vs current % change, max absolute movement, direction |
| **Situational context** | Rest days, midweek flag, fatigue index (14/21/30d congestion) |
| **League statistics** | Home win rate, draw rate, avg goals, over-2.5 rate, BTTS rate |
| **Weather** | Temperature, wind, precipitation (Open-Meteo, optional) |

## Risk Management

| Mechanism | Description |
|---|---|
| **Paper trading mode** | Every pick flagged `is_paper`, excluded from the live record, Telegram banner. The master safety control. |
| **Gate registry — 4 on, 8 off** | Every gate is declared in [`gate_registry.py`](src/betting/gate_registry.py) with its origin, evidence and holdout verdict; `is_enabled()` refuses unknown names rather than silently disabling them. The 4 `risk` gates (`min_odds`, `max_odds`, `divergence_sanity`, `min_kelly_stake`) are a-priori and **on**. All 8 `edge` gates — derived from settled outcomes — are **off**: every verdict is `UNTESTABLE` or `INSUFFICIENT EVIDENCE`. |
| **Market correlation filter** | One predicate, two call sites (pre-persist and post-review). Covers 1X2 ↔ over/under, BTTS ↔ totals, Double Chance / DNB overlaps, and both same-direction and **opposite** rungs of a totals ladder |
| **Drawdown circuit breaker** | Scales stakes 100% → 0% as 30-pick ROI falls from −10% to −30%; pauses below −30% |
| **EV + confidence thresholds** | `min_ev: 5%`, `min_confidence: 55%`, sliding `EV × confidence ≥ 0.038` |
| **Model divergence guard** | Rejects picks where model prob / implied prob > 2.0× |
| **Model agreement scaling** | Kelly × unanimous 1.0, majority 0.80, split 0.60, solo/unknown 0.75 |
| **Extreme confidence dampening** | Above 90%, only 30% of the excess is retained (98% → 92.4%) |
| **Per-match / per-league caps** | Max 2 picks per match (enforced across reruns), 5 per league per day |
| **Odds range** | 1.50 – 10.0 |
| **Excluded markets** | `under_1.5`, `btts_no`, `over_3.5`. Enforced by `betting.excluded_markets` at three selection sites — a **separate mechanism** from the gate registry, which is why `over_3.5` is still excluded even though the registry's `exclude_over_3_5` gate defaults off. Under 2.5/3.5 were re-enabled 2026-08-02 by request despite negative settled ROI |
| **Evidence bar** | Any parameter change must clear a paired bootstrap on [`scripts/run_baseline.py`](scripts/run_baseline.py). Inconclusive means revert. |

## Active Leagues

30 leagues: the Top-5 (EPL, LaLiga, Bundesliga, Serie A, Ligue 1), Eredivisie, Primeira Liga, Jupiler Pro League, Süper Lig, Greek Super League, Austrian Bundesliga, Swiss Super League, Scottish Premiership, the Nordics (Allsvenskan, Eliteserien, Superliga, Veikkausliiga), Ekstraklasa, Liga 1, efbet League, six second divisions (Championship, League One, League Two, LaLiga2, 2. Bundesliga, Serie B, Ligue 2) and the three UEFA competitions.

The engine is league-agnostic; the active set is `scraping.flashscore_leagues`.

## Quick Start

### Prerequisites

- Python 3.11+
- Google Chrome + Camoufox (anti-fingerprint Firefox for Flashscore)
- API keys: [API-Football](https://www.api-football.com/) (free, 100 req/day), [football-data.org](https://www.football-data.org/) (free), [The Odds API](https://the-odds-api.com/) (free, 500 credits/month)
- Telegram bot token + chat ID
- PostgreSQL ([Supabase](https://supabase.com/) free tier) or SQLite fallback
- Optional: `CLAUDE_CODE_OAUTH_TOKEN` (Claude Pro, $0) and/or `ANTHROPIC_API_KEY` for the pick review

### Setup

```bash
git clone <repo>
cd betting-agent
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m camoufox fetch
cp config/config.example.yaml config/config.yaml
```

> `config/config.yaml` is **gitignored and never deployed** — CI copies `config.example.yaml` over it. If you edit the local file, keep every key in `model_version.TRACKED_KEYS` identical to the example or you are running a different model from production. `tests/test_config_identity.py` enforces this.

```bash
export API_FOOTBALL_KEY="..."      export FOOTBALL_DATA_ORG_KEY="..."
export ODDS_API_KEY="..."          export TELEGRAM_BOT_TOKEN="..."
export TELEGRAM_CHAT_ID="..."      export DATABASE_URL="postgresql://..."
export CLAUDE_CODE_OAUTH_TOKEN="..."   # optional
export ANTHROPIC_API_KEY="sk-ant-..."  # optional
```

Without `DATABASE_URL` the agent falls back to SQLite at `data/football_betting.db`.

### Database migrations

Apply `migrations/*.sql` in order before first run. All are additive.

| Migration | Adds |
|---|---|
| `001` | history mirror + indexes |
| `002` | concurrency safety — unique index `(match_id, selection, pick_date)` |
| `003` | closing-line columns |
| `004` | prospective measurement — model snapshot, market probability |
| `005` | experiment metadata — `model_version`, capture status, `is_paper` |
| `006` | `pick_observations` — dual MODEL/FINAL attribution |

### First Run

```bash
python -m src.agent.betting_agent --init
python -m src.agent.betting_agent --train
```

## CLI Reference

| Command | Description |
|---|---|
| `--init` | First-run setup: initialize DB, collect seed data |
| `--update [--skip-ml-retrain]` | Daily scrape: fixtures, odds, injuries, stats |
| `--update-results` | Scrape Flashscore results |
| `--picks [--force] [--leagues eng1,esp1]` | Generate picks → observations → Claude review → Telegram |
| `--picks --shard i/N --out FILE` / `--picks --collect DIR` | Sharded analysis, then one portfolio pass over the union |
| `--settle [--no-settle]` | Settle pending picks (90-min score for knockouts) + auto-learn |
| `--report` | Performance report to Telegram |
| `--stats` | ROI, win rate, Brier — live picks only |
| `--train` / `--tune` | Retrain ML / tune ensemble weights |
| `--analyze <id>` | Prediction breakdown for one match |
| `--backtest-rolling` | Rolling-origin backtest |
| `--backfill-wc` / `--backfill-history` / `--backfill-stats` | Historical backfills |
| `--briefing` / `--prematch-briefing [min]` | Manually post a narrative briefing article (not used by the pipeline) |
| `--telegram-setup` / `--telegram-test` / `--telegram-welcome` | Telegram helpers |

### Experiment scripts

| Script | Purpose |
|---|---|
| `python -m scripts.paper_trading_report [--days N] [--model-version V]` | The experiment's instrument: MODEL and FINAL CLV series, paired subset, review-action breakdown, coverage cross-tab, separate checkpoints. Read-only. |
| `python -m scripts.refresh_and_capture [--dry-run] [--status]` | Refresh imminent odds within quota, then capture closing lines |
| `python -m scripts.capture_closing_lines [--stats]` | Closing capture on its own |
| `python -m scripts.simulate_odds_quota` | Replay fixture history against the credit model |
| `python -m scripts.run_baseline` / `run_clean_baseline` | Immutable evaluation baselines |

## Configuration Reference

```yaml
betting:
  paper_trading_mode: true          # PAPER ONLY — see the banner at the top
  min_expected_value: 0.05
  min_confidence: 0.55
  min_ev_confidence_score: 0.038    # sliding scale: EV × confidence
  kelly_fraction: 0.25
  max_stake_percentage: 4.0
  max_picks_per_league: 5
  max_total_kelly_pct: 0            # daily portfolio cap (0 = disabled)
  min_odds: 1.50
  max_odds: 10.0
  excluded_markets: [under_1.5, btts_no, over_3.5]
  gates:                            # overrides gate_registry defaults.
    club_btts_yes_ban: false        # all 8 `edge` gates are OFF — none
    club_pick_min_ev: false         # survived holdout validation. The 4
    club_pick_min_blend: false      # a-priori `risk` gates stay ON and are
    split_agreement_low_conf: false # not listed here.

models:
  ensemble_weights: { poisson: 0.25, xgboost: 0.35, random_forest: 0.20, elo: 0.20 }
  bookmaker_blend_weight: 0.80      # 80% bookmaker / 20% model
  goals_ml_blend_weight: 0.25
  extreme_confidence_ceiling: 0.90
  dixon_coles_rho: 0.0              # tuned to zero on clean data
  strength_half_life_days: 540
  poisson_use_xg: true
  poisson_xg_min_coverage: 0.35
  probability_calibration_enabled: false   # dormant; predictions already calibrated
  bayesian_weight_half_life_days: 90
  ml_retrain_days: 3
  drawdown_lookback_picks: 30
  drawdown_reduce_threshold: -0.10
  drawdown_pause_threshold: -0.30

odds_api:
  monthly_credit_budget: 400        # of a 500 free tier
  safety_margin_credits: 50

briefings:
  enabled: true
  finalize_picks: true              # Claude makes the final KEEP/CHANGE call
  send_to_telegram: false           # decision-only, no article
  backend: claude_code              # claude_code (Pro, $0) or anthropic_api (paid)
```

Changing any key in `TRACKED_KEYS` changes `model_version` and starts a new experimental cohort. Full reference: [`config/config.example.yaml`](config/config.example.yaml).

## CI/CD Pipeline

Three scheduled workflows on `main`. All times UTC.

| Workflow | Schedule | Does |
|---|---|---|
| **Daily Betting Picks** | `37 9 * * *` | update → settle → train → picks (+ Claude review) → update-results → settle → Sunday report |
| **Closing Line Capture** | `17 11,13,15,17,19,21,23 * * *` | refresh imminent odds within quota → capture MODEL and FINAL closing lines |
| **Paper Trading Report** | `47 10 * * *` | the experiment's instrument; read-only, zero API credits |

Each has its own `concurrency` group (queue, never cancel). Every core step in the daily job uses `continue-on-error`, and a dedicated failure-alert step sends **one** Telegram message if anything failed — a green-but-broken run is never silent. Tests run in CI with `conftest.py` stripping `DATABASE_URL`, so they can never touch production.

Closing capture is separate from the daily job because the cadences are irreconcilable: predictions are once-daily, but a closing line must be taken within 180 minutes of each kickoff across a 12-hour evening window. Folding it into the daily run would collect nothing.

### Odds API quota

Claim-before-spend ledger in the `api_budget` table, reconciled against the provider's own `x-requests-used` header (they diverged by 95 credits once).

```
credits = requests × regions × markets = requests × 1 × 2   (eu; h2h,totals)

worst case:  min(7 runs/day × 24-credit ceiling × 31,  400 budget − 50 margin)
             = 350 credits/month   vs 500 free tier   → 30% headroom
measured:    212 credits/month, worst observed day 46
```

A credit is spent only on leagues with an imminent fixture **and** a pick awaiting a close — cheaper *and* better covered than refreshing every league (212 cr at 88% vs 340 cr at 85%).

### Required GitHub Secrets

`API_FOOTBALL_KEY`, `FOOTBALL_DATA_ORG_KEY`, `ODDS_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `DATABASE_URL`, and optionally `CLAUDE_CODE_OAUTH_TOKEN` / `ANTHROPIC_API_KEY`.

## Automated Learning Pipeline

After every `--settle`, on **live picks only**:

1. Refit Poisson/Elo with time decay
2. Tune ensemble weights (Bayesian, per league/market)
3. Update per-model calibration factors
4. Persist the auto-calibrated EV threshold
5. Retrain ML if models are >3 days old

Paper picks are excluded from all five. That isolation is what makes the frozen experiment trustworthy — without it the experiment's own output would rewrite the model it measures.

## Data Sources

| Source | Data | Limit |
|---|---|---|
| [Flashscore](https://www.flashscore.com/) | Fixtures, results, extended stats, referee, venue | Scraped (Camoufox + Chrome/Xvfb) |
| [API-Football](https://www.api-football.com/) | Fixture IDs, results, xG, odds, injuries, backfill | 100 req/day (free; seasons 2022–2024) |
| [football-data.org](https://www.football-data.org/) | Fixtures + results, 9 top leagues | 10 req/min (free) |
| [The Odds API](https://the-odds-api.com/) | 1X2 + Over/Under, incl. the pre-kickoff closing refresh | 500 credits/month (free) |
| [football-data.co.uk](https://football-data.co.uk/) | Historical results + Bet365/Pinnacle odds from 2016 | CSV, no key |
| [Open-Meteo](https://open-meteo.com/) | Match-day weather | No key |
| **Claude** (Anthropic) | Web research + KEEP/CHANGE ruling per pick | Pro subscription ($0) or paid API |

## Database

Supabase serverless PostgreSQL (free tier); SQLite fallback when `DATABASE_URL` is unset. Queries are column-projected throughout — a Stage-3 egress pass cut usage ~97%.

| Table | Purpose |
|---|---|
| `teams` | Team registry with API-Football ID mapping |
| `matches` | Scores, regulation (90') score, xG, referee, venue, fixture/result flag |
| `odds` | Bookmaker odds, `opening_odds` frozen at first sight. **Unique on (match_id, bookmaker, market_type, selection)** — one row per book, overwritten on refresh, so there is no price history |
| `injuries` / `players` | Injury status fetched before picks |
| `saved_picks` | Pick, odds, EV, Kelly, result — plus `is_paper`, `model_version`, the pre-review model snapshot, market probability and closing-capture status |
| `pick_observations` | One row per `(pick, attribution)`: `model` or `final`, with its own taken price and closing observation |
| `api_budget` | Claim-before-spend credit ledger, per provider per period |

**Odds pruning:** rows older than 400 days are deleted; odds for matches with saved picks are preserved.

**numpy-safe writes:** psycopg2 adapters are registered at startup — a numpy 2.x scalar in SQL otherwise raises a phantom `schema "np" does not exist`.

## Settlement Notes

- **90-minute grading:** knockouts going to extra time or penalties are settled on the **regulation score** (bookmaker convention). The final score is kept for display.
- **Stuck-pick sweeper:** picks whose match never produced a result are voided after ~10 days.

## Telegram Notifications

| Message | Trigger |
|---|---|
| **Daily picks** | `--picks` — grouped by league, with the **PAPER TRADING — DO NOT BET REAL MONEY** banner while paper mode is on |
| **Settlement report** | `--settle` — record, ROI, P/L |
| **Performance report** | `--report` or the Sunday CI run |
| **Failure alerts** | CI failure-check steps in all three workflows |

Reports show genuine CLV **only** when a validated closing line exists. Model-probability-minus-implied-probability is reported under its own name as model-vs-market divergence — it was once sent as "Avg CLV" and read +6.3%, which was not CLV.

## Performance Metrics

`--stats` / `--report` (live picks only): win rate and ROI all-time / 7d / 30d / by market / by league, Brier score, log loss, model-agreement tags, drawdown status.

The **paper-trading report** is the experiment's real instrument and reports separately: MODEL and FINAL CLV with cluster-bootstrapped CIs, effective sample size and design effect, the paired subset (`final − model`, reported as an observed difference, never a causal claim), a `none`/`KEEP`/`CHANGE` breakdown, a coverage cross-tab, and independent MODEL/FINAL checkpoint counters.

## Project Structure

```
src/
├── agent/betting_agent.py           # Orchestrator, CLI, learning pipeline, observation writes
├── models/
│   ├── ensemble.py                  # Weighted ensemble + bookmaker blend + calibration
│   ├── poisson_model.py             # Dixon-Coles, xG-enhanced, time-decayed
│   ├── ml_models.py                 # LR + RF + XGBoost (1X2 + over/under)
│   ├── elo_system.py                # Elo with home advantage + season regression
│   ├── bayesian_weights.py          # Hedge weight learner, market-qualified scopes
│   ├── probability_calibration.py   # Isotonic calibration (dormant; live picks only)
│   └── model_version.py             # TRACKED_KEYS fingerprint + CODE_REVISION
├── features/                        # 14-section pipeline, weather, injuries, H2H
├── scrapers/                        # Flashscore, API-Football, football-data.org, The Odds API
├── betting/
│   ├── value_calculator.py          # EV, Kelly, model agreement, market specs
│   └── gate_registry.py             # Every outcome-derived gate + its holdout verdict
├── evaluation/
│   ├── clv.py                       # Closing-line validity rules and coverage
│   ├── attribution.py               # MODEL / FINAL series resolution
│   ├── clean_dataset.py             # Corruption-free evaluation dataset
│   └── baseline.py                  # Immutable baseline harness
├── data/
│   ├── database.py                  # SQLAlchemy manager, numpy adapters, pruning
│   ├── models.py                    # ORM incl. PickObservation, ApiBudget
│   ├── market_spec.py               # Market arity, overround, de-vigging
│   ├── odds_quota.py                # Odds API credit ledger
│   └── api_budget.py                # Claim-before-spend primitive
└── reporting/                       # Claude review, Telegram, dashboards
scripts/                             # Capture, refresh, reports, baselines, migrations
migrations/                          # 001–006 + rollbacks
docs/                                # Stage 1–11 audit reports
```

## Tests

```bash
pytest -q      # 626 tests
```

`tests/conftest.py` strips `DATABASE_URL` so DB-backed tests always use a temp SQLite database. Two AST regression tests guard against a script calling `load_dotenv()` at import time — that once let a unit test write to production.

Notable suites: `test_experiment_invariants.py` (the 10 invariants the experiment rests on), `test_dual_clv_attribution.py`, `test_pick_observations.py`, `test_pick_save_atomicity.py`, `test_paper_live_isolation.py`, `test_config_identity.py`, `test_odds_quota_and_refresh.py`.

## Requirements

- Python 3.11+
- Google Chrome / Chromium + Xvfb
- Camoufox (anti-fingerprint Firefox)
- See [`requirements.txt`](requirements.txt)
