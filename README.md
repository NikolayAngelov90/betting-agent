# Football Betting Agent

An automated football betting prediction system that combines statistical models, machine learning, and real-time data scraping to identify value bets across 21+ European leagues. Runs daily via GitHub Actions with a Neon PostgreSQL database and delivers picks to Telegram.

## How It Works

```
Flashscore (Camoufox) + API-Football + football-data.org
                    ↓
     Neon PostgreSQL (matches, odds, stats)
                    ↓
        Feature Engineering (14 sections)
                    ↓
   Ensemble Prediction (Poisson + Elo + ML + bookmaker blend)
        + Bayesian per-league adaptive weights
                    ↓
     Value Calculation (EV, Kelly fraction, confidence filter)
        + Drawdown circuit breaker + contrarian signals
                    ↓
            Telegram picks message
                    ↓
    Settlement → Automated learning pipeline
        (refit models, tune weights, recalibrate)
                    ↓
         Performance reporting (weekly)
```

## Models

| Model | Role |
|---|---|
| **Poisson (Dixon-Coles)** | Score matrix with low-score correction (rho=-0.13), time-decay (180d half-life), xG-enhanced when data available |
| **Elo** | Team strength ratings updated after every match |
| **XGBoost / Random Forest** | 1X2 outcome classifier trained on 14 feature sections |
| **GoalsMLModel** | Binary over/under 2.5 classifier blended at 25% weight |
| **Bookmaker blend** | Implied probabilities from real odds mixed at 40% for both goals and 1X2 markets |
| **Bayesian weight learner** | Per-league adaptive ensemble weights with Beta-distribution priors and temporal decay |

## Features (14 sections)

- Rolling form with exponential decay
- Poisson attack/defence strength + Elo ratings
- Bookmaker implied probabilities (1X2, over/under, BTTS, team goals)
- Head-to-head records
- Rest days, midweek flag, match stakes index
- xG tracking
- RSI/MACD momentum indicators
- League statistics (home win rate, avg goals, over 2.5 rate)
- Referee strictness (cards/fouls per match, over 2.5 rate)
- Injury reports
- Match-day weather via Open-Meteo (temperature, wind, precipitation)
- Cumulative fatigue index (14d/21d/30d match congestion, short rest count)
- Odds movement detection (opening vs current odds, % change across markets)
- Contrarian signal detection (model-vs-market divergence)

## Risk Management

- **Drawdown circuit breaker** — scales stakes 0-100% based on recent ROI (linear ramp between -10% and -30% thresholds)
- **Market correlation filtering** — drops the lower-EV pick when two picks on the same match are positively correlated
- **Daily exposure limits** — caps total Kelly exposure at 40% of bankroll per day
- **EV threshold auto-calibration** — adjusts minimum edge based on recent hit rate (persisted between runs)
- **Extreme confidence dampening** — caps probabilities at 90%, retaining 30% of excess
- **Per-match pick cap** — max 2 picks per match across all runs in a day

## Covered Leagues

**Top 5 + second divisions:** Premier League, Championship, La Liga, La Liga 2, Bundesliga, 2. Bundesliga, Serie A, Serie B, Ligue 1, Ligue 2

**Strong European:** Eredivisie, Primeira Liga, Jupiler Pro League, Super Lig, Scottish Premiership, Swiss Super League, Ekstraklasa, and more

**European competitions:** Champions League, Europa League, Europa Conference League

## Quick Start

### Prerequisites

- Python 3.10+
- Google Chrome + Camoufox (for Flashscore scraping)
- API keys: [API-Football](https://www.api-football.com/) (free, 100 req/day) and [football-data.org](https://www.football-data.org/client/register) (free)
- Telegram bot token + chat ID (optional but recommended)
- PostgreSQL database (recommended: [Neon](https://neon.tech/) free tier) or SQLite fallback

### Setup

```bash
git clone <repo>
cd betting-agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m camoufox fetch  # Download anti-fingerprint Firefox binary
```

### Configuration

```bash
cp config/config.example.yaml config/config.yaml
```

Set environment variables (override config file values):

```bash
export API_FOOTBALL_KEY="your_key"
export FOOTBALL_DATA_ORG_KEY="your_key"
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"
export DATABASE_URL="postgresql://user:pass@host/db"  # Neon connection string
```

If `DATABASE_URL` is not set, the agent falls back to a local SQLite database at `data/football_betting.db`.

### First Run

```bash
# Initialize database and collect historical data
python -m src.agent.betting_agent --init

# Train ML models
python -m src.agent.betting_agent --train
```

### Daily Workflow

```bash
python -m src.agent.betting_agent --settle   # Settle yesterday's pending picks + auto-learn
python -m src.agent.betting_agent --update   # Scrape fixtures, results, odds
python -m src.agent.betting_agent --train    # Retrain ML models if stale (>3 days)
python -m src.agent.betting_agent --picks    # Generate and send today's picks
```

## CLI Reference

```
python -m src.agent.betting_agent <command>
```

| Command | Description |
|---|---|
| `--init` | First-run setup: initialize DB and collect seed data |
| `--update` | Daily scrape: fixtures, results, odds, stats enrichment |
| `--update --skip-ml-retrain` | Daily scrape without ML retraining (used in CI) |
| `--picks` | Generate value picks, save to DB, send to Telegram |
| `--picks --leagues eng1,esp1` | Filter picks by league codes |
| `--settle` | Settle pending picks + run automated learning pipeline |
| `--report` | Send comprehensive performance report to Telegram |
| `--stats` | Print prediction stats (ROI, win rate, Brier score, CLV) |
| `--train` | Retrain ML models on all historical data |
| `--tune` | Tune ensemble weights from recent results |
| `--analyze <id>` | Deep-dive prediction breakdown for a specific match ID |
| `--backfill-history` | Fetch historical stats for low-coverage teams via API-Football |
| `--telegram-setup` | Interactive Telegram bot configuration guide |
| `--telegram-test` | Send a test message to verify Telegram is working |

## Configuration Reference

Key parameters in `config/config.yaml`:

```yaml
betting:
  min_expected_value: 0.03         # Minimum 3% edge required
  min_confidence: 0.58             # Minimum model probability (58%)
  kelly_fraction: 0.25             # Fractional Kelly (25% of full Kelly)
  max_stake_percentage: 4.0        # Hard cap: max 4% of bankroll per bet
  max_total_kelly_pct: 40.0        # Daily exposure limit (% of bankroll)

models:
  bookmaker_blend_weight: 0.40     # 40% bookmaker / 60% model for all markets
  goals_ml_blend_weight: 0.25      # 25% GoalsMLModel for over/under
  extreme_confidence_ceiling: 0.90 # Dampen probabilities above 90%
  dixon_coles_rho: -0.13           # Low-score correction strength
  strength_half_life_days: 180     # Time-decay: match 6mo ago = half weight
  weather_features_enabled: true   # Open-Meteo weather features
  poisson_use_xg: true             # Use xG instead of raw goals when available
  bayesian_weight_half_life_days: 90
  bayesian_prior_strength: 10
  ml_retrain_days: 3               # Auto-retrain ML after N days
  drawdown_lookback_picks: 30
  drawdown_reduce_threshold: -0.10
  drawdown_pause_threshold: -0.30
  ev_calibration_lookback: 40
```

Full reference: [`config/config.example.yaml`](config/config.example.yaml)

## CI/CD Pipeline

The agent runs automatically via GitHub Actions every day at **06:30 UTC** (~09:30 Kyiv summer / ~08:30 Kyiv winter).

**Pipeline steps:**

1. Restore ML models + historical cache from rolling cache
2. Install Chrome, Xvfb, Camoufox (for Flashscore scraping)
3. Verify Neon PostgreSQL connection
4. Run test suite (`pytest tests/`)
5. `--settle` — settle pending picks + trigger automated learning
6. `--update --skip-ml-retrain` — scrape fixtures, results, odds (ML deferred)
7. `--train` — retrain ML models in dedicated step (15 min timeout)
8. DB health check (match counts, fixture counts, pending picks)
9. `--picks` — generate picks and send to Telegram
10. `--report` — weekly performance report (Sundays only)
11. Save ML models cache for next run
12. Notify Telegram on failure

**Timeouts:** settle 10 min, update 30 min, train 15 min, picks 45 min, job 90 min total.

**Required GitHub Secrets:**

| Secret | Description |
|---|---|
| `API_FOOTBALL_KEY` | API-Football free tier key |
| `FOOTBALL_DATA_ORG_KEY` | football-data.org free tier key |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token |
| `TELEGRAM_CHAT_ID` | Target chat/channel ID |
| `DATABASE_URL` | Neon PostgreSQL connection string |

## Automated Learning Pipeline

After `--settle`, the agent automatically runs a learning pipeline:

1. **Refit Poisson/Elo** — update team strength ratings with newly settled results
2. **Tune ensemble weights** — Bayesian weight updates per league + model calibration factors
3. **Persist EV threshold** — save auto-calibrated minimum edge to `data/models/ev_threshold.json`
4. **Retrain ML** — if models are stale (>3 days), defer to `--train` step

This ensures the system continuously adapts without manual intervention.

## Data Sources

| Source | Usage | Limit |
|---|---|---|
| [Flashscore](https://www.flashscore.com/) | Fixtures, results, live odds, match stats | Scraped (Camoufox + Chrome fallback via Xvfb) |
| [API-Football](https://www.api-football.com/) | Odds, fixture IDs, team history, xG data | 100 req/day (free tier) |
| [football-data.org](https://www.football-data.org/) | Fixtures + results for 9 top leagues | No daily limit, 10 req/min (free) |
| [Open-Meteo](https://open-meteo.com/) | Match-day weather forecasts | No limit, no key required |

## Project Structure

```
src/
├── agent/
│   └── betting_agent.py          # Main orchestrator + CLI entry point
├── models/
│   ├── ensemble.py               # EnsemblePredictor (weighted blend + calibration)
│   ├── poisson_model.py          # Dixon-Coles Poisson (xG-enhanced, time-decay)
│   ├── ml_models.py              # XGBoost/RF classifiers (1X2 + goals)
│   ├── elo_system.py             # Elo ratings
│   └── bayesian_weights.py       # Per-league adaptive ensemble weights
├── features/
│   ├── feature_engineer.py       # 14-section feature pipeline
│   ├── h2h_features.py           # Head-to-head feature extraction
│   ├── team_features.py          # Team-level feature extraction
│   ├── injury_features.py        # Injury impact features
│   └── weather_service.py        # Open-Meteo integration
├── scrapers/
│   ├── flashscore_scraper.py     # Camoufox-based Flashscore scraper
│   ├── apifootball_scraper.py    # API-Football client
│   ├── footballdataorg_scraper.py # football-data.org client
│   ├── historical_loader.py      # Historical data backfill
│   ├── injury_scraper.py         # Injury data scraper
│   └── news_scraper.py           # News + sentiment scraper
├── betting/
│   ├── value_calculator.py       # EV + Kelly sizing + contrarian signals
│   └── bankroll_manager.py       # Bankroll tracking
├── reporting/
│   ├── telegram_bot.py           # Picks + settlement + performance reports
│   ├── dashboard.py              # Dashboard utilities
│   └── match_report.py           # Match analysis reports
└── data/
    ├── database.py               # SQLAlchemy + PostgreSQL/SQLite management
    └── models.py                 # Match, Team, Odds, SavedPick, Prediction
scripts/
    └── migrate_to_neon.py        # SQLite → PostgreSQL migration (COPY protocol)
config/
    └── config.example.yaml       # Full configuration reference
```

## Performance Metrics

Track via `--stats` or Telegram `--report`:

- Win rate and ROI by period, market, and league
- Brier score (probability calibration)
- Closing Line Value (CLV) — compares pick odds to closing odds
- Model agreement voting (consensus tags on each pick)
- Calibration buckets (predicted probability vs actual frequency)
- Drawdown tracking and circuit breaker status

## Requirements

- Python 3.10+
- Chrome / Chromium + Xvfb (for Selenium fallback scraping)
- Camoufox (anti-fingerprint Firefox, primary scraper)
- See [`requirements.txt`](requirements.txt) for full dependency list
