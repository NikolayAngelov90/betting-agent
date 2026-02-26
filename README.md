# Football Betting Agent

An automated football betting prediction system that combines statistical models, machine learning, and real-time data scraping to identify value bets across 21+ European leagues. Runs daily via GitHub Actions and delivers picks to Telegram.

## How It Works

```
Flashscore + API-Football + football-data.org
         ↓
   SQLite (matches, odds, stats)
         ↓
   Feature Engineering (14 sections)
         ↓
   Ensemble Prediction (Poisson + Elo + ML + bookmaker blend)
         ↓
   Value Calculation (EV, Kelly fraction, confidence filter)
         ↓
   Telegram picks message
         ↓
   Settlement + Performance reporting
```

## Models

| Model | Role |
|---|---|
| **Poisson (Dixon-Coles)** | Score matrix with low-score correction (ρ=−0.13) and time-decay |
| **Elo** | Team strength ratings updated after every match |
| **XGBoost / Random Forest** | 1X2 outcome classifier trained on 14 feature sections |
| **GoalsMLModel** | Binary over/under 2.5 classifier blended at 25% weight |
| **Bookmaker blend** | Implied probabilities from real odds mixed in at 40% for goals markets |

## Features (14 sections)

- Rolling form with exponential decay
- Poisson attack/defence strength + Elo ratings
- Bookmaker implied probabilities (1X2, over/under, BTTS, team goals)
- Head-to-head records
- Rest days, midweek flag, match stakes index
- xG tracking
- RSI/MACD momentum
- League statistics (home win rate, avg goals, over 2.5 rate)
- Referee strictness (cards/fouls per match)
- Injury reports
- Match-day weather via Open-Meteo (temperature, wind, precipitation)

## Covered Leagues

**Top 5 + second divisions:** Premier League, Championship, La Liga, Bundesliga, Serie A, Ligue 1 + their second tiers

**Strong European:** Eredivisie, Primeira Liga, Jupiler Pro League, Süper Lig, Scottish Premiership, Ekstraklasa, and more

**European competitions:** Champions League, Europa League, Europa Conference League

## Quick Start

### Prerequisites

- Python 3.10+
- Google Chrome (for Flashscore scraping)
- API keys: [API-Football](https://www.api-football.com/) (free, 100 req/day) and [football-data.org](https://www.football-data.org/client/register) (free)
- Telegram bot token + chat ID (optional but recommended)

### Setup

```bash
git clone <repo>
cd betting-agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

```bash
cp config/config.example.yaml config/config.yaml
```

Edit `config/config.yaml` and fill in your API keys, or set them as environment variables (env vars override the config file):

```bash
export API_FOOTBALL_KEY="your_key"
export FOOTBALL_DATA_ORG_KEY="your_key"
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

### First Run

```bash
# Initialize database and collect historical data
python -m src.agent.betting_agent --init

# Train ML models
python -m src.agent.betting_agent --train
```

### Daily Workflow

```bash
python -m src.agent.betting_agent --update   # Scrape fixtures, results, odds
python -m src.agent.betting_agent --settle   # Settle yesterday's pending picks
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
| `--picks` | Generate value picks, save to DB, send to Telegram |
| `--picks --leagues eng1,esp1` | Filter picks by league codes |
| `--settle` | Match pending picks against actual results |
| `--report` | Send comprehensive performance report to Telegram |
| `--stats` | Print prediction stats (ROI, win rate, Brier score, CLV) |
| `--train` | Retrain ML models on all historical data |
| `--tune` | Tune ensemble weights from recent results (last 30 days) |
| `--analyze <id>` | Deep-dive prediction breakdown for a specific match ID |
| `--backfill-history` | Fetch historical stats for low-coverage teams via API-Football |
| `--telegram-setup` | Interactive Telegram bot configuration guide |
| `--telegram-test` | Send a test message to verify Telegram is working |

## Configuration Reference

Key parameters in `config/config.yaml`:

```yaml
betting:
  min_expected_value: 0.03    # Minimum 3% edge required
  min_confidence: 0.58        # Minimum model probability (58%)
  kelly_fraction: 0.25        # Fractional Kelly (25% of full Kelly)
  max_stake_percentage: 4.0   # Hard cap: max 4% of bankroll per bet

models:
  bookmaker_blend_weight: 0.40      # 40% bookmaker / 60% Poisson for goals markets
  goals_ml_blend_weight: 0.25       # 25% GoalsMLModel for over/under
  extreme_confidence_ceiling: 0.90  # Dampen probabilities above 90%
  dixon_coles_rho: -0.13            # Low-score correction strength
  strength_half_life_days: 180      # Time-decay: match 6mo ago = half weight
  weather_features_enabled: true    # Open-Meteo weather features
```

Full reference: [`config/config.example.yaml`](config/config.example.yaml)

## CI/CD Pipeline

The agent runs automatically via GitHub Actions every day at **08:30 UTC** (~11:00 Kyiv time).

**Pipeline steps:**

1. Restore SQLite DB and ML models from rolling cache
2. Verify DB integrity (auto-recover on corruption)
3. Run test suite (`pytest tests/`)
4. `--settle` pending picks from previous days
5. `--update` — scrape fixtures, results, and odds
6. `--train` — retrain ML models (skipped if < 100 matches)
7. `--picks` — generate picks and send to Telegram
8. Save DB and models to cache for next run
9. Export DB + models as downloadable artifact (7-day retention)

**Required GitHub Secrets:**

| Secret | Description |
|---|---|
| `API_FOOTBALL_KEY` | API-Football free tier key |
| `FOOTBALL_DATA_ORG_KEY` | football-data.org free tier key |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token |
| `TELEGRAM_CHAT_ID` | Target chat/channel ID |

## Data Sources

| Source | Usage | Limit |
|---|---|---|
| [Flashscore](https://www.flashscore.com/) | Fixtures, results, live odds, match stats | Scraped (headless Chrome) |
| [API-Football](https://www.api-football.com/) | Odds fallback, fixture IDs, team history | 100 req/day (free tier) |
| [football-data.org](https://www.football-data.org/) | Fixtures + results for 9 top leagues | No daily limit, 10 req/min (free) |
| [Open-Meteo](https://open-meteo.com/) | Match-day weather forecasts | No limit, no key required |

## Project Structure

```
src/
├── agent/
│   └── betting_agent.py       # Main orchestrator + CLI entry point
├── models/
│   ├── ensemble.py            # EnsemblePredictor
│   ├── poisson_model.py       # Dixon-Coles Poisson model
│   ├── ml_models.py           # XGBoost/RF classifiers
│   └── elo_system.py          # Elo ratings
├── features/
│   ├── feature_engineer.py    # 14-section feature pipeline
│   └── weather_service.py     # Open-Meteo integration
├── scrapers/
│   ├── flashscore_scraper.py  # Chrome-based Flashscore scraper
│   ├── apifootball_scraper.py # API-Football client
│   └── footballdataorg_scraper.py
├── betting/
│   ├── value_calculator.py    # EV + Kelly sizing
│   └── bankroll_manager.py
├── reporting/
│   └── telegram_bot.py        # Daily picks + settlement + performance reports
└── data/
    ├── database.py            # SQLAlchemy session management
    └── models.py              # Match, Team, Odds, SavedPick, Prediction
```

## Performance Metrics

Track via `--stats` or Telegram `--report`:

- Win rate and ROI by period, market, and league
- Brier score (probability calibration)
- Cumulative Loss Value (CLV) — compares pick odds to closing odds
- Model agreement voting (consensus tags on each pick)
- Calibration buckets (predicted probability vs actual frequency)

## Requirements

- Python 3.10+
- Chrome / Chromium (for Selenium scraping)
- See [`requirements.txt`](requirements.txt) for full dependency list
