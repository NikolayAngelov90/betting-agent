# Football Betting Agent

An automated football betting prediction system combining statistical models, machine learning, and multi-source data scraping to identify value bets across 28 European leagues. Runs daily via GitHub Actions with a Neon PostgreSQL database and delivers picks to Telegram.

## How It Works

```
Flashscore (Camoufox) + API-Football + football-data.org + The Odds API
                              ↓
          Neon PostgreSQL (matches, odds, injuries, stats)
                              ↓
             Feature Engineering (14 sections, 80+ features)
                              ↓
    Ensemble Prediction (Poisson + Elo + ML + bookmaker blend)
         + Bayesian per-league adaptive weights
         + Model calibration factors
                              ↓
      Value Calculation (EV, Kelly fraction, confidence filter)
         + Drawdown circuit breaker
         + Market correlation filtering
         + Daily exposure cap
                              ↓
               Telegram picks message
                              ↓
       Settlement → Automated learning pipeline
           (refit models, tune weights, recalibrate EV threshold)
                              ↓
            Performance reporting (weekly, Sundays)
```

## Models

| Model | Algorithm | Role |
|---|---|---|
| **Poisson (Dixon-Coles)** | Score matrix | Match outcome probabilities; low-score correction (rho=-0.13), exponential time-decay (180d half-life), xG-enhanced when ≥50% of matches have xG data |
| **Elo** | Rating system | Team strength ratings (K=32, home advantage=65 Elo pts), season regression, used for 1X2 probability conversion |
| **ML Classifier** | Logistic Regression + Random Forest (+ optional XGBoost) | 1X2 outcome classifier trained on 14 feature sections; isotonic calibration; ensemble weights: XGBoost 35%, RF 20%, LR via calibration |
| **GoalsMLModel** | Binary XGBoost/RF/LR | Over/Under 2.5 goals; blended at 25% weight alongside Poisson |
| **Bookmaker blend** | Implied probability extraction | Real bookmaker probabilities mixed at 40% into both goals and 1X2 markets to anchor predictions to market consensus |
| **Bayesian weight learner** | Beta-distribution temporal decay | Per-league and per-market adaptive ensemble weights with global prior; half-life 90 days, persisted to `data/models/bayesian_weights.json` |

## Features (14 sections)

| Section | Key Features |
|---|---|
| **Team form** | Rolling 10-game home/away/overall win rate, goals scored/conceded (exponential decay) |
| **Poisson strengths + Elo** | Attack/defence strength ratings, Elo rating differential |
| **Head-to-head** | H2H win rate, average goals, recent form in H2H |
| **League position** | Table rank difference, relegation gap, title gap |
| **International competition** | CL/EL/ECL experience and quality differential, is_international_match flag |
| **xG-based** | xG for/against averages, xG differential, xG over/underperformance |
| **Extended stats** | Dangerous attacks, goalkeeper saves, offsides, free kicks (Flashscore) |
| **Referee** | Cards/fouls per match, over 2.5 rate, avg yellow/red cards (7 features) |
| **Momentum** | RSI and MACD indicators for home and away teams; RSI/MACD differential |
| **Bookmaker implied probs** | 1X2, over/under, BTTS, team goals implied probabilities from real odds |
| **Odds movement** | Opening vs current odds % change for home/away/over 2.5; max abs movement; direction signal |
| **Situational context** | Rest days, midweek flag, fatigue index (14d/21d/30d congestion), short rest count |
| **League statistics** | Home win rate, draw rate, avg goals, over 2.5 rate, BTTS rate |
| **Weather** | Temperature, wind speed, precipitation, is_raining, is_windy (Open-Meteo, optional) |

## Risk Management

| Mechanism | Description |
|---|---|
| **Drawdown circuit breaker** | Scales stakes linearly from 100% → 0% as recent ROI (30-pick window) drops from -10% to -30%; pauses all picks below -30% |
| **Market correlation filter** | Detects positively-correlated picks on the same match (e.g. Home Win + Over 2.5); drops the lower-EV one |
| **Daily exposure cap** | Trims lowest-EV picks until total Kelly exposure stays ≤ 40% of bankroll |
| **EV + confidence thresholds** | `min_ev: 3%`, `min_confidence: 58%`; sliding scale: `EV × confidence ≥ 0.035`; hard floor 45% confidence |
| **Model divergence guard** | Rejects picks where model probability / implied probability > 2.0× (prevents miscalibration outliers) |
| **Model agreement scaling** | Kelly fraction scaled: unanimous 1.0×, majority 0.80×, split 0.60× |
| **EV auto-calibration** | Hit rate > 60% tightens min_EV (up to +2pp); < 45% loosens (−0.5pp max); persisted across runs |
| **Extreme confidence dampening** | Probabilities above 90% retain only 30% of excess (e.g. 98% → 92.4%) |
| **Per-match pick cap** | Max 2 picks per match per day, enforced across multiple `--picks` runs |
| **Odds range filter** | Only bets with odds between 1.30 and 10.0 |
| **Per-league cap** | Max 5 picks per league per day |

## Covered Leagues (28)

**Top 5:** Premier League, La Liga, Bundesliga, Serie A, Ligue 1

**English lower divisions:** Championship, League One, League Two

**Second divisions:** La Liga 2, 2. Bundesliga, Serie B, Ligue 2

**Strong European:** Eredivisie, Primeira Liga, Jupiler Pro League, Super Lig, Scottish Premiership, Austrian Bundesliga, Swiss Super League, Greek Super League, Danish Superliga, Romanian Liga 1

**Nordic / Eastern Europe:** Eliteserien (Norway), Allsvenskan (Sweden), Ekstraklasa (Poland)

**European competitions:** Champions League, Europa League, Europa Conference League

## Quick Start

### Prerequisites

- Python 3.11+
- Google Chrome + Camoufox (anti-fingerprint Firefox for Flashscore scraping)
- API keys: [API-Football](https://www.api-football.com/) (free, 100 req/day), [football-data.org](https://www.football-data.org/client/register) (free), [The Odds API](https://the-odds-api.com/) (free, 500 credits/month)
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

Set environment variables (these override config file values):

```bash
export API_FOOTBALL_KEY="your_key"
export FOOTBALL_DATA_ORG_KEY="your_key"
export ODDS_API_KEY="your_key"
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
python -m src.agent.betting_agent --settle          # Settle yesterday's picks + auto-learn
python -m src.agent.betting_agent --update --skip-ml-retrain  # Scrape fixtures, odds, injuries
python -m src.agent.betting_agent --train           # Retrain ML models if stale (>3 days)
python -m src.agent.betting_agent --picks           # Generate and send today's picks
python -m src.agent.betting_agent --update-results  # Scrape Flashscore results after matches finish
python -m src.agent.betting_agent --settle          # Settle any newly available results
```

## CLI Reference

```
python -m src.agent.betting_agent <command>
```

| Command | Description |
|---|---|
| `--init` | First-run setup: initialize DB and collect seed data |
| `--update` | Daily scrape: fixtures, odds, injuries, API-Football stats (skips Flashscore results) |
| `--update --skip-ml-retrain` | Same as above without triggering ML retraining (used in CI) |
| `--update-results` | Scrape Flashscore match results for all configured leagues |
| `--picks` | Generate value picks, save to DB, send to Telegram |
| `--picks --leagues eng1,esp1` | Filter picks by league codes |
| `--settle` | Settle pending picks against results + run automated learning pipeline |
| `--report` | Send comprehensive performance report to Telegram |
| `--stats` | Print prediction stats (ROI, win rate, Brier score, CLV) broken down by period/market/league |
| `--train` | Retrain ML models on all historical data |
| `--tune` | Tune ensemble weights from recent results |
| `--analyze <id>` | Deep-dive prediction breakdown for a specific match ID |
| `--backfill-history` | Fetch historical stats for low-coverage teams via football-data.org + API-Football |
| `--telegram-setup` | Guided Telegram bot configuration |
| `--telegram-test` | Send a test message to verify Telegram is working |
| `--telegram-welcome` | Send bot capabilities overview + commands list to Telegram |

## Configuration Reference

Key parameters in `config/config.yaml`:

```yaml
betting:
  min_expected_value: 0.03          # Minimum 3% edge required
  min_confidence: 0.58              # Minimum model probability (58%)
  min_ev_confidence_score: 0.035    # Sliding scale: EV × confidence must exceed this
  kelly_fraction: 0.25              # Fractional Kelly (25% of full Kelly)
  max_stake_percentage: 4.0         # Hard cap: max 4% of bankroll per bet
  max_picks_per_league: 5           # Cap per league per day
  max_total_kelly_pct: 40.0         # Daily portfolio exposure cap (% of bankroll)
  min_odds: 1.30                    # Reject odds below this
  max_odds: 10.0                    # Reject odds above this
  excluded_markets: [under_1.5, under_2.5, under_3.5]  # Never bet these

models:
  ensemble_weights:
    poisson: 0.25
    xgboost: 0.35
    random_forest: 0.20
    elo: 0.20
  bookmaker_blend_weight: 0.40      # 40% bookmaker / 60% model for all markets
  goals_ml_blend_weight: 0.25       # 25% GoalsMLModel for over/under
  extreme_confidence_ceiling: 0.90  # Dampen probabilities above 90%
  dixon_coles_rho: -0.13            # Low-score correction strength
  strength_half_life_days: 180      # Time-decay: match 6mo ago = half weight
  weather_features_enabled: true    # Open-Meteo weather features
  poisson_use_xg: true              # Use xG instead of raw goals when available
  poisson_xg_min_coverage: 0.50     # Min % of matches needing xG before xG-based fitting
  intl_goals_dampen: 0.30           # Blend toward priors for CL/EL/ECL matches
  bayesian_weight_half_life_days: 90
  bayesian_prior_strength: 10
  ml_retrain_days: 3                # Auto-retrain ML after N days
  min_training_samples: 1000
  drawdown_lookback_picks: 30
  drawdown_reduce_threshold: -0.10
  drawdown_pause_threshold: -0.30
  ev_calibration_lookback: 40
  max_injury_budget: 30             # Max API-Football injury requests per run
```

Full reference: [`config/config.example.yaml`](config/config.example.yaml)

## CI/CD Pipeline

The agent runs automatically via GitHub Actions every day at **06:00 UTC** (09:00 Kyiv summer / 08:00 Kyiv winter).

**Pipeline steps (in order):**

1. Checkout repo; restore ML models cache and Camoufox binary cache
2. Setup Python 3.11, install Chrome + Xvfb, start virtual display on :99
3. Install Python dependencies + NLTK vader lexicon
4. Download Camoufox browser binary
5. Create `config/config.yaml` from example; verify Neon PostgreSQL connection
6. `--update --skip-ml-retrain` — scrape fixtures, odds, injuries (30 min timeout)
7. `--settle` — settle pending picks against results (5 min timeout)
8. `--train` — retrain ML models if stale (35 min timeout)
9. DB health check — warn if fewer than 100 completed matches
10. `--picks` — generate picks and send to Telegram (45 min timeout)
11. `--update-results` — scrape Flashscore results after today's matches (60 min timeout, always runs)
12. `--settle` — settle any results just scraped in step 11 (5 min timeout)
13. Weekly `--report` — Sundays only
14. Save ML models + Camoufox to cache for next run
15. On failure: upload logs (7-day retention) + notify Telegram

**Total job timeout:** 180 minutes

**Required GitHub Secrets:**

| Secret | Description |
|---|---|
| `API_FOOTBALL_KEY` | API-Football free tier key |
| `FOOTBALL_DATA_ORG_KEY` | football-data.org free tier key |
| `ODDS_API_KEY` | The Odds API free tier key |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token |
| `TELEGRAM_CHAT_ID` | Target chat/channel ID |
| `DATABASE_URL` | Neon PostgreSQL connection string |

## Automated Learning Pipeline

After every `--settle`, the agent automatically runs:

1. **Refit Poisson/Elo** (60s) — recalculate team strengths from latest results with time decay
2. **Tune ensemble weights** — Bayesian per-league weight update; skipped if Poisson refit failed
3. **Update calibration factors** — per-model overconfidence ratios; persisted to `data/models/calibration.json`
4. **Persist EV threshold** — auto-calibrated minimum edge saved to `data/models/ev_threshold.json`; loaded on next run
5. **ML staleness check** — logs a warning if models are >3 days old; actual retraining deferred to `--train`

This closes the feedback loop so models adapt workflow-to-workflow without manual intervention.

## Data Sources

| Source | Data Fetched | Limit |
|---|---|---|
| [Flashscore](https://www.flashscore.com/) | Fixtures, results, extended match stats (dangerous attacks, saves, offsides), referee and venue data | Scraped (Camoufox anti-fingerprint Firefox + Chrome/Xvfb fallback) |
| [API-Football](https://www.api-football.com/) | Fixture IDs, results, xG, advanced stats, bookmaker odds (1X2/OU/BTTS/team goals), injuries, team history backfill | 100 req/day (free tier) |
| [football-data.org](https://www.football-data.org/) | Fixtures + results for 9 top leagues | No daily limit, 10 req/min (free) |
| [The Odds API](https://the-odds-api.com/) | Supplemental 1X2 + Over/Under odds for 27 leagues | 500 credits/month (free tier) |
| [football-data.co.uk](https://football-data.co.uk/) | Historical match results + Bet365/Pinnacle odds from 2016 (CSV) | No limit, no key required |
| [Open-Meteo](https://open-meteo.com/) | Match-day weather forecasts (temp, wind, precipitation) | No limit, no key required |

## Database

- **Primary**: Neon serverless PostgreSQL (free tier, eu-central-1); connection pooling with 5-min recycle for scale-to-zero
- **Fallback**: SQLite at `data/football_betting.db` (when `DATABASE_URL` is not set)
- **Auto-migration**: Missing columns and indexes are created on startup

| Table | Purpose |
|---|---|
| `teams` | Team registry with API-Football ID mapping |
| `matches` | Core match data: scores, xG, referee, venue, fixture vs result flag |
| `odds` | Bookmaker odds with `opening_odds` frozen at first-seen for movement tracking |
| `injuries` | Daily player injury status fetched before picks |
| `players` | Player registry (used for injury linking) |
| `saved_picks` | Bet history: pick, odds, EV, Kelly stake, result, settled timestamp |

**Odds pruning**: odds older than 400 days are deleted to stay within Neon's 500 MB free limit; odds for matches with saved picks are always preserved.

## Telegram Notifications

| Message | Trigger |
|---|---|
| **Daily picks** | `--picks` — grouped by league with match, bet type, odds, EV%, confidence, xG predictions, model agreement votes, reasoning |
| **Settlement report** | `--settle` — win/loss record, ROI, profit/loss, avg CLV, edge variance |
| **Performance report** | `--report` or Sunday CI run — all-time record, ROI, Brier score, avg CLV |
| **Welcome message** | `--telegram-welcome` — bot overview and command list |
| **Failure alert** | CI `on-failure` step — workflow name + run URL |

## Performance Metrics

Tracked via `--stats` or Telegram `--report`:

- Win rate and ROI: all-time, 7-day, 30-day, by market, by league
- Brier score (probability calibration quality)
- Closing Line Value (CLV) — model odds vs closing bookmaker odds
- Model agreement tags on each pick (unanimous / majority / split)
- Drawdown tracking and current circuit breaker status

## Project Structure

```
src/
├── agent/
│   └── betting_agent.py             # Main orchestrator, all CLI commands, learning pipeline
├── models/
│   ├── ensemble.py                  # Weighted ensemble predictor + bookmaker blend + calibration
│   ├── poisson_model.py             # Dixon-Coles Poisson (xG-enhanced, time-decay)
│   ├── ml_models.py                 # LR + RF + XGBoost classifiers (1X2 + over/under)
│   ├── elo_system.py                # Elo ratings with home advantage and season regression
│   └── bayesian_weights.py          # Per-league/per-market adaptive Bayesian weight learner
├── features/
│   ├── feature_engineer.py          # 14-section feature pipeline (80+ features)
│   ├── team_features.py             # Form, league position, international experience, momentum
│   ├── h2h_features.py              # Head-to-head statistics
│   ├── injury_features.py           # Player injury impact
│   └── weather_service.py           # Open-Meteo API integration (in-memory cache)
├── scrapers/
│   ├── flashscore_scraper.py        # Camoufox browser scraper (results, stats, referee, venue)
│   ├── apifootball_scraper.py       # API-Football client (fixtures, odds, xG, injuries)
│   ├── footballdataorg_scraper.py   # football-data.org client (9 leagues)
│   ├── theodds_scraper.py           # The Odds API client (supplemental odds, 27 leagues)
│   ├── historical_loader.py         # football-data.co.uk CSV bootstrap (2016+)
│   ├── injury_scraper.py            # API-Football /injuries endpoint
│   └── base_scraper.py              # Circuit breaker, retry logic, shared async utilities
├── betting/
│   ├── value_calculator.py          # EV calculation, Kelly sizing, model agreement, contrarian signals
│   └── bankroll_manager.py          # Bankroll tracking
├── reporting/
│   ├── telegram_bot.py              # All Telegram messages: picks, settlement, reports, alerts
│   ├── dashboard.py                 # Dashboard utilities
│   └── match_report.py              # Per-match analysis reports
├── data/
│   ├── database.py                  # SQLAlchemy manager, PostgreSQL/SQLite, odds pruning, migrations
│   └── models.py                    # ORM: Team, Match, Player, Injury, Odds, SavedPick
└── utils/
    ├── config.py                    # YAML config loader with env var override
    └── logger.py                    # Logging setup
scripts/
├── migrate_to_neon.py               # SQLite → PostgreSQL migration (COPY protocol, ~20x faster)
├── import_mcp_odds.py               # Import odds from MCP wagyu-sports JSON files into DB
└── sync_db.py                       # Pull CI database from GitHub Actions artifacts
config/
└── config.example.yaml              # Full configuration reference with all keys
```

## Requirements

- Python 3.11+
- Google Chrome / Chromium + Xvfb (for Selenium fallback scraping)
- Camoufox (anti-fingerprint Firefox, primary Flashscore scraper)
- See [`requirements.txt`](requirements.txt) for full Python dependency list
