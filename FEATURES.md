# Football Betting Agent — Complete Feature Reference

> Auto-generated feature inventory. Last updated: 2026-03-14

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `--init` | Initialize database and run first data collection |
| `--update` | Daily data update (fixtures + results + odds scraping) |
| `--picks` | Generate today's value picks (optional `--leagues england/premier-league,...`) |
| `--settle` | Settle pending picks against actual results |
| `--report` | Send comprehensive performance report to Telegram |
| `--stats` | Show prediction statistics (all-time, 7d, 30d, yesterday) |
| `--train` | Train ML models on historical data (up to 2000 samples) |
| `--tune` | Tune ensemble weights from recent results (Bayesian + calibration) |
| `--analyze <ID>` | Deep analysis of a specific match by database ID |
| `--backfill-history` | Fetch historical data for low-coverage teams |
| `--telegram-setup` | Interactive Telegram bot configuration |
| `--telegram-test` | Send a test message to verify Telegram |

---

## Prediction Models (5 total)

### 1. Dixon-Coles Poisson Model
**File:** `src/models/poisson_model.py`

- Time-decayed attack/defense strengths (half-life: 180 days)
- Per-league Dixon-Coles low-score correction (rho estimated via MLE)
- Bayesian shrinkage for small-sample teams (cap: 100 matches)
- xG-enhanced mode: uses expected goals instead of raw goals when ≥50% coverage
- Generates full score matrix (0-0 through 6-6) for all markets
- Per-league home/away goal averages
- International match dampening (CL/EL/ECL → 30% toward priors)

### 2. Elo Rating System
**File:** `src/models/elo_system.py`

- K-factor: 32, home advantage: 65 Elo points
- Season regression: 33% toward mean between seasons
- Win/draw/loss probabilities from rating differential
- 756 teams rated across all leagues

### 3. ML Models — 1X2 Classifier
**File:** `src/models/ml_models.py` (class `MLModels`)

- **Logistic Regression** — baseline calibrated classifier
- **Random Forest** (200 trees, depth 10) — isotonic calibration
- **XGBoost** (200 trees, depth 6, lr 0.1) — isotonic calibration
- **LightGBM** (200 trees, depth 8, lr 0.1) — optional
- Auto-retrains if >3 days stale
- HMAC-signed pickle persistence
- Sparse/zero-variance feature pruning + StandardScaler

### 4. Goals ML Model — Over/Under 2.5 Classifier
**File:** `src/models/ml_models.py` (class `GoalsMLModel`)

- Binary XGBoost/RF for Over 2.5 goals
- Blended at 25% weight into Poisson-based goals predictions
- Same persistence and retraining as 1X2 model

### 5. Ensemble Predictor
**File:** `src/models/ensemble.py`

- Weighted average of Poisson + Elo + ML for 1X2 outcomes
- Bookmaker blend: 40% bookmaker implied probs + 60% model for all markets
- Per-league Bayesian adaptive weights (learned from settled picks)
- Per-model calibration factors (0.6-1.0, overconfident models discounted)
- Extreme confidence dampening: ceiling 90%, excess retained at 30%
- Goals market: Poisson + bookmaker + GoalsML blend

### 6. Bayesian Weight Learner
**File:** `src/models/bayesian_weights.py`

- Per-league, per-market Beta-distribution weight learner
- Global priors from config ensemble_weights
- Temporal decay (half-life: 90 days)
- Minimum 15 observations before league-specific weights activate
- Persisted to `data/models/bayesian_weights.json`

---

## Feature Engineering (14 sections, ~267 features)

**File:** `src/features/feature_engineer.py`

| # | Section | Key Features | Count |
|---|---------|-------------|-------|
| 1 | **Team Form** | Rolling 10-game: wins, draws, losses, avg goals (overall + home/away split), exponential decay form score | ~20 |
| 2 | **Head-to-Head** | Total meetings, home/draw/away win %, avg goals scored | 6 |
| 3 | **Injuries** | Total injured, key player % (skipped in training) | 4 |
| 4 | **League Position** | Position, points, goal diff, relegation gap, title gap, position difference | ~11 |
| 5 | **News Sentiment** | Sentiment score, article count per team (skipped in training) | 4 |
| 6 | **International Form** | CL/EL/ECL matches, points/match, form string, experience diff | ~8 |
| 7 | **Expected Goals (xG)** | Rolling xG for/against, overperformance, xG differential | ~10 |
| 8 | **Extended Stats** | Dangerous attacks, saves, offsides, corners, possession (Flashscore rolling) | ~9 |
| 9 | **Referee Stats** | Cards/match, fouls/match, goals/match, over25 rate, yellow/red avg (30 matches) | 7 |
| 10 | **Momentum (RSI/MACD)** | RSI, MACD per team + differentials | 6 |
| 11 | **Bookmaker Implied Probs** | 1X2, O/U 1.5/2.5, BTTS, team goals implied probabilities | ~15 |
| 11b | **Odds Movement** | Home/away/over25 odds % change, max absolute movement, direction signal | 5 |
| 12 | **Situational** | Rest days, midweek flag, matches in 14d/21d/30d, fatigue index (0-1), short rest count | ~14 |
| 13 | **League Baseline Rates** | Home win rate, draw rate, avg goals, over25 rate, BTTS rate (200-match window) | 7 |
| 14 | **Weather** | Temperature, wind, precipitation, rain/wind flags (Open-Meteo, skipped in training) | 6 |

**Training Optimizations:**
- `for_training=True` skips weather, injuries, news (no historical API data)
- League standings cached per (league, 1st-of-month) for training queries
- Thread pool parallelism via `run_in_executor` for sync DB queries

---

## Markets Supported (13 selections)

| Market | Selections | DB `market_type` |
|--------|-----------|-----------------|
| **Match Result** | Home, Draw, Away | `1X2` |
| **Over/Under 1.5** | Over 1.5, Under 1.5 | `over_under` |
| **Over/Under 2.5** | Over 2.5, Under 2.5 | `over_under` |
| **Over/Under 3.5** | Over 3.5, Under 3.5 | `over_under` |
| **Both Teams Score** | Yes, No | `btts` |
| **Home Team Goals** | Home Over 1.5, Home Under 1.5 | `team_goals` |
| **Away Team Goals** | Away Over 1.5, Away Under 1.5 | `team_goals` |

**Excluded by default:** Under 1.5, Under 2.5, Under 3.5 (configurable)

---

## Betting Logic

**File:** `src/betting/value_calculator.py`

### Value Detection
1. Model probability vs bookmaker implied probability
2. EV = `(prob x odds) - 1`, minimum 3%
3. Sliding confidence gate: `EV x confidence >= 0.035`
4. Hard confidence floor: 45% minimum
5. Divergence guard: reject if model/implied ratio > 2.0x (miscalibration)

### Kelly Criterion
- Formula: `(b*p - q) / b` where b = odds-1
- Fractional Kelly: 25% (professional standard)
- Agreement scaling: unanimous 1.0x, majority 0.80x, split 0.60x
- Hard cap: 4.0% per bet

### Portfolio Risk Management
- **Per-match cap:** Max 2 picks per match
- **Per-league cap:** Max 5 picks per league per day
- **Daily exposure cap:** Max 40% total Kelly across all picks
- **Drawdown circuit breaker:** Linear stake reduction from -10% ROI to full pause at -30%
- **Market correlation filter:** Drops lower-EV pick when two picks on same match are positively correlated

### Auto-Calibration
- EV threshold adjusts based on last 40 picks' hit rate
- Hot (>60% hit rate) → tightens threshold (more selective)
- Cold (<45% hit rate) → loosens threshold (fewer but safer picks)
- Persisted to `data/models/ev_threshold.json`

### Contrarian Detection
- Tracks model-vs-market divergence (1.3x-2.0x)
- Unanimous + high divergence gets 1.10x sort boost
- Telegram indicator: contrarian signal emoji

### Parlay Suggestions
- Finds 2-3 uncorrelated picks from different leagues
- Requires unanimous/majority model agreement
- Combined odds and probability calculation

---

## Data Sources (3 scrapers)

### 1. Flashscore (Primary)
**File:** `src/scrapers/flashscore_scraper.py`
- Selenium Chrome + Camoufox fallback
- Results, fixtures, odds, extended stats (dangerous attacks, saves, corners, etc.)
- Referee names and venues
- 26 configured leagues
- Time budgets: 25 min results, 8 min fixtures

### 2. API-Football
**File:** `src/scrapers/apifootball_scraper.py`
- REST API with rate limiting (10/min, 100/day free tier)
- Bookmaker odds (Bet365, Pinnacle, 1xBet, FlashScore)
- Match statistics, xG data, referee info
- Historical backfill (2022-2024 seasons)
- Filtered to configured leagues only

### 3. Football-Data.org
**File:** `src/scrapers/footballdataorg_scraper.py`
- Free supplementary API (no daily limit)
- 9 top leagues covered
- Fills gaps when Flashscore scraping fails

---

## Database Schema (7 tables)

**Engine:** PostgreSQL (Neon serverless, free tier) with SQLite fallback

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `teams` | Team registry (756 teams) | name, league, flashscore_id, apifootball_team_id |
| `matches` | Fixtures + results | home/away_team_id, match_date, league, goals, xG, referee, venue |
| `odds` | Bookmaker odds | match_id, bookmaker, market_type, selection, odds_value, opening_odds |
| `players` | Player registry | name, team_id, position, is_key_player |
| `injuries` | Injury reports | player_id, injury_type, expected_return, status |
| `news` | Team news articles | team_id, headline, sentiment_score |
| `saved_picks` | Daily picks tracking | match_id, market, selection, odds, EV, kelly, result |

**Maintenance:**
- Odds pruning: keeps 400 days, preserves picks-linked odds
- Duplicate team prevention: name mappings in all 3 scrapers

---

## Performance Tracking

### Statistics Computed (`--stats`)
- **Win rate** (all-time, 7d, 30d, yesterday)
- **ROI** per period and per market/league
- **Brier score** (calibration accuracy)
- **Average CLV** (Closing Line Value — edge at kickoff)
- **Model coverage** (Poisson teams, Elo teams, ML status)
- **Odds source breakdown** (real vs fallback)
- **Calibration buckets** (predicted vs actual by confidence range)

### Settlement Logic
- Supports: win, loss, void, push
- Under/over goals lines (1.5, 2.5, 3.5)
- BTTS (both teams to score)
- Team goals (home/away over/under 1.5)
- Auto-learning after settlement (Bayesian weights + calibration + EV threshold)

---

## Telegram Notifications

**File:** `src/reporting/telegram_bot.py`

| Message Type | Content |
|-------------|---------|
| **Daily Picks** | League-grouped picks with xG, odds, EV, confidence, model agreement, risk level |
| **Settlement Report** | W-L record, hit rate, ROI, per-market and per-league breakdown |
| **Performance Report** | Weekly comprehensive stats (Sunday) — calibration, Brier, CLV |
| **Parlay Suggestions** | 2-3 leg combinations with combined odds/probability |
| **Failure Alert** | CI run URL on workflow failure |

**Features:** HTML formatting, league flag emojis, contrarian signal indicator, 4096-char chunking

---

## CI/CD Pipeline

**File:** `.github/workflows/daily-picks.yml`

**Schedule:** Daily at 06:30 UTC (09:30 Kyiv summer)

**Steps:**
1. Checkout + restore ML model cache
2. Setup Python 3.11 + Chrome + Xvfb
3. Install dependencies + camoufox
4. Verify Neon DB connection
5. Run tests
6. `--settle` (25 min timeout)
7. `--update` (40 min timeout)
8. `--settle` again (catch API-Football-only results)
9. DB health check
10. `--picks` (45 min timeout)
11. `--report` (Sundays only)
12. Save ML cache + upload logs on failure
13. Telegram alert on failure

**Total timeout:** 120 minutes

---

## Configuration Reference

**File:** `config/config.example.yaml`

```yaml
# === Scraping ===
scraping:
  flashscore_leagues:           # 26 leagues (see Leagues section)
  update_interval_hours: 6
  headless: true
  request_delay: 3

# === Models ===
models:
  ensemble_weights:
    poisson: 0.25
    xgboost: 0.35
    random_forest: 0.20
    elo: 0.20
  bookmaker_blend_weight: 0.40
  goals_ml_blend_weight: 0.25
  extreme_confidence_ceiling: 0.90
  dixon_coles_rho: -0.13
  strength_half_life_days: 180
  intl_goals_dampen: 0.30
  shrinkage_sample_cap: 100
  dc_rho_min_matches: 50
  bayesian_weight_half_life_days: 90
  bayesian_prior_strength: 10
  poisson_use_xg: true
  poisson_xg_min_coverage: 0.50
  ml_retrain_days: 3
  drawdown_lookback_picks: 30
  drawdown_reduce_threshold: -0.10
  drawdown_pause_threshold: -0.30
  ev_calibration_lookback: 40
  weather_features_enabled: true

# === Betting ===
betting:
  min_odds: 1.30
  max_odds: 10.0
  min_expected_value: 0.03
  min_confidence: 0.58
  min_ev_confidence_score: 0.035
  kelly_fraction: 0.25
  max_stake_percentage: 4.0
  max_picks_per_league: 5
  max_total_kelly_pct: 40.0
  excluded_markets: [under_1.5, under_2.5, under_3.5]
  kelly_agreement_scale:
    unanimous: 1.0
    majority: 0.80
    split: 0.60
    unknown: 0.75

# === Notifications ===
notifications:
  telegram_enabled: true
  telegram_bot_token: ${TELEGRAM_BOT_TOKEN}
  telegram_chat_id: ${TELEGRAM_CHAT_ID}
  send_daily_picks: true
  send_live_alerts: false

# === Data Sources ===
data_sources:
  apifootball_key: ${API_FOOTBALL_KEY}
  footballdataorg_key: ${FOOTBALL_DATA_ORG_KEY}

# === Database ===
database:
  sqlite_path: data/football_betting.db
  # url: ${DATABASE_URL}  (Neon PostgreSQL)

# === Logging ===
logging:
  level: INFO
  log_file: logs/betting_agent.log
  log_rotation: "1 week"
```

---

## Leagues (26 active)

### Top 5
- england/premier-league, spain/laliga, germany/bundesliga, italy/serie-a, france/ligue-1

### European Competitions
- europe/champions-league, europe/europa-league, europe/europa-conference-league

### Strong European
- netherlands/eredivisie, portugal/primeira-liga, belgium/jupiler-pro-league, turkey/super-lig, scotland/premiership

### Lower Divisions & Smaller Leagues
- england/championship, england/league-one, england/league-two
- spain/laliga2, germany/2-bundesliga, italy/serie-b, france/ligue-2
- austria/bundesliga, switzerland/super-league, greece/super-league
- poland/ekstraklasa, romania/liga-1

### Off-Season
- denmark/superliga (until ~July 2026)
