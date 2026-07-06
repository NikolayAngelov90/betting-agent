# Football Betting Agent

An automated football betting prediction system combining statistical models, machine learning, and multi-source data scraping to identify value bets. Each pick is then verified by Claude (web research + a final KEEP/CHANGE ruling) before it goes out. Runs daily via GitHub Actions with a PostgreSQL database (Supabase/Neon) and delivers a predictions summary to Telegram.

> **Current mode (as of July 2026):** the pipeline is scoped to the **FIFA World Cup 2026** plus the **summer club leagues that have resumed** — the two Nordic leagues (Allsvenskan, Eliteserien) and the three UEFA qualifying competitions (Champions League, Europa League, Conference League). The engine itself is league-agnostic; the active set is just `scraping.flashscore_leagues` in the config, and the full ~28-league club list is commented out in [`config/config.example.yaml`](config/config.example.yaml) ready to re-enable as those seasons restart (~August 2026).

## How It Works

```
Flashscore (Camoufox) + API-Football + football-data.org + The Odds API
                              ↓
        PostgreSQL — Supabase/Neon (matches, odds, injuries, stats)
                              ↓
             Feature Engineering (14 sections, 80+ features)
                              ↓
    Ensemble Prediction (Poisson + Elo + ML + 60% bookmaker blend)
         + Bayesian per-league/per-market adaptive weights
         + per-model calibration factors
                              ↓
      Value Calculation (EV, Kelly fraction, confidence filter)
         + WC "pick every match" (national-team competitions)
         + drawdown circuit breaker + correlation filter
                              ↓
   Claude pick review — web research → verify or switch each pick
         (decision-only: KEEP / CHANGE, no article, English)
                              ↓
            Telegram predictions summary (picks only)
                              ↓
       Settlement (90-min score for knockouts) → auto-learning
           (refit models, tune weights, recalibrate EV threshold)
                              ↓
            Performance reporting (weekly, Sundays)
```

## AI Pick Review (Claude)

Every saved pick — WC "every-match" picks **and** club-league value picks — is reviewed by Claude inside the `--picks` step, **before** the Telegram summary is sent:

- **What it does:** Claude uses server-side web search to research the match (recent form, head-to-head, injuries/suspensions by name, what's at stake, current bookmaker odds), then returns a machine-readable **`KEEP` / `CHANGE`** decision choosing the selection most likely to win at odds ≥ 1.50 from the priced menu. If it switches, the tracked pick in the DB is rewritten and the in-memory pick is synced so the summary reflects the final selection.
- **Decision-only — no briefings, no translation.** The review returns *only* the decision block. It does **not** write a briefing article and does **not** translate anything; the Telegram output is the plain predictions summary. (A separate `--briefing`/`--prematch-briefing` command can still post full narrative articles manually, but the pipeline no longer uses it.)
- **Backends** (`briefings.backend`):
  - `claude_code` (default) — headless Claude Code CLI billed to a **Claude Pro subscription** via the `CLAUDE_CODE_OAUTH_TOKEN` secret. **$0 in API credits.**
  - `anthropic_api` (fallback) — the direct Anthropic API (Opus 4.8 + web search) using `ANTHROPIC_API_KEY`, per-token credits. Fires only when Claude Code comes back empty (e.g. the shared Pro 5-hour session limit) and `briefings.api_fallback: true`.
- **Prompt caching:** on the Anthropic-API path the (stable) system prompt + web-search tool are cached with one ephemeral breakpoint, so calls 2..N of a run read the prefix at ~0.1× instead of full input price.
- **Fails safe:** if no Claude auth is present or the call fails, the review no-ops and the model's own pick is sent unchanged.

Controlled by `briefings.enabled`, `briefings.finalize_picks`, and `briefings.send_to_telegram: false`.

## Models

| Model | Algorithm | Role |
|---|---|---|
| **Poisson (Dixon-Coles)** | Score matrix | Match outcome probabilities; low-score correction (rho=-0.13), exponential time-decay (180d half-life), xG-enhanced when ≥35% of matches have xG data |
| **Elo** | Rating system | Team strength ratings (home advantage in Elo pts), season regression, used for 1X2 probability conversion |
| **ML Classifier** | Logistic Regression + Random Forest (+ optional XGBoost) | 1X2 outcome classifier trained on 14 feature sections; isotonic calibration; ensemble weights: XGBoost 35%, RF 20%, Elo 20%, Poisson 25% |
| **GoalsMLModel** | Binary XGBoost/RF/LR | Over/Under 2.5 goals; blended at 25% weight alongside Poisson |
| **Bookmaker blend** | Implied probability extraction | Real bookmaker probabilities mixed at **60%** into both goals and 1X2 markets to anchor predictions to market consensus |
| **Bayesian weight learner** | Beta-distribution temporal decay | Per-league and per-market adaptive ensemble weights with global prior; half-life 90 days, persisted to `data/models/bayesian_weights.json` |
| **National-team handling** | `NATIONAL_TEAM_LEAGUES` frozenset | WC / qualifiers / friendlies / continental cups get international-goals dampening, coverage-gate bypass (every match gets a pick), and marginal-mode "most likely score" |

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
| **WC pick-every-match** | National-team competitions save a tracked pick for **every** match even with no value edge (highest-EV selection, staked at the Kelly minimum). Club leagues stay value-only. |
| **WC mismatch guard** | In strong-favourite routs (fav win prob ≥ 65% AND underdog xG ≤ 1.10) the forced pick drops BTTS/underdog markets and backs the favourite's Over 1.5/2.5 — the weak side rarely scores |
| **Drawdown circuit breaker** | Scales stakes linearly 100% → 0% as recent ROI (30-pick window) drops from -10% to -30%; pauses all picks below -30% |
| **Market correlation filter** | Detects positively-correlated picks on the same match (e.g. Home Win + Over 2.5); drops the lower-EV one |
| **Daily exposure cap** | `max_total_kelly_pct` (0 = disabled) trims lowest-EV picks until total Kelly exposure stays within the cap |
| **EV + confidence thresholds** | `min_ev: 5%`, `min_confidence: 55%`; sliding scale: `EV × confidence ≥ 0.038` |
| **Model divergence guard** | Rejects picks where model probability / implied probability > 2.0× (prevents miscalibration outliers) |
| **Model agreement scaling** | Kelly fraction scaled: unanimous 1.0×, majority 0.80×, split 0.60×, solo/unknown 0.75× |
| **EV auto-calibration** | Hit rate > 60% tightens min_EV; < 45% loosens; persisted to `data/models/ev_threshold.json` across runs |
| **Extreme confidence dampening** | Probabilities above 90% retain only 30% of the excess (e.g. 98% → 92.4%) |
| **Per-match pick cap** | One pick per WC match per day (dedup by match_id); max 2 per club match, enforced across multiple `--picks` runs |
| **Odds range filter** | Only bets with odds between 1.50 and 10.0 |
| **Per-league cap** | Max 5 picks per league per day |
| **Excluded markets** | `under_1.5`, `under_2.5`, `under_3.5`, `btts_no`, `over_3.5` (proven loser in settled data) |

## Active Leagues

**International:** FIFA World Cup 2026

**Nordic summer leagues:** Eliteserien (Norway), Allsvenskan (Sweden)

**UEFA qualifying:** Champions League, Europa League, Conference League

The remaining ~28 club leagues (Top-5, second divisions, lower English divisions, other European leagues) are commented out in the config and re-enabled as their seasons restart. The scraper/model stack supports all of them unchanged — the active set is purely `scraping.flashscore_leagues`.

## Quick Start

### Prerequisites

- Python 3.11+
- Google Chrome + Camoufox (anti-fingerprint Firefox for Flashscore scraping)
- API keys: [API-Football](https://www.api-football.com/) (free, 100 req/day), [football-data.org](https://www.football-data.org/client/register) (free), [The Odds API](https://the-odds-api.com/) (free, 500 credits/month)
- Telegram bot token + chat ID (recommended)
- PostgreSQL database (recommended: [Supabase](https://supabase.com/) or [Neon](https://neon.tech/) free tier) or SQLite fallback
- **Optional (for the Claude pick review):** a Claude Pro subscription token (`CLAUDE_CODE_OAUTH_TOKEN`, $0 extra) and/or an `ANTHROPIC_API_KEY` for the paid fallback. Without either, the review is skipped and model picks are sent unchanged.

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
export DATABASE_URL="postgresql://user:pass@host/db"   # Supabase or Neon connection string
export CLAUDE_CODE_OAUTH_TOKEN="your_pro_token"        # optional — Claude review via Pro ($0)
export ANTHROPIC_API_KEY="sk-ant-..."                  # optional — paid API fallback for the review
```

If `DATABASE_URL` is not set, the agent falls back to a local SQLite database at `data/football_betting.db`.

### First Run

```bash
python -m src.agent.betting_agent --init    # Initialize DB and collect seed data
python -m src.agent.betting_agent --train   # Train ML models
```

### Daily Workflow

```bash
python -m src.agent.betting_agent --settle                    # Settle yesterday's picks + auto-learn
python -m src.agent.betting_agent --update --skip-ml-retrain  # Scrape fixtures, odds, injuries
python -m src.agent.betting_agent --train                     # Retrain ML models if stale (>3 days)
python -m src.agent.betting_agent --picks --force             # Generate picks → Claude review → send
python -m src.agent.betting_agent --update-results            # Scrape results after matches finish
python -m src.agent.betting_agent --settle                    # Settle newly available results
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
| `--picks` | Generate value picks, run the Claude pick review, save to DB, send predictions to Telegram |
| `--picks --force` | Regenerate today's picks even if some already exist (CI default) |
| `--picks --leagues eng1,esp1` | Filter picks by league codes |
| `--settle` | Settle pending picks against results (90-min score for knockouts) + run the automated learning pipeline |
| `--report` | Send comprehensive performance report to Telegram |
| `--stats` | Print prediction stats (ROI, win rate, Brier score, CLV) broken down by period/market/league |
| `--train` | Retrain ML models on historical data |
| `--tune` | Tune ensemble weights from recent results |
| `--analyze <id>` | Deep-dive prediction breakdown for a specific match ID |
| `--backfill-wc` | Backfill national-team match history via API-Football (used when WC coverage is low) |
| `--backfill-history` | Fetch historical stats for low-coverage teams via football-data.org + API-Football |
| `--backfill-stats` | Backfill xG / advanced stats for completed matches |
| `--backtest-rolling` | Rolling-origin backtest over historical results |
| `--briefing` / `--prematch-briefing [min]` | Manually post a full AI briefing **article** to Telegram (not used by the pipeline) |
| `--telegram-setup` / `--telegram-test` / `--telegram-welcome` | Telegram configuration, test message, capabilities overview |

## Configuration Reference

Key parameters in `config/config.yaml`:

```yaml
betting:
  min_expected_value: 0.05          # Minimum 5% edge required
  min_confidence: 0.55              # Minimum model probability (55%)
  min_ev_confidence_score: 0.038    # Sliding scale: EV × confidence must exceed this
  kelly_fraction: 0.25              # Fractional Kelly (25% of full Kelly)
  max_stake_percentage: 4.0         # Hard cap: max 4% of bankroll per bet
  max_picks_per_league: 5           # Cap per league per day
  max_total_kelly_pct: 0            # Daily portfolio exposure cap (0 = disabled)
  min_odds: 1.50                    # Reject odds below this
  max_odds: 10.0                    # Reject odds above this
  wc_pick_every_match: true         # National-team competitions: a pick on every match
  wc_mismatch_fav_prob: 0.65        # Rout detection: favourite win prob threshold
  wc_mismatch_dog_xg: 1.10          # Rout detection: underdog xG ceiling
  excluded_markets: [under_1.5, under_2.5, under_3.5, btts_no, over_3.5]

models:
  ensemble_weights: { poisson: 0.25, xgboost: 0.35, random_forest: 0.20, elo: 0.20 }
  bookmaker_blend_weight: 0.60      # 60% bookmaker / 40% model
  goals_ml_blend_weight: 0.25       # 25% GoalsMLModel for over/under
  extreme_confidence_ceiling: 0.90  # Dampen probabilities above 90%
  dixon_coles_rho: -0.13            # Low-score correction strength
  strength_half_life_days: 180      # Time-decay: match 6mo ago = half weight
  weather_features_enabled: true
  poisson_use_xg: true
  poisson_xg_min_coverage: 0.35     # Min % of matches needing xG before xG-based fitting
  intl_goals_dampen: 0.30           # Blend toward priors for international matches
  bayesian_weight_half_life_days: 90
  ml_retrain_days: 3                # Auto-retrain ML after N days
  drawdown_lookback_picks: 30
  drawdown_reduce_threshold: -0.10
  drawdown_pause_threshold: -0.30

briefings:
  enabled: true                     # master switch for the Claude pick review
  send_to_telegram: false           # false = decision-only (no briefing article, predictions-only)
  finalize_picks: true              # Claude makes the final KEEP/CHANGE call on each pick
  backend: claude_code              # claude_code (Pro, $0) or anthropic_api (paid)
  api_fallback: true                # retry via the paid API when Claude Code comes back empty
  language: Bulgarian               # only affects the manual --briefing article path

notifications:
  suppress_picks_summary: false     # false = send the predictions summary to Telegram
```

Full reference: [`config/config.example.yaml`](config/config.example.yaml)

## CI/CD Pipeline

The agent runs automatically via GitHub Actions on two schedules (GitHub cron runs 0.5–5.7h late in practice, so a backup is included):

- **Primary:** `37 9 * * *` — 09:37 UTC (12:37 Sofia)
- **Backup:** `23 12 * * *` — 12:23 UTC (15:23 Sofia)

A `concurrency` group ensures the two runs never overlap. Every core step uses `continue-on-error` so one failure never half-aborts the pipeline; a dedicated **failure-alert** step then inspects each step's outcome and sends **one** Telegram alert if anything failed (so a green-but-broken run is never silent).

**Pipeline steps (in order):**

1. Checkout; restore ML-models, Camoufox, and pip caches
2. Setup Python 3.11, install Chrome + Xvfb (virtual display), install dependencies, fetch Camoufox
3. Create `config/config.yaml` from example; verify PostgreSQL connection; mark tables created
4. **Run tests** (`pytest`) — conftest strips `DATABASE_URL` so tests can never touch the production DB
5. National-team coverage check → conditional `--backfill-wc`
6. `--update --skip-ml-retrain` — scrape fixtures, odds, injuries
7. `--settle` — settle pending picks (90-min score for knockouts) + auto-learn
8. `--train` — retrain ML models if stale
9. DB health check
10. **Install Claude Code CLI** (before picks, for the review)
11. `--picks --force` — generate picks → **Claude review (KEEP/CHANGE)** → send predictions summary (150-min timeout)
12. `--update-results` — scrape Flashscore results after today's matches (always runs)
13. `--settle` — settle any results just scraped
14. Failure-alert step (always) + Sunday-only `--report`

**Total job timeout:** 240 minutes (raised to fit the per-match Claude review)

**Required GitHub Secrets:**

| Secret | Description |
|---|---|
| `API_FOOTBALL_KEY` | API-Football free tier key |
| `FOOTBALL_DATA_ORG_KEY` | football-data.org free tier key |
| `ODDS_API_KEY` | The Odds API free tier key |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token |
| `TELEGRAM_CHAT_ID` | Target chat/channel ID |
| `DATABASE_URL` | Supabase/Neon PostgreSQL connection string |
| `CLAUDE_CODE_OAUTH_TOKEN` | Claude Pro token for the pick review ($0 API cost) — optional |
| `ANTHROPIC_API_KEY` | Paid Anthropic API key, fallback for the review — optional |

## Automated Learning Pipeline

After every `--settle`, the agent automatically runs:

1. **Refit Poisson/Elo** — recalculate team strengths from latest results with time decay
2. **Tune ensemble weights** — Bayesian per-league/per-market weight update; skipped if Poisson refit failed
3. **Update calibration factors** — per-model overconfidence ratios; persisted to `data/models/calibration.json`
4. **Persist EV threshold** — auto-calibrated minimum edge saved to `data/models/ev_threshold.json`; loaded on next run
5. **ML staleness check** — auto-retrains ML if models are >3 days old

This closes the feedback loop so models adapt run-to-run without manual intervention.

## Data Sources

| Source | Data Fetched | Limit |
|---|---|---|
| [Flashscore](https://www.flashscore.com/) | Fixtures, results, extended match stats (dangerous attacks, saves, offsides), referee and venue data | Scraped (Camoufox anti-fingerprint Firefox + Chrome/Xvfb) |
| [API-Football](https://www.api-football.com/) | Fixture IDs, results, xG, advanced stats, bookmaker odds (1X2/OU/BTTS/team goals), injuries, national-team + team history backfill | 100 req/day (free tier, seasons 2022–2024) |
| [football-data.org](https://www.football-data.org/) | Fixtures + results for 9 top leagues | No daily limit, 10 req/min (free) |
| [The Odds API](https://the-odds-api.com/) | Supplemental 1X2 + Over/Under odds | 500 credits/month (free tier) |
| [football-data.co.uk](https://football-data.co.uk/) | Historical match results + Bet365/Pinnacle odds from 2016 (CSV) | No limit, no key required |
| [Open-Meteo](https://open-meteo.com/) | Match-day weather forecasts (temp, wind, precipitation) | No limit, no key required |
| **Claude** (Anthropic) | Live web research + final KEEP/CHANGE ruling per pick | Claude Pro subscription ($0) or paid API fallback |

## Database

- **Primary:** Supabase (or Neon) serverless PostgreSQL, free tier; connection pooling with 5-min recycle for scale-to-zero
- **Fallback:** SQLite at `data/football_betting.db` (when `DATABASE_URL` is not set)
- **Auto-migration:** missing columns and indexes are created on startup
- **numpy-safe writes:** psycopg2 adapters for numpy scalars are registered at startup (a numpy 2.x scalar in SQL otherwise raises a phantom `schema "np" does not exist`)
- **Test isolation:** `tests/conftest.py` strips `DATABASE_URL`, so DB-backed tests always use a temp SQLite database and can never read/write production

| Table | Purpose |
|---|---|
| `teams` | Team registry with API-Football ID mapping (national + club partition, youth-team exclusion) |
| `matches` | Core match data: scores, regulation (90-min) score, xG, referee, venue, round, fixture vs result flag |
| `odds` | Bookmaker odds with `opening_odds` frozen at first-seen for movement tracking |
| `injuries` | Daily player injury status fetched before picks |
| `players` | Player registry (used for injury linking) |
| `saved_picks` | Bet history: pick, odds, EV, Kelly stake, model agreement, result, settled timestamp |

**Odds pruning:** odds older than 400 days are deleted to stay within the free storage limit; odds for matches with saved picks are always preserved.

## Settlement Notes

- **90-minute grading:** knockouts that go to extra time / penalties are settled on the **regulation (90') score** (bookmaker convention). API-Football's final score includes extra-time goals, so the regulation score is stored separately and used for 1X2 / Over-Under / BTTS grading; the final score is kept for display.
- **Stuck-pick sweeper:** picks whose match never produced a result are voided after ~10 days so they stop polluting win/ROI stats.

## Telegram Notifications

| Message | Trigger |
|---|---|
| **Daily predictions** | `--picks` — the picks summary grouped by league (match, bet, odds, EV%, confidence, xG, model-agreement votes, reasoning). Reflects Claude's finalized selection. **No briefing articles.** |
| **Settlement report** | `--settle` — win/loss record, ROI, profit/loss, avg CLV |
| **Performance report** | `--report` or Sunday CI run — all-time record, ROI, Brier score, avg CLV |
| **Failure alert** | CI failure-check step — which step(s) failed + run URL |

## Performance Metrics

Tracked via `--stats` or Telegram `--report`:

- Win rate and ROI: all-time, 7-day, 30-day, by market, by league
- Brier score (probability calibration quality)
- Closing Line Value (CLV) — model odds vs closing bookmaker odds
- Model agreement tags on each pick (unanimous / majority / split)
- Drawdown tracking and current circuit-breaker status

## Project Structure

```
src/
├── agent/
│   └── betting_agent.py             # Main orchestrator, all CLI commands, learning pipeline, pick review call
├── models/
│   ├── ensemble.py                  # Weighted ensemble predictor + bookmaker blend + calibration
│   ├── poisson_model.py             # Dixon-Coles Poisson (xG-enhanced, time-decay, national-team leagues)
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
│   ├── apifootball_scraper.py       # API-Football client (fixtures, odds, xG, injuries, national-team backfill)
│   ├── footballdataorg_scraper.py   # football-data.org client (9 leagues)
│   ├── theodds_scraper.py           # The Odds API client (supplemental odds)
│   ├── historical_loader.py         # football-data.co.uk CSV bootstrap (2016+)
│   ├── injury_scraper.py            # API-Football /injuries endpoint
│   └── base_scraper.py              # Circuit breaker, retry logic, shared async utilities
├── betting/
│   ├── value_calculator.py          # EV, Kelly sizing, model agreement, WC mismatch/forced-pick logic
│   └── bankroll_manager.py          # Bankroll tracking
├── reporting/
│   ├── match_briefing.py            # Claude pick review (decision-only) + optional briefing articles
│   ├── telegram_bot.py              # All Telegram messages: predictions, settlement, reports, alerts
│   ├── dashboard.py                 # Dashboard utilities
│   └── match_report.py              # Per-match analysis reports
├── data/
│   ├── database.py                  # SQLAlchemy manager, PostgreSQL/SQLite, numpy adapters, pruning, migrations
│   └── models.py                    # ORM: Team, Match, Player, Injury, Odds, SavedPick
└── utils/
    ├── team_names.py                # Cross-source team-name matching + national-team aliases
    ├── config.py                    # YAML config loader with env var override
    └── logger.py                    # Logging setup
config/
└── config.example.yaml              # Full configuration reference with all keys
```

## Requirements

- Python 3.11+
- Google Chrome / Chromium + Xvfb (for Selenium fallback scraping)
- Camoufox (anti-fingerprint Firefox, primary Flashscore scraper)
- Optional: `CLAUDE_CODE_OAUTH_TOKEN` (Claude Pro) and/or `ANTHROPIC_API_KEY` for the AI pick review
- See [`requirements.txt`](requirements.txt) for the full Python dependency list
