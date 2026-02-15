# Football Betting Prediction Agent - Claude Code Prompt

## Project Overview

Build a comprehensive football betting prediction agent that analyzes multiple data sources to generate value betting predictions. The agent should scrape data from Flashscore, aggregate statistics, monitor injuries, read news, and produce intelligent betting recommendations.

---

## Core Requirements

### 1. Data Collection Layer

#### 1.1 FlashscoreScraping Integration
Clone and integrate the FlashscoreScraping repository:
```
https://github.com/gustavofariaa/FlashscoreScraping
```

Extend the scraper to collect:
- **Match Results**: Historical match outcomes (home/away goals, halftime scores)
- **Fixtures**: Upcoming matches with dates, times, venues
- **League Standings**: Current table positions, points, goal difference
- **Team Statistics**: Goals scored/conceded, shots, possession, corners, cards
- **H2H Data**: Head-to-head records between teams (last 10-20 meetings)
- **Form Data**: Last 5-10 matches for each team (home and away separately)
- **Odds History**: Opening and closing odds if available

#### 1.2 Additional Data Sources to Integrate
Create scrapers or API integrations for:

**Injury & Squad Data:**
- Transfermarkt (https://www.transfermarkt.com) - injuries, suspensions, market values
- Soccerway - squad information
- Team official websites for confirmed lineups

**News Sources:**
- Goal.com, ESPN FC, BBC Sport, Sky Sports
- Team-specific news feeds
- Social media monitoring (Twitter/X for official team accounts)

**Odds Data:**
- The Odds API (https://the-odds-api.com) - requires API key
- Oddschecker scraping
- Betfair Exchange data for market movements

---

### 2. Database Schema

Design a PostgreSQL or SQLite database with the following tables:

```sql
-- Teams table
CREATE TABLE teams (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    country VARCHAR(50),
    league VARCHAR(100),
    flashscore_id VARCHAR(50),
    transfermarkt_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Matches table (historical + fixtures)
CREATE TABLE matches (
    id SERIAL PRIMARY KEY,
    home_team_id INTEGER REFERENCES teams(id),
    away_team_id INTEGER REFERENCES teams(id),
    match_date TIMESTAMP,
    league VARCHAR(100),
    season VARCHAR(20),
    -- Results (NULL for fixtures)
    home_goals INTEGER,
    away_goals INTEGER,
    ht_home_goals INTEGER,
    ht_away_goals INTEGER,
    -- Statistics
    home_shots INTEGER,
    away_shots INTEGER,
    home_shots_on_target INTEGER,
    away_shots_on_target INTEGER,
    home_possession DECIMAL(5,2),
    away_possession DECIMAL(5,2),
    home_corners INTEGER,
    away_corners INTEGER,
    home_fouls INTEGER,
    away_fouls INTEGER,
    home_yellow_cards INTEGER,
    away_yellow_cards INTEGER,
    home_red_cards INTEGER,
    away_red_cards INTEGER,
    -- Match status
    is_fixture BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Players table
CREATE TABLE players (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    team_id INTEGER REFERENCES teams(id),
    position VARCHAR(50),
    market_value DECIMAL(15,2),
    is_key_player BOOLEAN DEFAULT FALSE,
    transfermarkt_id VARCHAR(50)
);

-- Injuries table
CREATE TABLE injuries (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(id),
    team_id INTEGER REFERENCES teams(id),
    injury_type VARCHAR(100),
    start_date DATE,
    expected_return DATE,
    status VARCHAR(50), -- 'out', 'doubtful', 'available'
    source VARCHAR(200),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Odds table
CREATE TABLE odds (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(id),
    bookmaker VARCHAR(50),
    market_type VARCHAR(50), -- '1X2', 'over_under', 'btts', 'asian_handicap'
    selection VARCHAR(50),
    odds_value DECIMAL(8,3),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- News table
CREATE TABLE news (
    id SERIAL PRIMARY KEY,
    team_id INTEGER REFERENCES teams(id),
    headline TEXT,
    content TEXT,
    source VARCHAR(100),
    url VARCHAR(500),
    sentiment_score DECIMAL(5,3),
    published_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Predictions table
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(id),
    prediction_type VARCHAR(50),
    predicted_outcome VARCHAR(50),
    confidence DECIMAL(5,3),
    expected_value DECIMAL(8,3),
    model_version VARCHAR(50),
    features_used JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

### 3. Feature Engineering Module

Create a feature engineering pipeline that calculates:

#### 3.1 Team Form Features
```python
# Calculate for each team:
- points_last_5_matches (home/away/overall)
- goals_scored_last_5 (home/away/overall)
- goals_conceded_last_5 (home/away/overall)
- clean_sheets_last_5
- failed_to_score_last_5
- win_streak / losing_streak
- unbeaten_run
- shots_per_game_avg
- shots_on_target_per_game_avg
- possession_avg
- corners_per_game_avg
```

#### 3.2 H2H Features
```python
# Head-to-head statistics:
- h2h_home_wins_last_10
- h2h_away_wins_last_10
- h2h_draws_last_10
- h2h_avg_total_goals
- h2h_btts_percentage
- h2h_over_25_percentage
- h2h_home_team_scored_percentage
- h2h_away_team_scored_percentage
```

#### 3.3 Injury Impact Score
```python
# Calculate injury impact:
- total_injured_players_count
- key_players_injured_count
- injured_players_market_value_sum
- injured_players_market_value_percentage (of total squad)
- goalkeeper_available (boolean)
- top_scorer_available (boolean)
- captain_available (boolean)
- defensive_stability_score (based on injured defenders)
- attacking_threat_score (based on injured attackers)
```

#### 3.4 League Position Features
```python
- league_position
- points
- goal_difference
- points_per_game
- home_points_per_game
- away_points_per_game
- position_difference_between_teams
```

#### 3.5 Contextual Features
```python
- days_since_last_match
- is_derby (boolean)
- is_rivalry_match (boolean)
- travel_distance (for away team)
- match_importance_score
- european_competition_hangover (played midweek in Europe)
```

#### 3.6 News Sentiment Features
```python
- team_news_sentiment_score (last 7 days)
- manager_pressure_score
- transfer_activity_score
- squad_morale_indicator
```

---

### 4. Prediction Models

Implement multiple prediction models:

#### 4.1 Poisson Model
```python
class PoissonModel:
    """
    Classic Poisson distribution model for predicting goals.
    - Calculate attack/defense strength for each team
    - Predict expected goals (xG) for home and away
    - Generate probability distributions for exact scores
    - Calculate 1X2, Over/Under, BTTS probabilities
    """
```

#### 4.2 Machine Learning Models
```python
# Implement ensemble of models:
- XGBoost Classifier (for 1X2 prediction)
- Random Forest Classifier
- Logistic Regression (baseline)
- Neural Network (for complex patterns)

# For each model:
- Train on historical data (70/15/15 split)
- Cross-validate with time-series aware splits
- Optimize hyperparameters
- Track feature importance
- Monitor prediction accuracy over time
```

#### 4.3 Elo Rating System
```python
class EloRatingSystem:
    """
    Maintain Elo ratings for all teams.
    - Update after each match
    - Use home advantage factor
    - Calculate win/draw/lose probabilities
    - Track rating history
    """
```

#### 4.4 Ensemble Model
```python
class EnsemblePredictor:
    """
    Combine predictions from all models:
    - Weighted average based on recent accuracy
    - Bayesian model averaging
    - Output confidence intervals
    """
```

---

### 5. Value Betting Calculator

#### 5.1 Expected Value Calculation
```python
def calculate_expected_value(predicted_probability, odds):
    """
    EV = (Probability × Odds) - 1
    
    Only recommend bets where:
    - EV > 0.05 (5% edge minimum)
    - Confidence > 0.6
    - Odds are between 1.3 and 10.0
    """
```

#### 5.2 Kelly Criterion for Stake Sizing
```python
def kelly_criterion(probability, odds, bankroll, fraction=0.25):
    """
    Calculate optimal stake using Kelly Criterion.
    Use fractional Kelly (25%) for risk management.
    
    Kelly % = (bp - q) / b
    where:
    - b = odds - 1
    - p = probability of winning
    - q = 1 - p
    """
```

#### 5.3 Bet Recommendation Output
```python
@dataclass
class BetRecommendation:
    match: str
    market: str  # '1X2', 'Over 2.5', 'BTTS', 'Asian Handicap'
    selection: str
    odds: float
    predicted_probability: float
    expected_value: float
    confidence: float
    kelly_stake_percentage: float
    recommended_stake: float
    reasoning: str
    risk_level: str  # 'low', 'medium', 'high'
    injury_impact: str
    h2h_insight: str
    form_insight: str
    news_insight: str
```

---

### 6. Agent Architecture

#### 6.1 Main Agent Class
```python
class FootballBettingAgent:
    def __init__(self, config):
        self.scraper = FlashscoreScraper()
        self.injury_tracker = InjuryTracker()
        self.news_aggregator = NewsAggregator()
        self.odds_collector = OddsCollector()
        self.feature_engineer = FeatureEngineer()
        self.predictor = EnsemblePredictor()
        self.value_calculator = ValueBettingCalculator()
        self.db = DatabaseManager()
        
    async def daily_update(self):
        """Run daily data collection and updates."""
        await self.scraper.update_results()
        await self.scraper.update_fixtures()
        await self.injury_tracker.update_all_teams()
        await self.news_aggregator.collect_latest_news()
        await self.odds_collector.update_odds()
        
    async def analyze_fixture(self, match_id):
        """Complete analysis of a single fixture."""
        match = self.db.get_match(match_id)
        
        # Gather all data
        home_team_data = await self.gather_team_data(match.home_team_id)
        away_team_data = await self.gather_team_data(match.away_team_id)
        h2h_data = await self.get_h2h_data(match.home_team_id, match.away_team_id)
        injuries = await self.injury_tracker.get_injuries(match_id)
        news = await self.news_aggregator.get_team_news([match.home_team_id, match.away_team_id])
        odds = await self.odds_collector.get_odds(match_id)
        
        # Generate features
        features = self.feature_engineer.create_features(
            home_team_data, away_team_data, h2h_data, injuries, news
        )
        
        # Make predictions
        predictions = self.predictor.predict(features)
        
        # Calculate value bets
        recommendations = self.value_calculator.find_value_bets(predictions, odds)
        
        return MatchAnalysis(
            match=match,
            features=features,
            predictions=predictions,
            recommendations=recommendations,
            injury_report=injuries,
            news_summary=news
        )
        
    async def get_daily_picks(self, date=None, min_ev=0.05, max_picks=10):
        """Get top value betting picks for a specific date."""
        fixtures = await self.db.get_fixtures(date or datetime.today())
        
        all_recommendations = []
        for fixture in fixtures:
            analysis = await self.analyze_fixture(fixture.id)
            all_recommendations.extend(analysis.recommendations)
        
        # Sort by expected value and confidence
        sorted_picks = sorted(
            all_recommendations,
            key=lambda x: (x.expected_value * x.confidence),
            reverse=True
        )
        
        return sorted_picks[:max_picks]
```

#### 6.2 Automated Pipeline
```python
class AutomatedPipeline:
    """
    Scheduled tasks:
    - 06:00: Update overnight results
    - 08:00: Collect injury updates
    - 10:00: Aggregate news
    - 12:00: Update odds
    - 14:00: Generate predictions for today's matches
    - 18:00: Update odds again (pre-match)
    - 22:00: Collect results, update models
    """
    
    def __init__(self, agent):
        self.agent = agent
        self.scheduler = AsyncIOScheduler()
        
    def setup_schedule(self):
        self.scheduler.add_job(self.agent.daily_update, 'cron', hour=6)
        # ... add other scheduled tasks
```

---

### 7. Output & Reporting

#### 7.1 Match Analysis Report
```markdown
# Match Analysis: [Home Team] vs [Away Team]
**Date:** [Date] | **League:** [League] | **Kickoff:** [Time]

## Team Form
### [Home Team] (Home Form: W-W-D-L-W)
- Last 5 home: 12 pts, 8 goals scored, 3 conceded
- League position: 5th (42 pts)
- Key stat: Won 4 of last 5 home games

### [Away Team] (Away Form: L-D-W-W-L)
- Last 5 away: 7 pts, 5 goals scored, 6 conceded
- League position: 11th (31 pts)
- Key stat: Failed to score in 3 of last 5 away

## Head-to-Head (Last 10 meetings)
- Home wins: 5 | Draws: 3 | Away wins: 2
- Avg goals: 2.7 per game
- BTTS: 60% | Over 2.5: 50%

## Injury Report
### [Home Team]
- ❌ [Key Player] (ACL) - Market Value: €45M
- ⚠️ [Player] (Doubtful - Hamstring)

### [Away Team]
- ❌ [Goalkeeper] (Concussion)
- ✅ [Star Player] (Returned from injury)

## News & Sentiment
- [Summary of relevant news]
- Team morale: [High/Medium/Low]
- Manager pressure: [High/Medium/Low]

## Predictions
| Market | Prediction | Probability | Odds | EV | Confidence |
|--------|------------|-------------|------|-----|------------|
| 1X2 | Home Win | 52% | 1.85 | +3.8% | High |
| Over 2.5 | Yes | 58% | 1.95 | +13.1% | Medium |
| BTTS | Yes | 61% | 1.80 | +9.8% | High |

## Value Bet Recommendations
1. **BTTS Yes @ 1.80** ⭐⭐⭐
   - EV: +9.8%
   - Reasoning: Both teams score regularly, H2H supports this
   - Stake: 2.5% of bankroll

2. **Home Win @ 1.85** ⭐⭐
   - EV: +3.8%
   - Reasoning: Strong home form, key away player injured
   - Stake: 1.5% of bankroll
```

#### 7.2 Daily Summary Dashboard
```python
class DashboardGenerator:
    """
    Generate daily dashboard with:
    - Today's fixtures with predictions
    - Top value bets
    - Injury alerts
    - Breaking news
    - Yesterday's results vs predictions (accuracy tracking)
    - Weekly/Monthly performance metrics
    - Bankroll tracking
    """
```

---

### 8. Configuration

```yaml
# config.yaml
scraping:
  flashscore_leagues:
    - england/premier-league
    - spain/laliga
    - germany/bundesliga
    - italy/serie-a
    - france/ligue-1
    - netherlands/eredivisie
    - portugal/primeira-liga
    - champions-league
    - europa-league
  
  update_interval_hours: 6
  headless: true

database:
  type: postgresql  # or sqlite
  host: localhost
  port: 5432
  name: football_betting
  
models:
  ensemble_weights:
    poisson: 0.25
    xgboost: 0.35
    random_forest: 0.20
    elo: 0.20
  
  retrain_interval_days: 7
  
betting:
  min_odds: 1.30
  max_odds: 10.0
  min_expected_value: 0.05
  min_confidence: 0.55
  kelly_fraction: 0.25
  max_stake_percentage: 5.0
  
notifications:
  telegram_bot_token: "your_token"
  telegram_chat_id: "your_chat_id"
  send_daily_picks: true
  send_live_alerts: true
```

---

### 9. Project Structure

```
football-betting-agent/
├── src/
│   ├── scrapers/
│   │   ├── __init__.py
│   │   ├── flashscore_scraper.py
│   │   ├── injury_scraper.py
│   │   ├── news_scraper.py
│   │   └── odds_scraper.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── database.py
│   │   ├── models.py (SQLAlchemy models)
│   │   └── migrations/
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_engineer.py
│   │   ├── team_features.py
│   │   ├── h2h_features.py
│   │   └── injury_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── poisson_model.py
│   │   ├── ml_models.py
│   │   ├── elo_system.py
│   │   └── ensemble.py
│   ├── betting/
│   │   ├── __init__.py
│   │   ├── value_calculator.py
│   │   ├── kelly_criterion.py
│   │   └── bankroll_manager.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── betting_agent.py
│   │   └── scheduler.py
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── match_report.py
│   │   ├── dashboard.py
│   │   └── telegram_bot.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── logger.py
├── tests/
│   ├── test_scrapers.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_betting.py
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── model_training.ipynb
│   └── backtesting.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── config/
│   └── config.yaml
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---

### 10. Requirements

```txt
# requirements.txt

# Scraping
selenium>=4.15.0
beautifulsoup4>=4.12.0
requests>=2.31.0
aiohttp>=3.9.0
playwright>=1.40.0

# Database
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
alembic>=1.13.0

# Data Processing
pandas>=2.1.0
numpy>=1.26.0

# Machine Learning
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.1.0
tensorflow>=2.15.0  # optional, for neural network

# Statistics
scipy>=1.11.0
statsmodels>=0.14.0

# NLP (for news sentiment)
transformers>=4.35.0
nltk>=3.8.0

# Scheduling
apscheduler>=3.10.0

# Notifications
python-telegram-bot>=20.0

# Utilities
pyyaml>=6.0.0
python-dotenv>=1.0.0
loguru>=0.7.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

---

### 11. Getting Started Commands

```bash
# 1. Clone and setup
git clone https://github.com/gustavofariaa/FlashscoreScraping.git
cd FlashscoreScraping

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup database
python -m src.data.database init

# 5. Run initial data collection
python -m src.agent.betting_agent --init

# 6. Start scheduled agent
python -m src.agent.betting_agent --run

# 7. Get today's picks
python -m src.agent.betting_agent --picks
```

---

### 12. Key Implementation Notes

1. **Rate Limiting**: Implement respectful scraping with delays between requests (2-5 seconds)

2. **Error Handling**: Robust error handling for network failures, missing data, website changes

3. **Data Validation**: Validate all scraped data before storing

4. **Model Monitoring**: Track model performance over time, alert on degradation

5. **Backtesting**: Before live betting, backtest strategies on historical data

6. **Bankroll Management**: Never bet more than configured maximum stake

7. **Legal Compliance**: Ensure compliance with local gambling laws and website ToS

8. **Caching**: Cache API responses and scraping results to reduce load

9. **Logging**: Comprehensive logging for debugging and audit trails

10. **Testing**: Unit tests for all critical components, integration tests for pipelines

---

## Example Usage Session

```python
# Initialize agent
agent = FootballBettingAgent(config='config/config.yaml')

# Get today's best picks
picks = await agent.get_daily_picks(min_ev=0.05, max_picks=5)

for pick in picks:
    print(f"""
    Match: {pick.match}
    Bet: {pick.selection} @ {pick.odds}
    EV: {pick.expected_value:.1%}
    Confidence: {pick.confidence:.1%}
    Reasoning: {pick.reasoning}
    Stake: {pick.recommended_stake:.2f} units
    """)

# Detailed analysis of specific match
analysis = await agent.analyze_fixture(match_id=12345)
print(analysis.to_report())
```

---

## Success Metrics

Track these metrics to evaluate agent performance:

- **ROI (Return on Investment)**: Target > 5% long-term
- **Hit Rate**: Percentage of winning bets
- **Yield**: Profit / Total Stakes
- **CLV (Closing Line Value)**: Compare predicted odds vs closing odds
- **Brier Score**: Probability calibration accuracy
- **Model Accuracy**: Per-model and ensemble accuracy

---

*Remember: Gambling involves risk. This tool is for informational and educational purposes. Always bet responsibly and never bet more than you can afford to lose.*
