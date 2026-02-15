# Match Analysis Workflow - Instructions

<critical>The workflow execution engine is governed by: {project-root}/.bmad/core/tasks/workflow.xml</critical>
<critical>You MUST have already loaded and processed: {project-root}/.bmad/sports-betting/workflows/match-analysis/workflow.yaml</critical>
<critical>Communicate in {communication_language} throughout the analysis process</critical>
<critical>Embody the Football Betting Analyst persona: Evidence-based systematic approach. Patterns and correlations.</critical>

<workflow>

<step n="1" goal="Identify match to analyze">
  <ask>Which match would you like to analyze?

Please provide:
- **Home Team**: (e.g., Arsenal, Barcelona, Bayern Munich)
- **Away Team**: (e.g., Chelsea, Real Madrid, Dortmund)
- **Match Date**: (e.g., 2025-11-30, or "today", "tomorrow")

Or provide a match ID if you have one from the database.
  </ask>

  <action>Store match details as {{home_team}}, {{away_team}}, and {{match_date}}</action>
  <action>Normalize team names and verify match exists in upcoming fixtures</action>
  <action if="match not found">Inform user and ask for clarification or alternative match</action>

  <template-output>match_details</template-output>
</step>

<step n="2" goal="Gather comprehensive match data">
  <action>Connect to the football betting database and gather all available data for this match</action>

  <action>Collect the following data systematically:

**Match Context:**
- League/competition information
- Venue details
- Kickoff time
- Match importance (derby, rivalry, title implications)

**Team Statistics (for both teams):**
- Recent form (last 10 matches overall, last 5 home/away)
- League position and points
- Goals scored and conceded averages
- Shots, possession, corners statistics
- Discipline record (yellow/red cards)
- Current win/draw/loss streaks

**Head-to-Head Data:**
- Last 10 meetings between these teams
- H2H at this venue (last 5 matches)
- Goal patterns in H2H matches
- BTTS (Both Teams To Score) percentage
- Over 2.5 goals percentage

**Injury & Squad Information:**
- Current injuries and suspensions for both teams
- Key player availability (top scorers, captains, goalkeepers)
- Expected return dates for injured players
- Player market values (for injury impact assessment)

**News & Sentiment:**
- Recent news articles about both teams (last 7 days)
- Manager quotes and press conference highlights
- Transfer activity or squad changes
- Team morale indicators

**Odds Data:**
- Current odds from multiple bookmakers
- Market movements (opening vs current odds)
- Available markets (1X2, Over/Under, BTTS, Asian Handicap, etc.)
  </action>

  <action>Report data collection status:
- What data was successfully retrieved
- Any missing or unavailable data sources
- Data freshness (when was it last updated)
  </action>

  <action if="critical data missing">Warn user that analysis may be limited and explain impact</action>

  <template-output>data_summary</template-output>
</step>

<step n="3" goal="Analyze team form and context">
  <action>Conduct systematic form analysis using collected data:

**Home Team Form Analysis:**
- Calculate points per game (overall, home, last 5)
- Goals scored/conceded trends
- Win/draw/loss distribution
- Form trajectory (improving, declining, stable)
- Home record specifics (fortress or vulnerable?)
- Recent performance against similar opponents

**Away Team Form Analysis:**
- Calculate points per game (overall, away, last 5)
- Goals scored/conceded trends
- Win/draw/loss distribution
- Form trajectory analysis
- Away record specifics (strong travelers or struggles?)
- Recent performance in similar away fixtures

**Contextual Factors:**
- Is this a derby or local rivalry? (emotional intensity)
- Schedule congestion (midweek European matches, quick turnaround)
- Travel distance for away team
- Weather conditions if significant
- Historical patterns at this time of season
- Motivation factors (fighting relegation, chasing title, etc.)

**League Position Context:**
- Position difference between teams
- Points gap and its significance
- Form of teams around them in table
- Pressure situations (must-win scenarios)
  </action>

  <action>Synthesize form analysis into clear insights:
- What patterns emerge from the data?
- Which team has momentum?
- Are there notable strengths or weaknesses?
- How do recent trends compare to season averages?
  </action>

  <template-output>form_analysis</template-output>
</step>

<step n="4" goal="Assess injury impact and news sentiment">
  <action>Calculate injury impact scores systematically:

**Injury Impact Assessment:**
For each team, evaluate:
- **Total players unavailable** (injured + suspended)
- **Key player status** (goalkeeper, captain, top scorer, star players)
- **Positional impact** (defensive stability, attacking threat, midfield control)
- **Market value impact** (sum of injured players' market values)
- **Market value percentage** (injured value / total squad value)

Create injury impact score (0-10 scale):
- 0-2: Minimal impact (non-key players only)
- 3-5: Moderate impact (some rotation players or one key player)
- 6-8: Significant impact (multiple key players or critical position)
- 9-10: Severe impact (star players, goalkeeper, or defensive crisis)
  </action>

  <action>Analyze news sentiment and morale indicators:

**News Sentiment Analysis:**
- Positive news (good form, player returns, manager confidence)
- Negative news (poor results, internal issues, criticism)
- Neutral/tactical news (lineup speculation, opponent analysis)

**Manager Pressure Assessment:**
- Is manager under pressure? (recent poor results, media criticism)
- Managerial stability vs. turmoil

**Squad Morale Indicators:**
- Recent transfer activity (strengthening or sales)
- Team unity signals from quotes/reports
- Confidence levels from recent performances
  </action>

  <action>Integrate injury and sentiment analysis:
- How do injuries affect tactical setup?
- Does news sentiment suggest psychological edge?
- Are there hidden factors that could influence performance?
  </action>

  <template-output>injury_impact</template-output>
  <template-output>news_sentiment</template-output>
</step>

<step n="5" goal="Generate predictions using multiple models">
  <action>Run prediction models systematically:

**Note:** If prediction models are not yet implemented, use statistical analysis and historical patterns to provide probability estimates. Clearly indicate methodology used.

**1. Statistical/Poisson Analysis:**
- Calculate attack and defense strength ratings
- Expected goals (xG) for home and away team
- Probability distribution for different scorelines
- Market probabilities (1X2, Over/Under, BTTS)

**2. Form-Based Predictions:**
- Weight recent form heavily
- Adjust for home/away splits
- Factor in H2H patterns
- Consider injury adjustments

**3. Historical Pattern Analysis:**
- H2H outcome patterns
- Performance in similar matchups
- Seasonal trends
- Derby/rivalry outcome tendencies

**4. Ensemble Approach:**
- Combine multiple prediction methods
- Weight based on confidence levels
- Generate probability ranges
- Calculate confidence intervals
  </action>

  <action>Generate predictions for key markets:

**1X2 Market:**
- Home Win probability: X%
- Draw probability: X%
- Away Win probability: X%

**Over/Under Goals:**
- Over 2.5 goals probability: X%
- Under 2.5 goals probability: X%
- Total goals expected: X.XX

**BTTS (Both Teams To Score):**
- BTTS Yes probability: X%
- BTTS No probability: X%

**Other Markets (if applicable):**
- Asian Handicap suggestions
- Correct score most likely
- Goalscorer probabilities
  </action>

  <action>Document model confidence:
- Which predictions have high confidence?
- Where is there uncertainty?
- What factors could shift probabilities?
  </action>

  <template-output>predictions</template-output>
</step>

<step n="6" goal="Identify value betting opportunities">
  <action>Calculate expected value (EV) for all available markets:

**Expected Value Formula:**
EV = (Probability × Decimal Odds) - 1

**Minimum Thresholds:**
- Minimum EV: +5% (+0.05)
- Minimum Confidence: 55%
- Odds range: 1.30 to 10.0

For each market where odds are available:
1. Get our predicted probability
2. Get current best odds from bookmakers
3. Calculate EV
4. Flag if EV > minimum threshold
  </action>

  <action>Apply Kelly Criterion for stake sizing:

**Kelly Criterion Formula:**
Kelly % = (bp - q) / b

Where:
- b = decimal odds - 1
- p = probability of winning
- q = 1 - p

**Use Fractional Kelly (25%) for risk management:**
Recommended Stake % = (Kelly % × 0.25)

**Stake Limits:**
- Maximum stake: 5% of bankroll per bet
- Minimum stake: 1% of bankroll
  </action>

  <action>Assess risk levels for each bet:

**Risk Classification:**
- **Low Risk**: High confidence (>65%), Low odds (1.50-2.50), Strong EV (>8%)
- **Medium Risk**: Medium confidence (55-65%), Medium odds (2.00-4.00), Moderate EV (5-8%)
- **High Risk**: Lower confidence (50-55%), Higher odds (4.00+), Marginal EV (5-6%)
  </action>

  <action>Create value bet recommendations:

For each value bet identified:
- **Market & Selection**: What are we betting on?
- **Odds**: Current best odds available
- **Predicted Probability**: Our calculated probability
- **Expected Value**: +X.X%
- **Confidence**: High/Medium/Low
- **Kelly Stake**: X.XX% of bankroll
- **Recommended Stake**: X.XX% (fractional Kelly)
- **Risk Level**: Low/Medium/High
- **Reasoning**: Why this bet has value (2-3 sentences)
  </action>

  <action>Rank recommendations by quality:
- Sort by: (Expected Value × Confidence Score)
- Highlight top 3 value bets
- Note any hedge opportunities
  </action>

  <template-output>value_bets</template-output>
</step>

<step n="7" goal="Generate comprehensive match analysis report">
  <action>Compile all findings into the structured template</action>

  <action>Format the report with clear sections:
- Executive Summary (key findings in 3-5 bullets)
- Match Overview (teams, date, league, context)
- Team Form Analysis (recent performance, trends)
- Head-to-Head Insights (historical patterns)
- Injury Report & Impact (key absences, severity)
- News & Sentiment Summary (morale, pressure, context)
- Predictions (probabilities for all markets)
- Value Bet Recommendations (top opportunities with reasoning)
  </action>

  <action>Ensure all variables are populated in the template:
- {{home_team}}, {{away_team}}, {{match_date}}
- {{match_details}}
- {{data_summary}}
- {{form_analysis}}
- {{injury_impact}}, {{news_sentiment}}
- {{predictions}}
- {{value_bets}}
  </action>

  <action>Save the completed report to: {output_folder}/match-analysis-{{home_team}}-vs-{{away_team}}-{{date}}.md</action>

  <action>Provide summary to {user_name} in {communication_language}:
- Confirm report has been generated
- Highlight top 2-3 value betting opportunities
- Note any critical findings (major injuries, strong value bets, etc.)
- Remind about responsible gambling principles
  </action>
</step>

</workflow>

<critical>Remember: All betting recommendations are for informational and educational purposes. Always bet responsibly and never bet more than you can afford to lose.</critical>
