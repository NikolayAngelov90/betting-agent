# Football Betting Agent

A football prediction pipeline that runs once a day on GitHub Actions, **in
paper trading mode**. It records picks, sends them to a private Telegram
chat and grades them after the match. **No real money is staked, and no live
wager has ever been placed** — including on the picks made before
2026-08-11, when the paper flag was introduced. `betting.paper_trading_mode`
is `true` in [`config/config.example.yaml`](config/config.example.yaml).

The short version of what it found: the in-house models add nothing
measurable to the bookmakers' price, and no inefficiency found in the price
is larger than what it costs to bet. The details, with their numbers, are in
[What was measured](#what-was-measured).

## How picks are made now

Every value in this section is read from
[`config/config.example.yaml`](config/config.example.yaml), the file CI
deploys. A test (`tests/test_readme_matches_config.py`) fails if this README
and that file disagree on any of them.

```yaml
betting:
  selection_mode: confident_value
  confident_value:
    min_price: 1.70
    p_min: 0.54
    require_agreement: false
    weights:
      "1X2": {market: 0.94, elo: 0.04, poisson: 0.02}
      over_under: {market: 1.00, poisson: 0.00}
      btts: {market: 1.00, poisson: 0.00}
      team_goals: {market: 0.97, poisson: 0.03}
```

1. **A blended probability for every market leg** (home/draw/away, over/under
   goals, both teams to score, team goals). It is a weighted average of the
   bookmakers' price with the margin removed and two in-house models (an Elo
   rating and a Poisson goals model). The market takes **94–100%** of the
   weight:

   | market family | market | Elo | Poisson |
   |---|---|---|---|
   | home / draw / away | 0.94 | 0.04 | 0.02 |
   | over / under | 1.00 | — | 0.00 |
   | both teams to score | 1.00 | — | 0.00 |
   | team goals | 0.97 | — | 0.03 |

   The weights were fitted on fixtures from 2026-02-28 to 2026-08-31 and
   then tested, unchanged, on held-out fixtures from 2026-09-01 to
   2026-11-09. The machine-learning classifier is not part of the blend.
2. **The taken price** is the median price across bookmakers inside one
   snapshot of the odds feed.
3. **A leg qualifies** when its taken price is **at least 1.70** and its
   blended probability is **at least 0.54**.
4. **One pick per match**: the qualifying leg with the highest blended
   probability.
5. **No fixed daily count.** A day on which nothing qualifies has zero
   picks. The run records that as an outcome and sends one plain line
   instead of a card.

The previous rule (`standard`) is still computed for every fixture and
stored for comparison. Its picks are never sent. Changing any value above
changes the model version stamped on every pick, so results from different
settings are never mixed.

## Pipeline

All workflows run on `main`. Times are UTC. GitHub often starts scheduled
runs late.

| step | what it does | workflow, schedule |
|---|---|---|
| discovery | fetches fixtures from Flashscore, football-data.org and API-Football | [`daily-picks.yml`](.github/workflows/daily-picks.yml), Mon–Fri 01:00, Sat–Sun 00:30 |
| odds capture | stores bookmaker prices from API-Football and The Odds API | same run |
| settlement | grades earlier picks on the 90-minute score | same run, before and after picks |
| picks | applies the rule above and sends the day's picks to Telegram | same run |
| closing lines | refreshes prices near kickoff and records the closing price, within a monthly credit budget | [`closing-lines.yml`](.github/workflows/closing-lines.yml), 10:47, then 11:17 to 23:17 every two hours |
| audit | reads each run's log and alerts the operator on a new or worsening problem | [`ci-audit.yml`](.github/workflows/ci-audit.yml), after each picks or closing-lines run, with a 12:00 backstop |
| reminders | runs the date-bound and live checks | [`reminders.yml`](.github/workflows/reminders.yml), after each picks run, with a 13:00 backstop |

Two more workflows report but change nothing:
[`paper-trading-report.yml`](.github/workflows/paper-trading-report.yml)
(10:23) and [`non-gating-tests.yml`](.github/workflows/non-gating-tests.yml)
(after each picks run).

The picks run starts by running the test suite. If it fails, nothing after
it runs. `config/config.yaml` is gitignored: **CI copies
`config/config.example.yaml` over it** before anything reads the
configuration, so the example file is what runs in production.

## What was measured

Every figure below says what it is, how many observations it rests on and
when it was taken. *Measured* means read from recorded outcomes;
*simulated* means a replay of a rule over past fixtures. Figures computed
from bookmaker prices supplied by providers whose terms do not permit
publication are **withheld**: the finding and its method are stated, the
numbers are not (see [Data sources](#data-sources)).

**1. The in-house models add no information beyond the market price.**
Each test compares log-loss against the margin-free price. Its figures are
withheld because they are computed from provider prices.

* *Measured 2026-10-23.* Walk-forward test of home/draw/away log-loss, five
  folds from 2025-08-01 to 2026-07-01. Priced matches fall in only three of
  the folds, all in one odds period. Every mix of model and market scored
  worse than the market alone. The penalty shrank steadily as the market's
  weight rose, reaching zero at 100%.
* *Measured 2026-11-12.* Each candidate feature tested against the price,
  walk-forward by month, on home-win outcomes scored from 2026-05-01. Elo
  difference, rest difference, recent form and the stored model probability
  each failed to improve out-of-sample log-loss.
* *Measured 2026-11-12.* The blend weights above, fitted per family: "market
  only" lies inside every weight's 95% interval. On the held-out fixtures
  the blend was never better than the market.

**2. Proportional de-vigging manufactures a favourite–longshot pattern.**
Removing the bookmaker's margin *proportionally* (dividing each implied
probability by their sum) creates the pattern even on a perfectly calibrated
book. Suppose a book adds its margin M in equal shares across n outcomes.
Then, by arithmetic alone, the de-vigged probability misses the true one by
gap = M × (p − 1/n) / (1 + M). Favourites look underpriced and longshots
overpriced, and the sign flips at p = 1/n. *Simulated 2026-11-12, synthetic
data only:* take a perfectly calibrated three-way book with a 6.6% margin
added in equal shares. De-vigged proportionally, its worst calibration gap
in any band is 2.76 points; the estimator that matches its margin model
shows none. An earlier market-wide test in this project used proportional
de-vigging. **Its figures, and those of its correction, are withheld** until
the test is re-run under the Shin and power methods as well, on a source
whose terms permit publication (see [Data sources](#data-sources)).

**3. The cost of betting is larger than any inefficiency found.** *Measured
2026-11-06 on the settled paper picks from 2026-08-11 (n = 1,639).* The
bookmaker margin built into the prices this system took (the sum of 1/odds
minus 1 over each taken price set) exceeds every pricing inefficiency the
project has found. **The figures are withheld** for the same reason as
item 2: they are measured on prices supplied by providers whose terms do not
permit their publication.

**4. The settled paper record, by selection rule.** *Measured 2026-11-15.*
One pick is one observation. Voids are excluded, as are picks marked as
never issued or as built on corrupted data. Intervals are 95%, bootstrapped
by fixture (4,000 draws). The rules are never pooled.

| rule | period | n | hit rate | predicted | avg odds | return per 1-unit pick |
|---|---|---|---|---|---|---|
| standard + Claude review, before the paper flag | 2026-02-28 to 08-10 | 1,067 | 51.5% (48.5–54.7) | 59.9% | 1.94 | −4.1% (−10.1 to +2.2) |
| standard + Claude review, paper | 2026-08-11 to 11-09 | 1,816 | 58.6% (56.3–60.8) | 61.1% | 1.62 | −6.0% (−9.6 to −2.3) |
| standard, review removed, paper | 2026-11-13 to 11-14 | 20 | 55.0% | 61.2% | 1.58 | −13.8% — two days, no conclusion |
| `confident_value` (current), paper | from 2026-11-15 | 0 settled | — | — | — | — |

"Predicted" is the mean probability the system gave its own picks. The
review (removed 2026-11-12) could replace a pick; it reviewed 232 of the
first group's picks and 1,372 of the second's. No stake was placed on any
of them.

**5. What the current rule is expected to do.** *Simulated: a backtest on the
held-out fixtures, kickoffs 2026-09-01 to 2026-11-09, registered 2026-11-13.*
448 picks: hit rate **57.4%** (52.7–61.8) against 55.6% predicted, average
odds **1.73**, return **−0.5%** per 1-unit pick (−8.5 to +7.2). That is the
whole claim: about 57% at about 1.73, a return near zero. It is not a claim
of an edge. For comparison, *measured*: the previous rule's picks as
actually saved on the same fixtures were 1,456 picks, 58.3% (55.8–60.9)
against 62.8% predicted, at 1.63, with a return of −5.5% (−9.6 to −1.2).

**The no-edge level, as a principle.** A pick at price *o* has no edge when
it wins exactly as often as the price implies once the margin is removed.
That is about 1 / (*o* × (1 + *m*)), where *m* is the margin of the taken
price over a margin-free price, so at 1.70 it is 1 / (1.70 × (1 + *m*)). To
return money at all, a pick at 1.70 must win more than 1 / 1.70 = 58.8% of
the time. The measured *m*, the no-edge level it gives, and the number of
picks needed to tell 57% apart from that level are **withheld**, because *m*
is computed from provider prices. Telling 57% apart from the previous rule's
58.3% would take five to eleven thousand picks, which is years.

Detail: [the decision paper](docs/decision-2026-11-01.md) ·
[the predictive audit](docs/predictive-audit-2026-08-07.md) ·
[the current rule's registration](docs/confident-value-prospective-registration.md) ·
[the ledger](docs/ci-audit-ledger.md), which records every stage with its corrections.

## Setup

CI uses **Python 3.11** on Ubuntu, with Chrome and Xvfb for the Flashscore
scraper.

```bash
python -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements.txt -c constraints.txt
python -m camoufox fetch
python -c "import nltk; nltk.download('vader_lexicon')"
cp config/config.example.yaml config/config.yaml
```

Environment variables, by name. Values belong in your shell or in GitHub
repository secrets, never in a file in this repository:

| variable | used for |
|---|---|
| `DATABASE_URL` | PostgreSQL connection |
| `API_FOOTBALL_KEY` | API-Football |
| `FOOTBALL_DATA_ORG_KEY` | football-data.org |
| `ODDS_API_KEY` | The Odds API |
| `TELEGRAM_BOT_TOKEN` | the Telegram bot |
| `TELEGRAM_CHAT_ID` | the chat that receives picks |
| `TELEGRAM_OPERATOR_CHAT_ID` | the chat that receives operational alerts |
| `MODEL_HMAC_KEY` | integrity check on saved ML model files |

Without `DATABASE_URL` the program refuses to start; `BETTING_AGENT_ALLOW_SQLITE=1`
opts in to a local SQLite file. The test suite sets it and removes
`DATABASE_URL`, so tests never reach production.

```bash
python -m src.agent.betting_agent --update     # fixtures and odds
python -m src.agent.betting_agent --picks      # select and send
python -m src.agent.betting_agent --settle     # grade
python -m pytest tests/
```

Every configuration key, with its reason, is in [`config/config.example.yaml`](config/config.example.yaml).

## Data sources

| provider | supplies | terms, as checked 2026-11-11 |
|---|---|---|
| [football-data.org](https://www.football-data.org/) | fixtures and results for top leagues, and the kickoff schedule used to time the daily run | attribution required (below) |
| [API-Football](https://www.api-football.com/) | fixtures, odds, expected goals, injuries | does not license publication of its data; the terms page refused a direct fetch, so it was read via search |
| [Flashscore](https://www.flashscore.com/) | fixtures, results, match statistics (scraped) | forbids commercial use, automated access and making its content public |
| [The Odds API](https://the-odds-api.com/) | prices near kickoff, for closing lines | values derived from its data may be displayed; reselling the raw data is not allowed |
| [football-data.co.uk](https://www.football-data.co.uk/) | historical results and odds (CSV) | not yet checked |

**This README publishes no data from Flashscore or API-Football**: no odds,
results, rows or tables. Figures computed from prices that include their
data are withheld: the log-loss comparisons (item 1), the de-vigging band
test (item 2), the cost of betting (item 3), and the measured margin with
the no-edge level that follows from it (item 5). They return only after a
re-run on football-data.co.uk closing odds or The Odds API, with that
source's terms checked first. The terms check, with clauses quoted and fetch
dates, and the registered re-run for items 2 and 3 are in
[`docs/findings-inventory.md`](docs/findings-inventory.md). Items 1 and 5
wait on the same re-run.

Football data provided by the Football-Data.org API.

## Disclaimer

This is a research project, not betting advice. Every result above is a
paper result: nothing was staked. Betting involves the risk of losing money,
and on the evidence here this system has no demonstrated edge over the
bookmakers' price. If gambling is affecting you or someone close to you,
contact a gambling support service in your country.
