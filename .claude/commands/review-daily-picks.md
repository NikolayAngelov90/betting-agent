Use the `bmad-code-review` skill to perform a full production code review of the daily World Cup football picks pipeline.

Context:
This project is a Football Betting Agent. The production daily GitHub Action is `.github/workflows/daily-picks.yml`. Its critical execution path runs:

- `python -m src.agent.betting_agent --update --skip-ml-retrain`
- `python -m src.agent.betting_agent --settle`
- `python -m src.agent.betting_agent --train`
- `python -m src.agent.betting_agent --picks --force`
- `python -m src.agent.betting_agent --briefing`
- `python -m src.agent.betting_agent --update-results`
- a second `python -m src.agent.betting_agent --settle`
- Sunday-only `python -m src.agent.betting_agent --report`

Review scope:
Trace the real execution path from the GitHub Action through `src/agent/betting_agent.py` and every module that participates in daily picks:

- data ingestion and scrapers
- database sessions, migrations, and SQLAlchemy models
- feature engineering
- Poisson, Elo, ML, GoalsML, Bayesian weights, and ensemble prediction
- value bet selection, EV/Kelly/risk filters, correlation filtering, exposure caps
- pick saving, deduplication, idempotency, and rerun safety
- Telegram output
- Claude match briefings
- result scraping and settlement
- model retraining, cache restore/save, calibration, and stale model guards

Review goals:

1. Verify whether the daily CI pipeline works correctly end-to-end.
2. Find bugs, race conditions, idempotency issues, incorrect assumptions, silent failures, risky `continue-on-error` behavior, timezone/date-window bugs, data leakage, stale cache problems, duplicate picks, database consistency issues, missing migrations/indexes, and World Cup-specific risks.
3. Review the model and betting logic: ensemble weights, bookmaker blending, calibration, ML training gates, feature availability, odds matching, EV thresholds, exposure cap, and correlation filtering.
4. Check whether tests cover the critical business rules. Identify missing tests.
5. Recommend high-impact improvements without getting distracted by cosmetic refactors.

Output requirements:

- Start with findings, ordered by severity: Critical, High, Medium, Low.
- For every finding, include exact file and line references.
- Explain why the issue matters in the daily CI / production betting context.
- Give a concrete recommended fix.
- Add a separate `Recommended Tests` section.
- Add a separate `Pipeline Improvements` section.
- Do not modify code yet. This run is review and improvement planning only.
- Do not focus on style, formatting, or minor cleanup unless it creates production risk.
