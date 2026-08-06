"""`--train` must honour models.ml_retrain_days.

The bug
-------
``daily_update()`` asked ``_ml_models_stale()`` before retraining, but CI runs
``--update --skip-ml-retrain`` so the retrain is deferred to a dedicated
``--train`` step with its own timeout budget. The staleness decision was
therefore made inside the ``--update`` process and thrown away, and the
``--train`` entry point — the one that actually trains — never asked. Result:
the pipeline rebuilt features for 500 matches and refit Poisson/Elo *every
day*, despite ``ml_retrain_days: 3``, on two days out of three for nothing.

Classification: a Python bug in the CLI branch. The workflow step is even
named "Retrain ML models (if stale)", so the intent was always conditional;
the config key simply had no effect on that code path.
"""

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.utils.logger import utcnow


def _agent(trained_days_ago, retrain_days=3):
    """A stand-in agent whose models were last trained N days ago."""
    from src.agent.betting_agent import FootballBettingAgent

    agent = MagicMock()
    stamp = (utcnow() - timedelta(days=trained_days_ago)).isoformat()
    agent.predictor.ml_models.trained_at = stamp
    agent.predictor.goals_model.trained_at = stamp
    agent.config.get.side_effect = lambda key, default=None: (
        retrain_days if key == "models.ml_retrain_days" else default
    )
    # Use the real staleness check — that logic is not what was broken.
    agent._ml_models_stale = lambda max_age_days=3: (
        FootballBettingAgent._ml_models_stale(agent, max_age_days=max_age_days)
    )
    agent.train_ml_models = AsyncMock()
    agent.shutdown = AsyncMock()   # main() awaits this in its finally block
    return agent


async def _run_train(agent, argv):
    from src.agent.betting_agent import main
    with patch("src.agent.betting_agent.FootballBettingAgent", return_value=agent), \
         patch("src.agent.betting_agent._configure_cli_runtime"), \
         patch("sys.argv", argv):
        await main()


class TestTrainRespectsRetrainInterval:

    def test_fresh_models_are_not_retrained(self):
        agent = _agent(trained_days_ago=1, retrain_days=3)
        asyncio.run(_run_train(agent, ["betting_agent", "--train"]))
        agent.train_ml_models.assert_not_awaited()

    def test_stale_models_are_retrained(self):
        agent = _agent(trained_days_ago=4, retrain_days=3)
        asyncio.run(_run_train(agent, ["betting_agent", "--train"]))
        agent.train_ml_models.assert_awaited_once()

    def test_exactly_at_the_interval_retrains(self):
        """`age_days >= max_age_days` — day 3 with a 3-day interval trains."""
        agent = _agent(trained_days_ago=3, retrain_days=3)
        asyncio.run(_run_train(agent, ["betting_agent", "--train"]))
        agent.train_ml_models.assert_awaited_once()

    def test_never_trained_models_are_retrained(self):
        agent = _agent(trained_days_ago=0)
        agent.predictor.ml_models.trained_at = None
        agent.predictor.goals_model.trained_at = None
        asyncio.run(_run_train(agent, ["betting_agent", "--train"]))
        agent.train_ml_models.assert_awaited_once()

    def test_force_overrides_the_interval(self):
        agent = _agent(trained_days_ago=0, retrain_days=3)
        asyncio.run(_run_train(agent, ["betting_agent", "--train", "--force"]))
        agent.train_ml_models.assert_awaited_once()

    def test_a_longer_interval_is_honoured(self):
        """Config drives the decision — not a hard-coded 3."""
        agent = _agent(trained_days_ago=5, retrain_days=14)
        asyncio.run(_run_train(agent, ["betting_agent", "--train"]))
        agent.train_ml_models.assert_not_awaited()

    def test_one_stale_model_retrains_both(self):
        """The 1X2 classifier is fresh but the goals model is not."""
        agent = _agent(trained_days_ago=0, retrain_days=3)
        agent.predictor.goals_model.trained_at = (
            utcnow() - timedelta(days=9)).isoformat()
        asyncio.run(_run_train(agent, ["betting_agent", "--train"]))
        agent.train_ml_models.assert_awaited_once()
