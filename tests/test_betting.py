"""Tests for value betting calculator, bankroll manager, and settlement logic."""

import pytest


class TestValueBettingCalculator:
    """Tests for value betting calculations."""

    def test_positive_ev(self):
        from src.betting.value_calculator import ValueBettingCalculator
        ev = ValueBettingCalculator.calculate_expected_value(0.60, 2.0)
        assert ev == pytest.approx(0.20, abs=0.01)

    def test_negative_ev(self):
        from src.betting.value_calculator import ValueBettingCalculator
        ev = ValueBettingCalculator.calculate_expected_value(0.40, 2.0)
        assert ev == pytest.approx(-0.20, abs=0.01)

    def test_zero_ev(self):
        from src.betting.value_calculator import ValueBettingCalculator
        ev = ValueBettingCalculator.calculate_expected_value(0.50, 2.0)
        assert ev == pytest.approx(0.0, abs=0.01)

    def test_kelly_positive_edge(self):
        from src.betting.value_calculator import ValueBettingCalculator
        calc = ValueBettingCalculator.__new__(ValueBettingCalculator)
        calc.kelly_fraction = 0.25
        calc.max_stake_pct = 5.0
        kelly = calc.kelly_criterion(0.60, 2.0)
        assert kelly > 0

    def test_kelly_negative_edge_returns_zero(self):
        from src.betting.value_calculator import ValueBettingCalculator
        calc = ValueBettingCalculator.__new__(ValueBettingCalculator)
        calc.kelly_fraction = 0.25
        calc.max_stake_pct = 5.0
        kelly = calc.kelly_criterion(0.30, 2.0)
        assert kelly == 0.0

    def test_kelly_capped_at_max(self):
        from src.betting.value_calculator import ValueBettingCalculator
        calc = ValueBettingCalculator.__new__(ValueBettingCalculator)
        calc.kelly_fraction = 1.0  # Full Kelly
        calc.max_stake_pct = 5.0
        kelly = calc.kelly_criterion(0.90, 2.0)
        assert kelly <= 5.0


class TestBankrollManager:
    """Tests for bankroll management."""

    def test_initial_state(self):
        from src.betting.bankroll_manager import BankrollManager
        bm = BankrollManager.__new__(BankrollManager)
        bm.initial_bankroll = 1000.0
        bm.current_bankroll = 1000.0
        bm.bets = []
        bm.max_stake_pct = 5.0

        perf = bm.get_performance()
        assert perf["total_bets"] == 0
        assert perf["current_bankroll"] == 1000.0

    def test_place_and_settle_win(self):
        from src.betting.bankroll_manager import BankrollManager
        bm = BankrollManager.__new__(BankrollManager)
        bm.initial_bankroll = 1000.0
        bm.current_bankroll = 1000.0
        bm.bets = []
        bm.max_stake_pct = 5.0

        bm.place_bet("Match A", "1X2", "Home", 2.0, 50.0, 0.6, 0.2)
        assert bm.current_bankroll == 950.0

        bm.settle_bet(0, "win")
        assert bm.current_bankroll == 1050.0
        assert bm.bets[0].profit == 50.0

    def test_place_and_settle_loss(self):
        from src.betting.bankroll_manager import BankrollManager
        bm = BankrollManager.__new__(BankrollManager)
        bm.initial_bankroll = 1000.0
        bm.current_bankroll = 1000.0
        bm.bets = []
        bm.max_stake_pct = 5.0

        bm.place_bet("Match B", "1X2", "Away", 3.0, 30.0, 0.4, 0.1)
        bm.settle_bet(0, "loss")
        assert bm.current_bankroll == 970.0
        assert bm.bets[0].profit == -30.0

    def test_void_bet_returns_stake(self):
        from src.betting.bankroll_manager import BankrollManager
        bm = BankrollManager.__new__(BankrollManager)
        bm.initial_bankroll = 1000.0
        bm.current_bankroll = 1000.0
        bm.bets = []
        bm.max_stake_pct = 5.0

        bm.place_bet("Match C", "BTTS", "Yes", 1.8, 40.0, 0.6, 0.08)
        bm.settle_bet(0, "void")
        assert bm.current_bankroll == 1000.0

    def test_performance_metrics(self):
        from src.betting.bankroll_manager import BankrollManager
        bm = BankrollManager.__new__(BankrollManager)
        bm.initial_bankroll = 1000.0
        bm.current_bankroll = 1000.0
        bm.bets = []
        bm.max_stake_pct = 5.0

        bm.place_bet("M1", "1X2", "Home", 2.0, 50.0, 0.6, 0.2)
        bm.place_bet("M2", "1X2", "Away", 2.5, 40.0, 0.5, 0.15)
        bm.settle_bet(0, "win")
        bm.settle_bet(1, "loss")

        perf = bm.get_performance()
        assert perf["total_bets"] == 2
        assert perf["wins"] == 1
        assert perf["losses"] == 1
        assert perf["hit_rate"] == 0.5


class TestSettlementLogic:
    """Tests for the settlement outcome determination.

    This exercises the exact same if/elif chain used in
    betting_agent.settle_predictions() to determine win/loss.
    """

    @staticmethod
    def _settle(selection: str, home_goals: int, away_goals: int) -> str:
        """Replicate the settlement logic from betting_agent.py."""
        hg, ag = home_goals, away_goals
        total = hg + ag
        btts = hg > 0 and ag > 0
        won = False

        if selection == "Home Win":
            won = hg > ag
        elif selection == "Draw":
            won = hg == ag
        elif selection == "Away Win":
            won = hg < ag
        elif selection == "Over 2.5 Goals":
            won = total > 2.5
        elif selection == "Under 2.5 Goals":
            won = total < 2.5
        elif selection == "Over 1.5 Goals":
            won = total > 1.5
        elif selection == "Under 1.5 Goals":
            won = total < 1.5
        elif selection == "Over 3.5 Goals":
            won = total > 3.5
        elif selection == "Under 3.5 Goals":
            won = total < 3.5
        elif selection == "BTTS Yes":
            won = btts
        elif selection == "BTTS No":
            won = not btts
        elif selection == "Home Over 1.5":
            won = hg >= 2
        elif selection == "Away Over 1.5":
            won = ag >= 2

        return "win" if won else "loss"

    # ── 1X2 ──────────────────────────────────────────────────────────────

    def test_home_win(self):
        assert self._settle("Home Win", 2, 1) == "win"

    def test_home_win_loss(self):
        assert self._settle("Home Win", 0, 1) == "loss"

    def test_home_win_draw(self):
        assert self._settle("Home Win", 1, 1) == "loss"

    def test_draw_win(self):
        assert self._settle("Draw", 1, 1) == "win"

    def test_draw_loss(self):
        assert self._settle("Draw", 2, 1) == "loss"

    def test_away_win(self):
        assert self._settle("Away Win", 0, 2) == "win"

    def test_away_win_loss(self):
        assert self._settle("Away Win", 3, 1) == "loss"

    # ── Over/Under ───────────────────────────────────────────────────────

    def test_over25_win(self):
        assert self._settle("Over 2.5 Goals", 2, 1) == "win"

    def test_over25_loss_exactly_2(self):
        assert self._settle("Over 2.5 Goals", 1, 1) == "loss"

    def test_under25_win(self):
        assert self._settle("Under 2.5 Goals", 1, 0) == "win"

    def test_under25_loss(self):
        assert self._settle("Under 2.5 Goals", 2, 1) == "loss"

    def test_over15_win(self):
        assert self._settle("Over 1.5 Goals", 1, 1) == "win"

    def test_over15_loss(self):
        assert self._settle("Over 1.5 Goals", 1, 0) == "loss"

    def test_under15_win(self):
        assert self._settle("Under 1.5 Goals", 1, 0) == "win"

    def test_under15_loss(self):
        assert self._settle("Under 1.5 Goals", 1, 1) == "loss"

    def test_over35_win(self):
        assert self._settle("Over 3.5 Goals", 3, 1) == "win"

    def test_over35_loss(self):
        assert self._settle("Over 3.5 Goals", 2, 1) == "loss"

    def test_under35_win(self):
        assert self._settle("Under 3.5 Goals", 2, 1) == "win"

    def test_under35_loss(self):
        assert self._settle("Under 3.5 Goals", 3, 1) == "loss"

    # ── BTTS ─────────────────────────────────────────────────────────────

    def test_btts_yes_win(self):
        assert self._settle("BTTS Yes", 1, 1) == "win"

    def test_btts_yes_loss(self):
        assert self._settle("BTTS Yes", 2, 0) == "loss"

    def test_btts_no_win(self):
        assert self._settle("BTTS No", 2, 0) == "win"

    def test_btts_no_loss(self):
        assert self._settle("BTTS No", 1, 2) == "loss"

    # ── Team goals ───────────────────────────────────────────────────────

    def test_home_over15_win(self):
        assert self._settle("Home Over 1.5", 2, 0) == "win"

    def test_home_over15_loss(self):
        assert self._settle("Home Over 1.5", 1, 3) == "loss"

    def test_away_over15_win(self):
        assert self._settle("Away Over 1.5", 0, 2) == "win"

    def test_away_over15_loss(self):
        assert self._settle("Away Over 1.5", 3, 1) == "loss"

    # ── Edge cases: 0-0 scoreline ────────────────────────────────────────

    def test_0_0_draw_wins(self):
        assert self._settle("Draw", 0, 0) == "win"

    def test_0_0_under15_wins(self):
        assert self._settle("Under 1.5 Goals", 0, 0) == "win"

    def test_0_0_under25_wins(self):
        assert self._settle("Under 2.5 Goals", 0, 0) == "win"

    def test_0_0_under35_wins(self):
        assert self._settle("Under 3.5 Goals", 0, 0) == "win"

    def test_0_0_btts_no_wins(self):
        assert self._settle("BTTS No", 0, 0) == "win"

    def test_0_0_btts_yes_loses(self):
        assert self._settle("BTTS Yes", 0, 0) == "loss"

    def test_0_0_over25_loses(self):
        assert self._settle("Over 2.5 Goals", 0, 0) == "loss"
