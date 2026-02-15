"""Tests for value betting calculator and bankroll manager."""

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
