"""The Telegram reports count from the experiment, not the frozen live record.

WHAT THIS REPLACED. `get_stats()["all_time"]` reads `live_only()`, which is the
PRE-PAPER-TRADING live record: 1,074 settled picks at 51.676%, last pick
2026-08-10. It froze when paper trading began and was reported daily in three
messages as though it were current.

THE THREE THINGS THE OLD BLOCK DID NOT DO, each pinned below:

  1. name the cohort span — a single rate across six `model_version`
     fingerprints measures none of them, and pooling them is exactly what
     `model_version` exists to prevent;
  2. print `n` beside every rate, and the current cohort's `n` separately,
     because a few days of picks is noise and must look like noise;
  3. lead with CLV — Stage 16 established win-rate and ROI segments are all
     p > 0.15 and that CLV is the instrument this experiment turns on.

AND THE ONE IT MUST NOT DO: widen `live_only()`. That predicate gates the
LEARNERS. Reporting a paper outcome is not learning from one.
"""

import inspect

import pytest


# ── the predicate that must not move ────────────────────────────────────────
def test_live_only_still_excludes_paper_picks():
    """If this ever admits paper picks, the frozen experiment retrains itself."""
    src = inspect.getsource(__import__(
        "src.data.pick_filters", fromlist=["live_only"]).live_only)
    assert "is_paper" in src and "disposition" in src, (
        "live_only() no longer gates on is_paper/disposition — the reporting "
        "change was supposed to add a SEPARATE series, not widen this one")


def test_experiment_record_does_not_touch_live_only():
    """Checks USAGE, not mention.

    The module's docstring names `live_only()` on purpose, to say why it does
    not use it. A substring search would match that and pass for the wrong
    reason — the same "grep found it, so it must be handled" error the alias
    that never fired was made of. This walks the AST instead.
    """
    import ast as _ast
    from src.reporting import experiment_record

    tree = _ast.parse(inspect.getsource(experiment_record))
    used = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Name):
            used.add(node.id)
        elif isinstance(node, _ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, (_ast.Import, _ast.ImportFrom)):
            for a in node.names:
                used.add(a.asname or a.name.split(".")[-1])

    assert "live_only" not in used, (
        "the experiment record must not reuse the learner gate; it queries "
        "is_paper directly so the two can never drift into one predicate")
    assert "valid_evidence" not in used


def test_the_exp1_paths_are_still_gated():
    """Paper outcomes must not reach the Claude review prompt.

    EXP-1 was exactly this: paper results leaking into the prompt through
    _recent_selection_stats / _recent_review_stats. Nothing in the reporting
    change may re-open it.
    """
    from src.reporting.match_briefing import MatchBriefingService
    for name in ("_recent_selection_stats", "_recent_review_stats"):
        src = inspect.getsource(getattr(MatchBriefingService, name))
        assert "live_only()" in src, f"{name} lost its live_only() gate"
        assert "valid_evidence()" in src, f"{name} lost its valid_evidence() gate"


# ── the block itself ────────────────────────────────────────────────────────
def _rec(**kw):
    from src.reporting.experiment_record import ExperimentRecord, Series
    r = ExperimentRecord(**{k: v for k, v in kw.items()
                            if k not in ("model", "final")})
    if "model" in kw:
        r.model = kw["model"]
    if "final" in kw:
        r.final = kw["final"]
    return r


def test_none_is_not_an_empty_record():
    """`None` means unmeasured; the message must say so rather than omit it."""
    from src.reporting.experiment_record import format_block
    out = "\n".join(format_block(None, html=False))
    assert "unavailable" in out and "not the same as empty" in out


def test_clv_leads_the_record():
    from src.reporting.experiment_record import Series, format_block
    rec = _rec(settled=444, wins=247, losses=179, cohorts=6,
               current_cohort="stage5_baseline_20260807.694a60", current_n=62,
               current_wins=36, current_losses=23,
               model=Series("MODEL", n=49, fixtures=49, mean=-0.0051,
                            lo=-0.0128, hi=0.0030))
    out = "\n".join(format_block(rec, html=False))
    assert out.index("MODEL") < out.index("247W-179L"), (
        "the settled record appears ABOVE the CLV series — a message "
        "headlining win rate invites the reasoning four audits corrected")


def test_the_cohort_span_is_named_so_nobody_reads_it_as_one_system():
    from src.reporting.experiment_record import format_block
    rec = _rec(settled=444, wins=247, losses=179, cohorts=6)
    out = "\n".join(format_block(rec, html=False))
    assert "n=444" in out, "the record's n must appear beside its rate"
    assert "6 cohorts" in out, "the cohort count must be named"
    assert "measures none of them" in out, (
        "the span must be explained, not merely printed — a reader who sees "
        "'6 cohorts' without being told what it means will still pool them")


def test_the_current_cohort_n_is_printed_separately():
    from src.reporting.experiment_record import format_block
    rec = _rec(settled=444, wins=247, losses=179, cohorts=6,
               current_cohort="stage5_baseline_20260807.694a60",
               current_n=62, current_wins=36, current_losses=23)
    out = "\n".join(format_block(rec, html=False))
    assert "n=62" in out and "694a60" in out, (
        "s5.9's own n must be visible; a rate on a few days of picks is noise "
        "and has to look like noise")


def test_every_rate_carries_an_n():
    """No bare percentage anywhere in the block."""
    import re
    from src.reporting.experiment_record import Series, format_block
    rec = _rec(settled=444, wins=247, losses=179, cohorts=6,
               current_cohort="x.694a60", current_n=62,
               current_wins=36, current_losses=23,
               model=Series("MODEL", n=49, fixtures=49, mean=-0.0051,
                            lo=-0.0128, hi=0.0030))
    for line in format_block(rec, html=False):
        if re.search(r"\d+\.\d%", line) and "CI" not in line:
            assert re.search(r"n=\d+", line), f"rate without an n: {line!r}"


def test_an_interval_is_refused_rather_than_faked_on_a_thin_sample():
    from src.reporting.experiment_record import Series, format_block
    rec = _rec(settled=10, wins=6, losses=4, cohorts=1,
               model=Series("MODEL", n=3, fixtures=3, mean=0.01))
    out = "\n".join(format_block(rec, html=False))
    assert "CI needs" in out, (
        "a mean without an interval must say why, not print the mean alone")


# ── the messages ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("method", [
    "send_daily_picks", "send_settlement_report", "send_performance_report"])
def test_no_message_still_prints_the_frozen_live_all_time(method):
    """All three carried it. Removing it from two would leave it circulating."""
    from src.reporting.telegram_bot import TelegramNotifier
    src = inspect.getsource(getattr(TelegramNotifier, method))
    assert "All time" not in src, (
        f"{method} still prints the frozen pre-paper-trading all-time figure")
    assert "experiment" in src, f"{method} does not take the experiment record"


def test_the_audit_can_see_a_report_that_drops_the_block():
    """Registered before the code; this asserts it still discriminates."""
    from scripts.ci_audit import assertions, extract
    with_block = ("Performance report sent to Telegram!\n"
                  "EXPERIMENT RECORD: n=444 cohorts=6")
    without = "Performance report sent to Telegram!"
    assert not assertions(extract(with_block), [])
    assert assertions(extract(without), []), (
        "a report that silently stops carrying the experiment block is "
        "invisible to the audit — the defect the pattern was registered for")


def test_the_current_cohort_caveat_is_DERIVED_not_asserted():
    """A claim beside a number must follow from that number.

    The first version hardcoded "the current cohort's n is small enough that
    its rate is noise". Four days later n was 262 and the sentence was false —
    a claim that had stopped following from the data it sat beside, which is
    exactly the defect the frozen all-time was. The threshold is Stage 16's:
    17 observations suffice to exclude a decision-relevant effect.
    """
    from src.reporting.experiment_record import format_block
    thin = "\n".join(format_block(
        _rec(settled=100, wins=55, losses=45, cohorts=3,
             current_cohort="x.abc123", current_n=9,
             current_wins=6, current_losses=3), html=False))
    assert "is noise" in thin, "a 9-pick cohort must be called noise"

    fat = "\n".join(format_block(
        _rec(settled=644, wins=363, losses=257, cohorts=6,
             current_cohort="x.694a60", current_n=262,
             current_wins=152, current_losses=101), html=False))
    assert "is noise" not in fat, (
        "n=262 is not noise, and saying so would be a false claim printed "
        "daily — the same failure mode as the frozen all-time")
    assert "describes the configuration running now" in fat


# ── item 2: a win rate never appears without its economics ─────────────────
def test_the_settled_record_carries_roi_odds_and_pl():
    """A win rate alone is not interpretable — it moves with price.

    60% at 1.55 and 52% at 1.89 can be the same outcome or the reverse. The
    figure this block replaced at least carried its ROI; the first version of
    the replacement dropped it.
    """
    from src.reporting.experiment_record import format_block
    rec = _rec(settled=644, wins=363, losses=257, cohorts=6,
               avg_odds=1.646, pl_units=-26.45,
               current_cohort="x.694a60", current_n=262,
               current_wins=152, current_losses=101,
               current_avg_odds=1.649, current_pl_units=-2.19)
    out = "\n".join(format_block(rec, html=False))
    for token in ("flat ROI", "avg odds", "u"):
        assert token in out, f"the record is missing {token!r}"
    assert "-4.2" in out and "-0.8" in out, (
        "both the series ROI and the current cohort's ROI must be printed")


def test_the_price_mix_caveat_fires_when_paper_wins_more_at_shorter_prices():
    """MEASURED 2026-09-09: paper 58.5% @1.646 vs live 51.7% @1.939.

    The higher win rate is a price-mix effect and the paper ROI is WORSE. A
    reader shown only the win rates concludes the model improved.
    """
    from src.reporting.experiment_record import format_block
    rec = _rec(settled=644, wins=363, losses=257, cohorts=6,
               avg_odds=1.646, pl_units=-26.45,
               live_avg_odds=1.939, live_win_rate=0.5168)
    out = "\n".join(format_block(rec, html=False))
    assert "PRICE-MIX" in out and "Compare ROI, not win rate" in out


def test_the_caveat_stays_quiet_when_prices_are_comparable():
    """It must not fire on a difference it cannot attribute to price."""
    from src.reporting.experiment_record import format_block
    rec = _rec(settled=644, wins=363, losses=257, cohorts=6,
               avg_odds=1.93, pl_units=-26.45,
               live_avg_odds=1.939, live_win_rate=0.5168)
    out = "\n".join(format_block(rec, html=False))
    assert "PRICE-MIX" not in out, (
        "same prices, so a win-rate gap is NOT a price-mix effect and the "
        "message must not claim it is")


def test_clv_is_named_as_the_measurement_and_the_record_as_context():
    from src.reporting.experiment_record import Series, format_block
    rec = _rec(settled=644, wins=363, losses=257, cohorts=6,
               model=Series("MODEL", n=102, fixtures=102, mean=-0.0009,
                            lo=-0.0067, hi=0.0048))
    out = "\n".join(format_block(rec, html=False))
    assert "THE MEASUREMENT" in out and "context" in out, (
        "the block must say which figure decides anything — Stage 16 found "
        "win-rate AND ROI segments alike at p > 0.15")


def test_disposition_is_filtered_PER_SERIES_not_globally():
    """A consolidated row leaves FINAL and STAYS IN MODEL.

    `paper_trading_report._Pick` states the rule once: "Kept in the MODEL
    series, excluded from FINAL." A consolidated pick was never a bet, so it
    is not part of what was actually staked — but it remains the frozen
    model's own record of the price it took, and that is the one thing
    deleting the row would destroy.

    Filtering it out of BOTH silently shrank MODEL from 102 to 101 the first
    time a pick was consolidated (2026-09-09).
    """
    import inspect
    from src.reporting import experiment_record

    src = inspect.getsource(experiment_record.build)
    assert "sp.disposition IS NULL" not in src.split("pick_observations")[-1], (
        "the observation query filters disposition in SQL, which applies it to "
        "both series; it must be filtered per attribution instead")
    assert 'attribution == "final"' in src, (
        "the per-series disposition rule is missing — MODEL must keep "
        "consolidated rows, FINAL must drop them")
