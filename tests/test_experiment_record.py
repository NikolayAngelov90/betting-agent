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
