"""The Claude pick review is REMOVED — cohort s5.22, 2026-11-12.

WHY, AS A NUMBER. Over pick dates 2026-10-13..2026-11-12, grain=PICK:

    682 picks   459 reviewed (67.3%: 331 KEEP, 128 CHANGE)
                223 NOT reviewed (32.7%) -- seven of 22 pick-days wholly

The Pro session limit and the API balance both run out, so one fingerprint
covered picks finalised under TWO rules, and the review's paired value was
never measurable. The saved pick is now the model's pick.

THE CONTROL IS DRIVEN, NOT ASSERTED. The review service is run against the
deployed configuration with every route to Claude booby-trapped; and the
same driver is then run against a configuration with the review ON, where it
MUST trip. A control that cannot fire proves nothing about a call that did
not happen.
"""
import ast
import asyncio
import pathlib

import pytest
import yaml

from src.reporting.match_briefing import MatchBriefingService
from src.utils.config import Config

EXAMPLE = "config/config.example.yaml"

#: `measured` 2026-11-12, saved_picks, pick_date 2026-10-13..2026-11-12,
#: disposition IS NULL, grain=PICK.
REVIEW_COVERAGE_30D = {
    "picks": 682, "reviewed": 459, "keep": 331, "change": 128,
    "not_reviewed": 223, "pick_days": 22, "days_wholly_unreviewed": 7,
    "api_fallback_share": "UNEVALUATED",
    "why_unevaluated": ("the backend that answered a review is not stored "
                        "per pick, so the share that went through the API "
                        "fallback cannot be read from the record"),
}

#: Workflow entries that have no consumer once the review is off. LISTED, and
#: stopped: deleting a secret is the operator's action. Keyed by the token the
#: workflow file carries, so the list can be checked against the file.
UNUSED_AFTER_REMOVAL = {
    "ANTHROPIC_API_KEY": "env entry on the job (daily-picks.yml) and the "
                         "repository secret behind it",
    "CLAUDE_CODE_OAUTH_TOKEN": "env entry on the job (daily-picks.yml) and "
                               "the repository secret behind it",
    "@anthropic-ai/claude-code": "the `npm install -g` step before picks",
}
#: Not a credential and not Claude-named, so the scan below cannot see it:
#: the `briefings-sent` cache restore/save pair. The review was the only
#: WRITER of data/briefings_sent.json; the freeze reader in `finalize_picks`
#: still reads it and now always finds nothing for today.
ALSO_INERT = "the briefings-sent guard cache restore/save steps"

_CLAUDE_TOKEN = r"ANTHROPIC_[A-Z0-9_]+|CLAUDE_[A-Z0-9_]+|@anthropic-ai/[a-z0-9-]+"


class _Tripwire(Exception):
    pass


class _Pick:
    match_id = 123


def _service(config) -> MatchBriefingService:
    """The service with the deployed config and every route to Claude armed.

    Built without `__init__`, which takes a whole agent: the only state
    `finalize_picks_with_claude` reads before its gate is `config`.
    """
    svc = object.__new__(MatchBriefingService)
    svc.config = config
    svc._claude_code_exhausted = False
    svc._last_review_failures = []

    async def _trap(*_a, **_k):
        raise _Tripwire("a Claude review was attempted")

    def _trap_sync(*_a, **_k):
        raise _Tripwire("a Claude backend was resolved")

    svc._briefing_for = _trap
    svc._resolve_backend = _trap_sync
    svc._sync_recs_from_db = lambda *_a, **_k: None
    return svc


#: The values the `briefings:` block held when it was DELETED at s5.23
#: (2026-11-13), read from config.example.yaml at commit 3db1099.
BRIEFINGS_HELD = {"enabled": False, "finalize_picks": False,
                  "api_fallback": True, "backend": "claude_code",
                  "language": "Bulgarian"}


def test_the_BRIEFINGS_BLOCK_IS_DELETED_from_the_DEPLOYED_config():
    """s5.23: deleted rather than tracked. `config.example.yaml`, because CI
    copies it over the local file."""
    cfg = yaml.safe_load(pathlib.Path(EXAMPLE).read_text(encoding="utf-8"))
    assert "betting" in cfg, "anchor: the deployed config was read"
    assert set(cfg).isdisjoint({"briefings"}), sorted(cfg)


def test_every_CODE_DEFAULT_equals_the_value_its_DELETED_KEY_held():
    """Deleting the block must change nothing. From the AST of
    `match_briefing.py`: every `config.get("briefings.X", default)` names a
    key that was in the block, and its default is the value that key held --
    so the review stays off, the fallback stays as it was, and nothing that
    reads one of these keys behaves differently with the block gone. The
    first two defaults were True; left that way, deleting the block would
    have switched the review back ON."""
    src = pathlib.Path("src/reporting/match_briefing.py").read_text(
        encoding="utf-8")
    reads = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") \
                == "get" and node.args and isinstance(node.args[0],
                                                      ast.Constant) \
                and str(node.args[0].value).startswith("briefings."):
            key = node.args[0].value.split(".", 1)[1]
            assert len(node.args) == 2, f"briefings.{key} has no default"
            reads[key] = ast.literal_eval(node.args[1])
    assert set(reads) == set(BRIEFINGS_HELD), sorted(reads)
    assert reads == BRIEFINGS_HELD


def test_DRIVEN_no_Claude_call_is_made_under_the_deployed_config():
    """Every route to Claude raises. The review must return 0 untouched."""
    svc = _service(Config(EXAMPLE))
    reviewed = asyncio.run(svc.finalize_picks_with_claude([_Pick()]))
    assert reviewed == 0


def test_KNOWN_WRONG_the_same_driver_TRIPS_when_the_review_is_ON(tmp_path):
    """The control for the control. With the review enabled the driver must
    hit a tripwire -- otherwise the test above passes because the traps are
    not on the path, not because the path is closed."""
    cfg = yaml.safe_load(pathlib.Path(EXAMPLE).read_text(encoding="utf-8"))
    cfg.setdefault("briefings", {})
    cfg["briefings"]["enabled"] = True
    cfg["briefings"]["finalize_picks"] = True
    f = tmp_path / "on.yaml"
    f.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    svc = _service(Config(str(f)))
    with pytest.raises(_Tripwire):
        asyncio.run(svc.finalize_picks_with_claude([_Pick()]))


def test_the_PICKS_PATH_HAS_NO_REVIEW_CALL_SITE():
    """s5.23: the call site is gone, not gated. From the AST of the agent:
    no call to `finalize_picks_with_claude` anywhere -- ANCHORED by the
    manual `--briefing` command, which still constructs the service, so a
    scan that found nothing because it read the wrong file would fail
    here."""
    src = pathlib.Path("src/agent/betting_agent.py").read_text(
        encoding="utf-8")
    calls = [getattr(c.func, "attr", None) or getattr(c.func, "id", "")
             for c in ast.walk(ast.parse(src)) if isinstance(c, ast.Call)]
    assert "MatchBriefingService" in calls
    assert calls.count("finalize_picks_with_claude") == 0


def test_the_REASON_IS_A_NUMBER_and_the_parts_add_up():
    c = REVIEW_COVERAGE_30D
    assert c["reviewed"] + c["not_reviewed"] == c["picks"]
    assert c["keep"] + c["change"] == c["reviewed"]
    assert c["not_reviewed"] / c["picks"] == pytest.approx(0.327, abs=0.001)
    assert c["change"] / c["reviewed"] == pytest.approx(0.279, abs=0.001), (
        "the share of reviewed picks that were CHANGED is what makes the "
        "review selection-affecting; it was 27.9%")
    assert c["api_fallback_share"] == "UNEVALUATED" and c["why_unevaluated"]
    assert c["days_wholly_unreviewed"] < c["pick_days"]


def _claude_tokens(text):
    import re
    return set(re.findall(_CLAUDE_TOKEN, text))


def test_EVERY_CLAUDE_ENTRY_in_the_workflows_IS_ON_THE_UNUSED_LIST():
    """Listed and stopped: deleting them is the operator's action.

    A SUBSET, NOT AN EQUALITY, AND THAT IS THE POINT. This file is in the
    capture gate. Had it asserted the entries are still PRESENT, the operator
    deleting a secret -- the very action this list invites -- would have
    reddened the gate and lost that day's card. So it passes when they are
    there and passes when they are gone, and fails on the one thing that
    matters: a Claude entry in a workflow that nobody declared unused.
    """
    found = set()
    for wf in sorted(pathlib.Path(".github/workflows").glob("*.yml")):
        found |= _claude_tokens(wf.read_text(encoding="utf-8"))
    assert found <= set(UNUSED_AFTER_REMOVAL), (
        f"undeclared Claude entries in the workflows: "
        f"{sorted(found - set(UNUSED_AFTER_REMOVAL))}")
    assert ALSO_INERT


def test_KNOWN_WRONG_the_workflow_scan_SEES_a_planted_entry():
    """The scan above is a matcher, and a matcher with zero hits verified
    nothing. Planted text: a new credential and a new package are both found,
    and an unrelated secret is not."""
    planted = ("      CLAUDE_NEW_THING: ${{ secrets.CLAUDE_NEW_THING }}\n"
               "      run: npm install -g @anthropic-ai/some-tool\n"
               "      ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}\n")
    assert _claude_tokens(planted) == {
        "CLAUDE_NEW_THING", "@anthropic-ai/some-tool"}
    assert not _claude_tokens(planted) <= set(UNUSED_AFTER_REMOVAL)


def test_the_REVIEW_REMOVAL_is_the_PREVIOUS_cohort_and_its_GAP_is_RECORDED():
    """s5.22 moved by the revision bump alone, because no `briefings.*` key
    was tracked. At s5.23 the keys are deleted, not tracked -- so the gap is
    closed by removal, and the record of it stays."""
    from src.models.model_version import TRACKED_KEYS
    from tests.experiment_pins import (PREVIOUS_CODE_REVISION,
                                       PREVIOUS_MODEL_VERSION)
    assert PREVIOUS_CODE_REVISION == "s5.22"
    assert PREVIOUS_MODEL_VERSION.endswith("7f9e82")
    assert "betting.selection_mode" in TRACKED_KEYS, "anchor: keys were read"
    assert [k for k in TRACKED_KEYS if k.startswith("briefings.")] == []


# ---------------------------------------------------------------- the save

def test_DRIVEN_picks_are_STILL_SAVED_one_per_fixture_and_UNREVIEWED(tmp_path):
    """The save path is driven end to end on a scratch SQLite database, with
    no review anywhere in it. Two fixtures in, two picks out, one each, and
    the `review_action` of both is NULL: the saved pick is the model pick.

    The per-fixture cap itself is `betting.max_picks_per_match`, read from the
    DEPLOYED config, and it is 1.
    """
    from datetime import date

    from src.data.models import PickObservation, SavedPick
    from tests.test_pick_save_atomicity import _agent, _mgr, _pick

    mgr = _mgr(tmp_path, "removed.db", n_matches=2)
    _agent(mgr)._save_picks(
        [_pick(1, "Over 2.5", "Over 2.5 Goals", 1.85),
         _pick(2, "1X2", "Home Win", 2.10)], date(2026, 9, 1))

    with mgr.get_session() as s:
        saved = [(p.match_id, p.selection, p.review_action, p.model_selection)
                 for p in s.query(SavedPick).order_by(SavedPick.match_id)]
        obs = [(o.attribution, o.selection)
               for o in s.query(PickObservation).all()]
    assert [m for m, *_ in saved] == [1, 2], saved
    assert len({m for m, *_ in saved}) == len(saved), "two picks on one fixture"
    assert {a for _, _, a, _ in saved} == {None}, (
        f"a saved pick carries a review verdict with the review removed: "
        f"{saved}")
    for _mid, selection, _a, model_selection in saved:
        assert model_selection == selection, (
            "the saved selection is not the model selection")
    # model and final observations describe the SAME leg on every pick
    by_attr = {}
    for attribution, selection in obs:
        by_attr.setdefault(attribution, []).append(selection)
    assert sorted(by_attr["model"]) == sorted(by_attr["final"])

    deployed = yaml.safe_load(pathlib.Path(EXAMPLE).read_text(
        encoding="utf-8"))["betting"]
    assert deployed["max_picks_per_match"] == 1


def test_the_PICKS_PATH_SAVES_and_SENDS_with_NO_review_between():
    """From the AST of `main()`: picks are produced by `get_daily_picks` and
    sent by `send_daily_picks`, and nothing that reviews them is called in
    between, because nothing that reviews them is called at all."""
    src = pathlib.Path("src/agent/betting_agent.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)

    def _lines(attr):
        return [c.lineno for c in ast.walk(tree) if isinstance(c, ast.Call)
                and getattr(c.func, "attr", "") == attr]

    assert len(_lines("get_daily_picks")) >= 1
    assert len(_lines("send_daily_picks")) >= 1
    assert _lines("finalize_picks_with_claude") == []


# ---------------------------------------------------------------- the card

def _rec(**over):
    import datetime as dt

    from src.betting.value_calculator import BetRecommendation
    kw = dict(
        match="A vs B", match_id=1, market="Over 2.5",
        selection="Over 2.5 Goals", odds=1.8, predicted_probability=0.6,
        expected_value=0.08, confidence=0.65, kelly_stake_percentage=1.5,
        recommended_stake=1.5, reasoning="", risk_level="low",
        league="england/premier-league", model_agreement="solo",
        match_date=dt.datetime(2026, 11, 4, 20, 0))
    kw.update(over)
    return BetRecommendation(**kw)


def _card(rec, monkeypatch):
    from src.reporting import telegram_bot as tb
    monkeypatch.setattr(tb, "_picks_sent_today", lambda: False)
    n = object.__new__(tb.TelegramNotifier)
    n.config = None
    n.enabled = True
    return n._build_daily_picks_message([rec], force=True, paper_mode=True)


def test_DRIVEN_the_CLIENT_CARD_IS_UNCHANGED_IN_SHAPE(monkeypatch):
    """The card is built by ONE formatter from the recommendation alone. It is
    driven here on a model pick, and again on the same pick carrying a review
    verdict: byte-identical, because the formatter never read one. The
    known-wrong half changes the price and must change the card, so equality
    above is not the equality of two empty strings."""
    plain = _card(_rec(), monkeypatch)
    marked = _rec()
    object.__setattr__(marked, "review_action", "KEEP")
    assert _card(marked, monkeypatch) == plain
    message, _header = plain
    assert "A vs B" in message and "Over 2.5 Goals" in message
    assert _card(_rec(odds=2.4), monkeypatch) != plain


def test_the_CARD_FORMATTER_READS_NO_REVIEW_FIELD():
    """Set equality on the AST, not a substring search: the attribute names
    the notifier module reads, intersected with the review fields, is empty.
    And the same scan FINDS `review_action` in the module that does read it."""
    review_fields = {"review_action", "review_reason", "model_selection",
                     "model_result"}

    def _attrs(path):
        tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
        return {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}

    assert _attrs("src/reporting/telegram_bot.py") & review_fields == set()
    assert "review_action" in _attrs("src/agent/betting_agent.py"), (
        "the scan cannot see a review field where one is certainly read")


# ---------------------------------------------------------------- the time

#: `measured` 2026-11-12 from the logs of the 22 SCHEDULED daily-picks runs
#: started 2026-10-22..2026-11-12 (the parallel-review regime), grain=RUN,
#: unit=minutes. The review interval is the log line before the first
#: `match_briefing` line to the `Claude reviewed N/M` line.
REVIEW_TIME_MEASURED = {
    "scheduled_runs": 22,
    "review_ran": 14,          # 5 runs failed before picks, 3 had no fixtures
    "failed_before_picks": 5,
    "zero_fixture_runs": 3,    # 2026-11-10..12, the international window
    "review_min": {"mean": 2.27, "sd": 1.74, "median": 1.76,
                   "min": 0.23, "max": 6.43},
    "share_of_analysis_plus_review": 0.29,
    "seconds_per_fixture": 4.21,           # n=10 runs that reviewed > 0
    #: ran, spent the time, reviewed NOTHING (0 of 49, 13, 83 and 3 fixtures)
    "ran_and_reviewed_zero": {"runs": 4, "mean_min": 2.02, "max_min": 5.15},
    #: `simulated`: each run's time-to-picks minus its own review minutes,
    #: over the runs where the review ran. NOT substituted into the margin.
    "time_to_picks_bound_modelled": {
        "weekday": {"n": 10, "max": (40.35, 38.62),
                    "mean_plus_3sd": (48.34, 45.08)},
        "weekend": {"n": 4, "max": (54.95, 50.75),
                    "mean_plus_3sd": (71.65, 63.33)},
    },
    "margin_constants_substituted": False,
}


def test_the_TIME_REMOVED_is_recorded_with_its_population():
    t = REVIEW_TIME_MEASURED
    assert (t["review_ran"] + t["failed_before_picks"]
            + t["zero_fixture_runs"]) == t["scheduled_runs"]
    assert t["ran_and_reviewed_zero"]["runs"] < t["review_ran"]
    for day_type, m in t["time_to_picks_bound_modelled"].items():
        for stat in ("max", "mean_plus_3sd"):
            with_review, without = m[stat]
            assert without < with_review, (day_type, stat)
    wk = t["time_to_picks_bound_modelled"]["weekday"]
    assert wk["max"][0] - wk["max"][1] == pytest.approx(1.73, abs=0.01)
    assert (wk["mean_plus_3sd"][0] - wk["mean_plus_3sd"][1]
            == pytest.approx(3.26, abs=0.01))
    assert t["margin_constants_substituted"] is False, (
        "the margin's duration constants were re-pointed. That is an "
        "authorised, forward-only substitution, not part of the removal")
