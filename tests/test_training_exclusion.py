"""Stage 13 (s5.3) — contaminated matches must not reach any fitting path.

29 matches carry a participant whose row belongs to a different club. Their
results are attributed to teams that did not play them, so every strength, form
and H2H figure derived from them is wrong about both sides.

The danger this file guards is not the filter being absent — it is the filter
being present in *most* places. A flag honoured in four of five sites is worse
than no flag, because the report then says it was fixed.
"""

import pathlib
import re

import pytest

from src.data.models import Match

#: An exemption must name exactly ONE category. The generic form was
#: paste-able anywhere; this one has to assert something specific, so a
#: wrong paste is visible to a reviewer instead of invisible.
EXEMPT_MARKER = re.compile(
    r"training-exclusion:\s*NOT GATED\s*\((populates|repairs|resolves)\)")

EXCLUSION = "training_exclusion_reason"

#: Every query that reads completed matches for fitting must carry the filter.
#: `match_history` is the shared projection; the rest carry hand-copied
#: predicates and must be patched individually.
FITTING_SITES = [
    "src/data/match_history.py",
    "src/features/feature_engineer.py",
    "src/agent/betting_agent.py",
    "src/evaluation/clean_dataset.py",
]


def test_the_column_exists_on_the_model():
    assert hasattr(Match, EXCLUSION)


@pytest.mark.parametrize("path", FITTING_SITES)
def test_every_fitting_site_carries_the_filter(path):
    text = pathlib.Path(path).read_text(encoding="utf-8")
    assert EXCLUSION in text, (
        f"{path} reads completed matches for fitting but does not exclude "
        "contaminated ones")


def test_no_unguarded_completed_match_query():
    """The generalisation — same shape as the bulk-delete ban.

    Any query pairing `is_fixture == False` with `home_goals.isnot(None)` is
    reading completed matches, which is the fitting shape. Each such site must
    carry the exclusion filter within a few lines, or a future refactor
    reintroduces the contamination silently.

    Deliberately scoped to that PAIR rather than to `is_fixture` alone: querying
    fixtures, or counting matches for cache invalidation, is not fitting and
    must not be forced to carry a filter that would be meaningless there.
    """
    nl = chr(10)
    offenders = []
    for path in list(pathlib.Path("src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for i, line in enumerate(lines):
            code = line.split("#", 1)[0]
            if "is_fixture == False" not in code:
                continue
            window = "\n".join(lines[max(0, i - 6):i + 12])
            if "home_goals.isnot(None)" not in window:
                continue                      # not a completed-match read
            if EXCLUSION in window or EXEMPT_MARKER.search(window):
                continue
            offenders.append(f"{path}:{i + 1}: {line.strip()}")

    assert not offenders, (
        "completed-match query with neither the exclusion filter nor an "
        "exemption marker." + nl + nl +
        "THE RULE: training_exclusion_reason gates paths that LEARN FROM "
        "or MEASURE a match. It does NOT gate paths that POPULATE, REPAIR "
        "or RESOLVE a real-world fact about it." + nl + nl +
        "  learns/measures     -> add "
        "Match.training_exclusion_reason.is_(None)," + nl +
        "  populates/resolves  -> mark it at the query:" + nl +
        "      # training-exclusion: NOT GATED - populates/resolves"
        + nl + nl + "Unguarded:" + nl + "  " + (nl + "  ").join(offenders))


def test_the_shared_projection_is_not_the_only_guard():
    """feature_engineer must carry its OWN filter.

    It calls `get_completed_matches` twice but ALSO runs a direct historical
    query with a hand-copied `is_fixture == False, home_goals.isnot(None)`.
    Filtering only the shared projection would leave team form, H2H and rolling
    goals contaminated — the pipeline that feeds every future pick involving an
    affected team, including its legitimate matches.
    """
    text = pathlib.Path("src/features/feature_engineer.py").read_text(
        encoding="utf-8")
    assert EXCLUSION in text, (
        "feature_engineer relies on match_history alone — but it has its own "
        "historical query, which match_history does not cover")


def test_the_refit_mechanism_is_replay_but_not_always_from_the_database():
    """The corrected mechanism. The original claim is kept visible.

    OVERTURNED: "Poisson and Elo need no artifact surgery because fit() replays
    from `self.ratings = {}` against the DATABASE, so excluding a match removes
    it from the next fit by construction."

    Half right, and wrong in the case that matters. Both DO replay from an empty
    state — no saved rating or strength table was edited, and none needed to be.
    But they replay from whatever `get_completed_matches` returns, and that is
    the Parquet mirror whenever one is warm; the database is the fallback. So a
    stale mirror would have fed the excluded matches straight back into a fit
    that believed it had excluded them.

    The corrected statement: exclusion is sufficient ONLY because the mirror is
    stamped with the filter's generation and refuses itself when it does not
    match. Two mechanisms, not one.

    This is the second claim in this stage that a measurement reversed, and the
    pattern is worth more than either claim: both were plausible, both were
    stated confidently, and both were only caught because something was built
    that could contradict them.
    """
    import pathlib as _p
    import re as _re

    elo = _p.Path("src/models/elo_system.py").read_text(encoding="utf-8")
    assert _re.search(r"self\.ratings\s*=\s*\{\}", elo), (
        "Elo no longer replays from an empty state")

    # ...and the mirror must be able to refuse itself, or the replay is not safe
    hm = _p.Path("src/data/history_mirror.py").read_text(encoding="utf-8")
    assert "filter_generation" in hm, (
        "the mirror no longer carries a filter-generation stamp — a stale "
        "Parquet could feed excluded matches back into a fit")


def test_the_number_of_exemptions_is_pinned():
    """Shape is not enough; the population must be observed.

    An exemption marker travels with the code, which beats a central allowlist
    that drifts out of date. But the failure mode inverts: someone silences the
    guard by copying the comment. Requiring a named category makes a wrong paste
    assert something specific; pinning the count makes a NEW paste a deliberate
    edit here, which a reviewer sees.
    """
    from tests.experiment_pins import TRAINING_EXCLUSION_EXEMPTIONS

    found = []
    for path in pathlib.Path("src").rglob("*.py"):
        for n, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            if EXEMPT_MARKER.search(line):
                found.append(f"{path}:{n}")

    assert len(found) == TRAINING_EXCLUSION_EXEMPTIONS, (
        f"{len(found)} exemption(s) found, pin says "
        f"{TRAINING_EXCLUSION_EXEMPTIONS}. If the new one is legitimate, say so "
        "by editing TRAINING_EXCLUSION_EXEMPTIONS in tests/experiment_pins.py."
        + chr(10) + "  " + (chr(10) + "  ").join(found))


def test_every_exemption_names_a_real_category():
    """`populates/resolves` fits everywhere, which is what made it paste-able."""
    for path in pathlib.Path("src").rglob("*.py"):
        for n, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            # Scoped to THIS marker family. `evidence-gate:` markers are
            # a different gate with its own guard and its own pin;
            # sharing the "NOT GATED" phrase must not make either test
            # police the other.
            if ("training-exclusion:" in line and "NOT GATED" in line
                    and "<" not in line):
                assert EXEMPT_MARKER.search(line), (
                    f"{path}:{n} claims an exemption without naming exactly "
                    "one of populates / repairs / resolves")


def test_the_exclusion_is_deliberately_not_write_once():
    """Stated so it is not "fixed" by someone pattern-matching on siblings.

    `disposition` and `evidence_status` carry write-once validators because they
    record judgements about past events. This column records current data
    quality, and the 29 are meant to become repairable once authoritative ids
    exist — so clearing it must stay possible.

    The danger is the opposite one, and it is recorded in the column comment: a
    repair re-includes matches in the fitting set, which changes predictions
    while `model_version` — which fingerprints configuration, not training data
    — stays put. A repair is a cohort event, not a data fix.
    """
    import inspect

    from src.data.models import Match, SavedPick

    src = inspect.getsource(Match)
    assert '@validates("training_exclusion_reason")' not in src, (
        "a write-once validator was added to training_exclusion_reason — that "
        "makes the 29 permanently unrepairable, which is the opposite of why "
        "detaching was chosen over reassignment")
    assert "cohort event" in src, (
        "the warning that lifting an exclusion is prediction-affecting has "
        "been removed from the column comment")

    # the siblings must still have theirs
    picks = inspect.getsource(SavedPick)
    assert '@validates("disposition")' in picks
    assert '@validates("evidence_status")' in picks
