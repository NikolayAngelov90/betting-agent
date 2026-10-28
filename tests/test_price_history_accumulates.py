"""Stage 18 Part D — a storage change that silently does nothing looks exactly
like one that works.

This is the L2 lesson applied before deployment rather than after. L2 shipped
inert and would have measured as "implemented, 0 credits saved" — indistinguishable
from success. The checks here are the ones that would fail if the snapshot
machinery were wired up and dead.

WHAT CANNOT BE CHECKED HERE. "After one real day, at least one key has three
observations" needs a real day of real runs. The query for it is in the ledger.
Everything that can be established without waiting is established here.
"""

import pathlib

from tests.conftest import code_of
import re

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from src.data.models import Base, InjuryObservation, Odds, OddsSnapshot
from src.data.price_history import record_injury, record_price, stamp_first_seen
from src.utils.logger import utcnow


@pytest.fixture
def session():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    s = sessionmaker(bind=eng)()
    yield s
    s.close()


def test_snapshots_accumulate_rather_than_overwrite(session):
    """THE load-bearing test. `odds` holds two observations and never a third;
    this table must hold as many as arrive."""
    from datetime import timedelta
    t0 = utcnow()
    for i, price in enumerate((2.10, 2.05, 1.98)):
        record_price(session, match_id=1, bookmaker="Pinnacle", market_type="1X2",
                     selection="Home", odds_value=price,
                     observed_at=t0 + timedelta(hours=i))
    session.commit()
    rows = session.query(OddsSnapshot).order_by(OddsSnapshot.observed_at).all()
    assert len(rows) == 3, (
        f"three observations of one key produced {len(rows)} rows — the table is "
        "behaving like `odds` and momentum will remain untestable")
    assert [r.odds_value for r in rows] == [2.10, 2.05, 1.98]
    assert rows[0].observed_at < rows[1].observed_at < rows[2].observed_at


def test_the_natural_key_is_deliberately_not_unique():
    """A unique constraint here would silently restore overwrite semantics."""
    idx = OddsSnapshot.__table__.indexes
    for i in idx:
        cols = {c.name for c in i.columns}
        if {"match_id", "bookmaker", "market_type", "selection"} <= cols:
            assert not i.unique, (
                f"index {i.name} is UNIQUE on the natural key — the second "
                "observation of a key will be rejected and this table becomes "
                "`odds` with extra steps")
    for c in OddsSnapshot.__table__.constraints:
        assert "unique" not in type(c).__name__.lower() or len(c.columns) <= 1


def test_injury_observations_accumulate(session):
    from datetime import timedelta
    t0 = utcnow()
    for i, status in enumerate(("Questionable", "Out", "Available")):
        record_injury(session, team_id=7, player_id=42, injury_type="Knee",
                      status=status, source="api-football",
                      observed_at=t0 + timedelta(days=i))
    session.commit()
    rows = session.query(InjuryObservation).order_by(
        InjuryObservation.observed_at).all()
    assert len(rows) == 3, (
        "injury status changed three times and the history kept fewer than "
        "three rows — the 'when it was known' is exactly what this table is for")
    assert [r.status for r in rows] == ["Questionable", "Out", "Available"]


def test_first_seen_at_is_write_once(session):
    row = Odds(match_id=1, bookmaker="Pinnacle", market_type="1X2",
               selection="Home", odds_value=2.0)
    stamp_first_seen(row)
    first = row.first_seen_at
    assert first is not None
    stamp_first_seen(row)
    assert row.first_seen_at == first, (
        "first_seen_at moved on a second write — it would then record most "
        "recently seen, which is what `timestamp` already does")


def test_first_seen_at_is_not_updated_on_conflict():
    """The bulk upsert must not refresh it, or it stops meaning first-seen."""
    src = pathlib.Path("src/scrapers/apifootball_scraper.py").read_text(encoding="utf-8")
    m = re.search(r"on_conflict_do_update\((.*?)\n        \)", src, re.S)
    assert m, "could not locate the ON CONFLICT clause"
    assert "first_seen_at" not in m.group(1), (
        "first_seen_at appears in the ON CONFLICT set_ — every refresh would "
        "overwrite it and it would record last-seen, not first-seen")


@pytest.mark.parametrize("scraper", [
    "src/scrapers/theodds_scraper.py",
    "src/scrapers/apifootball_scraper.py",
])
def test_every_odds_writer_records_a_snapshot(scraper):
    """Wired but uncalled is the exact failure mode this file exists for."""
    src = pathlib.Path(scraper).read_text(encoding="utf-8")
    assert "_record_price(" in src, (
        f"{scraper} persists odds but never appends to the price path — its "
        "observations are still being discarded")


def _code_lines(text: str) -> str:
    """Source with comment-only lines and trailing comments removed.

    A MENTION IS NOT A READ. On 2026-10-01 this check fired on
    `src/data/odds_quota.py` because a comment there RECORDS HOW A MEASUREMENT
    WAS MADE — the September spend was inferred from the append-only snapshot
    history and cross-checked against the ledger. A text scan cannot tell that
    prose from a query.

    The crude scan stays crude on CODE, which is where the cohort risk is. What
    it must not do is create pressure to delete provenance in order to pass:
    this project's whole discipline is that a number carries how it was
    obtained. Fourth instance of this class — the same fix was applied to the
    `definition is not an occurrence` checks.
    """
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0]
        out.append(line)
    return "\n".join(out)


#: Files permitted to reference the snapshot history. EVERY ENTRY NEEDS A
#: REASON, because this list is the whole content of the successor contract
#: below.
_SNAPSHOT_READERS = {
    # the table and the writer
    "models.py": "defines OddsSnapshot",
    "price_history.py": "the only writer",
    # the MARKING script, 2026-10-10: reads the duplicate-Home signature to
    # separate CONFIRMED from PRESUMED corruption. Writes nothing to the
    # table and feeds no prediction.
    "mark_1x2_assignment_corruption.py": (
        "dates the 1X2 corruption from the duplicate-Home signature"),
    # the ALARM, 2026-10-18. It counts trajectories per day to decide whether
    # H1 collected nothing, which is the check that did not exist when three
    # collection days produced zero rows and reported RUNNING_AND_FINE. It
    # MEASURES the history and feeds no prediction, so it is not a cohort
    # event — the same standing as h1_collection_check.
    "h1_precondition.py": (
        "H1's zero-collection alarm. Counts trajectories per day from "
        "odds_snapshots to tell a silent zero from a real one; no session is "
        "held open past the read and nothing reaches a model"),
    # the DETECTOR, 2026-10-11. It names the table in its coverage table and
    # in a comment; it issues no query and opens no session. It was caught by
    # THIS guard on its first full run, which is the guard working: a file that
    # mentions the snapshot history must say why, even when the reason is
    # "to report that nobody reads it".
    "pnc_scan.py": (
        "the PNC-1 detector. It records `odds_snapshots` as the KIND T "
        "instance it was built to find - 47 days written and unread - and "
        "reads the SCHEMA, never the rows. No session, no prediction"),
    # the RESEARCH consumer, wired 2026-10-10
    "h1_collection_check.py": (
        "H1's analysis. It MEASURES the price path; it does not feed a "
        "prediction, so it is not a cohort event"),
}


def test_only_DECLARED_consumers_read_the_snapshot_history():
    """SUCCESSOR CONTRACT, written 2026-10-10 when the predicate flipped.

    THIS TEST USED TO ASSERT THAT **NOTHING** READ THE TABLE. That was right
    for 46 days and then became the defect: Stage 18 built `odds_snapshots` so
    Stage 19 could study the price path, H1 is that study, and H1 read `odds`
    instead — which is UNIQUE on
    (match_id, bookmaker, market_type, selection) and overwritten, so it holds
    one instant per series and `n` was structurally zero forever.

    > A test asserting that nothing consumes a producer records PNC-1 as a
    > SPECIFICATION. It cannot distinguish "not wired yet" from "must never be
    > wired", and it defends the first as if it were the second.

    CLR-2, STATED: the clearing action is WIRING THE CONSUMER, and wiring it
    changes this predicate. The old guard could not be cleared by the remedy it
    prompted — any correct fix made it fail — so it is replaced rather than
    deleted, and what it actually protected is kept.

    WHAT IT PROTECTED, AND STILL DOES: cohort neutrality. A MODEL reading
    history is a cohort event needing a decision, not a deploy. So the contract
    inverts from "nobody reads it" to "only declared consumers read it, and no
    consumer sits on the prediction path" — the second half is the test below.
    """
    offenders = []
    for base in ("src", "scripts"):
        for path in pathlib.Path(base).rglob("*.py"):
            if path.name in _SNAPSHOT_READERS:
                continue
            # ── `code_of`, NOT `_code_lines`. 2026-10-28. ──────────────
            #
            # `_code_lines` strips COMMENTS and is blind to DOCSTRINGS, and
            # `src/data/h1_window.py` names `odds_snapshots` in its MODULE
            # DOCSTRING while explaining why the grain of a snapshot count
            # decided an overlap question. The scan called it an undeclared
            # reader.
            #
            # SEVENTEENTH INSTANCE of the comment-blind class, and the fifth
            # time THIS test has had to tell prose from a query. `code_of`
            # tokenises both away, which is the whole reason it exists.
            txt = code_of(path.read_text(encoding="utf-8"))
            if "OddsSnapshot" in txt or "odds_snapshots" in txt:
                offenders.append(str(path))
    assert not offenders, (
        "an UNDECLARED file reads the snapshot history: " + ", ".join(offenders)
        + "\nAdd it to _SNAPSHOT_READERS with a reason, and check it is not on "
          "the prediction path — Stage 18 stored this so Stage 19 could study "
          "it, not so the model could use it.")


def test_no_snapshot_READER_sits_on_the_prediction_path():
    """The half of the old guard that still has teeth.

    The research consumer is allowed; a feature builder or a model is not. If
    `src/models/` or `src/features/` ever reads the price path, that is a
    cohort event and `CODE_REVISION` must move — which is a decision, not a
    deploy.
    """
    forbidden = []
    for name, _reason in _SNAPSHOT_READERS.items():
        for path in pathlib.Path("src").rglob(name):
            parts = path.parts
            if "models" in parts and path.name != "models.py":
                forbidden.append(str(path))
            if "features" in parts:
                forbidden.append(str(path))
    assert not forbidden, (
        "a declared snapshot reader lives on the prediction path: "
        + ", ".join(forbidden) + " — bump CODE_REVISION or move the read out")
    # and the research consumer must be a SCRIPT, not library code
    assert pathlib.Path("scripts/h1_collection_check.py").exists(), (
        "h1_collection_check.py moved — re-check which tree it now lives in")


def test_the_analysis_reads_SNAPSHOTS_and_NOT_the_overwritten_table():
    """The positive half: H1 must read the append-only table, by inspection.

    Asserted on the source rather than by running a query, so it holds without
    a database and fails loudly if the read is ever pointed back at `odds`.
    """
    txt = _code_lines(
        pathlib.Path("scripts/h1_collection_check.py").read_text(encoding="utf-8"))
    assert "FROM odds_snapshots" in txt, (
        "H1's analysis no longer reads odds_snapshots — if it reads `odds`, n "
        "is structurally zero because that table keeps one instant per series")
    assert "FROM odds o" not in txt, (
        "H1's analysis reads the OVERWRITTEN `odds` table again")
    assert "o.observed_at" in txt, (
        "the instant column is `observed_at` on odds_snapshots; `timestamp` "
        "belongs to `odds`")


def test_odds_table_shape_is_unchanged_for_existing_consumers():
    """Stage 3's egress work is preserved BY CONSTRUCTION, not by care."""
    cols = Odds.__table__.c
    for required in ("match_id", "bookmaker", "market_type", "selection",
                     "odds_value", "opening_odds", "timestamp"):
        assert required in cols, f"`odds` lost {required}"
    assert cols.first_seen_at.nullable, (
        "first_seen_at is NOT NULL — every existing row would need a backfilled "
        "value, and a guessed first-sight time looks like evidence")
    uniques = [i for i in Odds.__table__.indexes if i.unique]
    assert uniques, "`odds` lost its unique constraint; it is now append-only too"


def test_POSITIVE_CONTROL_the_scan_still_catches_a_REAL_read():
    """The relaxation must not have made the guard vacuous.

    A comment is exempt; a query is not. If this stops failing, `_code_lines`
    has started stripping code.
    """
    assert "odds_snapshots" in _code_lines(
        'q = session.execute(text("SELECT * FROM odds_snapshots"))')
    assert "OddsSnapshot" in _code_lines("rows = session.query(OddsSnapshot).all()")
    # and a trailing comment on a real read is still caught
    assert "odds_snapshots" in _code_lines(
        'rows = q("odds_snapshots")  # reads the price path')
    # while a pure mention is not
    assert "odds_snapshots" not in _code_lines(
        "# inferred from odds_snapshots, cross-checked against the ledger")
