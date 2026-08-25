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


def test_nothing_reads_the_snapshot_table_yet():
    """Cohort neutrality. If a model starts reading history, that is a cohort
    event and it needs a decision, not a deploy."""
    offenders = []
    for base in ("src", "scripts"):
        for path in pathlib.Path(base).rglob("*.py"):
            if path.name in ("models.py", "price_history.py"):
                continue
            txt = path.read_text(encoding="utf-8")
            if "OddsSnapshot" in txt or "odds_snapshots" in txt:
                offenders.append(str(path))
    assert not offenders, (
        "something reads the snapshot history: " + ", ".join(offenders) +
        "\nNothing may read it without a cohort decision — Stage 18 stored it "
        "so Stage 19 could study it, not so the model could use it.")


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
