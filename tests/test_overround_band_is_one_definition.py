"""Stage 18, option 4 — the overround band is declared ONCE, and enforced at the write.

WHY THIS FILE EXISTS. Stage 18 found THREE copies of `(1.005, 1.25)`:
`market_spec`, `feature_engineer` and `baseline`. That is the fifth data-layer
instance of THE HABIT and the worst-placed one, because the constant DEFINES
what "contaminated" means. Three copies drifting apart would change which rows
each consumer trusts, in different directions, with nothing failing anywhere.

The corrective is not "we consolidated them" — it is a test that fails when a
fourth appears.

COHORT NEUTRALITY IS AN IDENTITY, AND IT IS VERIFIED HERE RATHER THAN ASSERTED.
The write gate refuses exactly the books the readers already discard, so no
consumer's input changes. The replay below is the 20-of-20 standard the Stage 13
identity gate met: 20 real contaminated books must be refused, 20 real
legitimate books must be admitted, and the refused set must equal the
currently-discarded set exactly.
"""

import pathlib
import re

import pytest

from src.data.market_spec import (
    OVERROUND_2WAY, OVERROUND_3WAY, check_overround, overround, validate_write)

# Real (home, draw, away) 1X2 books sampled from production on 2026-08-25.
CONTAMINATED = [
    ("Bet365", 1.44, 3.60, 2.62), ("Bet365", 1.67, 3.00, 2.10),
    ("Unibet", 1.16, 3.80, 4.50), ("Bet365", 1.28, 3.40, 3.50),
    ("888Sport", 1.50, 3.10, 2.40), ("Bet365", 1.12, 5.00, 6.00),
    ("Betfair", 1.17, 4.33, 4.50), ("Bet365", 2.10, 3.50, 1.67),
    ("Betano", 2.40, 3.40, 1.53), ("Bet365", 1.62, 3.25, 2.20),
    ("Bet365", 1.44, 3.60, 2.62), ("William Hill", 1.70, 3.10, 2.05),
    ("Bet365", 1.44, 3.70, 2.62), ("888Sport", 1.36, 3.60, 2.90),
    ("Bet365", 1.40, 3.40, 2.75), ("Pinnacle", 1.44, 3.17, 2.89),
    ("William Hill", 3.40, 3.50, 1.29), ("Bet365", 1.28, 4.10, 3.50),
    ("Unibet", 1.62, 3.20, 2.12), ("Bet365", 1.36, 3.60, 3.00),
]
LEGITIMATE = [
    ("TheOddsAPI-williamhill", 2.50, 3.00, 2.75), ("TheOddsAPI-pinnacle", 1.30, 6.15, 9.36),
    ("TheOddsAPI-matchbook", 3.00, 3.90, 2.36), ("Pinnacle", 4.26, 3.77, 1.83),
    ("TheOddsAPI-leovegas_se", 2.60, 3.30, 2.75), ("TheOddsAPI-codere_it", 2.28, 3.15, 2.90),
    ("Pinnacle", 3.44, 2.97, 2.31), ("1xBet", 2.55, 3.36, 2.61),
    ("TheOddsAPI-tipico_de", 3.00, 3.00, 2.50), ("TheOddsAPI-unibet_se", 2.55, 3.20, 3.00),
    ("TheOddsAPI-betonlineag", 1.60, 4.10, 5.00), ("TheOddsAPI-unibet_fr", 3.50, 3.90, 1.92),
    ("TheOddsAPI-codere_it", 1.99, 3.75, 3.55), ("TheOddsAPI-betonlineag", 2.85, 3.62, 2.45),
    # Bet365 is 92% contaminated but NOT always — this book is real and must pass.
    ("Bet365", 1.11, 6.25, 6.50),
    ("TheOddsAPI-betclic_fr", 2.72, 3.17, 2.33), ("TheOddsAPI-betsson", 1.11, 8.60, 26.0),
    ("TheOddsAPI-pmu_fr", 2.18, 3.25, 3.00), ("TheOddsAPI-unibet_nl", 1.30, 5.80, 11.5),
    # Overround 1.0059 — just inside the 1.005 floor. The tightest real case.
    ("Betfair", 2.38, 3.10, 3.80),
]
SELS = ["Home", "Draw", "Away"]


def _refused(prices):
    ok, _ = validate_write("1X2", "Match Winner", SELS, prices=prices)
    return not ok


def _reader_discards(prices):
    """What feature_engineer / baseline / clean_dataset already throw away."""
    lo, hi = OVERROUND_3WAY
    return not (lo <= overround(prices) <= hi)


@pytest.mark.parametrize("book", CONTAMINATED, ids=lambda b: f"{b[0]}-{b[1]}")
def test_contaminated_books_are_refused(book):
    prices = list(book[1:])
    assert _refused(prices), (
        f"{book[0]} at overround {overround(prices):.4f} was ADMITTED — this is "
        "the two-way trap writing into a three-way slot")


@pytest.mark.parametrize("book", LEGITIMATE, ids=lambda b: f"{b[0]}-{b[1]}")
def test_legitimate_books_are_admitted(book):
    prices = list(book[1:])
    assert not _refused(prices), (
        f"{book[0]} at overround {overround(prices):.4f} was REFUSED — the gate "
        "is rejecting real markets, which loses data and IS prediction-affecting")


def test_refused_set_equals_currently_discarded_set():
    """THE COHORT-NEUTRALITY IDENTITY. Verified, not asserted.

    If these two sets ever differ, the write gate is removing something a reader
    would have used, or keeping something a reader throws away. Either way it
    stops being cohort-neutral and needs a decision, not a deploy.
    """
    for name, h, d, a in CONTAMINATED + LEGITIMATE:
        prices = [h, d, a]
        assert _refused(prices) == _reader_discards(prices), (
            f"{name} {prices}: write gate refused={_refused(prices)} but readers "
            f"discard={_reader_discards(prices)} — NOT cohort-neutral")


def test_the_band_has_exactly_one_definition():
    """Fails when a fourth copy appears. This is the whole point of the file."""
    pat = re.compile(r"=\s*\(\s*1\.005\s*,")
    decls = []
    for base in ("src", "scripts"):
        for path in pathlib.Path(base).rglob("*.py"):
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                if pat.search(line):
                    decls.append(f"{path}:{i + 1}: {line.strip()}")
    assert len(decls) == 2, (
        "the overround band must be declared exactly twice — OVERROUND_3WAY and "
        "OVERROUND_2WAY, both in src/data/market_spec.py. Import them; do not "
        "retype them. This constant defines what 'contaminated' means, so copies "
        "drifting apart silently change which rows each consumer trusts.\n\n  "
        + "\n  ".join(decls))
    assert all("market_spec.py" in d for d in decls), (
        "a copy of the band lives outside market_spec.py:\n  " + "\n  ".join(decls))


def test_readers_import_the_band_rather_than_retyping_it():
    fe = pathlib.Path("src/features/feature_engineer.py").read_text(encoding="utf-8")
    bl = pathlib.Path("src/evaluation/baseline.py").read_text(encoding="utf-8")
    assert "from src.data.market_spec import" in fe and "OVERROUND_3WAY" in fe
    assert "from src.data.market_spec import OVERROUND_3WAY" in bl


def test_a_partial_book_is_not_refused_for_overround():
    """A missing leg is a different defect; an overround needs every leg."""
    ok, _ = validate_write("1X2", "Match Winner", SELS, prices=None)
    assert ok, "a book with an incomplete leg set must not be refused HERE"
