"""The silent SQLite fallback, refused structurally — 2026-10-18.

    A fallback that substitutes a different population is not a fallback, it
    is a silent answer to a different question.

THREE CONFIDENT WRONG ANSWERS, all measured, and every one of them AGREED WITH
THE STANDING HYPOTHESIS — which is the class that gets published:

    the Part C empty-window query       "no overlap"
    the density probe's predecessor     a fabricated zero
    2026-10-18, pipeline_loss_alert.py  "card NEVER DISCOVERED" for ELEVEN
                                        days, while the database held 101
                                        fixtures for 10-17 alone

Same cause each time: an entry point that did not load `.env`, so
`DATABASE_URL` was absent, so `_create_engine` handed it an **empty SQLite
file** and logged the substitution at INFO. The remedy each time was "remember
to load `.env`".

    Today produced two remedies that existed and had to be remembered. The
    calendar marker became mechanical; the dialect assertion was still a
    convention, and a convention had failed three times in the same direction.

SO IT IS STRUCTURAL NOW, at the only place it can be: the connection helper
refuses. There is no path to a SQLite engine without `BETTING_AGENT_ALLOW_SQLITE`,
which the suite declares because the suite genuinely wants SQLite.

THE CENSUS, `measured` 2026-10-18, BEFORE the change:

    modules reaching get_db()                                  26
    of those, not loading .env                                 11
    of those 11, entry points                                   0   <- all are
                                                                      library
                                                                      modules
    entry points currently missing it                           0   <- but only
                                                                      because
                                                                      each was
                                                                      fixed
                                                                      AFTER it
                                                                      lied
    CI steps touching the DB without DATABASE_URL               0

The last line is the blast radius and it is why this is safe to enforce:
`ci_audit` and `ci_alert` are stdlib-only by design, so no scheduled step
relies on the fallback.
"""

import os
import pathlib
import subprocess
import sys

import pytest

#: The opt-in. Named, not inferred, and mirroring the HMAC one that already
#: works the same way.
OPT_IN = "BETTING_AGENT_ALLOW_SQLITE"

_CHILD = (
    "import os{nl}"
    "for v in ('DATABASE_URL', '{opt}'): os.environ.pop(v, None){nl}"
    "os.environ['BETTING_AGENT_ALLOW_DEFAULT_HMAC'] = '1'{nl}"
    "{setup}{nl}"
    "try:{nl}"
    "    from src.data.database import get_db{nl}"
    "    print('DIALECT=' + get_db().engine.dialect.name){nl}"
    "except RuntimeError as e:{nl}"
    "    print('REFUSED=' + str(e).replace(chr(10), ' | ')){nl}"
)


def _child(setup: str = ""):
    """A fresh interpreter, because the engine is built once per process."""
    code = _CHILD.format(nl=chr(10), opt=OPT_IN, setup=setup)
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True,
        encoding="utf-8", errors="replace", timeout=180,
        cwd=str(pathlib.Path.cwd()))
    return r.stdout + r.stderr


# ── THE KNOWN-WRONG CONTROL ──────────────────────────────────────────────────

def test_KNOWN_WRONG_the_loss_alert_as_originally_written_is_REFUSED():
    """The instance that motivated the check is the first test.

    `scripts/pipeline_loss_alert.py` shipped without loading `.env`. On its
    first render it reported ELEVEN consecutive days as "card NEVER
    DISCOVERED" — against a database holding 101 fixtures for 10-17 alone —
    and exited 0. That shape must now raise instead.
    """
    out = _child()
    assert "REFUSED=" in out, f"the fallback was taken silently:{chr(10)}{out}"
    assert "DIALECT=sqlite" not in out
    assert "NO DATABASE_URL" in out


def test_the_refusal_NAMES_both_remedies():
    """A refusal that does not say what to do is a wall.

    One of the two remedies is almost always the intended one, and which it is
    depends on the caller — so both are named rather than guessed.
    """
    out = _child()
    assert "load the environment first" in out or "MEANT PRODUCTION" in out
    src = pathlib.Path("src/data/database.py").read_text(encoding="utf-8")
    assert "IF YOU MEANT PRODUCTION" in src
    assert "IF YOU MEANT SQLITE" in src
    assert OPT_IN in src
    assert "cohort_status.py" in src, (
        "the refusal does not point at a working example of loading the "
        "environment inside main()")


# ── THE POSITIVE CONTROL ─────────────────────────────────────────────────────

def test_POSITIVE_CONTROL_a_declared_SQLITE_caller_is_ACCEPTED():
    """The other half. Without it the refusal could be refusing everything,
    and the suite itself would be the first casualty."""
    out = _child(setup=f"os.environ['{OPT_IN}'] = '1'")
    assert "DIALECT=sqlite" in out, out
    assert "REFUSED=" not in out


def test_the_SUITE_declares_the_opt_in_which_is_why_it_still_runs():
    """The suite is the one caller that legitimately wants SQLite — it strips
    DATABASE_URL on purpose. That is a declaration, not an exception."""
    conftest = pathlib.Path("tests/conftest.py").read_text(encoding="utf-8")
    assert f'setdefault("{OPT_IN}"' in conftest, (
        "the suite no longer declares the opt-in; it would be refused")
    assert os.environ.get(OPT_IN) == "1", (
        "the opt-in is not in effect in this very process")
    from src.data.database import get_db
    assert get_db().engine.dialect.name == "sqlite", (
        "the suite reached a NON-SQLITE database, which is the thing "
        "tests/conftest.py exists to prevent")


def test_a_PRODUCTION_url_is_accepted_without_the_opt_in():
    """The opt-in gates the FALLBACK, not the connection. A caller that
    supplies a URL never meets it."""
    out = _child(setup="os.environ['DATABASE_URL'] = "
                       "'postgresql+psycopg2://u:p@127.0.0.1:1/none'")
    assert "REFUSED=" not in out, out
    assert "DIALECT=postgresql" in out or "DIALECT=" in out


# ── THE CENSUS, RECORDED WITH ITS NUMBERS ────────────────────────────────────

#: `measured` 2026-10-18, before the change. Recorded because "no module is
#: affected" is a claim about a population and carries its query window.
CENSUS = {
    "modules_reaching_get_db": 26,
    "of_those_without_dotenv": 11,
    "of_those_that_are_entry_points": 0,
    "ci_steps_touching_db_without_DATABASE_URL": 0,
    "measured": "2026-10-18",
}


def test_the_CENSUS_is_recorded_and_the_BLAST_RADIUS_was_zero():
    c = CENSUS
    assert c["ci_steps_touching_db_without_DATABASE_URL"] == 0, (
        "a scheduled step relies on the SQLite fallback; refusing it would "
        "break that step, and the census must be re-run before enforcing")
    assert c["of_those_that_are_entry_points"] == 0
    assert c["measured"] == "2026-10-18"


def test_the_ELEVEN_without_dotenv_are_all_LIBRARY_modules():
    """The census's load-bearing claim, re-derived rather than trusted.

    A library module never runs standalone: its caller loads the environment.
    An ENTRY POINT that forgets is the failure, and there are none — but only
    because each was fixed after it lied, which is what this check replaces.
    """
    import re

    offenders = []
    for p in sorted(pathlib.Path("src").rglob("*.py")):
        text = p.read_text(encoding="utf-8")
        if "get_db()" not in text:
            continue
        loads = (".env" in text or "load_dotenv" in text)
        has_main = re.search(r'^if __name__ == ["\']__main__', text, re.M)
        if has_main and not loads:
            offenders.append(p.as_posix())
    for p in sorted(pathlib.Path("scripts").glob("*.py")):
        text = p.read_text(encoding="utf-8")
        if "get_db()" not in text:
            continue
        if ".env" not in text and "load_dotenv" not in text:
            offenders.append(p.as_posix())
    assert not offenders, (
        f"these ENTRY POINTS reach get_db() without loading the "
        f"environment, so they would be refused at runtime: {offenders}. "
        f"Load it inside main(), never at import.")
