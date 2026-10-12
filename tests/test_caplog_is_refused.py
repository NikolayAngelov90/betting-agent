"""The caplog trap, refused by a mechanism rather than a document — 2026-10-12.

`caplog` captures stdlib logging; almost everything in `src/` logs through
loguru. An assertion on `caplog.records` therefore reads an **empty list**, and
an empty list is byte-identical to "the line never fired" — which is usually
the precise defect the test was written to exclude.

IT HAPPENED THREE TIMES. Twice with files already in the tree describing it:

    tests/test_credit_gate_first_refusal.py   documents the trap
    tests/test_discovery_instrumentation.py   documents the trap
    tests/test_logging_regime.py              classifies which modules use which

All three were in place when it happened the third time, on 2026-10-12, in
`tests/test_model_integrity_observable.py`.

    A trap documented three times and hit three times is not a documentation
    gap. The control has to be the mechanism that refuses it, not the file
    that describes it.

THE CENSUS, `measured` 2026-10-12. Five test files mention `caplog`; **one**
uses it as a fixture; and that one use is **correct** —
`src/scrapers/barren_leagues.py` is the single module in `src/` that uses
`logging.getLogger`, which `test_logging_regime` asserts and classifies. So:

    tests asserting on an EMPTY caplog today: 0

The count the addendum asked for is zero. Each of the three historical
instances was caught and rewritten in the same sitting, so none shipped — the
cost was time, not a wrong verdict, and that is also why documentation looked
like it was working.
"""

import os
import pathlib
import re
import subprocess
import sys

import tests.conftest as conftest

# THE PROBE MUST LIVE IN `tests/`, and two drafts of these controls had it
# elsewhere. A conftest applies to its own directory tree: a file written into
# `tmp_path` is outside it, so neither the hook nor the fixture is in scope and
# the control measured the ABSENCE of the control. Copying the conftest next to
# the probe did not help either — `pytest_collection_modifyitems` is honoured
# from the rootdir's conftest, and the rootdir moved with the file.
#
# So: plant in `tests/`, run in a subprocess, remove in a `finally`. The name
# carries the pid so two runs cannot collide.


def _run_probe(body: str):
    probe = pathlib.Path("tests") / f"test_zz_caplog_probe_{os.getpid()}.py"
    probe.write_text(body, encoding="utf-8")
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(probe),
             "-p", "no:cacheprovider"],
            capture_output=True, encoding="utf-8", errors="replace",
            cwd=str(pathlib.Path.cwd()), timeout=300)
        return r, r.stdout + r.stderr
    finally:
        probe.unlink(missing_ok=True)


# ── THE MECHANISM CARRIES ITS OWN REMEDY ─────────────────────────────────────

def test_the_SHARED_fixture_exists_and_the_ERROR_NAMES_IT():
    """The remedy must travel with the refusal.

    A message that says "don't use caplog" and stops is the documentation
    control again, one layer down.
    """
    src = pathlib.Path("tests/conftest.py").read_text(encoding="utf-8")
    assert "def capture_loguru" in src, "there is no shared fixture to name"
    message = src.split("UsageError", 1)[1][:2000]
    assert "capture_loguru" in message, (
        "the refusal message does not name the replacement fixture")
    assert "CAPLOG_ALLOWED" in message, (
        "the message does not say how a legitimate stdlib case is declared")


def test_the_ALLOWLIST_entries_each_carry_a_REASON():
    """An allowlist without reasons is a silence with extra steps — and the
    reason has to be the real one: that the code under test uses STDLIB
    logging, which `test_logging_regime` classifies."""
    assert conftest.CAPLOG_ALLOWED, "the allowlist is empty"
    for nodeid, reason in conftest.CAPLOG_ALLOWED.items():
        assert "::" in nodeid, f"{nodeid} is not a node id"
        assert len(reason) > 60, f"{nodeid} has no real reason"
        assert "stdlib" in reason.lower(), (
            f"{nodeid}'s reason is not about stdlib logging, which is the "
            f"only thing that makes caplog correct here")


# ── THE CONTROLS ─────────────────────────────────────────────────────────────

def test_POSITIVE_CONTROL_a_caplog_test_FAILS_AT_COLLECTION():
    """Plant one, run it, read the message.

    `no tests ran` is the proof that the refusal is at COLLECTION: the probe
    body would PASS if it executed — an empty `caplog.records` is exactly what
    the trap produces — so a passing body plus zero executions can only mean
    collection stopped it.
    """
    r, out = _run_probe(
        "def test_it(caplog):" + chr(10) +
        "    assert caplog.records == []" + chr(10))
    assert r.returncode != 0, "collection succeeded: " + out[-1500:]
    assert "caplog is REFUSED" in out, out[-1800:]
    assert "capture_loguru" in out, "the remedy is not in the message"
    assert "CAPLOG_ALLOWED" in out, "the escape hatch is not in the message"
    assert "no tests ran" in out, (
        "the probe body RAN - the refusal is at setup, not at collection, so "
        "a test that never executes would slip past it")


def test_NEGATIVE_CONTROL_the_same_test_on_the_LOGURU_fixture_PASSES():
    """The other half. Without it the control above would pass against a hook
    that refused every test in the suite."""
    r, out = _run_probe(
        "from loguru import logger" + chr(10) +
        chr(10) +
        "def test_it(capture_loguru):" + chr(10) +
        "    with capture_loguru('ERROR') as records:" + chr(10) +
        "        logger.error('MARKER here')" + chr(10) +
        "    assert any('MARKER' in r['message'] for r in records)" + chr(10))
    assert r.returncode == 0, out[-1500:]
    assert "1 passed" in out, out[-800:]


def test_an_ALLOWLISTED_caplog_test_still_COLLECTS():
    """The escape hatch works, checked on the real declared entry.

    Without this the allowlist could be dead code while the one legitimate
    stdlib test failed — which would read as an invitation to delete a correct
    test.
    """
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "tests/test_barren_league_cache.py::test_the_refusal_is_logged"],
        capture_output=True, encoding="utf-8", errors="replace",
        cwd=str(pathlib.Path.cwd()), timeout=300)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-1200:]
    assert "1 passed" in out, out[-600:]


def test_the_shared_fixture_really_CAPTURES_what_caplog_MISSES(capture_loguru):
    """The mechanism's premise, asserted rather than assumed.

    If this ever fails, the replacement is as blind as the thing it replaced
    and the refusal above is doing harm.
    """
    from loguru import logger

    with capture_loguru("ERROR") as records:
        logger.error("INTEGRITY-STYLE MARKER: model=x run=1")
    assert records, "the shared fixture captured nothing"
    assert any("INTEGRITY-STYLE MARKER" in r["message"] for r in records)
    assert records[0]["level"].name == "ERROR"


# ── THE CENSUS ───────────────────────────────────────────────────────────────

def test_the_CENSUS_of_existing_caplog_users_is_recorded():
    """`measured` 2026-10-12: five files mention it, one uses it as a fixture,
    and that one is correct — so **zero** tests assert on an empty list.

    Recorded because "no existing test depends on caplog" is a claim, and a
    claim about a population carries its query window.
    """
    files = [p for p in sorted(pathlib.Path("tests").glob("*.py"))
             if "caplog" in p.read_text(encoding="utf-8")]
    assert len(files) >= 4, [p.name for p in files]

    # A PARAMETER, not a name. A draft of this matched any `def test_` line
    # containing the string and counted its own test NAMES — five "uses" where
    # there is one.
    arg = re.compile(r"^def test_[A-Za-z0-9_]*\(([^)]*)\)", re.M | re.S)
    uses = []
    for p in files:
        text = p.read_text(encoding="utf-8")
        for m in arg.finditer(text):
            if "caplog" in [a.strip() for a in m.group(1).split(",")]:
                line = text[:m.start()].count(chr(10)) + 1
                uses.append(f"{p.as_posix()}:{line}")
    assert len(uses) == 1, (
        f"the caplog population moved: {uses}. Re-run the census and say "
        f"whether any of them is asserting on an empty list.")
    assert "test_barren_league_cache" in uses[0], uses
    assert any("test_barren_league_cache" in k
               for k in conftest.CAPLOG_ALLOWED), conftest.CAPLOG_ALLOWED
