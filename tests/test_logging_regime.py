"""This project has TWO logging systems, and `caplog` only sees one of them.

ENUMERATED 2026-09-11 after a test in the credit-gate harness asserted on
`caplog.records`, found them empty, and would have passed vacuously whatever the
code did. The messages were going through loguru; `caplog` captures stdlib
`logging` and nothing else.

THE ENUMERATION CAME BACK CLEAN. Exactly two test files reference `caplog`:

  * `test_credit_gate_first_refusal.py` — the one that found the problem. It
    uses a real loguru sink and documents why.
  * `test_barren_league_cache.py::test_the_refusal_is_logged` — VALID, because
    `src/scrapers/barren_leagues.py` is the one module that uses stdlib
    `logging.getLogger(__name__)`. Verified by positive control: renaming the
    EXCLUDING message fails that test, so it observes what it asserts.

So there is nothing to repair. What is worth pinning is the CONDITION that made
it clean, because it is not obvious and it is one edit away from changing:

    `caplog` works in `test_barren_league_cache` ONLY because that module uses
    stdlib logging. If it is ever converted to loguru — which would be a
    tidy-up, not a behaviour change — that test starts passing vacuously and
    nothing else in the suite would notice.

A vacuous assertion cannot be spotted by reading the test. It can only be
spotted by knowing which logger the module under test uses, which is what this
file records.
"""

import pathlib
import re

#: Modules that log through stdlib `logging`, where `caplog` is a valid
#: instrument. Everything else in src/ uses loguru and needs a sink.
STDLIB_LOGGING_MODULES = {
    "src/scrapers/barren_leagues.py",
}

_STDLIB = re.compile(r"^logger\s*=\s*logging\.getLogger\(", re.M)
_LOGURU = re.compile(r"^from loguru import|^\s*from src\.utils\.logger import", re.M)


def _scan():
    stdlib, loguru = set(), set()
    for p in sorted(pathlib.Path("src").rglob("*.py")):
        rel = str(p).replace("\\", "/")
        text = p.read_text(encoding="utf-8")
        if _STDLIB.search(text):
            stdlib.add(rel)
        elif _LOGURU.search(text) or "logger = get_logger()" in text:
            loguru.add(rel)
    return stdlib, loguru


def test_the_stdlib_logging_modules_are_the_pinned_set():
    """A module moving between regimes changes which test instrument is valid."""
    stdlib, _loguru = _scan()
    assert stdlib == STDLIB_LOGGING_MODULES, (
        "the set of stdlib-logging modules changed.\n"
        f"  found:  {sorted(stdlib)}\n"
        f"  pinned: {sorted(STDLIB_LOGGING_MODULES)}\n\n"
        "This matters because pytest's `caplog` captures stdlib logging ONLY. "
        "A test asserting on caplog against a loguru module passes vacuously — "
        "it observes nothing and reports success. If a module LEFT this set, "
        "check every caplog assertion aimed at it; if one JOINED, caplog is "
        "now a valid instrument for it.")


def test_loguru_is_the_majority_regime():
    """Sanity: the exception is the exception."""
    stdlib, loguru = _scan()
    assert len(loguru) > len(stdlib) * 5, (
        f"loguru modules {len(loguru)}, stdlib {len(stdlib)} — the assumption "
        "that loguru is the default no longer holds")


def test_caplog_is_only_used_where_it_can_observe():
    """Every caplog test must target a stdlib-logging module.

    Deliberately coarse: it flags a caplog test that names no pinned stdlib
    module anywhere in its file, which is the cheap proxy for "this assertion
    may be vacuous". A false flag is answered by naming the module in a comment;
    a missed vacuous test is answered by nobody, which is the asymmetry that
    decides the direction.
    """
    offenders = []
    for p in sorted(pathlib.Path("tests").glob("*.py")):
        text = p.read_text(encoding="utf-8")
        if "caplog" not in text:
            continue
        # A file that builds its own loguru sink has already solved this.
        if "loguru" in text:
            continue
        names = {pathlib.Path(m).stem for m in STDLIB_LOGGING_MODULES}
        if not any(n in text for n in names):
            offenders.append(str(p).replace("\\", "/"))
    assert not offenders, (
        "these tests assert on `caplog` but do not reference any module known "
        "to use stdlib logging, so the assertion may observe nothing:\n  "
        + "\n  ".join(offenders)
        + "\n\nEither the module logs through loguru — in which case use a "
          "loguru sink, as tests/test_credit_gate_first_refusal.py does — or "
          "add it to STDLIB_LOGGING_MODULES.")
