"""Shared pytest fixtures and environment setup."""

import os


# ─────────────── THE `calendar` MARKER (Stage 28, 2026-10-09) ───────────────
#
# A test whose PASS/FAIL CAN CHANGE WITH NO CODE CHANGE does not belong in the
# path that gates price capture. On 2026-10-09 two such tests failed — the
# empty-card suppression's expiry, firing exactly as designed, and three quota
# tests that inherited the allocator's first active day — and together they
# stopped `Run tests`, which stopped `--update`, which lost the first real card
# in nineteen days until it was recovered by hand with 3h01m to spare.
#
# The gate decision is UNCHANGED: capture on an untested tree is still the worse
# risk, because wrong odds rows never backfill. What moves is where a CALENDAR
# assertion lives.
#
# MEMBERSHIP IS DECLARED, NEVER INFERRED. A grep cannot find these: the three
# quota tests that broke contain no clock read at all — `remaining()` reads it.
# So a member carries the marker, and `tests/test_calendar_separation.py`
# asserts the split holds in both directions.
#
# A member asserts a DEPLOYMENT FACT — "this guard has expired, decide" — not a
# correctness property. A correctness test that happens to use a date stays in
# the gating suite.
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "calendar: a test whose outcome depends on today's date rather than on "
        "the code — it asserts a deployment fact and must NOT gate price "
        "capture. Run separately; a failure reds the run without blocking "
        "--update.")
    config.addinivalue_line(
        "markers",
        "live: a test that queries LIVE EXTERNAL STATE (the GitHub API, the "
        "network, a provider). Its outcome changes with no code change, so it "
        "must NOT gate price capture - the same rule as `calendar`, and the "
        "same cost when ignored. MEASURED 2026-10-18: "
        "test_the_detector_runs_against_the_LIVE_repo asserted a date was not "
        "missing from a rolling 14-day `gh run list` window; the date aged "
        "out, `Run tests` failed, and SIX consecutive cards were lost "
        "(10-13..10-18, 137 fixtures, 0 picks). Stage 28 had already moved "
        "the calendar assertions out of the gate for this exact reason, at a "
        "cost of one card.")

# Opt into the insecure default HMAC key for tests. Production runs require
# MODEL_HMAC_KEY to be set or this env var to be set explicitly — see
# src/models/ml_models.py for the rationale.
os.environ.setdefault("BETTING_AGENT_ALLOW_DEFAULT_HMAC", "1")

# AND SQLITE, DECLARED. The suite strips DATABASE_URL on purpose (below), so
# it is the one caller that legitimately wants SQLite. Declaring it is what
# lets `database.py` REFUSE the substitution everywhere else: three confident
# wrong answers came from an entry point that forgot `.env` and was handed an
# empty SQLite file instead of production.
os.environ.setdefault("BETTING_AGENT_ALLOW_SQLITE", "1")

# SAFETY: tests must NEVER touch a real database. CI (and local dev) set
# DATABASE_URL to the production Postgres, and DatabaseManager._create_engine
# prefers env DATABASE_URL over any sqlite_path passed in config — so a DB-backed
# test would read AND WRITE production. This actually happened: test_briefing_dedup
# seeded fixtures straight into the live Supabase DB during a CI run, which then
# re-briefed those rows. Remove the var at import (before any test constructs a
# DatabaseManager) so every test falls back to a local/temp SQLite database.
os.environ.pop("DATABASE_URL", None)


# SAFETY, the filesystem twin of the DATABASE_URL strip above — and it covers
# the CLASS, not the one file that was caught.
#
# THE INCIDENT. On 2026-09-10 `tests/test_odds_credit_gate.py` fed fabricated
# quota headers to `_absorb_quota_headers`, which calls `_persist_credits`,
# which writes data/models/theodds_credits.json. The suite overwrote that file
# with a made-up number and `git add -A` committed it. The true reading was 154;
# the repo shipped 100. `TheOddsScraper.update()` HARD-SKIPS the whole odds
# fetch when that file reads <= 10, and two of those tests write 0 — a different
# test ORDER would have silently disabled pick-time odds fetching in production.
#
# THE RULE, which is not file-specific:
#
#     A TEST MUST NOT WRITE TO ANY PATH PRODUCTION READS.
#
# So the paths are enumerated here ONCE rather than discovered one at a time.
# Every module-level `Path("data/...")` / `Path("config/...")` constant in src/
# is redirected into tmp_path for every test. `tests/test_no_test_writes_prod_state.py`
# fails if a new such constant appears in src/ and is not listed here, so the
# class cannot quietly regrow — the enumeration is enforced, not maintained.
#
# Verified by positive control, per path: disable the redirect and the file is
# clobbered; enable it and the md5 is identical across a full run.
import pathlib as _pathlib

import pytest as _pytest

#: (module, attribute, kind). `kind` is "dir" for directory roots, "file"
#: otherwise — a directory must exist before production writes into it.
PRODUCTION_STATE_PATHS = (
    ("src.data.history_mirror", "_DEFAULT_DIR", "dir"),
    ("src.models.ml_models", "MODELS_DIR", "dir"),
    ("src.models.bayesian_weights", "WEIGHTS_PATH", "file"),
    ("src.models.probability_calibration", "DEFAULT_PATH", "file"),
    ("src.reporting.match_briefing", "_SENT_PATH", "file"),
    ("src.reporting.telegram_bot", "_PICKS_SENT_STATE", "file"),
    ("src.reporting.telegram_bot", "_COLD_STREAK_STATE", "file"),
    ("src.scrapers.barren_leagues", "DEFAULT_PATH", "file"),
    ("src.scrapers.historical_loader", "CACHE_FILE", "file"),
    ("src.scrapers.theodds_scraper", "_CREDITS_STATE_PATH", "file"),
)


@_pytest.fixture(autouse=True)
def _no_test_writes_to_production_state(tmp_path, monkeypatch):
    """No test may write any path production reads."""
    import importlib

    root = _pathlib.Path(tmp_path) / "prodstate"
    root.mkdir(parents=True, exist_ok=True)
    for mod_name, attr, kind in PRODUCTION_STATE_PATHS:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            # A module that cannot even import cannot write anything either.
            continue
        current = getattr(mod, attr, None)
        if current is None:
            continue
        target = root / mod_name.replace(".", "_") / attr
        if kind == "dir":
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target = target.with_suffix(_pathlib.Path(str(current)).suffix)
        monkeypatch.setattr(mod, attr, target, raising=False)


# ─── THE ALLOCATOR'S REGIME IS DECLARED, NEVER INHERITED (2026-10-09) ────────
#
# On 2026-10-09 the H1 reservation turned itself on by calendar and THREE tests
# in `test_odds_quota_and_refresh.py` went red without a line of code changing.
# They were fixed by pinning `today=`. A census that same day found 22 MORE in
# the same position — tests calling a quota method with no explicit `today=`,
# surviving only because their budgets were small enough that withholding 168
# changed no assertion. Latent, not safe.
#
# Making the claim path actually enforce the reservation (the same day, one
# commit later) moved SEVEN of those 22 from latent to failing, which is the
# whole argument for not leaving the class alone: its membership changes when
# unrelated code changes, and nobody is told.
#
# SO THE IMPLICIT PATH RETURNS NO ALLOCATION, AND AN EXPLICIT DATE RETURNS THE
# TRUTH. `h1_reservation()` with no argument reads the wall clock; under test
# that is a test stating no regime, and it now gets zero. `h1_reservation(d)`
# is a test stating WHICH regime it measures, and it gets the real answer —
# including 168 inside the window, which is how `test_h1_credit_attribution`
# measures the allocator on purpose.
#
# This is not a frozen clock. A process-wide frozen clock was tried TWICE on
# this suite and failed both times — tests that write on the real clock and
# read on the pinned one, and `test_history_mirror`'s incremental sync, which
# needs the clock to ADVANCE rather than hold. Nothing here freezes time; one
# calendar predicate stops answering a question nobody asked.
@_pytest.fixture(autouse=True)
def _allocation_regime_must_be_declared(monkeypatch):
    import src.data.odds_quota as _oq

    _real = _oq.h1_reservation

    def _declared_only(today=None):
        if today is None:
            return 0
        return _real(today)

    monkeypatch.setattr(_oq, "h1_reservation", _declared_only)


# ─────────────────────────────────────────── THE caplog TRAP, REFUSED BY A HOOK
#
# `caplog` captures STDLIB logging. Almost everything in `src/` logs through
# LOGURU, which caplog does not see — so an assertion built on it reads an
# EMPTY list, and an empty list is byte-identical to "the line never fired",
# which is usually the exact defect the test was written to exclude.
#
# IT HAS HAPPENED THREE TIMES, and twice there were already files in the tree
# documenting it:
#
#     tests/test_credit_gate_first_refusal.py   documents the trap
#     tests/test_discovery_instrumentation.py   documents the trap
#     tests/test_logging_regime.py              CLASSIFIES which modules use which
#
# All three were in place when it happened the third time, on 2026-10-12, in
# tests/test_model_integrity_observable.py.
#
#     A trap documented three times and hit three times is not a documentation
#     gap. The control has to be the mechanism that refuses it, not the file
#     that describes it.
#
# So: using `caplog` is a COLLECTION ERROR unless the test is declared below,
# and the error names the replacement fixture so the remedy travels with the
# refusal instead of living in a file someone has to already know about.

#: Tests that may use `caplog`, each with the reason. The reason must be that
#: the code under test logs through STDLIB logging — `test_logging_regime`
#: classifies every module in `src/`, and as of 2026-10-12 exactly one does.
CAPLOG_ALLOWED = {
    "tests/test_barren_league_cache.py::test_the_refusal_is_logged":
        "src/scrapers/barren_leagues.py is the ONE module in src/ that uses "
        "stdlib `logging.getLogger`, which test_logging_regime asserts and "
        "classifies. caplog is the correct instrument here, not a trap.",
}


@_pytest.fixture()
def capture_loguru():
    """THE replacement for `caplog` in this project. Captures LOGURU records.

    Yields a list that fills with loguru record dicts — `r["message"]`,
    `r["level"].name`, `r["name"]`. Use it as a context manager factory:

        def test_x(capture_loguru):
            with capture_loguru("ERROR") as records:
                do_the_thing()
            assert any("MARKER" in r["message"] for r in records)

    It is here, in conftest, rather than copied into each file, because three
    local copies of this helper already exist and a fourth was about to be
    written. One rule, one definition.
    """
    import contextlib

    from loguru import logger as _loguru

    @contextlib.contextmanager
    def _capture(level="WARNING"):
        records = []
        sink = _loguru.add(lambda m: records.append(m.record), level=level)
        try:
            yield records
        finally:
            _loguru.remove(sink)

    return _capture


def pytest_collection_modifyitems(config, items):
    """Refuse `caplog` at COLLECTION, with the remedy in the message."""
    offenders = []
    for item in items:
        if "caplog" not in getattr(item, "fixturenames", ()):
            continue
        nodeid = item.nodeid.replace(chr(92), "/")
        if nodeid not in CAPLOG_ALLOWED:
            offenders.append(nodeid)
    if offenders:
        raise _pytest.UsageError(
            "caplog is REFUSED in this project: it captures stdlib logging, "
            "and almost everything in src/ logs through LOGURU. An assertion "
            "on caplog.records reads an EMPTY list, which is byte-identical "
            "to the defect such a test is usually written to exclude. This "
            "has happened three times.\n\n"
            "USE THE `capture_loguru` FIXTURE INSTEAD (tests/conftest.py):\n\n"
            "    def test_x(capture_loguru):\n"
            "        with capture_loguru('ERROR') as records:\n"
            "            do_the_thing()\n"
            "        assert any('MARKER' in r['message'] for r in records)\n\n"
            "If the code under test really does use stdlib logging — "
            "src/scrapers/barren_leagues.py is the only such module in src/, "
            "per tests/test_logging_regime.py — add the test to "
            "CAPLOG_ALLOWED in tests/conftest.py with that reason.\n\n"
            "Offending test(s):\n  " + "\n  ".join(offenders))


# ────────────── THE CALENDAR/LIVE MARKER, ENFORCED RATHER THAN DECLARED
#
# The `calendar` marker shipped 2026-10-09 after a date rolling over cost ONE
# CARD. A new rolling-window test entered the gate on 2026-10-13 and cost SIX
# — 137 fixtures, zero picks, because `Run tests` failed and `--update`,
# `--settle` and the picks were all SKIPPED.
#
#     A remedy that must be remembered when writing new code is not a remedy.
#
# Declared membership failed for the reason it always does: it asks the author
# to classify the thing at the moment they are not thinking about the
# classification. So `scripts/calendar_marker_scan.py` computes it, and this
# hook refuses an unmarked finding AT COLLECTION.
#
# WHY A COLLECTION-TIME REFUSAL IS NOT THE SAME DISEASE. The predicate is
# PURELY STATIC: it reads the test's own AST and nothing else. It therefore
# cannot fire on a date roll, a provider, or a window moving — only on a commit
# that introduces an unmarked time-varying test, and it fires on that commit,
# locally, in the same pytest the author is already running. That is the
# opposite failure mode from the one that cost six cards: immediate and
# code-dependent rather than delayed and calendar-dependent.

#: Tests the scan flags that are CORRECT as written, each with the reason.
#: Static analysis cannot see that both sides of the comparison move together.
#: `tests/conftest.py`'s own note is the governing rule here: "A correctness
#: test that happens to use a date stays in the gating suite." These are
#: correctness tests, so they are exempted rather than marked out of the gate.
CALENDAR_SCAN_ALLOWED = {
    "tests/test_models.py::test_state_file_written_after_send":
        "BOTH SIDES MOVE TOGETHER. It asserts "
        "`state_file.read_text() == date.today().isoformat()`, and the file "
        "was written by the code under test moments earlier WITH today's "
        "date. The scan sees a file read as 'fixed' because it cannot know "
        "the file is clock-derived. The 2026-10-13 defect was the opposite: a "
        "FROZEN literal against a moving window, where only one side moved.",
    # ── ADDED 2026-10-21, when `finding_register.evaluate` joined the live
    # set. Both are EXEMPTED rather than marked, because both are hermetic in
    # fact and marking them would move real coverage out of the capture gate.
    "tests/test_finding_register.py::test_the_COUNTS_and_the_MEDIAN_with_its_caveat":
        "THE ENVIRONMENT DEPENDENCE WAS REMOVED, NOT DECLARED. `evaluate()` "
        "is live only through finding 13, whose predicate is "
        "`not Path('data/models/feature_list.json').exists()`. This test no "
        "longer pins that: it pins the twelve source-only predicates and "
        "asserts 13 SEPARATELY as two-valued, with a guard that 13's "
        "predicate still reads the filesystem so the carve-out cannot go "
        "stale. Proven by control on 2026-10-21: it passes with "
        "feature_list.json present AND absent, which is the exact "
        "disagreement that cost that morning's card.",
    "tests/test_finding_register.py::test_KNOWN_WRONG_the_WEATHER_entry_appears_in_the_harvest":
        "IT ASSERTS ABOUT FINDING 6 ONLY, whose predicate reads "
        "`src/features/feature_engineer.py` -- committed source, so its "
        "verdict moves only when the code moves. Finding 13 is the sole "
        "working-tree-dependent member of the register and this test never "
        "touches it. The scan flags the call because `evaluate` is live "
        "wholesale; it cannot see which row is being asserted.",
}


def pytest_collection_modifyitems(config, items):
    """Refuse `caplog`, and refuse an unmarked time-varying test."""
    # ── caplog, as before ───────────────────────────────────────────────────
    offenders = []
    for item in items:
        if "caplog" not in getattr(item, "fixturenames", ()):
            continue
        nodeid = item.nodeid.replace(chr(92), "/")
        if nodeid not in CAPLOG_ALLOWED:
            offenders.append(nodeid)
    if offenders:
        raise _pytest.UsageError(
            "caplog is REFUSED in this project: it captures stdlib logging, "
            "and almost everything in src/ logs through LOGURU. An assertion "
            "on caplog.records reads an EMPTY list, which is byte-identical "
            "to the defect such a test is usually written to exclude. This "
            "has happened three times.\n\n"
            "USE THE `capture_loguru` FIXTURE INSTEAD (tests/conftest.py):\n\n"
            "    def test_x(capture_loguru):\n"
            "        with capture_loguru('ERROR') as records:\n"
            "            do_the_thing()\n"
            "        assert any('MARKER' in r['message'] for r in records)\n\n"
            "If the code under test really does use stdlib logging — "
            "src/scrapers/barren_leagues.py is the only such module in src/, "
            "per tests/test_logging_regime.py — add the test to "
            "CAPLOG_ALLOWED in tests/conftest.py with that reason.\n\n"
            "Offending test(s):\n  " + "\n  ".join(offenders))

    # ── the calendar/live marker ────────────────────────────────────────────
    # Only over the files actually collected, so a narrowed run is not
    # refused for a file it did not ask for.
    try:
        from scripts.calendar_marker_scan import refusal_text, scan, unmarked
    except Exception:                               # pragma: no cover
        return
    paths = sorted({_pathlib.Path(str(i.fspath)) for i in items
                    if str(getattr(i, "fspath", "")).endswith(".py")})
    if not paths:
        return
    root = _pathlib.Path.cwd()
    rel = []
    for p in paths:
        try:
            rel.append(p.relative_to(root))
        except ValueError:
            rel.append(p)
    bad = [f for f in unmarked(scan(rel))
           if f.nodeid not in CALENDAR_SCAN_ALLOWED]
    if bad:
        raise _pytest.UsageError(refusal_text(bad))


# ── CODE WITHOUT PROSE, BECAUSE TEXT SCANS KEEP MATCHING THEIR OWN COMMENTS ──
#
#     A remedy that must be remembered when writing new code is not a remedy.
#
# THREE TIMES on 2026-10-19 a test asserted `"<pattern>" not in source` and
# matched the COMMENT or DOCSTRING that documents why the pattern is absent.
# Enumerated rather than counted, because a bare ordinal is how this project
# has twice invented an instance number it could not support:
#
#   1. `test_the_fetch_uses_the_STDLIB_and_not_httpx` -- matched its own
#      docstring, which names `httpx` to explain that it is not used
#   2. `test_the_EIGHTH_surface_...` -- matched `DATABASE_URL` in the comment
#      that explains the audit holds no database credentials
#   3. `test_the_MOTIVATING_instance_is_still_fixed_in_ci_audit` -- matched
#      the repaired line quoted in the comment recording the repair
#
# Stripping `#` lines does not help: the longest explanations live in
# docstrings. The same shape had already cost a stage each for `::error::`
# and `date.today()`.
#
#     assert "if not any(crons.values())" not in src          # matches prose
#     assert "if not any(crons.values())" not in code_of(src) # matches code
#
# Use `code_of` for any assertion about what the code DOES. For assertions
# about what is DOCUMENTED, read the raw text deliberately and say so.
#
# ══════════════════════════════════════════════════════════════════════════
# IT IS REQUIRED, NOT OFFERED, AND A MECHANISM SAYS SO. 2026-10-23.
#
#     A remedy that must be remembered is not a remedy.
#
# ELEVEN instances of a negative assertion matching the test's own comment or
# docstring about why the thing is absent -- the eleventh INSIDE the test
# about that very failure. This helper existed through all of them and nothing
# required it.
#
# Third confirmation of the same lesson: the calendar marker and the dialect
# assertion both kept recurring until a mechanism REFUSED them.
#
# `tests/test_code_of_is_enforced.py` is that mechanism. It is a RATCHET, not
# a conversion: measured 2026-10-23, of 61 negative assertions over `.py`
# source, 48 read raw text, 8 hand-roll a `#`-stripper (and so are still
# blind to DOCSTRINGS, which is the half that bit twice) and only 5 use this
# function. Converting 56 blind would change what 56 tests assert, so the
# mechanism refuses GROWTH and a site that must match raw text declares
# itself with a reason.
# ══════════════════════════════════════════════════════════════════════════

def code_of(source: str) -> str:
    """`source` with comments and docstrings removed.

    LINE NUMBERS ARE NOT PRESERVED, and the first version of this docstring
    said they were -- measured on `ci_audit.py` the newline count differs,
    because a multi-line string that is NOT a docstring keeps its own
    newlines while a dropped one contributes a different number. Nothing here
    needs them, so the claim is removed rather than the behaviour fixed: an
    unverified convenience claim is how a helper gets trusted for the one
    thing it cannot do.

    Implemented by tokenising rather than by regex: a `#` inside a string
    literal is not a comment, and a regex that treats it as one introduces a
    second, quieter version of the same bug.
    """
    import io
    import tokenize

    out = []
    prev_end = (1, 0)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError):
        return source          # unparseable: better to over-match than to lie
    # A STRING is a docstring when it is the whole statement. Tracked by the
    # token before it: NEWLINE/INDENT/DEDENT/start means "statement begins".
    at_stmt_start = True
    for tok in toks:
        ttype, tstr, start, end, _line = tok
        if start[0] > prev_end[0]:
            out.append("\n" * (start[0] - prev_end[0]))
            prev_end = (start[0], 0)
        if start[1] > prev_end[1]:
            out.append(" " * (start[1] - prev_end[1]))
        drop = (ttype == tokenize.COMMENT
                or (ttype == tokenize.STRING and at_stmt_start))
        if drop:
            # keep the line count, drop the content
            out.append("\n" * (end[0] - start[0]))
        else:
            out.append(tstr)
        if ttype in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT,
                     tokenize.DEDENT):
            at_stmt_start = True
        elif ttype not in (tokenize.COMMENT,):
            at_stmt_start = False
        prev_end = end
    return "".join(out)
