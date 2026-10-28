"""Every registered pattern is tested at REGISTRATION, not at its condition.

    2026-10-21. A detector's health must be observable in its quiet state.

`model_integrity_failures` and `model_unsigned_loads` were registered on
2026-10-12, matched their log lines, and COULD NOT FIRE: neither pattern has a
numeric capture group, `int()` raised, the key never reached `facts`, and
`facts.get(...)` was always falsy. TEN DAYS.

The fallback printed "matched but produced no number ... NOT COUNTED" the whole
time — but ONLY WHEN THE PATTERN MATCHED, which needs a real integrity failure
to happen first. **A diagnostic gated on the condition cannot report that the
detector is incapable of firing.**

So every pattern carries an example line it must match, and the self-test
pushes that line through `extract()` and requires a usable value.
"""

import re

import pytest

import scripts.ci_audit as ci

OLD_INTEGRITY_PATTERN = "MODEL INTEGRITY FAILURE"
OLD_INTEGRITY_LINE = "MODEL INTEGRITY FAILURE path=data/models/x.pkl"


# ── the breakdown ──────────────────────────────────────────────────────────

def test_the_SELF_TEST_PASSES_for_every_registered_pattern():
    """Reported as a breakdown, because "it passes" without the counts does
    not say how many patterns were skipped for want of an example."""
    r = ci.selftest_patterns()
    assert r["failed"] == [], (
        "pattern(s) match their own example and still yield nothing — the "
        f"model-integrity shape: {r['failed']}")
    assert r["no_example"] == [], (
        f"pattern(s) with no example, so their health is UNEVALUATED rather "
        f"than proven: {r['no_example']}")
    # 35 -> 36 ON 2026-10-28, and the ratchet is asking what was added:
    # `reminder_collect_errors`, which matches the `##[warning]REMINDERS
    # COLLECTION ERRORS - N test` line that `2c537b1` added and nothing
    # consumed. It carries an example, so 36 of 36 still pass. The run that
    # prints it concludes GREEN, which is why no marker or red-classification
    # path could have caught it.
    assert len(r["passed"]) == len(ci.PATTERNS) == 36, (
        f"{len(r['passed'])} of {len(ci.PATTERNS)} patterns pass; the count "
        f"moved, so say what was added and whether it carries an example")


def test_EVERY_pattern_has_an_EXAMPLE_and_no_example_is_ORPHANED():
    assert set(ci.PATTERN_EXAMPLES) == set(ci.PATTERNS), (
        f"only in PATTERNS: {set(ci.PATTERNS) - set(ci.PATTERN_EXAMPLES)}; "
        f"only in PATTERN_EXAMPLES: "
        f"{set(ci.PATTERN_EXAMPLES) - set(ci.PATTERNS)}")


# ── KNOWN-WRONG CONTROL: the 2026-10-12 state must FAIL ───────────────────

def test_KNOWN_WRONG_model_integrity_AS_IT_WAS_fails_the_self_test(
        monkeypatch):
    """The pattern verbatim as registered on 2026-10-12, under a key that is
    NOT in `extract()`'s count-style list — which is exactly the state that
    made it incapable of firing. `PATTERNS` is patched so `extract()` really
    tries it; without that the key would fail merely for being unknown, which
    is a control passing for the wrong reason.
    """
    key = "zz_integrity_as_it_was_2026_10_12"
    monkeypatch.setitem(ci.PATTERNS, key, OLD_INTEGRITY_PATTERN)
    monkeypatch.setitem(ci.PATTERN_EXAMPLES, key, OLD_INTEGRITY_LINE)

    r = ci.selftest_patterns()
    bad = dict(r["failed"])
    assert key in bad, (
        "the 2026-10-12 shape PASSED the self-test, so the self-test would "
        "not have caught the ten-day gap it exists for")
    assert "model_integrity shape" in bad[key], bad[key]
    # and the real key, now fixed, is NOT among the failures
    assert "model_integrity_failures" not in bad


def test_the_SAME_PATTERN_under_the_REAL_key_now_PASSES():
    """The other half of the control: the fix is what makes the difference,
    not the pattern. Same regex, same example, different key — and the real
    key is in the count-style list."""
    r = ci.selftest_patterns()
    assert "model_integrity_failures" in r["passed"]
    assert "model_unsigned_loads" in r["passed"]
    assert ci.PATTERNS["model_integrity_failures"] == OLD_INTEGRITY_PATTERN, (
        "the pattern itself was changed, so the control above is comparing "
        "two different regexes rather than two different registrations")


# ── POSITIVE CONTROL: an example that rots must FAIL ──────────────────────

def test_POSITIVE_CONTROL_an_example_that_no_longer_matches_FAILS(monkeypatch):
    """So the example cannot rot either. Edit the pattern and leave the
    example behind, and the self-test must say so."""
    monkeypatch.setitem(ci.PATTERNS, "settled", r"Settled EXACTLY (\d+) picks")
    r = ci.selftest_patterns()
    bad = dict(r["failed"])
    assert "settled" in bad, (
        "the pattern was changed and its example left behind, and the "
        "self-test did not notice")
    assert "does NOT match its own example" in bad["settled"]


def test_POSITIVE_CONTROL_a_broken_regex_FAILS_rather_than_raising(monkeypatch):
    monkeypatch.setitem(ci.PATTERNS, "settled", r"Settled (\d+ picks")
    r = ci.selftest_patterns()
    bad = dict(r["failed"])
    assert "settled" in bad and "not a valid regex" in bad["settled"]


def test_a_MISSING_example_is_its_own_state_not_a_pass(monkeypatch):
    """Third state: "no example" is not "passes"."""
    monkeypatch.delitem(ci.PATTERN_EXAMPLES, "settled")
    r = ci.selftest_patterns()
    assert "settled" in r["no_example"]
    assert "settled" not in r["passed"]
    assert "settled" not in dict(r["failed"])


# ── it must actually reach the audit's alarm ──────────────────────────────

def test_the_SELF_TEST_RUNS_IN_main_and_ALARMS_on_a_failure():
    """Rule 4: observable at registration. A self-test only in the suite is
    observable to me; one in `main()` is observable to the operator."""
    import inspect
    src = inspect.getsource(ci.main)
    assert "selftest_patterns()" in src, (
        "the audit does not run the pattern self-test, so a dead detector is "
        "visible only when someone runs the test suite")
    assert "format_pattern_selftest" in src, "the result is not printed"
    i = src.index("selftest_patterns()")
    tail = src[i:i + 1200]
    assert "alarmed.append" in tail, (
        "a failing pattern prints but does not alarm, which is the same "
        "unread-diagnostic shape this check exists to end")
    assert "CLEARING ACTION" in tail, (
        "the alarm does not name the action that clears it")
    # and it must precede the early return on "no runs"
    assert src.index("selftest_patterns()") < src.index("No runs to audit"), (
        "the self-test sits after the early return, so it is unreachable on "
        "the day the audit has nothing else to do")


def test_the_BENIGN_case_no_longer_shares_the_RARE_message():
    """Part B's fix, asserted here because it is the same defect class: the
    `NOT COUNTED` text is how a dead detector reports itself, and it was
    printing on every Flashscore run."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        facts = ci.extract("Flashscore: 12 fixtures updated\n" + "q" * 400)
    assert "NOT COUNTED" not in buf.getvalue(), (
        "an ordinary Flashscore line still prints the dead-detector "
        "diagnostic, so the rare case is buried again")
    assert facts.get("src_flashscore_matched") == 1, (
        "suppressing the message also suppressed the value")
    assert "src_flashscore_matched" in ci._OWNED_BY_A_DEDICATED_BRANCH
