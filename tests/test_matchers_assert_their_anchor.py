"""Every matcher asserts its anchor — 2026-10-23.

> A matcher with zero hits has not verified anything. Passing is a claim about
> what was found, and nothing was found.

FOUR INCIDENTS, ONE SHAPE — a matcher that could no longer match passed
silently:

  1. `\\b` written through a non-raw string became a BACKSPACE byte, so the
     pattern matched nothing and `assert not offenders` passed;
  2. an ANSI-colour filter had its key stripped upstream by `fetch_log`, so it
     removed nothing and its unit test still passed;
  3. `pnc_scan`'s own `KNOWN_INSTANCES` table read as the evidence it was
     scanning, suppressing its findings;
  4. a blanket rename left `assert "REMINDERS_REGISTERED" not in yaml`
     checking for a name the workflow had never contained.

THEY NEED THREE DIFFERENT GUARDS, and saying so is the honest part — one
helper does not cover four shapes:

  * a needle that never existed      -> `anchored`: a SENTINEL in the same
                                        haystack, known to be present
  * a pattern that cannot match      -> `must_match`, or a PLANTED example
  * a filter with no effect          -> measured on REAL input, end to end
  * a scanner suppressed by its data -> a PLANTED finding must be detected

CENSUS FIRST, `measured` 2026-10-23 over every `tests/test_*.py` that derives
a text haystack (`read_text` / `getsource` / `fetch_log` / a regex) and then
claims ABSENCE or EMPTINESS:

    112  matcher sites with an absence/emptiness claim
     73  GUARDED -- the test establishes its haystack is the right one
     39  UNGUARDED -- zero hits reads as a pass

Assertions about a COMPUTED value (`assert exposure == 0` after a real
calculation) are deliberately out of scope: there is no anchor that can
vanish. A first, broader cut counted 336 of 535 by including those, and that
number would have been meaningless to ratchet against.
"""

import ast
import collections
import pathlib
import re

import pytest

from tests.conftest import anchored, code_of, must_match

READS = ("read_text(", "inspect.getsource(", "fetch_log(", "getsource(")
RE_CALLS = ("re.findall(", "re.search(", "re.finditer(", "re.match(")

#: MEASURED 2026-10-23. The ratchet: may fall, never rise.
BASELINE_UNGUARDED = 39

#: ── THE EXEMPTIONS, CAPPED, AND THE CAP IS THE POINT ─────────────────────
#:
#: A list that absorbs the backlog makes the mechanism vacuous — the same
#: reason the `code_of` ratchet caps its own at four. The slot exists for a
#: site whose SUBJECT is a zero-hit scan, where requiring hits would invert
#: the test.
#:
#: SHIPPED EMPTY, and the first draft is why it is worth saying. It carried
#: one entry, for `test_the_CENSUS_is_reproducible` — and that entry was
#: VACUOUS: the census only reaches the exemption check for a site that reads
#: a haystack AND claims absence, and that test does neither (`>=` on two
#: counts). It named a site the mechanism could never have counted. A vacuous
#: exemption inside the mechanism built to refuse vacuous checks, so it is
#: removed rather than reworded. Zero of four is the honest figure.
EXEMPT: dict = {}
EXEMPT_CAP = 4


def _negatives(fn):
    n = 0
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assert):
            continue
        t = node.test
        if isinstance(t, ast.Compare) and any(
                isinstance(o, ast.NotIn) for o in t.ops):
            n += 1
        elif isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not):
            n += 1
        elif isinstance(t, ast.Compare) and any(
                isinstance(o, ast.Eq) for o in t.ops):
            for c in t.comparators:
                if (isinstance(c, ast.Constant) and c.value in (0, "", None)) \
                        or (isinstance(c, (ast.List, ast.Dict, ast.Set))
                            and not (getattr(c, "elts", None)
                                     or getattr(c, "keys", None))):
                    n += 1
    return n


def _establishes_anchor(fn, seg):
    """A positive `in`, a `>=`/`>` count, a split/index that would raise, or
    one of the two conftest guards."""
    if "anchored(" in seg or "must_match(" in seg:
        return True
    for node in ast.walk(fn):
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
            ops = node.test.ops
            if any(isinstance(o, ast.In) for o in ops):
                return True
            if any(isinstance(o, (ast.GtE, ast.Gt)) for o in ops):
                return True
    return ".index(" in seg or ".split(" in seg


def _census():
    out = collections.Counter()
    unguarded = []
    for f in sorted(pathlib.Path("tests").glob("test_*.py")):
        txt = f.read_text(encoding="utf-8")
        for fn in [n for n in ast.walk(ast.parse(txt))
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            if not fn.name.startswith("test"):
                continue
            seg = ast.get_source_segment(txt, fn) or ""
            is_matcher = any(r in seg for r in READS) or \
                any(r in seg for r in RE_CALLS)
            if not is_matcher or not _negatives(fn):
                continue
            site = f"{f.name}::{fn.name}"
            if site in EXEMPT:
                out["exempt"] += 1
                continue
            out["matcher sites"] += 1
            if _establishes_anchor(fn, seg):
                out["guarded"] += 1
            else:
                out["unguarded"] += 1
                unguarded.append(site)
    return out, unguarded


def test_the_CENSUS_is_reproducible():
    """The population, so the ratchet's number means something."""
    out, _ = _census()
    assert out["matcher sites"] >= 100, out
    assert out["guarded"] >= 70, (
        f"guarded sites fell to {out['guarded']}; the suite is losing guards "
        f"rather than gaining them")


def test_the_UNGUARDED_BACKLOG_does_not_GROW():
    """The ratchet.

    CLEARING ACTION, pick one:
      * wrap the haystack in `tests.conftest.anchored(hay, sentinel, what)` —
        a sentinel known to be present, so a re-pointed or renamed haystack
        reports UNEVALUATED instead of passing;
      * for a positive scan, take hits through `must_match`, or prove the
        pattern against a PLANTED example;
      * if the site's subject IS a zero-hit scan, add it to `EXEMPT` with a
        reason — the list is capped at four;
      * if you guarded a site, LOWER the number here.
    """
    out, unguarded = _census()
    assert out["unguarded"] <= BASELINE_UNGUARDED, (
        f"unguarded matcher sites grew from {BASELINE_UNGUARDED} to "
        f"{out['unguarded']}. New ones include:\n  "
        + "\n  ".join(sorted(unguarded)[-6:]) + "\n"
        + (test_the_UNGUARDED_BACKLOG_does_not_GROW.__doc__ or ""))


def test_the_EXEMPTION_LIST_is_CAPPED_and_NO_ENTRY_IS_VACUOUS():
    """Capped, reasoned, and each entry must name a site the census would
    otherwise have COUNTED.

    The third clause is here because the first draft failed it. The entry
    named `test_the_CENSUS_is_reproducible`, which the census never reaches
    -- so the exemption excused nothing and would have sat in the list
    looking like a considered decision.
    """
    assert len(EXEMPT) <= EXEMPT_CAP, (
        f"{len(EXEMPT)} exemptions against a cap of {EXEMPT_CAP}; a list "
        f"that absorbs the backlog makes the mechanism vacuous")

    # the population the census WOULD count, with the exemptions put back
    counted = set()
    for f in sorted(pathlib.Path("tests").glob("test_*.py")):
        txt = f.read_text(encoding="utf-8")
        for fn in [n for n in ast.walk(ast.parse(txt))
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            if not fn.name.startswith("test"):
                continue
            seg = ast.get_source_segment(txt, fn) or ""
            if (any(r in seg for r in READS) or any(r in seg for r in RE_CALLS)) \
                    and _negatives(fn):
                counted.add(f"{f.name}::{fn.name}")

    for site, why in EXEMPT.items():
        assert len(why) > 40, f"{site} is exempt without a reason"
        assert site in counted, (
            f"the exemption for {site} is VACUOUS: the census does not count "
            f"that site in the first place, so the exemption excuses nothing "
            f"and spends one of {EXEMPT_CAP} slots looking like a decision")


# ── KNOWN-WRONG CONTROL 1: the renamed constant ──────────────────────────

def test_CONTROL_1_a_RENAMED_constant_reports_UNEVALUATED():
    """2026-10-23: a blanket rename turned
    `assert "REMINDERS_EXPECTED_MIN" not in yaml` into
    `assert "REMINDERS_REGISTERED" not in yaml` -- a name the workflow had
    never contained -- so it passed while saying nothing.

    AND WRITING THIS CONTROL REPRODUCED THE OTHER HALF: against RAW yaml the
    real needle is still there, in the comment that records its removal. The
    thirteenth comment-blind instance, found by the control rather than in
    production. So the haystack is BOTH stripped AND anchored -- two
    different failures, two different guards, and neither covers the other.
    """
    from tests.test_reminders_are_evaluated import _code_of_yaml

    raw = pathlib.Path(".github/workflows/reminders.yml").read_text(
        encoding="utf-8")

    # THE DEFECT: the needle never existed, so its absence proves nothing
    assert "REMINDERS_REGISTERED" not in raw          # passes, vacuously
    # THE GUARD: that same name as a SENTINEL reports UNEVALUATED, because it
    # is not in the haystack and so the haystack is not the right one.
    with pytest.raises(AssertionError, match="UNEVALUATED"):
        anchored(raw, "REMINDERS_REGISTERED", "reminders.yml")

    # THE SECOND HALF, measured rather than assumed: the REAL needle survives
    # in a comment, so raw text would pass an assertion that is false.
    assert "REMINDERS_EXPECTED_MIN" in raw, (
        "the comment recording the floor's removal is gone, so this control "
        "no longer reproduces the comment-blind half")
    assert "REMINDERS_EXPECTED_MIN" not in _code_of_yaml(raw)

    # AND BOTH TOGETHER is what the real assertion needs: stripped, then
    # anchored on something the counting path must contain.
    guarded = anchored(_code_of_yaml(raw), "reminders_selected=$sel",
                       "reminders.yml's counting path")
    assert "REMINDERS_EXPECTED_MIN" not in guarded


# ── KNOWN-WRONG CONTROL 2: the `\b` backspace regex ──────────────────────

def test_CONTROL_2_a_BACKSPACE_regex_reports_UNEVALUATED():
    """`"\\b870\\b"` through a NON-RAW string is `chr(8) + "870" + chr(8)`.
    It matched nothing, and `assert not offenders` passed."""
    hay = "RESOLVED_RUNS = { '38708288664': ('7514a50', 'the outage') }"
    broken = "\b870\b"                      # the bug, verbatim
    assert chr(8) in broken, "the fixture lost the backspace byte"
    assert re.findall(broken, hay) == [], "the broken pattern matched"
    # THE GUARD: a scan that must find something refuses zero hits
    with pytest.raises(AssertionError, match="UNEVALUATED"):
        must_match(broken, hay, "the RESOLVED_RUNS table")
    # and the CORRECT pattern, word-bounded, finds the run id it should
    assert must_match(r"(?<![0-9])38708288664(?![0-9])", hay, "x")


# ── KNOWN-WRONG CONTROL 3: the ANSI key stripped upstream ────────────────

def test_CONTROL_3_a_FILTER_whose_KEY_was_stripped_reports_UNEVALUATED():
    """The colour-based echo filter removed nothing, because `fetch_log` runs
    `ANSI.sub("", …)` first. Its unit test passed on a raw `gh` log.

    THE GUARD IS A MEASURED EFFECT ON REAL INPUT: a filter must be shown to
    remove something from the text it will actually be given.
    """
    import scripts.ci_audit as ci

    with_colour = f"job\tstep\t2026-10-23T00:00:00Z \x1b[36;1mecho hi\x1b[0m\n"
    stripped = ci.ANSI.sub("", with_colour)
    assert "\x1b[36;1m" in with_colour and "\x1b[36;1m" not in stripped

    # THE DEFECT: keyed on the colour, against text that no longer has it
    def colour_filter(log):
        return "".join(l for l in log.splitlines(keepends=True)
                       if "\x1b[36;1m" not in l)

    assert colour_filter(stripped) == stripped, (
        "the fixture no longer reproduces the no-op")
    with pytest.raises(AssertionError, match="UNEVALUATED"):
        must_match(re.escape("\x1b[36;1m"), stripped,
                   "the log as fetch_log returns it")

    # THE SHIPPED FILTER keys on the group boundaries, which SURVIVE, and its
    # effect on real input is measured rather than assumed
    real = ("job\tstep\t2026-10-23T00:00:00Z ##[group]Run echo hi\n"
            "job\tstep\t2026-10-23T00:00:01Z echo hi\n"
            "job\tstep\t2026-10-23T00:00:02Z ##[endgroup]\n"
            "job\tstep\t2026-10-23T00:00:03Z hi\n")
    out = ci.strip_non_output(ci.ANSI.sub("", real))
    assert out.count("\n") < real.count("\n"), (
        "the shipped filter removes nothing from the text it is really given")
    assert "hi\n" in out and "##[group]Run" not in out


# ── KNOWN-WRONG CONTROL 4: the scanner suppressed by its own data ────────

def test_CONTROL_4_a_SCANNER_SUPPRESSED_BY_ITS_OWN_TABLE_is_refused(tmp_path):
    """`pnc_scan`'s `KNOWN_INSTANCES` names symbols as STRINGS. A blanket
    string-as-reference visitor counted those names as consumers, so the
    scanner's own coverage table suppressed its own findings.

    `anchored` CANNOT GUARD THIS and saying so is the point: the claim is
    about findings, not about text, and zero findings from a scanner that
    cannot see is indistinguishable from a clean tree. The guard is a
    PLANTED FINDING -- the detector is run on a tree built to contain one,
    and must still report it.
    """
    import scripts.pnc_scan as pnc

    NL = chr(10)
    TABLE = "KNOWN_INSTANCES = ({'symbol': 'orphan_producer'},)"
    DEFN = "def orphan_producer():" + NL + "    return 1"

    def no_consumer(body, where):
        d = tmp_path / where
        d.mkdir()
        (d / "planted.py").write_text(body, encoding="utf-8")
        s = pnc.scan(prod_dirs=[str(d)], test_dirs=[])
        return {x.name for x, _ in s.no_consumer}

    # THE DISCRIMINATOR, measured 2026-10-23, because "is it reported at all"
    # does not distinguish anything -- the planted symbol appears in SOME
    # bucket either way. What the defect changed is WHICH bucket.
    #
    #   named only as a STRING in the table -> no_consumer  (correct)
    #   called for real                     -> orphan_cluster
    #
    # so if a string counts as a reference again, the first case starts
    # looking like the second and the finding silently disappears.
    string_only = no_consumer(TABLE + NL * 2 + DEFN + NL, "a")
    called = no_consumer(TABLE + NL * 2 + DEFN + NL * 2 +
                         "def caller():" + NL +
                         "    return orphan_producer()" + NL, "b")

    assert "orphan_producer" in string_only, (
        "UNEVALUATED, not clear: the planted producer left `no_consumer` "
        "although nothing calls it. A string naming a symbol is being "
        "counted as a reference again, so every finding the coverage table "
        "names is suppressed -- and a clean tree would look identical to "
        f"this one. no_consumer={sorted(string_only)}")
    assert "orphan_producer" not in called, (
        "a REAL call no longer moves a producer out of `no_consumer`, so the "
        "control above cannot tell the two cases apart and proves nothing")

    # AND THE SHAPE IS REFUSED STRUCTURALLY TOO, comment-blind, anchored on a
    # name the file must contain for the claim to mean anything.
    src = code_of(pathlib.Path("scripts/pnc_scan.py").read_text(
        encoding="utf-8"))
    assert "def visit_Constant" not in anchored(
        src, "KNOWN_INSTANCES", "pnc_scan.py")


# ── POSITIVE CONTROL ─────────────────────────────────────────────────────

def test_POSITIVE_CONTROL_a_matcher_with_its_anchor_PRESENT_still_asserts():
    """A guard that refuses everything is worse than none. With the anchor
    present, both helpers get out of the way."""
    src = code_of(pathlib.Path("scripts/ci_audit.py").read_text(
        encoding="utf-8"))
    hay = anchored(src, "def strip_non_output(", "ci_audit.py")
    assert hay is src, "anchored altered the haystack"
    assert "_ECHO_COLOUR" not in hay          # the real, meaningful claim
    hits = must_match(r"def strip_non_output\(", hay, "ci_audit.py")
    assert len(hits) == 1, hits


def test_the_RULE_is_RECORDED_beside_the_guards():
    """Recorded where the instrument lives, so the next reader is told the
    guards are required rather than offered."""
    conf = pathlib.Path("tests/conftest.py").read_text(encoding="utf-8")
    assert "A matcher with zero hits has not verified anything" in conf
    assert "test_matchers_assert_their_anchor" in conf, (
        "the guards do not point at the mechanism that requires them")
    # THE THREE SHAPES ARE NAMED, because one helper does not cover four
    # incidents. Matched case-insensitively and through `must_match`, so a
    # re-worded header reports UNEVALUATED rather than a bare failure -- this
    # test is a matcher too.
    for shape in ("sentinel", "planted example", "suppressed by its own"):
        must_match(re.escape(shape), conf.lower(), "conftest.py's header")


def test_the_BACKSPACE_RECORD_is_not_ITSELF_a_backspace():
    """Found while writing this file: `must_match`'s docstring was non-raw,
    so the two characters it quotes had become chr(8) and a grep for them
    found nothing. The record of the backspace bug was written in backspace.

    SEPARATE FROM THE TEST ABOVE ON PURPOSE. This is the only NEGATIVE claim
    in the pair, and it is about a `__doc__` object rather than file text --
    so `code_of` is inapplicable, and leaving it beside a `read_text` of a
    `.py` file made the raw-match census attribute it as a raw site. Split
    rather than exempted: the exemption list has four slots and this would
    have spent one on an attribution artefact.
    """
    from tests.conftest import must_match as mm
    doc = mm.__doc__
    assert chr(92) + "b" in doc, (
        "the docstring quoting the backspace bug has been re-mangled into an "
        "actual backspace; mark it r-prefixed")
    assert chr(8) not in doc
