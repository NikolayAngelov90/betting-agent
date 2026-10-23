"""A scan that matches text about itself — refused, not remembered — 2026-10-23.

Eleven instances, the latest **inside the test about that very failure**. The
tokenising helper `tests/conftest.py:code_of` exists and nothing requires it.

> A remedy that must be remembered is not a remedy. Third confirmation —
> after the calendar marker and the dialect assertion, both of which only
> stopped recurring once a mechanism refused them.

THE DEFECT SHAPE, narrowed by measurement rather than by taste: it is always a
NEGATIVE assertion (`assert "X" not in <source>`) false-firing on the test's
own comment or docstring explaining why X is absent. A POSITIVE assertion
cannot break the same way — at worst a comment satisfies it, which is weaker
and is often where the thing is legitimately recorded.

CENSUS FIRST, `measured` 2026-10-23 over every `tests/test_*.py` that reads a
`.py` source and makes a negative assertion:

    48  RAW text, no stripping at all
     8  HAND-ROLLED stripper (`l.split("#", 1)[0]`) -- comments only, and so
        still blind to DOCSTRINGS, which is the half that bit on 10-22 and
        10-23
     5  uses conftest.code_of  (correct)
    --
    61  total

**So the remedy exists in three forms and 5 of 61 use the right one.** 56
sites would be refused outright, which is why this is a RATCHET and not a
conversion: converting 56 blind would change what 56 tests assert, and the
load-bearing question cannot be settled statically for 29 of them (they read
through a helper or fixture, so the target file is not resolvable from the
AST).

So this refuses GROWTH, the shape `test_pnc_detector`'s backlog already uses
and the only one this project has made work.
"""

import ast
import collections
import pathlib

import pytest

from tests.conftest import code_of

#: Hand-rolled comment strippers seen in the suite. They handle `#` lines and
#: not docstrings.
HAND_ROLLED = ('split("#", 1)[0]', "split('#', 1)[0]",
               'startswith("#")', "startswith('#')")


def _classify():
    """`{bucket: [site, ...]}` over the whole suite."""
    out = collections.defaultdict(list)
    for f in sorted(pathlib.Path("tests").glob("test_*.py")):
        txt = f.read_text(encoding="utf-8")
        for fn in [n for n in ast.walk(ast.parse(txt))
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            seg = ast.get_source_segment(txt, fn) or ""
            reads = (("read_text(" in seg and ".py" in seg)
                     or "inspect.getsource(" in seg)
            if not reads:
                continue
            # a NEGATIVE membership comparison, by AST so a comment cannot
            # look like one
            if not any(isinstance(n, ast.Compare)
                       and any(isinstance(o, ast.NotIn) for o in n.ops)
                       for n in ast.walk(fn)):
                continue
            site = f"{f.name}::{fn.name}"
            if site in EXEMPT:
                out["exempt"].append(site)
            elif "code_of(" in seg:
                out["code_of"].append(site)
            elif any(h in seg for h in HAND_ROLLED):
                out["hand"].append(site)
            else:
                out["raw"].append(site)
    return out


#: ── THE EXEMPTIONS, EACH WITH A REASON ───────────────────────────────────
#:
#: The shape the dialect and marker exemptions already use: a site that must
#: match RAW text declares it here and says why. Prose is in scope for these.
EXEMPT = {
    "test_source_echo_is_stripped.py::"
    "test_CONTROL_3_a_digit_run_inside_a_longer_one_is_not_a_literal":
        "its subject IS a raw-text scan: it asserts the word boundary that "
        "stops `870` matching inside a run id, over a hand-made fixture "
        "rather than over a source file.",
    "test_code_of_is_enforced.py::"
    "test_KNOWN_WRONG_the_ELEVENTH_INSTANCE_verbatim_is_REFUSED":
        "it builds the eleventh instance on purpose, as the control that "
        "proves the refusal works.",
}

#: MEASURED 2026-10-23. The ratchet: these may fall, never rise.
#:
#: `raw` and `hand` are both defects -- `hand` handles comments and not
#: docstrings -- and they are counted apart because the conversion effort
#: differs and because a `hand` site regressing to `raw` should be visible.
BASELINE = {"raw": 48, "hand": 8}


def test_the_CENSUS_is_recorded_with_its_window():
    got = _classify()
    assert len(got["code_of"]) >= 5, (
        f"{len(got['code_of'])} sites use code_of, was 5 on 2026-10-23; the "
        f"correct instrument is being removed from sites that had it")


@pytest.mark.parametrize("bucket", sorted(BASELINE))
def test_the_RAW_MATCH_BACKLOG_does_not_GROW(bucket):
    """The ratchet. A NEW negative raw-text assertion is refused.

    CLEARING ACTION, pick one:
      * route the read through `tests.conftest.code_of` -- the correct
        instrument, which strips comments AND docstrings by tokenising;
      * if the site must match RAW text, add it to `EXEMPT` above with a
        reason, the way the dialect and marker exemptions do;
      * if you converted or deleted a site, LOWER the number here.
    """
    got = _classify()
    n = len(got[bucket])
    assert n <= BASELINE[bucket], (
        f"negative raw-source assertions in bucket {bucket!r} grew from "
        f"{BASELINE[bucket]} to {n}. New sites:\n  "
        + "\n  ".join(sorted(got[bucket])[-6:])
        + "\n" + (test_the_RAW_MATCH_BACKLOG_does_not_GROW.__doc__ or ""))


def test_the_HAND_ROLLED_sites_are_NAMED_because_they_miss_DOCSTRINGS():
    """Eight sites strip `#` lines by hand. That handles the comment half and
    not the docstring half -- and the docstring half is what bit on 2026-10-22
    (a quoted old code line in a docstring) and 2026-10-23 (the record that
    `_ECHO_COLOUR` had been deleted)."""
    got = _classify()
    assert len(got["hand"]) <= BASELINE["hand"]
    # the distinction is real: code_of removes docstrings, the hand-rolled
    # form does not
    src = 'def f():\n    """mentions FORBIDDEN"""\n    return 1\n'
    hand = chr(10).join(l.split("#", 1)[0] for l in src.splitlines())
    assert "FORBIDDEN" in hand, "the hand-rolled form stopped being blind"
    assert "FORBIDDEN" not in code_of(src), (
        "code_of no longer strips docstrings, which is the whole difference")


# ── KNOWN-WRONG CONTROL: the eleventh instance, verbatim ─────────────────

#: The real thing, from `tests/test_source_echo_is_stripped.py` as first
#: written on 2026-10-23. The comment in `ci_audit.py` that it matched is
#: reproduced too, because without it the control proves nothing.
ELEVENTH_INSTANCE_TARGET = '''
#: Pinned to what the CONSUMER observes. An `_ECHO_COLOUR` constant was
#: written first and DELETED when the strip widened to the whole region.
def strip_non_output(log):
    """Drops the echo."""
    return log
'''


def test_KNOWN_WRONG_the_ELEVENTH_INSTANCE_verbatim_is_REFUSED():
    """`assert "_ECHO_COLOUR" not in src` against RAW text, which matched the
    comment recording that `_ECHO_COLOUR` had been deleted.

    Both directions on one input, which is what makes it a control rather
    than an illustration.
    """
    raw = ELEVENTH_INSTANCE_TARGET
    # THE ELEVENTH INSTANCE: raw text, and it fails on the comment
    assert "_ECHO_COLOUR" in raw, "the fixture lost the trap"
    with pytest.raises(AssertionError):
        assert "_ECHO_COLOUR" not in raw
    # THE POSITIVE CONTROL: the same assertion through code_of passes
    assert "_ECHO_COLOUR" not in code_of(raw), (
        "code_of no longer strips the comment, so the remedy itself is broken")
    # and it still SEES the code it is meant to check
    assert "def strip_non_output" in code_of(raw)


def test_the_RATCHET_ITSELF_is_not_satisfiable_by_EXEMPTING_everything():
    """An exemption list that absorbs the backlog makes the mechanism
    vacuous. Two entries today, both of them controls."""
    assert len(EXEMPT) <= 4, (
        f"{len(EXEMPT)} exemptions; the list is absorbing the backlog "
        f"instead of the backlog being converted")
    for site, why in EXEMPT.items():
        assert len(why) > 40, f"{site} is exempt without a reason"


def test_the_RULE_is_RECORDED_where_the_instrument_lives():
    """So the next reader of `code_of` is told it is required, not offered."""
    conf = pathlib.Path("tests/conftest.py").read_text(encoding="utf-8")
    assert "A remedy that must be remembered is not a remedy" in conf, (
        "the rule is not recorded beside `code_of`; it has been rediscovered "
        "eleven times and a twelfth is the default outcome")
    assert "test_code_of_is_enforced" in conf, (
        "`code_of` does not point at the mechanism that requires it")
