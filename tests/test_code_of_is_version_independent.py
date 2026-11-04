"""`code_of` gave DIFFERENT ANSWERS on 3.11 and 3.13, and CI runs 3.11.

THE CARD LOST ON 2026-11-04. `test_the_GRAIN_PROVIDER_and_WINDOW_are_
PRINTED_beside_the_number` asserted `"window=" in code_of(source_of(main))`.
It passed locally on 3.13 and failed in CI on 3.11, so `Run tests` failed,
`--update` was skipped, no card was written, and H1 got `NO_CARD` for the day.

TWO DEFECTS, ONE SYMPTOM:

  `tokenize.NL` IS NOT A STATEMENT BOUNDARY INSIDE BRACKETS. NL is emitted
  both for a blank line between statements and for a line break inside
  ( ) [ ] { }. Treating it as "a statement begins" made the SECOND piece of a
  multi-line string look like a whole-statement string — a docstring — and it
  was dropped. This is version-INdependent and reproduces on 3.13 with a
  plain string.

  AN F-STRING IS A `STRING` TOKEN BEFORE 3.12 AND `FSTRING_*` FROM 3.12
  (PEP 701). So the f-string form was dropped on 3.11 and kept on 3.13 — and
  the version that mattered is the one this repo does not develop on. THAT
  ASYMMETRY IS WHY A 1,710-TEST LOCAL GATE COULD NOT FAIL.

The mitigation here is an EQUIVALENCE test rather than a version check,
because no 3.11 interpreter is installed locally: if `code_of` treats the
f-string and plain forms identically, the PEP 701 split cannot change its
answer.
"""

import glob
import os
import sys

import pytest
import yaml

from tests.conftest import code_of

#: CI's interpreter, pinned. Every workflow sets it; the test below refuses a
#: workflow that drifts off it, because `code_of` is version-sensitive and a
#: silent bump would reopen exactly this hole.
CI_PYTHON = "3.11"

#: The interpreter this file is running on, recorded rather than asserted.
#: ASSERTING IT WOULD BE A LIE WE CANNOT CURRENTLY KEEP: only 3.13 is
#: installed on the development machine, so "the local gate runs on CI's
#: version" is not yet true and is reported as an open gap, not as a pass.
LOCAL_PYTHON = f"{sys.version_info[0]}.{sys.version_info[1]}"


# ═══════════ the known-wrong case: the shape that cost the card ════════════

def test_KNOWN_WRONG_a_CONTINUATION_STRING_IN_A_CALL_SURVIVES():
    """FAILED BEFORE THE FIX, on this interpreter, with a plain string.

    Pre-fix, `code_of` returned `print("alpha "\\n          )` — the second
    piece deleted. Measured 2026-11-04 on Python 3.13.7.
    """
    src = 'def g():\n    print("alpha "\n          "window=X")\n'
    out = code_of(src)
    assert "window=X" in out, (
        "the second piece of a multi-line string was dropped. `tokenize.NL` "
        "inside brackets is being read as a statement boundary, so a "
        "continuation string looks like a docstring.")
    assert "alpha " in out


def test_THE_TWO_FORMS_AGREE_which_is_what_makes_it_VERSION_INDEPENDENT():
    """The f-string and plain forms must be treated the same.

    This is the whole mitigation. Before 3.12 an f-string is a STRING token
    and after it is not, so any rule that keys on STRING-ness gives two
    answers. If both forms survive, the PEP 701 split cannot matter.
    """
    plain = code_of('def g():\n    print("a "\n          "window=X")\n')
    fstr = code_of('def g():\n    print(f"a "\n          f"window={x}")\n')
    assert ("window=X" in plain) and ("window=" in fstr), (
        f"the forms disagree: plain={'window=X' in plain} "
        f"fstring={'window=' in fstr}. On the other interpreter the "
        f"surviving one would be the dropped one.")


def test_an_FSTRING_IS_NEVER_TREATED_AS_A_DOCSTRING():
    """A lone f-string statement is not a docstring on any version."""
    out = code_of('def g():\n    f"not a doc {x}"\n    return 1\n')
    assert "not a doc" in out, (
        "a lone f-string was dropped as a docstring. On 3.12+ it is "
        "FSTRING_* and cannot be; the two versions must agree.")


# ═══════════ and the fix must not have over-corrected ══════════════════════

@pytest.mark.parametrize("src,needle", [
    ('def g():\n    """secret"""\n    return 1\n', "secret"),
    ('"""module doc"""\nx = 1\n', "module doc"),
    ('class C:\n    """cls doc"""\n    x = 1\n', "cls doc"),
    ('def g():\n    x = 1\n\n    "lone"\n', "lone"),
])
def test_REAL_DOCSTRINGS_ARE_STILL_DROPPED(src, needle):
    """Widening the keep rule must not stop it dropping what it is for."""
    assert needle not in code_of(src), (
        f"{needle!r} survived; the docstring rule has been widened away and "
        f"every `not in code_of(...)` assertion now passes for free")


def test_COMMENTS_ARE_STILL_DROPPED_and_a_hash_in_a_string_is_not_one():
    assert "hidden" not in code_of("x = 1  # hidden\n")
    assert "not#acomment" in code_of('x = "not#acomment"\n')


# ═══════════ the interpreter the gate runs on ══════════════════════════════

def test_EVERY_WORKFLOW_PINS_THE_SAME_PYTHON_and_it_is_the_recorded_one():
    """A silent version bump would reopen this hole.

    `code_of` is version-sensitive by construction — it reads token types
    whose meaning changed in 3.12 — so the interpreter is part of the gate's
    definition and not an environment detail.
    """
    seen = {}
    for f in sorted(glob.glob(".github/workflows/*.yml")):
        d = yaml.safe_load(open(f, encoding="utf-8"))
        for job in (d.get("jobs") or {}).values():
            for step in job.get("steps") or []:
                v = (step.get("with") or {}).get("python-version")
                if v is not None:
                    seen.setdefault(str(v), []).append(os.path.basename(f))
    assert seen, "no workflow pins a python-version at all"
    assert set(seen) == {CI_PYTHON}, (
        f"workflows pin {sorted(seen)}, not just {CI_PYTHON!r}. `code_of` "
        f"answers differently across the 3.12 boundary, so a mixed or bumped "
        f"pin means the gate and CI are different gates: {seen}")
    assert len(seen[CI_PYTHON]) >= 8, (
        f"only {len(seen[CI_PYTHON])} workflow(s) pin a version; a new one "
        f"that does not is running on whatever the runner defaults to")


def test_THE_LOCAL_INTERPRETER_GAP_IS_RECORDED_not_asserted_away():
    """An open gap, stated as one.

    "The local gate runs on CI's Python version" is NOT yet true: only
    {LOCAL} is installed on the development machine and CI runs {CI}. The
    equivalence tests above are the mitigation; they are not the same thing
    as running the gate on the same interpreter, and this test exists so the
    difference is never mistaken for a pass.
    """
    if LOCAL_PYTHON == CI_PYTHON:
        return      # the gap closed; nothing to record
    assert LOCAL_PYTHON != CI_PYTHON
    # The gap is tolerated ONLY while the equivalence tests above pass, which
    # is what removes code_of's sensitivity to the boundary.
    plain = code_of('def g():\n    print("a "\n          "k=V")\n')
    fstr = code_of('def g():\n    print(f"a "\n          f"k={x}")\n')
    assert ("k=V" in plain) and ("k=" in fstr), (
        f"local Python {LOCAL_PYTHON} differs from CI's {CI_PYTHON} AND "
        f"`code_of` is still version-sensitive. One of the two must be fixed "
        f"before this gate means anything.")
