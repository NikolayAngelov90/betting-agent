"""The LOCAL config cannot be the source of truth for anything CI runs.

FOURTH INSTANCE OF THIS SURFACE — 2026-11-07.

CI does `cp config/config.example.yaml config/config.yaml` before it runs
anything, so **the example IS the deployed configuration** and the local file
is a scratch copy that is overwritten on every run. `test_config_identity`
already pins the copy, the gitignore and the ordering. What it does not pin is
the thing that nearly went wrong when `betting.min_odds` was moved to open the
FLB band: a change written to `config/config.yaml` alone would have been
silently overwritten, and the band would never have opened.

    The local file is not a weaker source of truth. It is NOT ONE.

So this file adds the scan the other three instances did not: any test or
script that reads `config/config.yaml` must ALSO read the example, or declare
why not. Reading the local file to COMPARE the two is correct; reading it to
learn a deployed value is the defect.

NOTE ON SCOPE. `src/` is deliberately NOT scanned. The application loads
`config/config.yaml` at runtime and that is right — after the copy it IS the
deployed file. The error is specific to tests and measurement scripts, which
run against a working tree where the copy may never have happened.
"""
import ast
import pathlib

import pytest
import yaml

EXAMPLE = pathlib.Path("config/config.example.yaml")
LOCAL = pathlib.Path("config/config.yaml")

#: A triple quote, built rather than written, so this file can plant a
#: docstring inside a test without closing its own.
Q = chr(34) * 3

#: Files that name the local path for a legitimate reason. Each must say what
#: it does with it, and "compares the two" is the only accepted answer.
DECLARED = {
    "test_config_identity.py": ("the identity test itself: it exists to "
                                "compare the two files and to pin the copy"),
    "test_contamination_and_margin.py": ("asserts the local file carries no "
                                         "authority"),
    "test_deployed_config_is_the_example.py": (
        "this scan; it names the path in order to look for it"),
    "test_exchange_terms.py": ("reads BOTH and asserts they agree on "
                               "betting.min_odds"),
}


def _sources():
    for base in ("tests", "scripts"):
        for f in sorted(pathlib.Path(base).rglob("*.py")):
            if "__pycache__" in f.parts:
                continue
            yield f


def _docstring_nodes(tree):
    """Every string constant that is a DOCSTRING, so prose is not a read."""
    out = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            out.add(id(first.value))
    return out


def _reads(path: pathlib.Path, want: str) -> bool:
    """Does this file READ `want`, as opposed to mentioning it?

    A TEXT MATCH IS THE WRONG INSTRUMENT, and it flagged three files on its
    first run -- two comments and one entry in a measured record -- none of
    which reads anything. Comments and docstrings are prose; a read needs the
    path as a live string constant.

    `code_of` is the register's usual answer but importing it from
    `tests.conftest` would pull this file into the CAPTURE GATE, and a config
    scan has no business on the capture path. AST reaches the same place.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return False
    docstrings = _docstring_nodes(tree)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                # EQUALITY, NOT CONTAINMENT. A read passes the path AS the
                # path; a recorded sentence that happens to name it does not.
                # Containment flagged two files whose only mention is inside
                # a prose string stored in a measured record.
                and node.value.strip().lstrip("./") == want):
            return True
    return False


def _reads_local_config(path: pathlib.Path) -> bool:
    return _reads(path, "config/config.yaml")


def _reads_example_config(path: pathlib.Path) -> bool:
    """THE OTHER SIDE OF THE SCAN, through the same predicate.

    It was `"config.example.yaml" in txt` -- a bare positive match whose
    zero-hit case is silent, which the unguarded-matcher ratchet counts. The
    needle and the haystack now both go through `_reads`, and the planted
    control below proves it fires on a read and not on prose. One predicate,
    two needles, one proof.
    """
    return _reads(path, "config/config.example.yaml")


def test_the_EXAMPLE_IS_THE_DEPLOYED_FILE_and_the_local_one_is_ignored():
    """The premise, re-derived rather than trusted."""
    assert EXAMPLE.exists(), "the example config is gone"
    ignore = pathlib.Path(".gitignore").read_text(encoding="utf-8")
    assert "config/config.yaml" in ignore, (
        "`config/config.yaml` is no longer gitignored, so it may now be "
        "committed — and then two files could disagree with CI preferring "
        "whichever the copy wrote last")
    wfs = list(pathlib.Path(".github/workflows").glob("*.yml"))
    assert wfs, "no workflows found"
    copiers = [w.name for w in wfs
               if "cp config/config.example.yaml config/config.yaml"
               in w.read_text(encoding="utf-8")]
    assert copiers, (
        "no workflow copies the example over the local file any more. If that "
        "is deliberate, the example has stopped being the deployed config and "
        "every assertion in this file is about the wrong object")


def test_ANY_FILE_READING_THE_LOCAL_CONFIG_also_reads_the_EXAMPLE():
    """The scan. A deployed value learned from the local file is a value that
    may never have been written by the copy."""
    offenders = []
    for f in _sources():
        if not _reads_local_config(f):
            continue
        if f.name in DECLARED:
            continue
        if _reads_example_config(f):
            continue
        offenders.append(f.as_posix())
    assert not offenders, (
        "these read `config/config.yaml` without reading the example, so "
        "they learn a deployed value from a file CI overwrites: "
        + ", ".join(offenders)
        + ". Read `config/config.example.yaml`, or add the file to DECLARED "
          "with the reason")


def test_EVERY_DECLARED_FILE_STILL_NAMES_THE_LOCAL_PATH():
    """A declaration for a file that no longer reads the local config is a
    stale exemption, and a stale exemption is how an exemption list grows
    past the point of being read."""
    stale = []
    for name, _why in DECLARED.items():
        hits = [f for f in _sources() if f.name == name]
        if not hits:
            stale.append(f"{name} (file is gone)")
            continue
        if not any(_reads_local_config(f) for f in hits):
            stale.append(f"{name} (no longer reads it)")
    assert not stale, f"stale declarations: {stale}"


def test_the_DECLARATIONS_EACH_CARRY_A_REASON():
    for name, why in DECLARED.items():
        assert why and len(why) > 10, f"{name}'s declaration says nothing"


def test_KNOWN_WRONG_THE_TWO_FILES_CAN_DISAGREE_so_the_scan_is_not_vacuous():
    """The control. If the two files were identical by construction the scan
    would be decoration, so this shows they are separate objects that CAN
    diverge — and that the one CI reads is the example.
    """
    if not LOCAL.exists():
        pytest.skip("no local config/config.yaml (fresh clone or CI) — "
                    "the divergence cannot be demonstrated, and that is "
                    "reported rather than passed over as agreement")
    ex = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    lo = yaml.safe_load(LOCAL.read_text(encoding="utf-8"))
    assert ex is not lo, "the same object was loaded twice"
    # they must AGREE on the key this stage moved, which is the live check
    assert ex["betting"]["min_odds"] == lo["betting"]["min_odds"], (
        f"the two configs disagree on betting.min_odds: example "
        f"{ex['betting']['min_odds']} vs local {lo['betting']['min_odds']}. "
        f"CI uses the EXAMPLE, so the example is the one that is live")
    # and the example is the one that must carry the derived FLB floor
    assert ex["betting"]["min_odds"] == 1.50, (
        "the example no longer carries the RESTORED floor. It went 1.50 -> "
        "1.14 for cohort s5.20 and back on 2026-11-11; 1.50 is derived from "
        "the record -- all fifteen cohorts before s5.20 ran at it")


def test_KNOWN_WRONG_A_MENTION_IN_PROSE_IS_NOT_A_READ():
    """The control for the predicate, and it earned its place twice.

    A plain TEXT match flagged three files on its first run:
    `test_gate_is_opt_in.py` (a comment), `test_reference_deadline.py` (a
    real read) and `scripts/exchange_terms.py` (the path inside a recorded
    SENTENCE). An AST match on CONTAINMENT still flagged the sentence. So
    the predicate requires the constant to BE the path, and both needles
    go through it.

    If it ever stops telling prose from a read it reports prose as a
    defect, which is how a scan loses its reader.
    """
    import tempfile

    nl = chr(10)
    prose = (Q + 'A docstring naming config/config.yaml.' + Q + nl
             + '# a comment: config/config.yaml' + nl
             + "X = 'min_odds in config/config.yaml, a gate'" + nl)
    reads = ('import pathlib' + nl
             + "Y = pathlib.Path('config/config.yaml').read_text()" + nl)
    example = ('import pathlib' + nl
               + "Z = pathlib.Path('config/config.example.yaml')"
               + '.read_text()' + nl)

    with tempfile.TemporaryDirectory() as d:
        a, b, c = (pathlib.Path(d) / n for n in
                   ('prose.py', 'reads.py', 'example.py'))
        a.write_text(prose, encoding='utf-8')
        b.write_text(reads, encoding='utf-8')
        c.write_text(example, encoding='utf-8')

        assert _reads_local_config(a) is False, (
            'a docstring, a comment and a prose string were reported as '
            'a read')
        assert _reads_local_config(b) is True, (
            'a real read was not detected, so the scan cannot fail on one')

        # THE OTHER NEEDLE, proven on planted input rather than trusted.
        # It was a bare `in` match over the file text, which the
        # unguarded-matcher ratchet counts: its zero-hit case is silent,
        # and a renamed needle would make every file look like it ignores
        # the deployed config.
        assert _reads_example_config(c) is True, (
            'a read of the EXAMPLE was not detected, so every file would '
            'look like it ignores the deployed config')
        assert _reads_example_config(b) is False
        assert _reads_local_config(c) is False, (
            'the example path matched the LOCAL needle; the two are told '
            'apart by equality, and one path contains the other as a '
            'substring, so this must hold')
