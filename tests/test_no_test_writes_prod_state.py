"""The enumeration of production state paths is ENFORCED, not maintained.

`conftest.PRODUCTION_STATE_PATHS` redirects every module-level
`Path("data/...")` / `Path("config/...")` constant in `src/` into tmp_path, so
no test can write a path production reads. A list like that rots the moment
someone adds an eleventh constant, so this fails when the list and the source
disagree.

WHY IT EXISTS. On 2026-09-10 `test_odds_credit_gate.py` fed fabricated quota
headers to a function that persists them, overwrote
data/models/theodds_credits.json with a made-up number, and committed it. The
true reading was 154; the repo shipped 100. `TheOddsScraper.update()` hard-skips
the entire odds fetch when that file reads <= 10, and two of those tests write 0
— a different test ORDER would have disabled pick-time odds fetching in
production, silently.

The first fix redirected that one file. This covers the class, which is the
difference between a guard and an anecdote.
"""

import pathlib
import re

from tests.conftest import PRODUCTION_STATE_PATHS

#: Module-level constants bound to a repo-relative data/ or config/ path.
_CONST = re.compile(
    r"^(_?[A-Z][A-Z0-9_]*)\s*(?::[^=]+)?=\s*(?:pathlib\.)?Path\(\s*[\"'](data|config)/",
    re.M)


def _declared_in_source():
    found = set()
    for path in sorted(pathlib.Path("src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for m in _CONST.finditer(text):
            module = str(path).replace("\\", "/")[:-3].replace("/", ".")
            found.add((module, m.group(1)))
    return found


def test_every_production_path_constant_is_redirected():
    declared = _declared_in_source()
    listed = {(m, a) for m, a, _kind in PRODUCTION_STATE_PATHS}
    missing = declared - listed
    assert not missing, (
        "these module-level production paths are NOT redirected in "
        "tests/conftest.py::PRODUCTION_STATE_PATHS, so a test that writes one "
        "would reach the file production reads:\n  "
        + "\n  ".join(f"{m}.{a}" for m, a in sorted(missing))
        + "\n\nAdd them. The rule is: a test must not write to any path "
          "production reads.")


def test_the_list_has_no_stale_entries():
    """A redirect for a constant that no longer exists is dead weight."""
    declared = _declared_in_source()
    listed = {(m, a) for m, a, _kind in PRODUCTION_STATE_PATHS}
    stale = listed - declared
    assert not stale, (
        "these entries no longer exist in src/ and should be removed:\n  "
        + "\n  ".join(f"{m}.{a}" for m, a in sorted(stale)))


def test_the_redirect_is_actually_in_force():
    """The positive control, baked in.

    Asserting the LIST is right proves nothing about whether the fixture bites.
    This reads the live value of each constant during a test and requires it to
    point somewhere that is not the repo's data/ or config/ tree.
    """
    import importlib

    repo = pathlib.Path.cwd().resolve()
    for mod_name, attr, _kind in PRODUCTION_STATE_PATHS:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        value = getattr(mod, attr, None)
        if value is None:
            continue
        resolved = pathlib.Path(str(value)).resolve()
        for guarded in ("data", "config"):
            under = repo / guarded
            assert under not in resolved.parents and resolved != under, (
                f"{mod_name}.{attr} still points inside {guarded}/ during a "
                f"test ({resolved}). The autouse redirect in conftest is not "
                f"in force, and a write through this constant would land on "
                f"the file production reads.")


# ── THE GUARD ENUMERATED A PATH AND PROTECTED NOTHING (found 2026-10-10) ─────

def test_every_guarded_module_resolves_its_path_at_CALL_time():
    """Listing a path in PRODUCTION_STATE_PATHS is not protecting it.

    `BarrenLeagueCache.__init__` was declared `path=DEFAULT_PATH`, binding the
    module value when the `def` ran. The conftest fixture monkeypatches that
    global, and the patch could not take effect — so this file reported a
    protected path for a class that ignored the patch entirely.

    HOW IT SURFACED, which is the part worth keeping: nothing here caught it.
    `test_h1_caller_reachability` drove `refresh_imminent` with a fetcher
    returning `[]`, which recorded three consecutive empty fetches and wrote
    `data/odds_barren_leagues.json` for real. TWO UNRELATED TESTS then read
    that file and failed — `test_dry_run_spends_nothing` and
    `test_refresh_reconciles_from_response_headers`, neither of which mentions
    barren leagues. A guard that fails silently is found by its collateral.

    SCANNED OVER THE AST, NOT OVER LIVE OBJECTS, and the first version of this
    test got that wrong in a way worth recording: it compared each parameter's
    default against `getattr(module, attr)`. Under the autouse fixture that
    attribute is ALREADY MONKEYPATCHED, so the captured original never matched
    the patched sentinel and the scan reported zero offenders WITH THE DEFECT
    PRESENT. A control caught it — reinstating the defect failed only the
    specific test below, never the general one. Reading the source is immune
    because the source does not change when an object is patched.
    """
    import ast
    import importlib

    from tests.conftest import PRODUCTION_STATE_PATHS

    offenders = []
    for mod_name, attr, _kind in PRODUCTION_STATE_PATHS:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        src_file = getattr(mod, "__file__", None)
        if not src_file:
            continue
        tree = ast.parse(pathlib.Path(src_file).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            a = node.args
            pairs = list(zip(a.posonlyargs + a.args, [None] * (
                len(a.posonlyargs + a.args) - len(a.defaults)) + list(a.defaults)))
            pairs += list(zip(a.kwonlyargs, a.kw_defaults))
            for arg, default in pairs:
                if isinstance(default, ast.Name) and default.id == attr:
                    offenders.append(
                        f"{mod_name}.{node.name}({arg.arg}={attr}) captures a "
                        f"guarded path as a default argument")
    assert not offenders, (
        "a guarded production path is bound at import and the conftest "
        "monkeypatch cannot reach it — resolve it inside the body "
        "(`path if path is not None else DEFAULT_PATH`):\n  "
        + "\n  ".join(offenders))


def test_the_barren_cache_under_the_fixture_does_NOT_point_at_data():
    """The specific case, pinned, and it is the positive control for the above.

    If the general scan ever stops working, this still fails.
    """
    import pathlib

    from src.scrapers.barren_leagues import BarrenLeagueCache

    got = pathlib.Path(BarrenLeagueCache().path).resolve()
    prod = pathlib.Path("data/odds_barren_leagues.json").resolve()
    assert got != prod, (
        f"BarrenLeagueCache() resolved to the PRODUCTION path {got} inside a "
        f"test — a test run now rewrites the barren-league record and can skip "
        f"real leagues on the next production run")
    assert "data" not in got.parts[-2:-1] or got != prod
