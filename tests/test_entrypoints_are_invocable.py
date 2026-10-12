"""Every script runs under the invocation its workflow uses — 2026-10-11.

THE DEFECT THIS EXISTS FOR. On 2026-10-10 `ci_audit.py` grew a module-level
`from scripts.schedule_deadline import ...`. Its workflow ran
`python scripts/ci_audit.py`, which puts `scripts/` on `sys.path[0]` rather than
the repo root, so the import raised `ModuleNotFoundError` **at import time**.

Three consecutive `ci-audit` runs went red reporting
`CI audit alarmed — see the audit step for the run ids` when there were **no run
ids** and **nothing had been audited for ~13 hours**. The audit's own output is
its only artifact, and it produced none.

**AND THE TEST SUITE COULD NOT HAVE CAUGHT IT.** `pytest` imports
`scripts.ci_audit` as a module from the repo root — exactly the invocation that
works. Every existing test of that file passed throughout. A packaging defect
is invisible to a test that does not use the real entry point, so this file
spawns SUBPROCESSES and runs what the workflows run.

> A crash boundary inside a module cannot catch its own import failing.

The same job already invoked `python -m scripts.ci_alert`. Two invocation forms
inside one file, and only one of them was exercised by anything.
"""

import pathlib
import subprocess
import sys

import pytest
import yaml

WORKFLOWS_DIR = pathlib.Path(".github/workflows")

#: Entry points a workflow invokes.
ENTRY_POINTS = [
    "scripts/ci_audit.py",
    "scripts/h1_collection_check.py",
    "scripts/h1_collect.py",
    "scripts/fixture_horizon_probe.py",
    "scripts/refresh_and_capture.py",
    "scripts/mark_1x2_assignment_corruption.py",
    # Added 2026-10-11. It was absent, and it is the ONLY entry point whose
    # sole caller is a person — the commit rules require it before every
    # commit — so the module form that masked the defect in the others was
    # never exercised here. `python scripts/cohort_status.py` raised
    # `ModuleNotFoundError: No module named src`.
    "scripts/cohort_status.py",
]

#: Entry points whose SCRIPT form is known to fail on a module-level `src.`
#: import. MEASURED 2026-10-11 by this file's own subprocess check — the first
#: version of this test demanded every entry point survive both forms and
#: immediately found these two.
#:
#: THEY ARE NOT LIVE DEFECTS: every workflow invokes `python -m scripts.x`, so
#: no production path uses the script form. `ci_audit.py` was the lone
#: exception, and that disagreement is what broke.
#:
#: SUCCESSOR CONTRACT, written now: both get the same `sys.path` bootstrap
#: `ci_audit.py` carries, AFTER 2026-10-17. They are left alone today because
#: `h1_collect.py` IS the live H1 collector, `refresh_and_capture.py` carries
#: its `--h1` path, H1 is mid-collection, and a no-op edit to a file running a
#: sized experiment is still an edit. The set is PINNED so it cannot grow
#: silently in the meantime.
SCRIPT_FORM_LATENT = {
    "scripts/h1_collect.py",
    "scripts/refresh_and_capture.py",
}


def _run(args):
    """A real subprocess from the repo root, as CI does. `--help` exits before
    any work, so this exercises IMPORT and argument parsing only — no network,
    no database."""
    return subprocess.run(
        [sys.executable, *args, "--help"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(pathlib.Path.cwd()), timeout=120,
    )


def _broken(r) -> bool:
    return ("ModuleNotFoundError" in r.stderr or "ImportError" in r.stderr
            or r.returncode != 0)


def _workflow_invocations():
    """`{entry point: {forms}}` read from the workflows, not from a list."""
    out = {}
    for wf in sorted(WORKFLOWS_DIR.glob("*.yml")):
        for line in wf.read_text(encoding="utf-8").splitlines():
            st = line.strip()
            if st.startswith("#"):
                continue
            for tok in st.split():
                if tok.startswith("scripts/") and tok.endswith(".py"):
                    out.setdefault(tok, set()).add("script")
                elif tok.startswith("scripts.") and " -m " in f" {st} ":
                    out.setdefault(tok.replace(".", "/") + ".py",
                                   set()).add("module")
    return out


def test_every_workflow_uses_the_MODULE_form():
    """The contract, measured from the workflows rather than assumed."""
    forms = _workflow_invocations()
    script_form = sorted(p for p, f in forms.items() if "script" in f)
    assert not script_form, (
        f"these workflows invoke a script directly: {script_form}. The script "
        f"form puts `scripts/` on sys.path[0], which breaks a module-level "
        f"`scripts.`/`src.` import AT IMPORT — three ci-audit reds and ~13h of "
        f"unaudited runs on 2026-10-10.")


def test_every_workflow_python_invocation_names_a_REAL_entry_point():
    """A workflow invoking a path that does not exist fails at run time.

    Scans the `run:` blocks rather than trusting a list, because the list is
    what drifts.
    """
    missing = [p for p in _workflow_invocations()
               if not pathlib.Path(p).exists()]
    assert not missing, f"workflow invokes a non-existent entry point: {missing}"


@pytest.mark.parametrize("path", ENTRY_POINTS)
def test_the_MODULE_form_imports_cleanly(path):
    """The form every workflow uses. This one must never fail."""
    mod = path.replace("/", ".").removesuffix(".py")
    r = _run(["-m", mod])
    assert not _broken(r), (
        f"{mod} cannot be run with -m, which is the form CI uses:\n"
        f"{r.stderr[-1200:]}")


@pytest.mark.parametrize("path", ENTRY_POINTS)
def test_the_SCRIPT_form_is_either_WORKING_or_PINNED(path):
    """Both forms should work; the ones that do not are an enumerated set.

    Rather than force edits into H1's live collector mid-window, the failures
    are measured and pinned. What the pin buys is that the set cannot GROW
    without this test failing — and it also fails if one starts working, so the
    successor contract gets discharged rather than forgotten.
    """
    r = _run([path])
    if path in SCRIPT_FORM_LATENT:
        assert _broken(r), (
            f"{path} now works as a script — remove it from "
            f"SCRIPT_FORM_LATENT and record the successor contract as "
            f"discharged")
        return
    assert not _broken(r), (
        f"{path} cannot be run as a script and is NOT in the pinned set:\n"
        f"{r.stderr[-1200:]}")


def test_ci_audit_survives_BOTH_forms():
    """The file that broke carries the bootstrap, so the form cannot matter.

    Belt to the workflow's braces, because the defect was precisely two forms
    disagreeing inside one job.
    """
    for args in (["scripts/ci_audit.py"], ["-m", "scripts.ci_audit"]):
        r = _run(args)
        assert r.returncode == 0, f"{args}: {r.stderr[-800:]}"
        assert "ModuleNotFoundError" not in r.stderr


def test_the_POSITIVE_CONTROL_a_broken_import_is_caught():
    """PROVE THE CHECKS ABOVE CAN FAIL.

    Without this they would pass against a script that imported nothing — the
    vacuous-filter shape one level up.
    """
    import tempfile

    with tempfile.TemporaryDirectory(dir="scripts") as d:
        bad = pathlib.Path(d) / "bad_entry.py"
        bad.write_text("import scripts.this_module_does_not_exist\n",
                       encoding="utf-8")
        r = _run([str(bad)])
        assert r.returncode != 0
        assert "ModuleNotFoundError" in r.stderr, (
            "a deliberately broken import was NOT caught, so the checks above "
            "prove nothing")


def test_ci_audit_distinguishes_a_CRASH_from_a_FINDING():
    """Exit 2 is machinery, exit 1 is a verdict.

    A red `ci-audit` is a verdict BY DESIGN, so the process must say which kind
    it is. On 2026-10-10 it could not, and three crashes were reported as
    findings with run ids that did not exist.
    """
    from scripts.ci_audit import EXIT_CLEAN, EXIT_CRASH, EXIT_FINDING

    assert (EXIT_CLEAN, EXIT_FINDING, EXIT_CRASH) == (0, 1, 2)

    wf = (WORKFLOWS_DIR / "ci-audit.yml").read_text(encoding="utf-8")
    assert "audit_exit=$code" in wf, (
        "the audit step no longer exports its exit code, so the fail step "
        "cannot tell a crash from a finding")
    assert "CI AUDIT CRASHED" in wf, "the fail step lost its crash branch"
    d = yaml.safe_load(wf)
    job = list(d["jobs"].values())[0]
    audit = next(s for s in job["steps"] if s.get("id") == "audit")
    assert "python -m scripts.ci_audit" in audit["run"]


def test_the_crash_message_says_NOTHING_WAS_AUDITED():
    """The reader's first question after a crash is what it covered.

    A crash reported as a finding is worse than silence: it sends the reader
    looking for run ids that do not exist.
    """
    import inspect

    from scripts.ci_audit import _main_guarded

    src = inspect.getsource(_main_guarded)
    assert "MACHINERY, not a finding" in src
    assert "NOTHING was audited" in src
    assert "clean bill of health" in src, (
        "the crash message does not warn against reading no findings as clean")


# ═════ PART C — a new workflow cannot be silently outside the audit ═════════

def test_every_workflow_on_disk_is_AUDITED_or_EXCLUDED():
    """THE THIRD AND FOURTH INSTANCE OF THE SAME STRUCTURAL DEFECT.

    `ci-audit.yml` was absent from `WORKFLOWS` until 2026-09-27 and cost EIGHT
    unrecorded reds. The same shape then repeated for both workflows created on
    2026-10-10 — `h1-collect.yml` and `fixture-horizon-probe.yml` — because
    creating a workflow and registering it for audit are two acts and only one
    of them was forced by anything.

    This test makes them one act. A new file under `.github/workflows/` fails
    the suite until it is either listed in `WORKFLOWS` or named in
    `UNAUDITED_WORKFLOWS` with a reason.
    """
    from scripts.ci_audit import UNAUDITED_WORKFLOWS, WORKFLOWS

    on_disk = {p.name for p in WORKFLOWS_DIR.glob("*.yml")}
    registered = set(WORKFLOWS) | set(UNAUDITED_WORKFLOWS)
    unregistered = sorted(on_disk - registered)
    assert not unregistered, (
        f"workflow file(s) outside the audit's scope: {unregistered}. Add them "
        f"to WORKFLOWS, or to UNAUDITED_WORKFLOWS with the reason. A workflow "
        f"silently outside the audit is the defect that cost eight unrecorded "
        f"ci-audit reds.")
    stale = sorted(registered - on_disk)
    assert not stale, (
        f"WORKFLOWS names file(s) that do not exist: {stale} — the audit will "
        f"query a workflow GitHub has never heard of and read the empty result "
        f"as 'no runs'")


def test_every_UNAUDITED_exclusion_carries_a_REASON():
    """An exclusion without a reason is an oversight with a comment."""
    from scripts.ci_audit import UNAUDITED_WORKFLOWS

    for name, reason in UNAUDITED_WORKFLOWS.items():
        assert reason and len(reason) > 20, (
            f"{name} is excluded from the audit with no usable reason: "
            f"{reason!r}")


def test_the_NEVER_ALARM_set_states_its_cost_for_each_member():
    """Each member buys silence and pays the same price, for its own reason.

    `ci-audit` is there to break a feedback loop; the probe is there because its
    red is already its alert. Both forgo self-alarming, and the cost is written
    beside the set rather than only in the ledger.
    """
    import inspect

    import scripts.ci_audit as ci

    assert set(ci.NEVER_ALARM_WORKFLOWS) == {"ci-audit", "fixture-horizon-probe"}
    src = inspect.getsource(ci)
    block = src[src.index("#: Workflows that are LISTED and LEDGERED"):
                src.index("NEVER_ALARM_WORKFLOWS = (")]
    assert "THE COST" in block, (
        "the never-alarm set no longer states what it gives up")
    for member in ci.NEVER_ALARM_WORKFLOWS:
        assert member in block, (
            f"{member} is in NEVER_ALARM_WORKFLOWS with no reason beside it")
    # and h1-collect must NOT be silenced: a red there is a lost collection day
    assert "h1-collect" not in ci.NEVER_ALARM_WORKFLOWS
    assert "h1-collect" in block, (
        "the set does not record WHY h1-collect is excluded from it, so the "
        "next reader may add it for symmetry")


def test_the_audit_step_needs_no_DATABASE_and_asks_for_none():
    """The audit is stdlib + `gh` only, and nothing in it may need the DB.

    A DB-dependent line was added on 2026-10-10 and could never produce its
    number: the step passes `GH_TOKEN` and nothing else, and `ci_audit.py`
    loads no `.env` because the step installs no packages. It silently fell
    back to SQLite and printed `UNKNOWN` every run.
    """
    import pathlib as _p

    raw = _p.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    # COMMENT-BLIND, the same correction a guard needed on 2026-10-02: a
    # literal in prose is provenance, and deleting the provenance to satisfy a
    # scan is the wrong repair. Only executable lines are scanned.
    src = chr(10).join(l.split("#", 1)[0] for l in raw.splitlines())
    for forbidden in ("src.data.database", "get_db(", "psycopg2", "sqlalchemy"):
        assert forbidden not in src, (
            f"ci_audit.py references {forbidden!r}, but its workflow step "
            f"installs nothing and passes no DATABASE_URL — the line will read "
            f"UNKNOWN on every run")
    import yaml as _y
    d = _y.safe_load((WORKFLOWS_DIR / "ci-audit.yml").read_text(encoding="utf-8"))
    job = list(d["jobs"].values())[0]
    audit = next(s2 for s2 in job["steps"] if s2.get("id") == "audit")
    assert "DATABASE_URL" not in (audit.get("env") or {}), (
        "the audit step now takes a database; if that is deliberate it also "
        "needs psycopg2 and SQLAlchemy installed, and the no-install property "
        "in this workflow's header is no longer true")


# ─────────────────────────── A WORKFLOW PyYAML ACCEPTS AND GITHUB REFUSES
#
# MEASURED 2026-10-12. A duplicate `env:` key was pushed to
# `fixture-horizon-probe.yml` and the test that "validated" the file PASSED,
# because `yaml.safe_load` takes the LAST duplicate and says nothing. GitHub
# refused the whole file:
#
#     HTTP 422: failed to parse workflow: (Line: 183, Col: 9):
#     'env' is already defined
#
# So the workflow could not be dispatched OR fire on its cron — a broken
# workflow that every local check called fine.
#
#     A parser that is more permissive than the one in production turns a
#     validation test into a formality.

def _strict_loader():
    """A SafeLoader that REFUSES duplicate mapping keys, as GitHub does."""
    import yaml

    class _Strict(yaml.SafeLoader):
        pass

    def _no_dupes(loader, node, deep=False):
        seen = set()
        for k, _v in node.value:
            key = loader.construct_object(k, deep=deep)
            if key in seen:
                raise yaml.YAMLError(
                    f"duplicate key {key!r} at line {k.start_mark.line + 1}")
            seen.add(key)
        return yaml.SafeLoader.construct_mapping(loader, node, deep)

    _Strict.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_dupes)
    return _Strict


@pytest.mark.parametrize("wf", sorted(
    p.name for p in pathlib.Path(".github/workflows").glob("*.yml")))
def test_no_workflow_has_a_DUPLICATE_KEY(wf):
    """The check `yaml.safe_load` cannot make.

    CLEARING ACTION is in the message: delete one of the two blocks. A
    duplicate `env:` usually means a second block was added beside an existing
    one rather than merged into it.
    """
    import yaml

    path = pathlib.Path(".github/workflows") / wf
    try:
        yaml.load(path.read_text(encoding="utf-8"), Loader=_strict_loader())
    except yaml.YAMLError as exc:
        pytest.fail(
            f"{wf} has a duplicate key, which GitHub refuses with HTTP 422 "
            f"while yaml.safe_load accepts it silently: {exc}. CLEARING "
            f"ACTION: merge the two blocks into one.")


def test_the_STRICT_loader_really_CATCHES_a_duplicate(tmp_path):
    """Positive control. Without it the test above passes against a loader
    that is as permissive as the one it replaces — which is the whole defect."""
    import yaml

    f = tmp_path / "dupe.yml"
    f.write_text("a:\n  env:\n    X: 1\n  env:\n    Y: 2\n", encoding="utf-8")
    # the PERMISSIVE loader accepts it, which is the problem
    assert yaml.safe_load(f.read_text(encoding="utf-8")) == {"a": {"env": {"Y": 2}}}
    with pytest.raises(yaml.YAMLError):
        yaml.load(f.read_text(encoding="utf-8"), Loader=_strict_loader())
