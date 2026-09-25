"""The SQLAlchemy upper bound must hold in all THREE places that install it.

FOUND 2026-09-25, while applying the bound. `requirements.txt` is read only by
`daily-picks`. `closing-lines` and `paper-trading-report` install by name:

    pip install sqlalchemy psycopg2-binary aiohttp pyyaml python-dotenv ...

So pinning `sqlalchemy<2.1` in requirements.txt alone would have fixed the one
workflow that did NOT fail and left the two that did — closing-lines at 20:28
and 23:51 on 09-24 — installing 2.1.0 bare and failing identically.

    A fix applied where the failure is not does not look different from
    a fix.

THE CAUSE, so this can be removed deliberately rather than as stale: SQLAlchemy
2.1.0 changed the default DBAPI for a bare `postgresql://` URL from psycopg2 to
psycopg (v3). This repo ships `psycopg2-binary` and not `psycopg`, so
`create_engine` raises `ModuleNotFoundError: No module named 'psycopg'` inside
`sqlalchemy/dialects/postgresql/psycopg.py`.

TO LIFT IT: add `psycopg[binary]`, or write the driver explicitly into
DATABASE_URL as `postgresql+psycopg2://`. Then delete this test with the bounds.
Verify against a real Postgres URL — SQLite never reaches that code path.

The bound is duplicated across three files because two of them install by name,
and duplication drifts. This test is the cost of that, and it is cheaper than
rediscovering the outage.
"""

import pathlib
import re

import pytest

REQUIREMENTS = pathlib.Path("requirements.txt")
WORKFLOWS = [
    pathlib.Path(".github/workflows/closing-lines.yml"),
    pathlib.Path(".github/workflows/paper-trading-report.yml"),
]
DAILY = pathlib.Path(".github/workflows/daily-picks.yml")

#: Any upper bound at all. The exact form is not pinned — `<2.1`, `<2.1.0` and
#: `!=2.1.*` are all fine — because pinning the spelling would fail on a
#: legitimate reformat while saying nothing about the risk.
UPPER_BOUND = re.compile(r"sqlalchemy\s*[><=!,.0-9\s]*?(?:<\s*2\.1|!=\s*2\.1)",
                         re.IGNORECASE)


def test_requirements_bounds_sqlalchemy_below_2_1():
    text = REQUIREMENTS.read_text(encoding="utf-8")
    assert UPPER_BOUND.search(text), (
        "requirements.txt has no upper bound on sqlalchemy — 2.1 resolves a "
        "bare postgresql:// URL to psycopg v3, which this repo does not ship")


@pytest.mark.parametrize("wf", WORKFLOWS, ids=lambda p: p.name)
def test_every_workflow_that_installs_BY_NAME_carries_the_bound(wf):
    """The two that bypass requirements.txt must carry it themselves.

    This is the half that would have been missed: these are the workflows that
    actually went red.
    """
    text = wf.read_text(encoding="utf-8")
    installs = [l for l in text.splitlines()
                if "pip install" in l and "sqlalchemy" in l.lower()]
    assert installs, f"{wf.name} no longer installs sqlalchemy by name — if it " \
                     f"now reads requirements.txt, delete this case"
    for line in installs:
        assert UPPER_BOUND.search(line), (
            f"{wf.name} installs sqlalchemy with no upper bound:\n  "
            f"{line.strip()}\nrequirements.txt is NOT read by this workflow, so "
            f"the bound there does not protect it")


def test_daily_picks_installs_from_the_file_so_it_needs_no_duplicate():
    """Pins the reason the third workflow is exempt, not just the fact.

    If this starts failing, daily-picks stopped installing from
    requirements.txt and needs the bound inline like the other two.
    """
    text = DAILY.read_text(encoding="utf-8")
    assert "pip install -r requirements.txt" in text, (
        "daily-picks no longer installs from requirements.txt — it now needs "
        "the sqlalchemy bound written into the workflow as well")


def test_psycopg_v3_is_still_absent_so_the_bound_is_still_load_bearing():
    """The bound exists because psycopg v3 is not installed.

    If `psycopg` is ever added, the bound can go — and this test says so
    rather than leaving a future reader to guess whether it is stale.
    """
    # COMMENT LINES ARE STRIPPED FIRST. The first version matched the word
    # `psycopg` inside this file's own explanation of the bound and failed on
    # it — a definition read as an occurrence, the same error `_is_provisional`
    # made on a ledger row and the ls-files meta-test made on its own controls.
    lines = [l.split("#", 1)[0].strip()
             for l in REQUIREMENTS.read_text(encoding="utf-8").splitlines()]
    declared = [l for l in lines
                if re.match(r"psycopg(?!2)[\[><=;\s]*", l, re.IGNORECASE)]
    assert not declared, (
        f"psycopg v3 appears in requirements.txt as {declared} — if it is "
        f"genuinely installed, the sqlalchemy<2.1 bound is no longer needed. "
        f"Remove the bound and this test together, after verifying against a "
        f"real Postgres URL.")


# ── CONSTRAINTS: every install site must use it ──────────────────────────────
#
# Added 2026-09-25 with constraints.txt. The file converts "a release breaks
# production" into "a release breaks the refresh", and it does that only for
# the install steps that actually reference it. One step left unwired is the
# same shape as pinning requirements.txt while two workflows install by name.

CONSTRAINTS = pathlib.Path("constraints.txt")
ALL_WORKFLOWS = [DAILY] + WORKFLOWS


def test_constraints_file_exists_and_pins_the_silent_set():
    """The packages whose failure mis-deserialises rather than raising.

    The pipeline reloads ml_models.pkl / goals_model.pkl across runs, and
    `_init_models()` guards the refit path rather than the load path — so a
    format change in these is quiet.
    """
    assert CONSTRAINTS.is_file(), "constraints.txt is gone"
    text = CONSTRAINTS.read_text(encoding="utf-8")
    for pkg in ("scikit-learn", "xgboost", "lightgbm", "numpy", "pandas"):
        assert re.search(rf"^{re.escape(pkg)}==", text, re.M), (
            f"{pkg} is not pinned — it is in the silent-failure set")


def test_constraints_pins_the_zero_rows_green_run_set():
    """camoufox/playwright/selenium — the presentation that hid for 88 days."""
    text = CONSTRAINTS.read_text(encoding="utf-8")
    for pkg in ("camoufox", "playwright", "selenium", "undetected-chromedriver"):
        assert re.search(rf"^{re.escape(pkg)}==", text, re.M), f"{pkg} unpinned"


def test_every_direct_requirement_is_constrained():
    """A requirement with no constraint is the gap the file was written to close."""
    def names(path, pattern):
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            m = re.match(pattern, line)
            if m:
                out.append(m.group(1).lower().replace("_", "-"))
        return set(out)

    required = names(REQUIREMENTS, r"([A-Za-z0-9_.+-]+)\s*[><=!]")
    pinned = names(CONSTRAINTS, r"([A-Za-z0-9_.+-]+)\s*==")
    missing = sorted(required - pinned)
    assert not missing, f"direct requirements with no constraint: {missing}"


@pytest.mark.parametrize("wf", ALL_WORKFLOWS, ids=lambda p: p.name)
def test_every_install_step_uses_the_constraints_file(wf):
    text = wf.read_text(encoding="utf-8")
    installs = [l for l in text.splitlines()
                if "pip install" in l and "--upgrade pip" not in l]
    for line in installs:
        assert "-c constraints.txt" in line, (
            f"{wf.name} installs without the constraints file:\n  {line.strip()}\n"
            f"that step resolves freely and a release breaks it the same day")
