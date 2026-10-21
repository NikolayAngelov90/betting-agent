"""The diagnosed-and-unactioned register — 2026-10-20.

    An unmarked entry is how seventy-four days happened.

The weather producer was diagnosed on 2026-08-07 at
`docs/predictive-audit-2026-08-07.md` line 650, and deleted on 2026-10-20.
**Seventy-four days.** Nothing counted it in between, so nothing could notice.

THE HARVEST PREDICATE, stated before it ran:
  1. a defect named with a MECHANISM, not a symptom
  2. a LOCATION — file, line, symbol, table or workflow step
  3. no subsequent entry recording a fix, deletion, declaration or exemption

QUERY WINDOW, and it is the honest limit: exactly ONE document in the repo
carries findings in a form the predicate can be applied to mechanically —
that audit's 13-row `# | File | Location | Defect` table at lines 645-657.
The ledger's 29,700 lines are prose, and a mechanical harvest of "diagnosis
without subsequent fix" from prose would be a confident guess. **So every age
is the same 74 days and the median carries no spread** — a property of the
source, not of the backlog.

THE RESULT, `measured` 2026-10-20 against today's tree:

| | |
|---|---|
| entries harvested | **13** |
| still live | **8** |
| fixed since | 5 (findings 1, 2, 3, 5, 9) |
| UNEVALUATED | 0 |
| untriaged | **0** |
| median age of the still-live | **74 days** (point mass) |

| outcome | n | findings |
|---|---|---|
| `FIX` | 1 | 13 — `data/models/` untracked, `feature_list.json` missing |
| `DELETE` | 1 | 6 — the injury half of the dead-work row |
| `DECLARE` | 6 | 4, 7, 8, 10, 11, 12 — all touch the predictive core |

**SIX OF THE EIGHT ARE `DECLARE`, AND THAT IS THE EXPLANATION OF THE 74 DAYS.**
The backlog did not stall because nobody looked; it stalled because most of it
is cohort-affecting, and this project will not change a fitted input or a
selection rule inside an observability stage. The two that were *not*
cohort-affecting are the two that moved: weather (deleted) and `data/models/`
(a bounded fix).

WHY A REGISTER AND NOT A MARKER IN THE LEDGER — Part C's question, answered
no. Annotating a 2026-08-07 entry with today's status **rewrites it**, and
historical entries are not rewritten; that rule has held four times. So the
register references doc and line and holds the status itself.
"""

import datetime as dt
import pathlib

import scripts.finding_register as fr


# ── THE KNOWN-WRONG CONTROL ──────────────────────────────────────────────────

def test_KNOWN_WRONG_the_WEATHER_entry_appears_in_the_harvest():
    """A harvest that cannot find the instance that motivated it has no
    measured coverage.

    Row 6 as written on 2026-08-07 named BOTH halves: *"Injury and weather
    features excluded from training but computed at predict time -> dropped by
    `_align_features`. Dead work on every fixture"*. It must be in the output,
    and it must still read LIVE — because the weather half was deleted on
    2026-10-20 and the **injury half is unchanged**.
    """
    rows = {r["id"]: r for r in fr.evaluate()}
    assert 6 in rows, "the weather/injury row is not in the harvest at all"
    r = rows[6]
    assert "weather" in r["diagnosis"] and "Injury" in r["diagnosis"], (
        "the row's diagnosis no longer names both halves verbatim")
    assert "Dead work on every fixture" in r["diagnosis"]
    assert r["file"] == "src/features/feature_engineer.py"
    assert r["is_live"] is True, (
        "row 6 reads as fixed. The WEATHER half was deleted 2026-10-20, but "
        "the INJURY half is still computed at predict time and dropped by "
        "alignment -- a half-fixed row is live, not fixed.")
    assert r["outcome"] == fr.DELETE
    assert "injury SCRAPER is NOT" in r["why"], (
        "the triage no longer distinguishes the injury FEATURES (no consumer, "
        "deletable) from the injury SCRAPER (reaches the Claude prompt)")


def test_the_DIAGNOSES_are_VERBATIM_from_the_source():
    """Nothing here paraphrases the original, so the harvest can be audited
    against the document it references."""
    doc = pathlib.Path(fr.SOURCE).read_text(encoding="utf-8").splitlines()
    for r in fr.FINDINGS:
        row = doc[r["line"] - 1]
        assert row.lstrip().startswith(f"| {r['id']} |"), (
            f"finding {r['id']} no longer points at its own row: {row[:60]!r}")
        # The row's own file cell must name the file the register records.
        # BACKTICKS STRIPPED: the source writes `config/config.yaml` vs
        # `config.example.yaml` with markdown code spans, so a raw substring
        # test fails on formatting rather than on content -- the same
        # distinction this file set has had to draw for emphasis and for line
        # wrapping.
        plain = row.replace("`", "")
        assert r["file"] in plain or r["file"].split("/")[-1] in plain, (
            f"finding {r['id']}'s file cell disagrees with the source row: "
            f"{plain[:70]!r}")


def test_the_SOURCE_is_REFERENCED_and_NOT_REWRITTEN():
    """Rule 3, mechanically. The register may not annotate history."""
    doc = pathlib.Path(fr.SOURCE).read_text(encoding="utf-8")
    for marker in ("FIX", "DELETE", "DECLARE"):
        # the audit doc may discuss these words in prose, but must carry no
        # per-row status column added by this register
        assert f"| {marker} |" not in doc, (
            f"a `{marker}` status column was added to the source document -- "
            f"that rewrites a 2026-08-07 entry, which is refused")
    assert "finding_register" not in doc, (
        "the source document now references the register; the dependency runs "
        "the other way, so history stays as written")


# ── THE COUNTS, AND THE THIRD STATE ─────────────────────────────────────────

def test_the_COUNTS_and_the_MEDIAN_with_its_caveat():
    rows = fr.evaluate(today=dt.date(2026, 10, 20))
    assert len(rows) == 13
    live = [r for r in rows if r["is_live"] is True]

    # ── FINDING 13 IS ENVIRONMENT-DEPENDENT BY CONSTRUCTION. 2026-10-21. ────
    #
    # Its predicate is `not Path("data/models/feature_list.json").exists()`,
    # which reads the WORKING TREE. On a dev box the file is absent, so the
    # finding is live; in CI `actions/cache` restores `data/models/` before the
    # tests run, so it is NOT live. Pinning it made this test pass locally and
    # fail in CI, and because it sits in the capture-gating suite it took
    # today's card with it (run 39532208006, "7 findings still live, expected
    # 8").
    #
    # IT IS NOT MARKED `live` AND NOT DELETED. Both would hide a true finding.
    # The split below pins the twelve predicates that read only committed
    # source, and asserts 13 SEPARATELY as two-valued with its reason named --
    # so its state is still reported, and nobody re-pins it.
    #
    # And the disagreement is itself finding 13's own point, measured: the
    # pickles exist in CI only because a cache restored them.
    _ENV_DEPENDENT = {13}
    stable_live = {r["id"] for r in live} - _ENV_DEPENDENT
    assert stable_live == {4, 6, 7, 8, 10, 11, 12}, (
        f"the source-only live set is {sorted(stable_live)}, expected "
        f"[4, 6, 7, 8, 10, 11, 12]. If one was fixed, move it and say so; if "
        f"one regressed, that is the finding.")
    assert len(stable_live) == 7

    f13 = next(r for r in rows if r["id"] == 13)
    assert f13["is_live"] in (True, False), "finding 13 lost its third state"
    assert f13["file"] == "data/models/"
    # the predicate must stay a filesystem read, or this carve-out is stale
    import inspect
    pred = next(f["live"] for f in fr.FINDINGS if f["id"] == 13)
    assert "exists()" in inspect.getsource(pred), (
        "finding 13's predicate no longer reads the filesystem -- fold it back "
        "into the pinned set above and delete this carve-out")
    assert all(r["age_days"] == 74 for r in rows), (
        "the ages moved off 74 days -- every row comes from one dated source")
    text = fr.format_register(rows)
    assert "point mass" in text and "NO SPREAD" in text, (
        "the register no longer warns that its median summarises nothing")
    assert "REFERENCED, never rewritten" in text


def test_NOTHING_IS_LEFT_UNMARKED():
    """Rule 4. An unmarked entry is how 74 days happened."""
    rows = fr.evaluate()
    assert fr.untriaged(rows) == [], (
        f"untriaged live findings: "
        f"{[r['id'] for r in fr.untriaged(rows)]}")
    for r in rows:
        assert r["outcome"] in fr.OUTCOMES
        assert len(r["why"]) > 40, f"finding {r['id']} has no reason"
        if r["outcome"] == fr.DECLARE:
            assert r.get("retriage"), (
                f"finding {r['id']} is DECLARE with no re-triage date")
            dt.date.fromisoformat(r["retriage"])      # parses
        if r["outcome"] in (fr.FIX, fr.DELETE):
            assert r.get("size"), (
                f"finding {r['id']} is {r['outcome']} with no estimated size")


def test_COULD_NOT_LOOK_is_UNEVALUATED_and_NOT_fixed():
    """The defect in this register's own first draft.

    `_config_diverges` returned False on ImportError, which would have
    reported finding 9 as FIXED in the audit job -- where `yaml` is not
    installed, because that step runs stdlib-only. A predicate that answers
    "no defect" when it could not evaluate is the third-state failure this
    register exists to catch, one level up.
    """
    import builtins

    real = builtins.__import__

    def _no_yaml(name, *a, **k):
        if name == "yaml":
            raise ImportError("simulated: the audit step installs no packages")
        return real(name, *a, **k)

    builtins.__import__ = _no_yaml
    try:
        rows = {r["id"]: r for r in fr.evaluate()}
    finally:
        builtins.__import__ = real
    assert rows[9]["is_live"] is None, (
        "finding 9 reads as FIXED when yaml is unavailable -- 'could not "
        "look' is not 'no defect'")
    assert "ImportError" in rows[9]["eval_error"]


def test_the_REGISTER_imports_only_STDLIB_at_MODULE_SCOPE():
    """`ci_audit` reads it and that job installs no packages."""
    import ast
    import sys

    tree = ast.parse(pathlib.Path("scripts/finding_register.py").read_text(
        encoding="utf-8"))
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add((node.module or "").split(".")[0])
    third = imported - set(sys.stdlib_module_names) - {"", "__future__"}
    assert not third, f"{sorted(third)} imported at module scope"


# ── THE MECHANISM: REPORTED DAILY, ALARMS ONLY ON UNTRIAGED ────────────────

def test_the_AUDIT_REPORTS_the_count_and_does_NOT_alarm_on_a_triaged_backlog():
    """Same disposition as `pnc_scan`: a triaged backlog is a decision, not a
    finding. What alarms is an UNTRIAGED live finding."""
    import inspect

    import scripts.ci_audit as ci

    src = inspect.getsource(ci.main)
    assert "diagnosed_findings[" in src, "the audit no longer reports it"
    assert "finding_register" in src
    block = src[src.index("diagnosed_findings["):]
    assert "untriaged" in block
    # the alarm is conditioned on the UNTRIAGED set, never on the count
    alarm_line = [l for l in src.splitlines()
                  if "diagnosed finding(s) UNTRIAGED" in l]
    assert alarm_line, "there is no untriaged alarm at all"
    assert "_fr_bad" in src and "if _fr_bad:" in src, (
        "the alarm is not gated on the untriaged set")
    assert "if _fr_live:" not in src, (
        "the audit alarms on the live COUNT -- a triaged backlog would then "
        "alarm every day, which is how a signal becomes noise")


def test_CLR2_the_clearing_action_CHANGES_the_predicate():
    """A fix or a deletion flips `is_live` to False; a declaration with a date
    empties `untriaged`. Both are actions a person can take, and both change
    the predicate rather than being recorded beside it.
    """
    rows = fr.evaluate()
    live = [r for r in rows if r["is_live"] is True]

    # a DECLARE without its date would be untriaged -> the predicate fires
    stripped = [dict(r, retriage=None) if r["outcome"] == fr.DECLARE else r
                for r in rows]
    assert fr.untriaged(stripped), (
        "removing every re-triage date leaves the untriaged set empty, so the "
        "declaration half of the clearing action changes nothing")

    # and a fixed finding leaves the live set, which is the other half
    assert len(live) < len(rows), (
        "no finding has ever been fixed, so the predicate has never been "
        "observed to clear")
    assert any(r["is_live"] is False for r in rows)


def test_the_AUDIT_step_still_installs_NOTHING():
    """The register is read from the stdlib-only audit job."""
    import yaml as _yaml

    wf = _yaml.safe_load(pathlib.Path(".github/workflows/ci-audit.yml")
                         .read_text(encoding="utf-8"))
    steps = list(wf["jobs"].values())[0]["steps"]
    assert not any("pip install" in str(s.get("run", "")) for s in steps), (
        "the audit job now installs packages; the register's stdlib-only "
        "constraint was the reason it could be read there")
