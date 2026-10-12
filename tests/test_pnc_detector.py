"""The PNC-1 detector, with its controls — 2026-10-11.

ELEVEN VERIFIED producers have shipped without consumers in this project and
**every one was found by accident**. The ledger claimed FOURTEEN until the
tally audit of 2026-10-12 enumerated the register — see `docs/rule-tallies.md`. `scripts/pnc_scan.py` is the pass that did not
exist. These are its controls, because:

    A detector that cannot find the instances already known has a measured
    coverage, not a verdict.

MEASURED COVERAGE, 2026-10-12, against the **eleven verified**: **4/11 = 36%**,
and 4 of 4 of the subset a name-or-table pass could ever reach.

    IT ROSE FROM 23% BECAUSE THE DENOMINATOR WAS WRONG, NOT BECAUSE THE
    DETECTOR IMPROVED.

Four of the fourteen claimed were misclassified — two were the INVERSE shape
(a consumer with no producer), one was a RULE filed as an instance, one was a
claim about a single run mode read as a claim about the tree — and two were
unrecorded. The misses that remain are the taxonomy, not a failure:

    KIND N  a defined name nothing references          2 instances, both FOUND
    KIND T  a table written and never queried          2 instances, both FOUND
    KIND V  a value appended/written nobody reads      2 instances, OUT
    KIND F  a field/column no consumer reads           3 instances, OUT
    KIND -  not Python at all                          1 instance,  OUT
    (`first_seen_at` is a COLUMN: the table has readers, the column has none,
     so the schema pass does not reach it.)

THE SCAN REPORTS, IT DOES NOT ALARM. 48 findings stand on today's tree, which
is a backlog and not an incident — an alarm that fires on a pre-existing
backlog every day is `fixtures_zero_active` for the sixth time. The gate here
is a RATCHET: the counts below are pinned, so the backlog cannot grow silently,
and each new entry must be wired, deleted, or declared with a reason.
"""

import pathlib
import re
import subprocess
import sys

import pytest

import scripts.pnc_scan as pnc


# ── THE CONTROLS ─────────────────────────────────────────────────────────────

def _tree(tmp_path, monkeypatch, files: dict):
    """A throwaway tree, scanned in place of the repo."""
    for rel, body in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    monkeypatch.chdir(tmp_path)


def test_POSITIVE_CONTROL_an_uncalled_function_appears(tmp_path, monkeypatch):
    """Introduce one; it must be reported. Delete it; it must not be.

    Without this, `no_consumer == []` would pass against a scan that reported
    nothing whatever — the shape of three vacuous guards in this project.
    """
    _tree(tmp_path, monkeypatch, {
        "pkg/__init__.py": "",
        "pkg/m.py": (
            "def used():\n    return 1\n\n\n"
            "def nobody_calls_me():\n    return 2\n\n\n"
            "def main():\n    return used()\n\n\n"
            "if __name__ == '__main__':\n    main()\n"
        ),
    })
    found = {d.name for d, _ in pnc.scan(prod_dirs=("pkg",),
                                         test_dirs=()).no_consumer}
    assert "nobody_calls_me" in found, found
    assert "used" not in found, "a called function was reported"

    # ...and the control's other half: remove it and the finding goes away.
    (tmp_path / "pkg" / "m.py").write_text(
        "def used():\n    return 1\n\n\n"
        "def main():\n    return used()\n\n\n"
        "if __name__ == '__main__':\n    main()\n", encoding="utf-8")
    after = {d.name for d, _ in pnc.scan(prod_dirs=("pkg",),
                                         test_dirs=()).no_consumer}
    assert "nobody_calls_me" not in after


def test_NEGATIVE_CONTROL_no_declared_entry_point_is_ever_reported():
    """An entry point has a caller — a workflow, or a person.

    `python scripts/cohort_status.py` is a caller as surely as `python -m
    scripts.ci_audit` is, and the entry point whose only caller is a person is
    the one no test exercises. If an entry symbol showed up as a finding the
    scan would be telling everyone to delete its own callers.
    """
    s = pnc.scan()
    reported = {d.qual for d, _ in s.no_consumer} \
        | {d.qual for d, _ in s.test_only} \
        | {d.qual for d, _ in s.orphan_cluster}
    assert s.forms, "no invocation form was discovered at all"
    for path, forms in s.forms.items():
        assert forms, f"{path} has no invocation form"
        for entry in ("main", "_main_guarded", "cli"):
            assert entry not in reported, (
                f"{entry} reported as unconsumed although {path} is invoked "
                f"as {sorted(forms)}")


def test_the_invocation_forms_are_ENUMERATED_BY_FORM_not_by_file():
    """Both forms must be present and distinguishable.

    A file listed once as "an entry point" loses exactly the distinction that
    cost three CI reds and a broken commit gate: the two forms put different
    things on `sys.path[0]`.
    """
    forms = pnc.discover_invocation_forms()
    kinds = set()
    for f in forms.values():
        kinds |= f
    assert "module" in kinds, "no `python -m` caller discovered"
    assert "human" in kinds, "no human-invoked entry point discovered"
    assert forms.get("scripts/ci_audit.py") == {"module"}
    assert "human" in forms.get("scripts/cohort_status.py", set()), (
        "cohort_status is required by hand before every commit; if its only "
        "caller is no longer a person, say so here")


def test_POSITIVE_AND_NEGATIVE_CONTROL_on_the_orphan_cluster_pass(
        tmp_path, monkeypatch):
    """Two functions that call each other, with nothing calling in.

    This is the shape reference counting cannot see — H1's apparatus shipped
    for three stages with a non-zero reference count on every name in it.
    """
    pair = ("def a():\n    return b()\n\n\n"
            "def b():\n    return a()\n\n\n")
    _tree(tmp_path, monkeypatch, {
        "pkg/__init__.py": "",
        "pkg/m.py": pair + "def main():\n    return 0\n\n\n"
                           "if __name__ == '__main__':\n    main()\n",
    })
    orphans = {d.name for d, _ in pnc.scan(prod_dirs=("pkg",),
                                           test_dirs=()).orphan_cluster}
    assert {"a", "b"} <= orphans, orphans

    # THE NEGATIVE HALF: call in from the entry point and both must clear.
    (tmp_path / "pkg" / "m.py").write_text(
        pair + "def main():\n    return a()\n\n\n"
               "if __name__ == '__main__':\n    main()\n", encoding="utf-8")
    after = {d.name for d, _ in pnc.scan(prod_dirs=("pkg",),
                                         test_dirs=()).orphan_cluster}
    assert not ({"a", "b"} & after), after


def test_POSITIVE_AND_NEGATIVE_CONTROL_on_the_table_pass(tmp_path, monkeypatch):
    """KIND T: written by production, read by nothing in production.

    This is the shape that cost 47 days. `odds_snapshots` was written on every
    refresh from 2026-08-25 and the analysis that needed it read `odds`, which
    cannot hold a price path at all.
    """
    model = ("class Snap(Base):\n"
             "    __tablename__ = 'snaps'\n"
             "    id = Column(Integer)\n\n\n")
    _tree(tmp_path, monkeypatch, {
        "pkg/__init__.py": "",
        "pkg/models.py": model,
        "pkg/writer.py": ("from pkg.models import Snap\n\n\n"
                          "def w(session):\n    session.add(Snap())\n"),
    })
    flagged = {f.table for f in pnc.scan_tables(prod_dirs=("pkg",))}
    assert "snaps" in flagged, flagged

    # THE NEGATIVE HALF: add a reader and the table must clear. Raw SQL,
    # because that is how this project reads it.
    (tmp_path / "pkg" / "reader.py").write_text(
        "from sqlalchemy import text\n\n\n"
        "def r(session):\n"
        "    return session.execute(text('SELECT id FROM snaps')).fetchall()\n",
        encoding="utf-8")
    after = {f.table for f in pnc.scan_tables(prod_dirs=("pkg",))}
    assert "snaps" not in after, after


def test_the_SQL_read_pattern_actually_MATCHES(tmp_path, monkeypatch):
    """REGRESSION. The word boundary was a BACKSPACE.

    Written through a non-raw Python string, `\\b` is chr(8). The pattern began
    with a literal backspace byte and matched nothing, so the table pass
    reported `odds_snapshots` as unread while `h1_collection_check.py:248`
    queries it — a vacuous filter returning a confident finding, caught only
    because that one answer was already known to be wrong.
    """
    assert "\x08" not in pnc._SQL_READ.pattern, (
        "a control character is back in the SQL pattern")
    assert pnc._SQL_READ.findall("FROM odds_snapshots o") == ["odds_snapshots"]
    assert pnc._SQL_READ.findall("JOIN matches m ON 1=1") == ["matches"]


def test_a_DATA_TABLE_naming_a_symbol_is_NOT_a_consumer():
    """REGRESSION, and the detector did this to itself.

    The first version counted every identifier-like string as a reference, on
    the reasoning that `getattr(x, "foo")` is a real consumer. `pnc_scan.py`'s
    own coverage table names each known instance as a string, so the detector
    suppressed exactly the instances it recorded — the first historical run
    reported MISS on two it had just been built to find.

        A detector whose own records are indistinguishable from the evidence
        it reads cannot measure itself.

    A string is a consumer only inside a reflection call, where it IS the call.
    """
    names = {i.get("symbol") for i in pnc.KNOWN_INSTANCES if i.get("symbol")}
    assert names, "the coverage table is empty"
    src = pathlib.Path("scripts/pnc_scan.py").read_text(encoding="utf-8")
    assert "visit_Constant" not in src, (
        "the blanket string-as-reference visitor is back; it makes this file's "
        "own KNOWN_INSTANCES table suppress its findings")
    assert "getattr" in src and "_callee_name(node.func) in (" in src, (
        "reflection calls are no longer recognised, so `getattr(x, 'foo')` "
        "now reads as a missing consumer")


# ── COVERAGE, AND THE MISSES THAT ARE THE POINT ──────────────────────────────

def test_the_COVERAGE_against_the_known_instances_is_MEASURED_AND_LOW():
    """3/13. Stated, not rounded up, and pinned so it cannot drift upward
    without someone measuring again.

    A hit is measured at the commit where the instance was LIVE. The first
    version of this scored every row against today's tree, where a fixed
    instance is by definition absent, and so reported `format_series_census`
    as a MISS one day after finding it.
    """
    rows = pnc.coverage_rows(set())
    assert len(rows) == 15, f"the register size moved: {len(rows)}"
    verified = [r for r in rows if r.get("verdict") != "MISCLASSIFIED"]
    assert len(verified) == 11, (
        f"the VERIFIED count moved to {len(verified)}. If an instance was "
        f"added or re-classified, update docs/rule-tallies.md in the same "
        f"commit - an ordinal with no enumeration is not a count.")
    hits = [r for r in verified if r["hit"]]
    assert len(hits) == 4, [r["id"] for r in hits]
    for r in hits:
        assert r.get("provenance", "").startswith("measured at "), r["id"]
    # every miss carries its REASON - that is the deliverable
    for r in rows:
        if not r["hit"]:
            assert r.get("why"), f"{r['id']} is a miss with no explanation"


def test_the_FOUR_misclassified_entries_stay_ON_FILE():
    """A register that silently drops its errors is the accretion problem with
    a tidier face. The claimed count and the corrected count must be readable
    side by side, which is why these stay in `KNOWN_INSTANCES` with a verdict
    instead of being deleted.

    Two were the INVERSE shape, one was a RULE filed as an instance, one was a
    per-run-mode claim read as a tree-wide one. All four corrected 2026-10-12.
    """
    by_id = {i["id"]: i for i in pnc.KNOWN_INSTANCES}
    wrong = ("report-discovered-margins", "scrape-league-fixtures",
             "lookup-table-not-consulted", "five-terminal-states")
    for ident in wrong:
        assert by_id[ident]["verdict"] == "MISCLASSIFIED", ident
        assert by_id[ident]["findable"] is False, ident
        assert len(by_id[ident]["why"]) > 60, ident
    assert sum(1 for i in pnc.KNOWN_INSTANCES
               if i.get("verdict") == "MISCLASSIFIED") == len(wrong)


def test_the_TALLY_REGISTER_exists_and_carries_both_counts():
    """Hard rule: a count that cannot be enumerated is not cited as a count.

    The register is the mechanism that makes that enforceable — without it,
    "the fourteenth instance" is a sentence, and four of the fourteen were
    wrong.
    """
    doc = pathlib.Path("docs/rule-tallies.md").read_text(encoding="utf-8")
    assert "claimed" in doc.lower() and "corrected" in doc.lower()
    assert "**11 verified**" in doc, "the corrected PNC-1 count is not stated"
    for rule in ("third-state family", "THE HABIT", "establish the clock",
                 "canonicalisation is not symmetric-safe"):
        assert rule in doc, f"{rule} is not audited in the register"


# ── CLR-2: WHAT CLEARS A FINDING ─────────────────────────────────────────────

def test_every_DECLARED_INTENTIONAL_entry_carries_a_REASON():
    """A declared exemption with no reason is a silence with extra steps."""
    assert pnc.DECLARED_INTENTIONAL, "the registry is empty"
    for key, reason in pnc.DECLARED_INTENTIONAL.items():
        assert ":" in key, f"{key} is not path:name"
        assert len(reason) > 40, f"{key} has no real reason: {reason!r}"


def test_DECLARING_a_finding_CHANGES_THE_PREDICATE(tmp_path, monkeypatch):
    """CLR-2. The clearing action must actually clear it.

    Three actions clear a finding and all three are checked somewhere: wiring
    a consumer (the orphan control's negative half), deleting the producer
    (the positive control's second half), and declaring it — here.
    """
    _tree(tmp_path, monkeypatch, {
        "pkg/__init__.py": "",
        "pkg/m.py": "WIDGET_STATES = ('a', 'b')\n\n\ndef main():\n    return 0\n",
    })
    before = {d.name for d, _ in pnc.scan(prod_dirs=("pkg",),
                                          test_dirs=()).no_consumer}
    assert "WIDGET_STATES" in before

    monkeypatch.setitem(pnc.DECLARED_INTENTIONAL, "pkg/m.py:WIDGET_STATES",
                        "a declared state set, enumerated so the set is read "
                        "rather than reconstructed from the code")
    s = pnc.scan(prod_dirs=("pkg",), test_dirs=())
    assert "WIDGET_STATES" not in {d.name for d, _ in s.no_consumer}
    # ...and it leaves carrying its reason, visible to a reader.
    reasons = [r for d, r in s.excluded if d.name == "WIDGET_STATES"]
    assert reasons and reasons[0].startswith("DECLARED INTENTIONAL: ")


def test_the_DEFERRED_set_is_H1s_and_carries_its_DATE():
    """Acknowledged, not exempt. H1 is mid-collection until 2026-10-17 and
    wiring or deleting any of these changes the apparatus under a running
    experiment.

    SUCCESSOR CONTRACT: after that date this set must be re-triaged into
    wire/delete/declare. It is not allowed to become permanent by inertia.
    """
    assert pnc.DEFERRED_UNTIL == "2026-10-17"
    assert len(pnc.DEFERRED) == 4, sorted(pnc.DEFERRED)
    for key, reason in pnc.DEFERRED.items():
        assert len(reason) > 40, f"{key} is deferred with no reason"
    s = pnc.scan()
    assert {d.name for d, _ in s.deferred} == {
        "allocate", "release_requests", "stamp_first_seen", "exhausted"}


# ── THE RATCHET, WHICH IS NOT AN ALARM ───────────────────────────────────────

#: MEASURED 2026-10-11 on today's tree. Pinned so the backlog cannot GROW
#: silently; deliberately NOT an alarm, because a check that fires every day on
#: a pre-existing backlog is the `fixtures_zero_active` shape for the sixth
#: time. Lower these as findings are wired or deleted.
BACKLOG = {"no_consumer": 33, "orphan_cluster": 14, "tables": 0}
#: `tables` fell 1 -> 0 on 2026-10-12, and NOT by a fix: `injury_observations`
#: moved into AWAITING_CONSUMER with a named consumer and a re-triage date.
#: The floor drops because the declaration is the clearing action, which is
#: CLR-2 working - and the substrate is still reported, in its own section.


def test_the_backlog_does_not_GROW():
    """The gate. A new producer with no consumer fails this and nothing else.

    CLEARING ACTION, in the failure text because an alarm names an action a
    person can take.
    """
    s = pnc.scan()
    actual = {"no_consumer": len(s.no_consumer),
              "orphan_cluster": len(s.orphan_cluster),
              "tables": len(pnc.scan_tables())}
    for bucket, pinned in BACKLOG.items():
        assert actual[bucket] <= pinned, (
            f"PNC-1: {bucket} grew from {pinned} to {actual[bucket]}. A "
            f"producer was added with no consumer. CLEARING ACTION, pick one: "
            f"wire a consumer, delete the producer, or add it to "
            f"DECLARED_INTENTIONAL in scripts/pnc_scan.py with a reason. "
            f"Run `python -m scripts.pnc_scan --list` to see which. "
            f"If you wired or deleted something, lower BACKLOG here.")
    for bucket, pinned in BACKLOG.items():
        if actual[bucket] < pinned:
            pytest.fail(
                f"GOOD NEWS, AND THE PIN IS STALE: {bucket} fell from "
                f"{pinned} to {actual[bucket]}. Lower BACKLOG so the ratchet "
                f"holds the new floor instead of the old one.")


def test_the_scan_REPORTS_and_does_not_ALARM():
    """Exit 0 with 48 findings on the tree.

    The decision, stated rather than inferred: this reports. It becomes a gate
    only through the ratchet above, which fires on GROWTH and not on the
    backlog. An alarm wired before the backlog was counted would have gone red
    on its first run and every run after.
    """
    r = subprocess.run([sys.executable, "-m", "scripts.pnc_scan"],
                       capture_output=True, encoding="utf-8",
                       errors="replace", timeout=180)
    assert r.returncode == 0, r.stderr[-800:]
    assert "NO CONSUMER" in r.stdout
    assert "::error::" not in r.stdout, (
        "the scan emits an annotation; it was decided to report, and the "
        "backlog has not been triaged to zero")


def test_the_scan_runs_in_BOTH_invocation_forms():
    """Its own lesson, applied to itself. Four entry points broke in the
    script form in two days, and the one whose only caller is a person is the
    one no test exercises — so this file is a person's tool and must work the
    way a person types it."""
    for args in (["-m", "scripts.pnc_scan"], ["scripts/pnc_scan.py"]):
        r = subprocess.run([sys.executable, *args, "--coverage"],
                           capture_output=True, encoding="utf-8",
                           errors="replace", timeout=180)
        assert r.returncode == 0, f"{args}: {r.stderr[-600:]}"
        assert re.search(r"COVERAGE\s+: 4/11", r.stdout), args


# ── THE DECLARED-BUT-UNREAD SUBSTRATES ──────────────────────────────────────

def test_every_AWAITING_CONSUMER_entry_names_a_CONSUMER_and_a_DATE():
    """An exemption is recorded with a named consumer and a re-triage date,
    never as a silence.

    Both entries are Stage 18 substrates that shipped unread. Deleting them
    would be wrong — each was built because a cheaper proxy was MEASURED and
    found insufficient (`created_at` is a backfill stamp for 91.8% of rows at
    mean +638 days; a current-status snapshot cannot date an absence). So the
    true state is recorded rather than resolved.
    """
    assert pnc.AWAITING_CONSUMER, "the registry is empty"
    for key, meta in pnc.AWAITING_CONSUMER.items():
        for field in ("rows", "question", "consumer", "build", "cohort",
                      "retriage", "basis"):
            assert meta.get(field), f"{key} has no {field}"
        assert re.fullmatch(r"20\d\d-\d\d-\d\d", meta["retriage"]), key
        assert meta["retriage"] > "2026-10-12", f"{key} re-triages in the past"
        assert "measured" in meta["rows"], (
            f"{key}'s row count carries no provenance")


def test_DECLARING_a_substrate_CHANGES_THE_PREDICATE_and_still_REPORTS_it():
    """CLR-2, on the table pass.

    `injury_observations` was a KIND T finding on 2026-10-11. The clearing
    action was to declare it, and declaring it removed it from the finding
    list — otherwise the declaration would be advice rather than an action.
    It is still printed, in its own section, with its date.
    """
    assert "__table:injury_observations__" in pnc.AWAITING_CONSUMER
    flagged = {f.table for f in pnc.scan_tables()}
    assert "injury_observations" not in flagged, (
        "a declared substrate is still reported as an open finding")
    assert "odds_snapshots" not in flagged, "the wired substrate regressed"
    rows = dict(pnc.awaiting_consumer_rows())
    assert len(rows) == 2


def test_the_UNREAD_COLUMN_is_recorded_even_though_NO_PASS_FINDS_IT():
    """`odds.first_seen_at` is the sharpest version of the taxonomy point.

    The TABLE has readers and the COLUMN does not, so the schema pass cannot
    reach it — a producer can be invisible to every detector here and still be
    real. Recording it is the only thing that keeps it from being lost.
    """
    meta = pnc.AWAITING_CONSUMER["__column:odds.first_seen_at__"]
    assert "H4" in meta["question"] or "H4" in meta["consumer"]
    assert "25.2%" in meta["rows"]
    by_id = {i["id"]: i for i in pnc.KNOWN_INSTANCES}
    assert by_id["first-seen-at-unread"]["findable"] is False, (
        "if a pass now finds the column, say which and update the coverage")


def test_the_stamp_first_seen_docstring_no_longer_claims_to_be_the_ONLY_site():
    """The lie, fixed. Two scrapers set the column inline and the function has
    no production caller, so the write-once guarantee was being asserted by a
    definition no execution passed through.

    The guarantee itself HOLDS — the column is absent from the upsert's update
    set — and the docstring now says that instead.
    """
    from src.data.price_history import stamp_first_seen

    doc = stamp_first_seen.__doc__ or ""
    assert "NOT** THE ONLY SITE" in doc or "NOT THE ONLY SITE" in doc, (
        "the docstring no longer records that it is not the only writer")
    assert "apifootball_scraper.py" in doc and "theodds_scraper.py" in doc, (
        "the real writers are not named")
    assert "on_conflict_do_update" in doc, (
        "the actual reason write-once holds is not stated")
    # and the reason is TRUE: the column must stay out of the update set
    scraper = pathlib.Path("src/scrapers/apifootball_scraper.py").read_text(
        encoding="utf-8")
    block = scraper.split("on_conflict_do_update", 1)[1][:600]
    assert "first_seen_at" not in block, (
        "first_seen_at entered the ON CONFLICT update set - write-once is "
        "broken and first-seen has become last-seen")
