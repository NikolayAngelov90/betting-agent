"""PNC-1, mechanically. A producer with no consumer, found by a pass not by luck.

    PNC-1. A value that is produced and never consumed presents as a working
    feature on every surface except the one the consumer was for.

ELEVEN VERIFIED instances have shipped in this project, and **every one was
found by accident** — three of them in the week of 2026-10-05, by someone
reading nearby code for an unrelated reason. This is the detector that did not
exist.

ELEVEN, NOT FOURTEEN. The ledger claimed fourteen on 2026-10-11. The tally
audit of 2026-10-12 enumerated the register and found FOUR misclassified — two
were the inverse shape (a consumer with no producer), one was a RULE filed as
an instance, one was a claim about a single run mode read as a claim about the
tree — and TWO unrecorded. A count maintained by accretion drifts upward
because each addition is cheap and none is checked.

WHAT IT IS NOT. It is not a verdict. An AST pass over names cannot see most of
PNC-1's shapes, and saying so is the point of `--coverage`: the measured
coverage against the eleven verified instances is the honest output, and the TAXONOMY OF
MISSES is more useful than the hit rate. Four kinds exist and only one is a
name:

    KIND N  a defined name nothing references            <- this scan finds it
    KIND V  a value appended or written to a structure
            nobody reads (`alarmed.append` with no
            `if alarmed`)                                <- invisible here
    KIND T  a database table written and never queried
            (`odds_snapshots`, 47 days)                  <- not a Python name
    KIND F  a field on a record the consumer never
            reads (`remaining` vs `updated`)             <- needs flow analysis

THE SIGNATURE IS NOT SILENCE, IT IS A GREEN TEST. All three of this week's
instances had passing tests on the producer. `format_series_census` had a test
asserting the attribution rule was "stated in the output" — of a string no
caller ever printed. So `TEST-ONLY CONSUMER` is a BUCKET, never an exclusion:

    A test on a formatter's output proves the formatter, not that anything is
    formatted.

EXCLUSIONS ARE DECLARED, NEVER INFERRED. Every excluded name is reported with
the reason and, for an entry point, WITH ITS INVOCATION FORM — because
`python scripts/x.py` is a caller too, and the entry point whose only caller
is a person is the one no test exercises. `cohort_status.py` was broken in the
human form for weeks while six machine-invoked entry points were guarded.

    python -m scripts.pnc_scan              # today's three bucket counts
    python -m scripts.pnc_scan --list       # every finding, with its sites
    python -m scripts.pnc_scan --coverage   # against the known instances
    python -m scripts.pnc_scan --json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import pathlib
import re
import sys
from typing import Dict, List, NamedTuple, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Scanned for definitions AND for consumers.
PROD_DIRS = ("src", "scripts")
#: Scanned for consumers only. A reference from here is a TEST-ONLY consumer,
#: which is PNC-1's signature rather than an exemption.
TEST_DIRS = ("tests",)
#: Where an invocation form is DISCOVERED rather than declared.
WORKFLOW_DIR = ".github/workflows"

# ── EXCLUSIONS, EACH WITH A REASON ───────────────────────────────────────────
#
# Rule: an exclusion is declared with a reason, never inferred. Anything whose
# only reference is a test is NOT here — that is the finding.

#: Entry points invoked only by a PERSON. No workflow runs them, so no
#: invocation form is discoverable from the tree and the list must be declared.
#: This is the category that hid a `ModuleNotFoundError` for weeks: the commit
#: rules require `cohort_status.py` by hand before every commit, and nothing
#: machine-invoked exercised it.
HUMAN_INVOKED: Dict[str, str] = {
    "scripts/cohort_status.py":
        "required BY HAND before every commit (commit rules); no workflow "
        "invokes it, so its only caller is a person",
    "scripts/h1_collection_check.py":
        "the daily H1 check, read by a person during the 10-09..10-17 window",
    "scripts/pnc_scan.py":
        "this file",
}

#: CLR-2. "Declared intentional" must be an ACTION, not a silence — so it is a
#: registry entry keyed by `path:name` with a reason, and adding one moves the
#: finding from a bucket into `excluded` WITH that reason attached. The three
#: clearing actions all change the predicate:
#:
#:     wire a consumer   -> the name gains a production reference
#:     delete it         -> there is no definition left to count
#:     declare it HERE   -> it leaves the bucket carrying a reason a reader sees
#:
#: An empty reason is not accepted; the test suite asserts every entry has one.
DECLARED_INTENTIONAL: Dict[str, str] = {
    "src/data/h1_window.py:RETIRED":
        "IT IS CONSUMED, and this entry records a SCAN LIMITATION rather than "
        "an intention. `odds_quota` reads it as `_h1_window.RETIRED` through "
        "a module alias and re-publishes it as "
        "`RETIRED_H1_RESERVATION_WINDOW`, which is verified at runtime and by "
        "`test_the_RETIRED_WINDOW_IS_HERE_AND_INERT`. The reference-collector "
        "does not resolve an attribute read through an aliased module import, "
        "so this reads as an orphan. Declared with the limitation named "
        "rather than deleted -- a figure measured inside the October window "
        "is not comparable with one measured inside November's, and the first "
        "attempt's n=4 belongs to those dates.",
    "scripts/h1_collection_check.py:COLLECTION_STATES":
        "a DECLARED state set. Its consumer is the test that enumerates the "
        "states, deliberately: `fixtures_zero_active` happened four times "
        "because a state set was inferred from code instead of written down.",
    "scripts/h1_collection_check.py:APPARATUS_STATES":
        "same: the apparatus states are enumerated so the alarm surface can "
        "be read rather than reconstructed (tests/test_alarm_surface.py).",
    "scripts/h1_collection_check.py:COLLECTION_TERMINAL_STATES":
        "the terminal subset of the above; four reachable states plus one "
        "dead branch were once claimed as five, which is why the subset is "
        "named separately and pinned.",
    "src/scrapers/flashscore_scraper.py:DISCOVERY_STATES":
        "the discovery instrumentation's state set, same contract.",
    "scripts/calendar_marker_scan.py:nodeid":
        "its consumer is the pytest_collection_modifyitems HOOK in "
        "tests/conftest.py, which runs on EVERY collection - production for "
        "the test harness, not a test. The scan classifies anything under "
        "tests/ as a test consumer, which is right for an assertion and "
        "wrong for a hook.",
    "scripts/h1_precondition.py:PRECONDITION_STATES":
        "a DECLARED state set, same contract as COLLECTION_STATES and "
        "DISCOVERY_STATES. Its consumer is the test that enumerates the four "
        "states, deliberately: H1 spent three days reporting `nothing to do` "
        "for three different causes, and naming the states is what separated "
        "NO_CARD from NO_FIXTURE_IN_BAND from UNEVALUATED.",
    "scripts/schedule_deadline.py:OBSERVED_EARLIEST_LOCAL_MIN":
        "a MEASURED anchor kept beside DEADLINE_PERCENTILES so a fallback "
        "can never drift laxer than the figure that justified the cron. Its "
        "consumer is the test that ties the two together - which is the "
        "point, because a drift is what the test exists to catch. LOCALISED "
        "2026-10-22: it is a LOCAL minute now, converted per day, because a "
        "stored UTC minute answers with a summer figure all winter.",
    "scripts/schedule_deadline.py:RETIRED_OBSERVED_EARLIEST_UTC_MIN":
        "the retired UTC pair, kept so the correction can be checked. A "
        "localisation that deletes what it localised cannot be audited, and "
        "both of these were summer figures - which IS the defect.",
    "scripts/ci_audit.py:UNAUDITED_WORKFLOWS":
        "an intentionally EMPTY registry with a successor contract: a "
        "workflow added to WORKFLOWS with no ledger rows is recorded here "
        "rather than reading as audited. Its consumer is the registration "
        "guard; it gains a production consumer the day it gains an entry.",
}

#: DEFERRED, with a date. Not an exemption: an acknowledged finding that must
#: not be touched yet. H1 is mid-collection and its window closes 2026-10-17;
#: wiring or deleting any of these now would change the apparatus under a
#: running experiment. They are REPORTED, in their own bucket, with the date.
DEFERRED_UNTIL = "2026-10-17"
DEFERRED: Dict[str, str] = {
    "src/data/odds_quota.py:allocate":
        "H1's credit allocator. Test-only since it shipped; the live path "
        "reads the reservation through OddsApiQuota instead. Wiring or "
        "removing it mid-collection changes the budget under the experiment.",
    "src/data/odds_quota.py:release_requests":
        "the compensating half of claim_requests. Same window.",
    "src/data/price_history.py:stamp_first_seen":
        "the opening_odds stamper. Reaching into the price path during H1's "
        "collection is exactly what must not happen.",
    "src/scrapers/theodds_scraper.py:exhausted":
        "the Refusal terminal predicate, on H1's provider path.",
}

#: AWAITING A CONSUMER. Not an exemption and not a silence: a substrate built
#: for a REGISTERED QUESTION that has not been tested yet. Deletion would be
#: the wrong call - both were built because a cheaper proxy was measured and
#: found insufficient - and "unread" is the true state, so it is recorded with
#: the question, what would have to be built, and the date the decision is
#: re-examined. Same shape as DEFERRED, with a per-entry date.
#:
#: THE SHAPE THIS REPLACES: three Stage 18 substrates shipped unread, and the
#: ledger tracked their ROW COUNTS for weeks and scored them CLEAN. A growth
#: check measures the writer. "The table is filling" and "the table is useful"
#: are different questions and only the first was ever asked.
AWAITING_CONSUMER: Dict[str, dict] = {
    "__table:injury_observations__": dict(
        rows="5,136 rows / 211 teams / 2026-08-27..10-11, measured 2026-10-12",
        question="does the AGE of an injury move a price? A current-status "
                 "snapshot cannot distinguish a two-week-old absence from "
                 "this morning's announcement, which is Stage 18 C3's stated "
                 "justification for recording observed_at at both write "
                 "sites including the UPDATE branch.",
        consumer="an injury-recency feature, or a research pass equivalent to "
                 "h1_collection_check: group observations by (team, player) "
                 "and measure price movement against the age of the status "
                 "change. Neither exists.",
        build="a reader over (team_id, player_id, status, observed_at) joined "
              "to odds_snapshots on match date; no schema change needed",
        cohort="a FEATURE would be a cohort event - Stage 14 established "
               "injuries reach only the reasoning string and never a fit, and "
               "that still holds (verified 2026-10-12). A research pass would "
               "not be.",
        retriage="2026-11-23",
        basis="six weeks from this declaration - the first point at which "
              "H1's analysis (window closes 2026-10-17) will have been "
              "written up. A REVIEW date, not a build commitment.",
    ),
    "__column:odds.first_seen_at__": dict(
        rows="123,043 of 488,312 odds rows (25.2%), measured 2026-10-12",
        question="how long before kickoff was this price taken? That is H4, "
                 "and it is UNTESTED. Stage 18 C2 measured that created_at "
                 "cannot proxy for it: 91.8% of rows created after kickoff, "
                 "mean +638 days.",
        consumer="H4. No analysis reads the column; `stamp_first_seen`, its "
                 "designated writer, has no production caller either - the "
                 "two scrapers set the column inline.",
        build="a lead-time distribution over (first_seen_at, match_date), "
              "then the same against pick outcomes. No schema change needed; "
              "coverage is 25.2% and the shortfall is itself a finding.",
        cohort="reading it for analysis is not a cohort event; using lead "
               "time as a FEATURE would be.",
        retriage="2026-11-23",
        basis="same review point. The structural choice for "
              "`stamp_first_seen` - wire it and drop the inline writes, or "
              "delete it - belongs to the 2026-10-17 H1 re-triage, because "
              "either one touches the odds write path.",
    ),
}

#: Names that a framework calls, so the absence of a reference is expected.
FRAMEWORK_REASONS: Dict[str, str] = {
    "main": "process entry; called from the `__main__` guard or by `-m`",
    "setup": "unittest/pytest hook",
    "teardown": "unittest/pytest hook",
    "pytest_configure": "pytest hook",
    "pytest_collection_modifyitems": "pytest hook",
    "pytest_addoption": "pytest hook",
}

#: Attribute values the ORM consumes by reflection, never by a Python read.
ORM_FACTORIES = ("Column", "relationship", "declarative_base", "Table",
                 "mapped_column", "Index", "UniqueConstraint")

#: Decorators that hand the function to a framework. The framework is the
#: consumer, and it calls by registration rather than by name — so a reference
#: can never exist and its absence proves nothing. MEASURED 2026-10-11: these
#: three accounted for 4 of the scan's first 39 findings.
DISPATCH_DECORATORS: Dict[str, str] = {
    "validates": "SQLAlchemy @validates listener; the ORM calls it on write",
    "listens_for": "SQLAlchemy @event.listens_for; the engine calls it",
    "event.listens_for": "SQLAlchemy engine event",
    "fixture": "pytest fixture; resolved by parameter name",
    "hookimpl": "pluggy hook",
    "setter": "property setter; reached by assignment, not by a call",
    "deleter": "property deleter",
}

#: A base class whose subclass methods are called by DISPATCH on the node type.
VISITOR_BASES = ("NodeVisitor", "NodeTransformer", "HTMLParser")


class Definition(NamedTuple):
    """A thing that could have a consumer."""
    name: str
    qual: str                 # Class.method, or the bare name
    path: str
    lineno: int
    end_lineno: int
    kind: str                 # function | method | class | constant
    decorators: Tuple[str, ...] = ()
    bases: Tuple[str, ...] = ()

    @property
    def site(self) -> str:
        return f"{self.path}:{self.lineno}"


class Reference(NamedTuple):
    path: str
    lineno: int
    name: str


def _py_files(dirs) -> List[pathlib.Path]:
    out = []
    for d in dirs:
        root = pathlib.Path(d)
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            out.append(p)
    return out


def _as_posix(p: pathlib.Path) -> str:
    return p.as_posix()


# ── INVOCATION FORMS, DISCOVERED ─────────────────────────────────────────────

def discover_invocation_forms() -> Dict[str, Set[str]]:
    """`{path: {"module", "script", "human"}}`, MEASURED from the workflows.

    ENUMERATED BY FORM, NOT BY FILE. The two forms put different things on
    `sys.path[0]`, which is a real difference in behaviour and not a style
    choice — it cost three CI reds and a broken commit gate in one week. A file
    listed once as "an entry point" loses exactly the distinction that matters.
    """
    forms: Dict[str, Set[str]] = {}
    wf = pathlib.Path(WORKFLOW_DIR)
    text = ""
    if wf.exists():
        for y in sorted(wf.glob("*.yml")) + sorted(wf.glob("*.yaml")):
            text += y.read_text(encoding="utf-8") + "\n"
    for mod in re.findall(r"python\s+-m\s+([A-Za-z_][\w.]*)", text):
        path = mod.replace(".", "/") + ".py"
        # `python -m pytest` / `-m pip` name INSTALLED packages. Only a path
        # that exists in this tree is a producer this scan can own.
        if pathlib.Path(path).exists():
            forms.setdefault(path, set()).add("module")
    for scr in re.findall(r"python\s+((?:src|scripts)/[\w/]+\.py)", text):
        if pathlib.Path(scr).exists():
            forms.setdefault(scr, set()).add("script")
    # A DECLARED entry always applies, because it carries a REASON the tree
    # cannot supply - "read by a person during the window" is not derivable.
    for path in HUMAN_INVOKED:
        forms.setdefault(path, set()).add("human")
    # DERIVED, NOT DECLARED. A file with a `__main__` guard that no workflow
    # invokes is, by construction, invoked only by a person. Listing them by
    # hand is how `cohort_status.py` stayed outside the entry-point guard while
    # six machine-invoked siblings were covered.
    for p in _py_files(PROD_DIRS):
        path = _as_posix(p)
        try:
            txt = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "__main__" in txt and not (forms.get(path, set()) - {"human"}):
            forms.setdefault(path, set()).add("human")
    return forms


# ── THE PASS ─────────────────────────────────────────────────────────────────

class _Collector(ast.NodeVisitor):
    """Definitions and references in one file, in a single walk."""

    def __init__(self, path: str, src: str):
        self.path = path
        self.defs: List[Definition] = []
        self.refs: List[Reference] = []
        self.dunder_all: Set[str] = set()
        self.orm_attrs: Set[str] = set()
        self._class_stack: List[str] = []
        self._base_stack: List[List[str]] = []
        self._fn_depth = 0
        self.tree = ast.parse(src)

    # -- definitions --
    def _record_fn(self, node):
        kind = "method" if self._class_stack else "function"
        if self._fn_depth:
            kind = "nested"
        qual = ".".join(self._class_stack + [node.name])
        self.defs.append(Definition(
            node.name, qual, self.path, node.lineno,
            getattr(node, "end_lineno", node.lineno), kind,
            _decorator_names(node),
            tuple(self._base_stack[-1]) if self._base_stack else ()))
        self._fn_depth += 1
        self.generic_visit(node)
        self._fn_depth -= 1

    visit_FunctionDef = _record_fn
    visit_AsyncFunctionDef = _record_fn

    def visit_ClassDef(self, node):
        self.defs.append(Definition(
            node.name, ".".join(self._class_stack + [node.name]), self.path,
            node.lineno, getattr(node, "end_lineno", node.lineno), "class"))
        self._class_stack.append(node.name)
        self._base_stack.append([_callee_name(b) for b in node.bases])
        self.generic_visit(node)
        self._base_stack.pop()
        self._class_stack.pop()

    def _record_assign(self, targets, node, value):
        # Module level and class level only: a local is consumed or not within
        # one function, which is the compiler's business and not PNC-1's.
        if self._fn_depth:
            return
        orm = (isinstance(value, ast.Call)
               and _callee_name(value.func) in ORM_FACTORIES)
        for t in targets:
            if isinstance(t, ast.Name):
                if t.id == "__all__":
                    self.dunder_all |= _string_set(value)
                    continue
                if orm:
                    self.orm_attrs.add(t.id)
                self.defs.append(Definition(
                    t.id, ".".join(self._class_stack + [t.id]), self.path,
                    node.lineno, getattr(node, "end_lineno", node.lineno),
                    "constant", (),
                    tuple(self._base_stack[-1])
                    if self._base_stack else ()))

    def visit_Assign(self, node):
        self._record_assign(node.targets, node, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self._record_assign([node.target], node, node.value)
        self.generic_visit(node)

    # -- references --
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.refs.append(Reference(self.path, node.lineno, node.id))
        self.generic_visit(node)

    def visit_Attribute(self, node):
        # `.attr`, unqualified. Coarse ON PURPOSE: resolving the receiver would
        # need type inference, and a missed consumer is a FALSE POSITIVE, which
        # is the failure mode that makes a detector get ignored.
        self.refs.append(Reference(self.path, node.lineno, node.attr))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for a in node.names:
            self.refs.append(Reference(self.path, node.lineno, a.name))
        self.generic_visit(node)

    def visit_Call(self, node):
        # A STRING IS A CONSUMER ONLY IN A REFLECTION CALL.
        #
        # This started as "any identifier-like string is a reference", on the
        # reasoning that `getattr(x, "foo")` is a real consumer. MEASURED
        # 2026-10-11: that made THIS FILE suppress its own findings. The
        # coverage table below names each known instance as a string
        # (`symbol="format_series_census"`), so every instance the detector
        # recorded looked consumed — and the first historical run reported
        # MISS on two instances it had just been built to find.
        #
        #     A detector whose own records are indistinguishable from the
        #     evidence it reads cannot measure itself.
        #
        # The same shape would hit any data table that names a function: the
        # audit's own regex dictionaries, a config of handler names, a ledger
        # constant. So the rule is narrowed to the reflection builtins, where
        # the string genuinely IS the call.
        if _callee_name(node.func) in ("getattr", "setattr", "hasattr",
                                       "delattr", "patch", "object",
                                       "setitem"):
            for a in node.args:
                if (isinstance(a, ast.Constant) and isinstance(a.value, str)
                        and a.value.isidentifier()):
                    self.refs.append(
                        Reference(self.path, a.lineno, a.value))
        self.generic_visit(node)


def _decorator_names(node) -> Tuple[str, ...]:
    out = []
    for d in getattr(node, "decorator_list", ()):
        target = d.func if isinstance(d, ast.Call) else d
        out.append(_callee_name(target))
    return tuple(n for n in out if n)


def _callee_name(fn) -> str:
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return ""


def _string_set(value) -> Set[str]:
    out = set()
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        for e in value.elts:
            if isinstance(e, ast.Constant) and isinstance(e.value, str):
                out.add(e.value)
    return out


class Scan(NamedTuple):
    no_consumer: List[Tuple[Definition, str]]
    test_only: List[Tuple[Definition, List[str]]]
    orphan_cluster: List[Tuple[Definition, str]]
    deferred: List[Tuple[Definition, str]]
    excluded: List[Tuple[Definition, str]]
    consumed: int
    forms: Dict[str, Set[str]]


def _reachable(defs: List[Definition], refs: List[Reference],
               roots: Set[str]) -> Set[str]:
    """Names reachable from an entry point, by BFS over the call graph.

    WHY THIS EXISTS AND REFERENCE COUNTING DOES NOT SUFFICE. H1's apparatus
    shipped across three stages with nothing calling it, and the functions
    inside it called EACH OTHER — so `h1_reservation` had a non-zero reference
    count the whole time. MEASURED at `0ed4cad`: the count-based pass finds
    `allocate` and misses the chain below it.

        A cluster of producers that reference one another, with no edge from
        any entry point, is invisible to a per-definition reference count.

    Roots are module-level code (it runs on import) plus every name a framework
    or an entry point calls - the same declared exclusions, reused as roots
    rather than as silence.
    """
    # edges: a definition's span -> the names referenced inside it
    spans = [(d.path, d.lineno, d.end_lineno, d.name) for d in defs]
    spans.sort(key=lambda t: (t[0], t[1]))
    edges: Dict[str, Set[str]] = {}
    for r in refs:
        if _is_test(r.path):
            continue
        # the INNERMOST definition containing this reference is the caller;
        # a reference at module scope has no caller and is a root edge.
        owner = None
        best = -1
        for path, lo, hi, name in spans:
            if path == r.path and lo <= r.lineno <= hi and lo > best:
                owner, best = name, lo
        edges.setdefault(owner or "__module__", set()).add(r.name)

    seen: Set[str] = set()
    queue = list(roots | {"__module__"})
    while queue:
        n = queue.pop()
        if n in seen:
            continue
        seen.add(n)
        queue.extend(edges.get(n, ()))
    return seen


def scan(prod_dirs=PROD_DIRS, test_dirs=TEST_DIRS) -> Scan:
    forms = discover_invocation_forms()
    defs: List[Definition] = []
    refs: List[Reference] = []
    dunder_all: Set[str] = set()
    orm_attrs: Set[str] = set()
    name_counts: Dict[str, int] = {}

    for p in _py_files(prod_dirs):
        path = _as_posix(p)
        try:
            c = _Collector(path, p.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        c.visit(c.tree)
        defs.extend(c.defs)
        refs.extend(c.refs)
        dunder_all |= c.dunder_all
        orm_attrs |= c.orm_attrs

    for p in _py_files(test_dirs):
        path = _as_posix(p)
        try:
            c = _Collector(path, p.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        c.visit(c.tree)
        refs.extend(c.refs)

    for d in defs:
        name_counts[d.name] = name_counts.get(d.name, 0) + 1

    by_name: Dict[str, List[Reference]] = {}
    for r in refs:
        by_name.setdefault(r.name, []).append(r)

    no_consumer: List[Tuple[Definition, str]] = []
    test_only: List[Tuple[Definition, List[str]]] = []
    orphan: List[Tuple[Definition, str]] = []
    deferred: List[Tuple[Definition, str]] = []
    excluded: List[Tuple[Definition, str]] = []
    consumed = 0
    roots: Set[str] = set()
    consumed_defs: List[Definition] = []

    seen: Set[Tuple[str, str, str]] = set()
    for d in defs:
        # One name, one finding. An if/elif that assigns the same constant in
        # both arms is a single producer with a single missing consumer.
        if d.kind == "constant":
            key = (d.name, d.path, d.qual)
            if key in seen:
                continue
            seen.add(key)
        key = f"{d.path}:{d.name}"
        if key in DEFERRED:
            deferred.append((d, DEFERRED[key]))
            roots.add(d.name)
            continue
        reason = _exclusion_reason(d, forms, dunder_all, orm_attrs, name_counts)
        if reason:
            excluded.append((d, reason))
            # A declared exclusion is a ROOT: the framework or the operator
            # calls it, so everything it reaches is reachable.
            roots.add(d.name)
            continue
        prod_sites, test_sites = [], []
        for r in by_name.get(d.name, ()):
            # A reference inside the definition's OWN span is not a consumer:
            # that is recursion, or a docstring naming itself.
            if r.path == d.path and d.lineno <= r.lineno <= d.end_lineno:
                continue
            (test_sites if _is_test(r.path) else prod_sites).append(
                f"{r.path}:{r.lineno}")
        if prod_sites:
            consumed += 1
            consumed_defs.append(d)
        elif test_sites:
            test_only.append((d, sorted(set(test_sites))))
        else:
            no_consumer.append((d, "referenced nowhere outside its definition"))

    # THE REACHABILITY PASS. Only over definitions that DO have a production
    # consumer - the other two buckets already have them, and reporting a
    # finding twice in different words is how an alarm surface inflates.
    reach = _reachable(defs, refs, roots)
    for d in consumed_defs:
        if d.name not in reach:
            orphan.append((d, "has production consumers, but no path from any "
                              "entry point reaches it"))

    return Scan(sorted(no_consumer, key=lambda t: t[0].site),
                sorted(test_only, key=lambda t: t[0].site),
                sorted(orphan, key=lambda t: t[0].site),
                sorted(deferred, key=lambda t: t[0].site),
                excluded, consumed, forms)


def _is_test(path: str) -> bool:
    return path.startswith("tests/") or "/tests/" in path


def _exclusion_reason(d: Definition, forms, dunder_all, orm_attrs,
                      name_counts) -> str:
    """The reason, or "" to scan it. Every branch states WHY."""
    declared = DECLARED_INTENTIONAL.get(f"{d.path}:{d.name}")
    if declared:
        return f"DECLARED INTENTIONAL: {declared}"
    if d.name.startswith("__") and d.name.endswith("__"):
        return "DUNDER: called by the language, not by a reference"
    if d.path in forms and d.name in ("main", "_main_guarded", "cli"):
        # BEFORE the framework label on purpose: "an entry point" is not the
        # useful fact, the FORM is. `python scripts/x.py` and `python -m
        # scripts.x` differ in what lands on sys.path[0], and the form whose
        # only caller is a person is the one no test exercises.
        return ("DECLARED ENTRY POINT ("
                + "/".join(sorted(forms[d.path])) + " form)")
    for dec in d.decorators:
        if dec in DISPATCH_DECORATORS:
            return f"DISPATCH DECORATOR @{dec}: {DISPATCH_DECORATORS[dec]}"
    if (d.name.startswith("visit_")
            and any(b in VISITOR_BASES for b in d.bases)):
        return ("VISITOR DISPATCH: a NodeVisitor subclass method, selected by "
                "node type and never referenced by name")
    if d.name in FRAMEWORK_REASONS:
        return f"FRAMEWORK HOOK: {FRAMEWORK_REASONS[d.name]}"
    if d.name in dunder_all:
        return "__all__: declared public API of its module"
    if d.name in orm_attrs:
        return "ORM: consumed by SQLAlchemy reflection, never by a Python read"
    if d.kind == "method" and name_counts.get(d.name, 0) > 1:
        return (f"POLYMORPHIC: {name_counts[d.name]} definitions share this "
                f"name, so a reference cannot be attributed to one")
    if d.name.startswith("test_"):
        return "TEST: pytest collects it by name"
    return ""


# ── KIND T: A TABLE WRITTEN AND NEVER READ ───────────────────────────────────
#
# The second pass, and the reason it exists is the most expensive instance on
# file. `odds_snapshots` was written on EVERY refresh from 2026-08-25 and read
# by nobody for 47 days; the analysis that needed it read `odds`, which cannot
# hold a price path at all. A name scan cannot see this — the writer IS called,
# and the reader does not exist to be unreferenced.
#
#     A producer whose consumer was never written leaves no name to count.

#: A read, in the two forms this project uses.
#:
#: THE WORD BOUNDARY IS LOAD-BEARING AND IT WAS A BACKSPACE. Written through a
#: non-raw Python string, `` is chr(8), so this pattern began with a literal
#: backspace byte and matched NOTHING. MEASURED 2026-10-11: the table pass then
#: reported `odds_snapshots` as unread although `h1_collection_check.py:248`
#: queries it - a VACUOUS FILTER returning a confident finding. Caught only
#: because that particular answer was known to be wrong.
_ORM_READ = ("query", "select", "scalar", "scalars", "get", "exists")
_SQL_READ = re.compile(r"\b(?:from|join|update|delete\s+from)\s+([a-z_][a-z0-9_]*)",
                       re.I)


class TableFinding(NamedTuple):
    table: str
    cls: str
    writes: List[str]
    reads: List[str]


def scan_tables(prod_dirs=PROD_DIRS, test_dirs=TEST_DIRS) -> List[TableFinding]:
    """Tables with writers and no reader, in production code.

    A read from a TEST does not count: a table nothing in production queries
    is unread however thoroughly its round trip is exercised, which is the
    whole of `odds_snapshots`' 47 days.
    """
    tables: Dict[str, str] = {}      # tablename -> class name
    for p in _py_files(prod_dirs):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if (isinstance(stmt, ast.Assign)
                        and any(isinstance(t, ast.Name)
                                and t.id == "__tablename__"
                                for t in stmt.targets)
                        and isinstance(stmt.value, ast.Constant)):
                    tables[str(stmt.value.value)] = node.name

    reads: Dict[str, List[str]] = {t: [] for t in tables}
    writes: Dict[str, List[str]] = {t: [] for t in tables}
    cls_to_table = {c: t for t, c in tables.items()}

    for p in _py_files(prod_dirs):
        path = _as_posix(p)
        try:
            text = p.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except (SyntaxError, UnicodeDecodeError):
            continue
        # RAW SQL. Any string mentioning the table after FROM/JOIN is a read.
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for m in _SQL_READ.finditer(node.value):
                    t = m.group(1).lower()
                    if t in reads:
                        reads[t].append(f"{path}:{node.lineno} (SQL)")
            # ORM. `query(Cls)` / `select(Cls)` is a read; a bare `Cls(...)`
            # constructor or `session.add(Cls(...))` is a write.
            if isinstance(node, ast.Call):
                callee = _callee_name(node.func)
                for a in node.args:
                    if isinstance(a, ast.Name) and a.id in cls_to_table:
                        t = cls_to_table[a.id]
                        if callee in _ORM_READ:
                            reads[t].append(f"{path}:{node.lineno} (ORM)")
                if callee in cls_to_table:
                    writes[cls_to_table[callee]].append(
                        f"{path}:{node.lineno} (construct)")
        # A bulk write through raw SQL.
        for m in re.finditer(r"insert\s+into\s+([a-z_][a-z0-9_]*)", text, re.I):
            t = m.group(1).lower()
            if t in writes:
                writes[t].append(f"{path} (INSERT)")

    out = []
    for t, cls in sorted(tables.items()):
        if writes[t] and not reads[t]:
            # CLR-2. A declared exemption leaves the finding list CARRYING ITS
            # REASON, which is what makes "declared" an action rather than a
            # silence. An undeclared one stays a finding forever.
            if f"__table:{t}__" in AWAITING_CONSUMER:
                continue
            out.append(TableFinding(t, cls, sorted(set(writes[t])), []))
    return out


def awaiting_consumer_rows() -> List[Tuple[str, dict]]:
    """The declared-but-unread substrates, so they are REPORTED, not hidden."""
    return sorted(AWAITING_CONSUMER.items())


# ── THE KNOWN INSTANCES, FOR MEASURING COVERAGE ──────────────────────────────
#
# A detector that cannot find what is already known has a measured coverage,
# not a verdict. `kind` is the taxonomy from the module docstring; `at` is a
# commit where the instance was still live.

KNOWN_INSTANCES: Tuple[dict, ...] = (
    # ── VERIFIED, 2026-10-12 ────────────────────────────────────────────────
    dict(id="alarmed-append", kind="V", at=None, verdict="VERIFIED",
         what="`alarmed.append(...)` landed; `if alarmed: return 1` did not",
         findable=False,
         why="a list APPEND with no read. The producer is a statement, not a "
             "name, and every name involved is referenced."),
    dict(id="h1-stop-no-caller", kind="N", at="0ed4cad", verdict="VERIFIED",
         what="H1's allocator/stop/terminal states, nothing called them",
         findable=True, symbol="allocate",
         measured=("FOUND", "0ed4cad", "2026-10-11",
                   "allocate TEST-ONLY @ src/data/odds_quota.py:120; "
                   "COLLECTION_TERMINAL_STATES ORPHAN CLUSTER"),
         why="PARTIAL, and the partiality is the honest part. At `0ed4cad` "
             "the scan flags `allocate` (TEST-ONLY) and "
             "`COLLECTION_TERMINAL_STATES` (ORPHAN CLUSTER). It does NOT "
             "flag `h1_reservation`, and must not: that had a live consumer "
             "at `OddsApiQuota.reserved_from_me` on the quota path. The "
             "ledger's 'nothing called them' was true of the COLLECTION "
             "being unwired, not of every name in the apparatus."),
    dict(id="odds-snapshots-unread", kind="T", at="8ab8b21",
         verdict="VERIFIED",
         what="`odds_snapshots` written every refresh since 2026-08-25, read "
              "by nobody", findable=True, symbol="__table:odds_snapshots__",
         measured=("FOUND", "8ab8b21", "2026-10-11",
                   "flagged at 8ab8b21; CLEAN at abb053e (the commit that "
                   "pointed the analysis at it) and today - a before/after "
                   "pair on the REAL history, not a synthetic control"),
         why="NOT findable by the name scan - a table is not a Python name, "
             "and the reader that would be unreferenced was never written. "
             "Found by the SEPARATE table pass (`--tables`), which is why "
             "that pass exists."),
    dict(id="format-series-census", kind="N", at="bafe876",
         verdict="VERIFIED",
         what="`format_series_census` carried the attribution rule and had no "
              "caller", findable=True, symbol="format_series_census",
         measured=("FOUND", "bafe876", "2026-10-11",
                   "TEST-ONLY CONSUMER @ scripts/ci_audit.py:1200 - the "
                   "exact bucket, and the exact signature")),
    dict(id="same-team-strict", kind="N", at=None, verdict="VERIFIED",
         what="`same_team_strict` never called on the footballdataorg path",
         findable=False,
         why="REAL when recorded, SINCE FIXED - the unified resolver calls it "
             "at src/data/team_resolution.py:265 (verified 2026-10-12). Not "
             "findable by a whole-tree scan either way: it was called on "
             "OTHER paths throughout, so the NAME always had consumers. A "
             "per-callsite pass would be needed."),
    dict(id="remaining-vs-updated", kind="F", at=None, verdict="VERIFIED",
         what="the credits gate read `remaining` and never read `updated`",
         findable=False,
         why="REAL when recorded, SINCE FIXED - `updated` is read at "
             "src/scrapers/theodds_scraper.py:88 (verified 2026-10-12). A "
             "FIELD on a persisted object; both names are referenced "
             "somewhere, so the gap is which consumer reads which key."),
    dict(id="conclusion-requested-unread", kind="F", at=None,
         verdict="VERIFIED",
         what="`conclusion` requested in the gh --json list and read nowhere",
         findable=False,
         why="RE-DESCRIBED 2026-10-12. This entry used to read "
             "'the step(s) FAILED alert text was never printed', which is an "
             "UNREACHABLE BRANCH and a different family. The PNC-1 at that "
             "site is the one the ledger actually stated: conclusion is "
             "requested at scripts/ci_audit.py:429 and read nowhere. STILL "
             "OPEN - verified 2026-10-12. Not findable: it is a JSON field "
             "name inside a command-line string."),
    dict(id="requirements-not-read", kind="T", at=None, verdict="VERIFIED",
         what="two workflows never read `requirements.txt`", findable=False,
         why="REAL when recorded, SINCE FIXED. Not Python at all, so no pass "
             "here can see it. Today three of six workflows omit it and all "
             "three are deliberate - ci-audit is stdlib-only by design, and "
             "the probe and h1-collect pip-install a narrower set."),
    dict(id="nothing-reads-snapshot-test", kind="V", at="8ab8b21",
         verdict="VERIFIED",
         what="a test ASSERTING nothing read the snapshot table, pinning "
              "PNC-1 as a specification", findable=False,
         why="an inverted guard. The scan cannot know that a passing test "
             "encodes the defect rather than the requirement."),
    dict(id="injury-observations-unread", kind="T", at=None,
         verdict="VERIFIED", findable=True,
         symbol="__table:injury_observations__",
         what="`injury_observations` written since 2026-08-27, read by nothing",
         measured=("FOUND", "36cbc7a", "2026-10-11",
                   "flagged by --tables on its first real run; 5,136 rows / "
                   "211 teams / 2026-08-27..10-11 measured 2026-10-12")),
    dict(id="first-seen-at-unread", kind="F", at=None, verdict="VERIFIED",
         findable=False,
         what="`odds.first_seen_at` written by two scrapers, read by nothing",
         why="MEASURED 2026-10-12: 123,043 of 488,312 odds rows carry it "
             "(25.2%) and no production site reads it. A COLUMN, not a "
             "table, so the schema pass does not reach it - the table has "
             "readers, this column does not. Its designated writer "
             "`stamp_first_seen` has no production caller either, so its "
             "docstring's 'this is the only site that sets it' is false; "
             "write-once nevertheless holds, by the column's ABSENCE from "
             "the ON CONFLICT update set."),

    # ── MISCLASSIFIED BY ME, corrected by the Part C tally audit ───────────
    #
    # These stay in the register. A register that silently drops its errors
    # is the accretion problem with a tidier face - the claimed count and the
    # corrected count have to be readable side by side.
    dict(id="lookup-table-not-consulted", kind="-", at=None,
         verdict="MISCLASSIFIED", findable=False,
         what="the lookup table the deciding caller never consults",
         why="A RULE FILED AS AN INSTANCE. `A lookup table is only as good as "
             "the earliest decision point that consults it` is a named rule, "
             "and the ledger says it explains THREE defects. Counting the "
             "rule as one occurrence of PNC-1 double-counts in both "
             "directions."),
    dict(id="five-terminal-states", kind="-", at="0ed4cad",
         verdict="MISCLASSIFIED", findable=False,
         what="five terminal states claimed, four reachable plus a dead branch",
         why="THE INVERSE SHAPE. The constant IS read; one value is "
             "unreachable because no input selects it. That is a dead branch "
             "- a consumer with no producer - not a producer with no "
             "consumer."),
    dict(id="report-discovered-margins", kind="-", at="844b0ea",
         verdict="MISCLASSIFIED", findable=False,
         what="`report_discovered_margins` could never evaluate in CI",
         why="It HAD a consumer at ci_audit.py:1962. The defect was the "
             "INVERSE of PNC-1 - a consumer whose producer could never "
             "evaluate, because the step passes no DATABASE_URL."),
    dict(id="scrape-league-fixtures", kind="-", at=None,
         verdict="MISCLASSIFIED", findable=False,
         what="`scrape_league_fixtures` never called",
         why="The ledger's 'never called' was about one RUN MODE (read-only "
             "by construction), not the whole tree: it is called at "
             "betting_agent.py:588 and flashscore_scraper.py:546. A per-mode "
             "claim cannot be checked by a whole-tree scan."),
)


def coverage_rows(found_names: Set[str]) -> List[dict]:
    """A row's verdict comes from the commit where the instance was LIVE.

    THIS WAS WRONG FIRST TIME AND THE ERROR IS WORTH KEEPING VISIBLE. The
    first version evaluated every row against TODAY's tree, where a fixed
    instance is by definition absent — so it scored `format_series_census` as
    a MISS one day after finding it. A detector's coverage is measured against
    the defect, which means against the tree that still had it.

    `measured=(verdict, sha, date, detail)` carries that measurement with its
    provenance. Rows with no `at` sha and no `measured` are evaluated against
    today's tree, which is only valid for an instance still live.
    """
    rows = []
    for inst in KNOWN_INSTANCES:
        if inst.get("verdict") == "MISCLASSIFIED":
            # Kept in the register, OUT of the denominator. A count that
            # includes its own corrections is the accretion problem again.
            rows.append(dict(inst, hit=False,
                             provenance="corrected 2026-10-12 by the tally "
                                        "audit"))
            continue
        m = inst.get("measured")
        if m:
            rows.append(dict(inst, hit=(m[0] == "FOUND"),
                             provenance=f"measured at {m[1]} on {m[2]}",
                             detail=m[3]))
            continue
        sym = inst.get("symbol")
        hit = bool(sym) and sym in found_names
        rows.append(dict(inst, hit=hit,
                         provenance="evaluated against today's tree"))
    return rows


# ── OUTPUT ───────────────────────────────────────────────────────────────────

def format_scan(s: Scan, show_list: bool = False) -> str:
    L = ["PNC-1 SCAN - a producer with no consumer, found by a pass",
         ""]
    L.append(f"  NO CONSUMER         {len(s.no_consumer):4d}   referenced "
             f"nowhere outside its own definition")
    L.append(f"  TEST-ONLY CONSUMER  {len(s.test_only):4d}   referenced ONLY "
             f"from tests - PNC-1's actual signature")
    L.append(f"  ORPHAN CLUSTER      {len(s.orphan_cluster):4d}   consumed, but "
             f"no path from any entry point reaches it")
    L.append(f"  DEFERRED            {len(s.deferred):4d}   acknowledged, "
             f"blocked until {DEFERRED_UNTIL} (H1 is collecting)")
    L.append(f"  DECLARED INTENTIONAL"
             f"{len([1 for _d, r in s.excluded if 'DECLARED INTENTIONAL' in r]):4d}"
             f"   a recorded exemption WITH a reason, never a silence")
    L.append(f"  DECLARED ENTRY POINT{len([1 for _d, r in s.excluded if 'ENTRY POINT' in r]):4d}   "
             f"excluded, invocation form named below")
    L.append(f"  other exclusions    {len([1 for _d, r in s.excluded if 'ENTRY POINT' not in r]):4d}   "
             f"each with a declared reason")
    L.append(f"  consumed            {s.consumed:4d}")
    L.append("")
    L.append("  INVOCATION FORMS (discovered from .github/workflows, plus the")
    L.append("  declared human-invoked list - a person is a caller):")
    for path in sorted(s.forms):
        L.append(f"    {'/'.join(sorted(s.forms[path])):16s} {path}")
    if show_list:
        L.append("")
        L.append("  NO CONSUMER:")
        for d, _r in s.no_consumer:
            L.append(f"    {d.site:52s} {d.kind:9s} {d.qual}")
        L.append("")
        L.append("  ORPHAN CLUSTER (references each other; nothing calls in):")
        for d, _r in s.orphan_cluster:
            L.append(f"    {d.site:52s} {d.kind:9s} {d.qual}")
        L.append("")
        L.append("  TEST-ONLY CONSUMER (the signature - a green test on an "
                 "unconsumed producer):")
        for d, sites in s.test_only:
            L.append(f"    {d.site:52s} {d.kind:9s} {d.qual}")
            L.append(f"      consumers: {', '.join(sites[:4])}"
                     + (" ..." if len(sites) > 4 else ""))
    return "\n".join(L)


def format_coverage(rows: List[dict]) -> str:
    verified = [r for r in rows if r.get("verdict") != "MISCLASSIFIED"]
    wrong = [r for r in rows if r.get("verdict") == "MISCLASSIFIED"]
    findable = [r for r in verified if r.get("findable")]
    hits = [r for r in verified if r["hit"]]
    L = ["COVERAGE against the known instances - MEASURED, not claimed", ""]
    L.append(f"  entries in the register : {len(rows)}")
    L.append(f"  MISCLASSIFIED BY ME    : {len(wrong)}   "
             f"(kept on file, out of the denominator)")
    L.append(f"  VERIFIED instances      : {len(verified)}  <- THE COUNT")
    L.append(f"  findable by this scan   : {len(findable)}")
    L.append(f"  actually found          : {len(hits)}")
    pct = 100.0 * len(hits) / len(verified) if verified else 0.0
    L.append(f"  COVERAGE                : {len(hits)}/{len(verified)} = "
             f"{pct:.0f}%  (of the findable subset: "
             f"{len(hits)}/{len(findable)})")
    L.append("")
    L.append("  THE LEDGER CLAIMED 14 ON 2026-10-11. The tally audit of")
    L.append("  2026-10-12 found 4 misclassified and 2 unrecorded, so the")
    L.append("  verified count is 11 and the coverage ROSE to 36% - not")
    L.append("  because the detector improved, but because the DENOMINATOR")
    L.append("  was wrong. A running tally is a claim like any other.")
    L.append("")
    L.append("  THE TAXONOMY OF MISSES IS THE REAL OUTPUT, and it says this")
    L.append("  scan addresses ONE of four shapes. KIND N is a name; V is a")
    L.append("  value nobody reads, T a table, F a field on a record. Only N")
    L.append("  is a reference-count question, and only 2 of 13 instances")
    L.append("  were ever KIND N - the rest need reachability on branches,")
    L.append("  a schema-vs-query pass, or per-callsite flow.")
    L.append("")
    for r in rows:
        mark = "FOUND" if r["hit"] else ("miss " if r.get("findable")
                                        else "OUT  ")
        L.append(f"  [{r['kind']}] {mark} {r['id']:30s} {r['what'][:58]}")
        if r.get("provenance") and r["hit"]:
            L.append(f"             {r['provenance']}: {r.get('detail', '')}")
        if r.get("why"):
            tag = "partial" if r["hit"] else "why not"
            L.append(f"             {tag}: {r['why']}")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="pnc_scan",
        description="Report producers with no consumer. REPORTS; it does not "
                    "alarm - see the ledger for why a pre-existing backlog "
                    "must be triaged before a gate is wired.")
    ap.add_argument("--list", action="store_true",
                    help="every finding with its definition site")
    ap.add_argument("--tables", action="store_true",
                    help="KIND T: tables written by production and read by "
                         "nothing in production")
    ap.add_argument("--coverage", action="store_true",
                    help="measured coverage against the known instances")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--symbol", help="report one name's consumers and exit")
    args = ap.parse_args()

    s = scan()
    found = ({d.name for d, _ in s.no_consumer}
             | {d.name for d, _ in s.test_only}
             | {d.name for d, _ in s.orphan_cluster}
             | {f"__table:{f.table}__" for f in scan_tables()})

    if args.symbol:
        for d, sites in s.test_only:
            if d.name == args.symbol:
                print(f"{args.symbol}: TEST-ONLY CONSUMER at {d.site}")
                print("  consumers: " + ", ".join(sites))
                return 0
        for d, _r in s.no_consumer:
            if d.name == args.symbol:
                print(f"{args.symbol}: NO CONSUMER at {d.site}")
                return 0
        print(f"{args.symbol}: consumed, excluded, or not defined in "
              f"{'/'.join(PROD_DIRS)}")
        return 0

    if args.json:
        print(json.dumps(dict(
            no_consumer=[dict(name=d.name, qual=d.qual, site=d.site,
                              kind=d.kind) for d, _ in s.no_consumer],
            test_only=[dict(name=d.name, qual=d.qual, site=d.site,
                            kind=d.kind, consumers=c)
                       for d, c in s.test_only],
            orphan_cluster=[dict(name=d.name, qual=d.qual, site=d.site,
                                 kind=d.kind) for d, _ in s.orphan_cluster],
            deferred=[dict(name=d.name, site=d.site) for d, _ in s.deferred],
            counts=dict(no_consumer=len(s.no_consumer),
                        test_only=len(s.test_only),
                        orphan_cluster=len(s.orphan_cluster),
                        deferred=len(s.deferred),
                        consumed=s.consumed, excluded=len(s.excluded)),
            coverage=coverage_rows(found),
        ), indent=2))
        return 0

    print(format_scan(s, show_list=args.list))
    if args.tables or args.list or args.coverage:
        t = scan_tables()
        print()
        print(f"KIND T - tables WRITTEN by production and READ by nothing in "
              f"production: {len(t)}")
        for f in t:
            print(f"  {f.table:24s} ({f.cls})  writes: "
                  + ", ".join(f.writes[:3]))
        if not t:
            print("  none that is not declared. Every table with a production "
                  "writer either has a production reader or an entry in "
                  "AWAITING_CONSUMER.")
        print()
        print(f"AWAITING A CONSUMER - declared, unread, with a named consumer "
              f"and a re-triage date: {len(AWAITING_CONSUMER)}")
        for key, meta in awaiting_consumer_rows():
            print(f"  {key}")
            print(f"      {meta['rows']}")
            print(f"      consumer : {meta['consumer'][:66]}")
            print(f"      re-triage: {meta['retriage']}  ({meta['basis'][:48]})")
    if args.coverage:
        print()
        print(format_coverage(coverage_rows(found)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
