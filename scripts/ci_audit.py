"""Mechanical half of the daily CI audit. Stdlib only.

    python -m scripts.ci_audit --unaudited          # runs not yet in the ledger
    python -m scripts.ci_audit --since 2026-08-11 --until 2026-08-13
    python -m scripts.ci_audit --run 31482430418    # one run

Stage 14, Part D. `.claude/commands/daily-ci-audit.md` drives this; the command
carries the judgement, this carries the counting.

WHY A SCRIPT AND NOT ONLY A PROMPT
----------------------------------
Part A's manual pass took a day and found 1 BROKEN and 9 DEGRADED runs that CI
had reported as `success`. The value was never in the reading — it was in
looking at every step of every run instead of the tail of the failed ones. That
part is mechanical and should not depend on anyone's stamina.

What is NOT mechanical, and stays in the command: deciding whether a zero is a
defect or a quiet day, and writing the note a future reader needs.

THE ASSERTIONS ARE SELF-CALIBRATING, DELIBERATELY
-------------------------------------------------
Every threshold here is relative to what this pipeline recently did, never to a
hand-maintained list. A hardcoded league list rots; "a unit that produced data
within the last N days produced none today" adapts as coverage changes and stays
silent on a card where nothing was ever expected.

This is the 2026-08-07 audit's lesson applied to alerting: thresholds fitted to
a snapshot are noise generators.

`conclusion: success` is not evidence. Every core step in `daily-picks` carries
`continue-on-error: true`, so a run is green whenever the runner survived.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Optional

LEDGER = pathlib.Path("docs/ci-audit-ledger.md")
LOGS = pathlib.Path("ci_logs")
WORKFLOWS = ("daily-picks.yml", "closing-lines.yml", "paper-trading-report.yml")

#: How far back "recently produced data" looks. Not a threshold on the metric —
#: a window on the pipeline's own history.
LOOKBACK_RUNS = 7

ANSI = re.compile(r"\x1b\[[0-9;]*m")


# ─────────────────────────────────────────────────────────── evidence sources

def _sh(*args: str) -> str:
    """Always decode UTF-8, never the locale codec.

    `text=True` alone decodes with the platform encoding. On a Windows console
    that is cp1251, and a CI log containing any byte outside it kills the reader
    thread, leaving `.stdout` as None and the caller writing None to a file.
    Third instance of this class in Stage 14 — the DEL-2 harness and the emoji
    in `ci_alert` were the other two. CI is Linux/UTF-8 and would never have
    shown it.
    """
    r = subprocess.run(args, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.stdout or ""


#: Verdicts that are NOT a result. A row carrying one of these is a note saying
#: "come back later", and until 2026-09-13 nothing came back.
PROVISIONAL_VERDICTS = ("IN_PROGRESS", "UNAUDITABLE")

#: Matches a ledger row's run id whether or not it is bolded. The original
#: pattern required a bare `| 123… |` and silently skipped every
#: bolded row — which is why the 09-10 pass re-listed six runs that were
#: already audited, and why this function over-reported ever since.
_LEDGER_ROW = re.compile(r"^\|\s*\**\s*(\d{9,})\s*\**\s*\|(.*)$", re.M)


def audited_run_ids() -> set:
    """Run ids carrying a FINAL verdict in the ledger.

    TWO DEFECTS, BOTH FOUND BY USING IT.

    1. It matched only unbolded rows, so `| **34199783169** | **daily-picks** |`
       read as unaudited. On 2026-09-10 that made `--unaudited` report 59 runs of
       which six already had verdicts. Over-reporting wastes work; it does not
       hide anything, which is why it survived.

    2. It counted ANY row as audited, including `IN_PROGRESS` and
       `UNAUDITABLE` — verdicts that explicitly mean "no verdict yet". Run
       34745992077 carried `IN_PROGRESS` on 2026-09-13 and was re-listed only
       because a human asked. **That one hides something**: a provisional
       verdict is permanent by default when the query that finds work treats it
       as finished.

    THE SECOND IS THE THIRD-STATE PROBLEM ONE LEVEL UP. Naming the state stopped
    it reading as health; nothing was obliged to act on the name. A rule that
    fires only when someone remembers it is not a practice.
    """
    if not LEDGER.exists():
        return set()
    out = set()
    for rid, rest in _LEDGER_ROW.findall(LEDGER.read_text(encoding="utf-8")):
        if _is_provisional(rest):
            continue          # re-list it: this row is a request, not a result
        out.add(rid)
    return out


def _is_provisional(row_rest: str) -> bool:
    """True when a ledger row's VERDICT CELL is provisional.

    Checked per CELL, never as a substring of the row. The first version tested
    `"IN_PROGRESS" in rest` and immediately re-listed 34745992077 — whose row
    says DEGRADED and whose NOTE says "Was IN_PROGRESS when first audited".

    That is the same error this ledger has recorded three times: a definition
    read as an occurrence. The word appearing in a row is not the row carrying
    that verdict, exactly as a workflow's YAML echoed into a log is not a failed
    step and a test named after a string is not that string occurring.

    A cell counts as provisional when, stripped of bold markers, it BEGINS with
    a provisional verdict — so `IN_PROGRESS` and `UNAUDITABLE — empty log` both
    match while a note mentioning either does not.
    """
    for cell in row_rest.split("|"):
        c = cell.replace("*", "").replace("~", "").strip().upper()
        if any(c.startswith(v) for v in PROVISIONAL_VERDICTS):
            return True
    return False


class RunListing(list):
    """A list of runs that knows whether it is the WHOLE list.

    THE THIRD STATE, NAMED. `returned == limit` does not mean "that is all
    there is" — it means the query stopped counting and nobody can tell which.
    A complete listing and a listing cut off at its ceiling are the same object
    until one of them says so.

    This is the fourth instance of the class in this file and the same one the
    odds path closed three times:

        ``[]`` vs ``None``                 measured-and-empty vs never-measured
        429-with-credits vs 429-with-zero  rate-limited vs exhausted
        no-hits vs no-log                  a clean run vs an unreadable one
        **returned == limit**              **complete vs truncated**

    `list_runs`'s own docstring already recorded this defect for
    `closing-lines` at ``limit=40`` — *"which read as 'no closing-lines runs in
    the window' rather than as a truncated query"* — and it recurred at 250 for
    `daily-picks` on 2026-09-16, hiding 24 unaudited runs. **Knowledge present,
    caller not consulting it.**

    RAISING THE LIMIT MOVES THE CLIFF; NAMING THE STATE REMOVES IT. A bigger
    number is right until the next workflow outgrows it, and it fails the same
    silent way when it does.
    """

    #: Workflows whose query came back at the cap, so their history is UNKNOWN
    #: beyond that point rather than exhausted. `{workflow: limit}`.
    truncated: Dict[str, int]

    def __init__(self, rows=(), truncated=None):
        super().__init__(rows)
        self.truncated = dict(truncated or {})

    @property
    def complete(self) -> bool:
        return not self.truncated

    def warning(self) -> Optional[str]:
        """The sentence a caller must print, or None when the listing is whole."""
        if self.complete:
            return None
        parts = ", ".join(f"{wf} (limit {n})"
                          for wf, n in sorted(self.truncated.items()))
        return (f"TRUNCATED LISTING: {parts} returned exactly as many runs as "
                f"were asked for, so anything older is UNKNOWN, not absent. "
                f"Counts below are a LOWER BOUND. Re-run with a higher --limit "
                f"to see further back.")


def list_runs(since: Optional[str], until: Optional[str],
              limit: int = 250) -> RunListing:
    """`limit` is per workflow and must outrun the busiest one.

    closing-lines fires roughly every two hours, so 40 covered five days and
    silently dropped everything older — which read as "no closing-lines runs in
    the window" rather than as a truncated query.

    THE RETURN VALUE NOW CARRIES WHETHER IT IS COMPLETE. See `RunListing`: a
    workflow that came back at exactly `limit` is recorded as truncated, and
    every caller that reports a COUNT must say so. The filtering below is
    applied AFTER the cap, so `since`/`until` cannot be used to argue the cap
    was not reached — the cap is a property of the query, not of the window.
    """
    out: List[dict] = []
    truncated: Dict[str, int] = {}
    for wf in WORKFLOWS:
        raw = _sh("gh", "run", "list", "--workflow", wf, "--limit", str(limit),
                  "--json", "databaseId,startedAt,conclusion,event,status")
        try:
            rows = json.loads(raw or "[]")
        except json.JSONDecodeError:
            rows = []
        if len(rows) >= limit:
            truncated[wf.replace(".yml", "")] = limit
        for r in rows:
            day = (r.get("startedAt") or "")[:10]
            if since and day < since:
                continue
            if until and day > until:
                continue
            r["workflow"] = wf.replace(".yml", "")
            out.append(r)
    return RunListing(sorted(out, key=lambda r: r.get("startedAt") or ""),
                      truncated)


def fetch_log(run_id: str) -> str:
    """Full log of every step. Cached under ci_logs/ so a re-audit is free."""
    d = LOGS / f"run_{run_id}"
    f = d / "full.log"
    if not f.exists() or f.stat().st_size == 0:
        d.mkdir(parents=True, exist_ok=True)
        f.write_text(_sh("gh", "run", "view", run_id, "--log"),
                     encoding="utf-8", errors="replace")
    return ANSI.sub("", f.read_text(encoding="utf-8", errors="replace"))


# ───────────────────────────────────────────────────────────── counted facts

PATTERNS = {
    "picks_saved": r"PAPER TRADING: saving (\d+) pick",
    "picks_saved_live": r"Saved (\d+) new pick",
    "observations": r"pick_observations.*?(\d+) written",
    "af_requests": r"API-Football update complete \((\d+) requests used",
    "injuries_saved": r"Injury update: saved (\d+) injuries from (\d+) fixtures",
    "odds_rows": r"TheOddsAPI update complete: (\d+) odds rows",
    "credits": r"= (\d+) credits",
    "clv_pairs": r"(\d+) valid CLV pair",
    "fixtures_created": r"creating new fixture",
    "reviews": r"Briefing decision \[",
    "settled": r"Settled (\d+) picks",
    # closing-lines evidence. Without these the audit reported 25 of 27 runs
    # CLEAN against a manual pass that found 9 DEGRADED — it was reading
    # daily-picks vocabulary against closing-lines logs and finding nothing.
    "credits_claimed": r"credits claimed\s*:\s*(\d+)",
    "captured": r"captured\s*:\s*(\d+)",
    "capture_missing": r"captured, (\d+) missing",
    "no_rows": r"result=no_rows",
    # A Claude KEEP/CHANGE verdict that was computed and then thrown away. The
    # manual pass caught these; the first version of this script did not, and
    # that was the single run where the two disagreed. The A1 cascade defect
    # surfaced here for four days as a swallowed NotNullViolation.
    "decisions_discarded": r"Could not apply",
    # Stage 21 follow-up. A fixture with no odds is indistinguishable from one
    # nobody wanted to bet. Alarms only on the identity class (no API-Football
    # id); odds-budget coverage is logged at INFO and deliberately not counted
    # here, because 15 of 18 on 2026-08-29 were budget and would have drowned
    # the 3 that were not.
    "unpriced_fixtures": r"UNPRICED FIXTURES: (\d+) row",
    "unpriced_check_dead": r"UNPRICED FIXTURE CHECK DID NOT RUN",
    "report_sent": r"(?:Performance report sent to Telegram|Settlement report sent to Telegram)",
    # s5.9's OWN refusal, distinct from the ordinary per-match cap. Both log
    # `PICK_REJECTED reason=same_fixture_limit`; only the duplicate case says
    # so, and on 2026-09-04 the cap fired and was nearly recorded as s5.9's
    # first catch.
    "duplicate_fixture_refusals": r"cause=duplicate_rows_one_fixture",
    # PRE-REGISTERED AND CURRENTLY MATCHES NOTHING. The picks-run guard was
    # reverted on 2026-09-04; this pattern waits for its redesign. It is here
    # BEFORE the guard rather than after because the guard shipped once already
    # with a log line this file had no pattern for, and five closing-lines runs
    # scored CLEAN while captures were stopped all day.
    "picks_run_guard_declined": r"PICKS-RUN GUARD: DECLINING",
    # The experiment record that replaces the frozen live all-time in the
    # Telegram reports. Registered BEFORE the code that emits it, deliberately:
    # the picks-run guard shipped with a log line this file had no pattern for
    # and five runs scored CLEAN while captures were stopped all day. A report
    # that silently stops carrying the block must be visible here.
    "experiment_record_built": r"EXPERIMENT RECORD: n=(\d+)",
    # Stage 19. The audit was blind to a day that discovered nothing: 2026-08-26
    # analysed 0 fixtures against a card of six real matches and was flagged
    # only for the API-Football suspension. It surfaced because Niki looked at a
    # football calendar.
    #
    # Keyed on the scraper's OWN warning, which already excludes
    # `off_season_leagues` — so a genuinely dormant league does not fire it, and
    # the assertion needs no threshold of its own to get wrong.
    # STAGE 20 Part C. Anchored on "NO ROWS AT ALL", which the scraper now
    # emits only when the PAGE yielded nothing — not when a league simply has
    # no fixtures inside the window it was queried for. The old pattern fired
    # 21 times on 2026-08-27, all false positives, which is how a check gets
    # ignored.
    "fixtures_zero_active": r"returned 0 fixtures for \S+ . the page yielded NO ROWS AT ALL, expected .1 for active season",
    "no_fixtures_at_all": r"No fixtures found for \d{4}-\d{2}-\d{2}",
    "fixtures_scraped": r"Scraped (\d+) fixtures from",
    # PER-SOURCE discovery. The aggregate assertion shipped earlier in Stage 19
    # would NOT have fired on 2026-05-31, the day Flashscore went silent:
    # flashscore=0, football-data.org=0, apifootball=13, so the TOTAL was
    # healthy and nothing alarmed. It then stayed silent for 88 days.
    #
    # A fallback that substitutes silently makes the primary's failure
    # invisible. Redundancy that is not checked PER COMPONENT is not
    # redundancy — it is one working source and two unverified claims. Each
    # source is therefore watched on its own, regardless of the others.
    "src_flashscore_fixtures": r"Scraped (\d+) fixtures from",
    "src_footballdataorg_fixtures": r"football-data\.org: \d+ scores updated, (\d+) new fixtures added",
    "src_apifootball_fixtures": r"API-Football: creating new fixture",
    # MATCHED, not created. A source that fetches 325 fixtures and successfully
    # matches every one onto an existing row CREATES nothing — and scored zero,
    # identical to a source that returned nothing at all. It fired a false
    # positive on every daily-picks run from 2026-09-03 to 09-08.
    #
    # Creation measures NOVELTY, and novelty legitimately falls to zero
    # whenever another source got there first. That is the normal steady state,
    # not a fault. Liveness is `created + matched > 0`.
    "src_apifootball_matched": r"API-Football fixtures [0-9-]+: \d+ created, (\d+) updated",
    "src_flashscore_matched": r"Fuzzy-merged|Flashscore: \d+ fixtures? updated",
    "src_footballdataorg_matched": r"football-data\.org: (\d+) scores updated",
}


def extract(log: str) -> Dict[str, object]:
    """Counts, not adjectives."""
    f: Dict[str, object] = {}
    for key, pat in PATTERNS.items():
        ms = re.findall(pat, log)
        if not ms:
            continue
        if key in ("fixtures_created", "reviews", "no_rows",
                   "decisions_discarded", "fixtures_zero_active",
                   "no_fixtures_at_all", "unpriced_check_dead",
                   "duplicate_fixture_refusals", "picks_run_guard_declined",
                   "report_sent"):
            f[key] = len(ms)
        elif key == "injuries_saved":
            f["injuries_saved"] = int(ms[-1][0])
            f["injury_fixtures"] = int(ms[-1][1])
        else:
            try:
                f[key] = int(ms[-1] if isinstance(ms[-1], str) else ms[-1][0])
            except (TypeError, ValueError):
                # A FALLBACK MUST RECORD THAT IT FIRED. This `pass` silently
                # dropped `unpriced_check_dead` for two days: the pattern has
                # no numeric capture group, so int() raised, the key never
                # reached `facts`, and the assertion built to catch a dead
                # check was itself dead. A pattern that matches but cannot be
                # counted is a bug in the pattern, and it now says so.
                print(f"  ci_audit: pattern {key!r} matched but produced no "
                      f"number — add it to the count-style keys above, or give "
                      f"it a numeric capture group. NOT COUNTED.")
    # Per-source: sum every occurrence (Flashscore logs one line per league),
    # count occurrences for API-Football (one line per fixture created).
    # Only set when the run ACTUALLY attempted fixture discovery. A
    # closing-lines run does not scrape fixtures, and reporting `fs=0` there
    # would conflate "did not report" with "reported nothing" — the exact
    # distinction this summary exists to preserve.
    if "Scraping fixtures:" in log or "fixtures from" in log:
        f["src_flashscore_fixtures"] = sum(
            int(x) for x in re.findall(PATTERNS["src_flashscore_fixtures"], log))
    _fdo = re.findall(PATTERNS["src_footballdataorg_fixtures"], log)
    if _fdo:
        f["src_footballdataorg_fixtures"] = sum(int(x) for x in _fdo)
    if "API-Football" in log:
        f["src_apifootball_fixtures"] = len(
            re.findall(PATTERNS["src_apifootball_fixtures"], log))
    # The matched counterparts. Summed, because a run reports one line per date
    # and a source is alive if it matched anything on any of them.
    _afm = re.findall(PATTERNS["src_apifootball_matched"], log)
    if _afm:
        f["src_apifootball_matched"] = sum(int(x) for x in _afm)
    _fsm = re.findall(PATTERNS["src_flashscore_matched"], log)
    if _fsm:
        f["src_flashscore_matched"] = len(_fsm)
    _fdm = re.findall(PATTERNS["src_footballdataorg_matched"], log)
    if _fdm:
        f["src_footballdataorg_matched"] = sum(int(x) for x in _fdm)

    # ---- DEL-3: per-report sequence integrity ---------------------------
    # Parsed as a RECORD, not counted as an occurrence. A report's delivery is
    # one fact with several fields, and the three assertions below each read a
    # different field. Counting matches would collapse them back together.
    #
    # `REPORT_DELIVERY report=<name> chunks=N sent=K failed=<list|none>
    #  terminator=<yes|no> attempts=N`
    _rd = re.findall(
        r"REPORT_DELIVERY report=(.+?) chunks=(\d+) sent=(\d+) "
        r"failed=(\S+) terminator=(yes|no) attempts=(\d+)", log)
    if _rd:
        f["report_deliveries"] = [
            {"report": r[0], "chunks": int(r[1]), "sent": int(r[2]),
             "failed": ([] if r[3] == "none"
                        else [int(x) for x in r[3].split(",") if x.isdigit()]),
             "terminator": r[4] == "yes", "attempts": int(r[5])}
            for r in _rd]
    f["report_incomplete"] = len(re.findall(r"REPORT INCOMPLETE:", log))

    _fx = re.findall(PATTERNS["fixtures_scraped"], log)
    if _fx:
        f["fixtures_scraped"] = sum(int(x) for x in _fx)
        f["fixture_attempts"] = len(_fx)
    f["errors"] = len(re.findall(r"\| ERROR +\|", log))
    f["tracebacks"] = len(re.findall(r"Traceback \(most recent call last\)", log))
    f["account_suspended"] = "account suspended" in log
    f["telegram_sent"] = len(re.findall(r"Telegram message sent", log))
    f["telegram_failed"] = len(re.findall(
        r"Failed to send Telegram message|alert NOT delivered", log))
    # Anchored to start with a STEP NAME, not a quote. GitHub echoes each
    # `run:` block into the log, so the workflow's own source line —
    #   msg = ("... step(s) FAILED — " + ", ".join(failed)
    # — matches a naive pattern and reports every run BROKEN. Found by running
    # this script against the window the manual pass had already audited: it
    # called two DEGRADED runs BROKEN. A definition is not an occurrence.
    f["steps_failed"] = re.findall(
        r"step\(s\) FAILED — ([A-Za-z][^\n]{0,120})", log)
    return f


# ───────────────────────────────────────────────── self-calibrating assertions

def assertions(facts: Dict[str, object],
               history: List[Dict[str, object]]) -> List[str]:
    """Fire only when a unit that RECENTLY produced data produces none.

    `history` is the same fact dict for the previous runs of this workflow, most
    recent last. An empty history means nothing can be said — and saying nothing
    is correct, not a pass.
    """
    hits: List[str] = []

    def produced_recently(key: str) -> bool:
        return any((h.get(key) or 0) > 0 for h in history[-LOOKBACK_RUNS:])

    for key, label in (("picks_saved", "picks"),
                       ("odds_rows", "Odds API rows"),
                       ("injuries_saved", "injuries"),
                       ("clv_pairs", "valid CLV pairs")):
        if key in facts and (facts.get(key) or 0) == 0 and produced_recently(key):
            hits.append(
                f"{label} = 0, but this workflow produced {label} within the "
                f"last {LOOKBACK_RUNS} runs")

    # PER-SOURCE discovery, on the day's first run only.
    #
    # The aggregate version shipped earlier in Stage 19 would NOT have fired on
    # 2026-05-31, the day Flashscore went silent: flashscore=0,
    # football-data.org=0, API-Football=13, so the TOTAL looked healthy. It then
    # stayed silent for 88 days.
    #
    # A fallback that substitutes silently makes the primary's failure
    # invisible. Redundancy that is not checked PER COMPONENT is not redundancy
    # — it is one working source and two unverified claims.
    if facts.get("is_first_run_of_day", True):
        for key, mkey, label in (
                ("src_flashscore_fixtures", "src_flashscore_matched",
                 "Flashscore fixtures"),
                ("src_footballdataorg_fixtures", "src_footballdataorg_matched",
                 "football-data.org fixtures"),
                ("src_apifootball_fixtures", "src_apifootball_matched",
                 "API-Football fixtures")):
            if key not in facts or not produced_recently(key):
                continue
            created = facts.get(key) or 0
            matched = facts.get(mkey) or 0
            # ALIVE IF created + matched > 0. Zero creations alone is novelty
            # falling to zero, which happens whenever another source reached
            # the fixture first — the steady state, not a fault.
            if created + matched > 0:
                continue
            hits.append(
                f"{label}: 0 created AND 0 matched while other sources still "
                f"produce — this source produced within the last "
                f"{LOOKBACK_RUNS} runs and is now silent on both counts")

    picks = facts.get("picks_saved") or facts.get("picks_saved_live") or 0
    obs = facts.get("observations")
    if picks and obs is not None and obs != 2 * picks:
        hits.append(f"pick_observations {obs} != 2 x {picks} picks saved")

    # NOT self-calibrating, and deliberately so: spending a credit and getting
    # no rows back is wrong on the first occurrence, not relative to history.
    # This is the condition the manual pass used to mark 9 runs DEGRADED.
    if facts.get("no_rows"):
        hits.append(
            f"{facts['no_rows']} league request(s) returned no_rows — credits "
            "were spent and the provider returned an empty event list")
    if (facts.get("credits_claimed") or 0) > 0 and (facts.get("captured") or 0) == 0:
        hits.append(
            f"{facts['credits_claimed']} credit(s) claimed, 0 closing lines "
            "captured")

    # Also not self-calibrating: a review decision that was computed and then
    # discarded is wrong on the first occurrence. The reviewed pick silently
    # keeps whatever the model chose, and the KEEP/CHANGE record gains a gap.
    if facts.get("decisions_discarded"):
        hits.append(
            f"{facts['decisions_discarded']} briefing decision(s) DISCARDED — "
            "the review ran and its verdict was thrown away")

    # Stage 19 — discovery. NOT self-calibrating: a fixture scrape that returns
    # nothing for a league the scraper itself calls in-season is wrong on the
    # first occurrence. MEASURED 2026-08-26: this had been true on EVERY run
    # since 2026-05-30 — 88 days, 200+ attempts, zero fixtures — and nothing
    # reported it, because API-Football was quietly covering for it until its
    # suspension on 08-19.
    if facts.get("fixtures_zero_active"):
        hits.append(
            f"{facts['fixtures_zero_active']} active-season league(s) returned "
            "0 fixtures — discovery produced nothing the scraper expected")
    if facts.get("fixture_attempts") and not facts.get("fixtures_scraped"):
        hits.append(
            f"{facts['fixture_attempts']} fixture scrape(s) attempted, "
            "0 fixtures found in total")
    if facts.get("no_fixtures_at_all"):
        hits.append("NO FIXTURES FOUND for the day — nothing was analysed")

    # NOT self-calibrating: a fixture that could not be priced because a team
    # would not resolve is wrong on the first occurrence, not relative to
    # history. It is the shape that hid for three days as "nobody bet that one".
    if facts.get("unpriced_fixtures"):
        hits.append(
            f"{facts['unpriced_fixtures']} row(s) carry NO ODDS while "
            "same-league peers do (cause not established: unresolved team, "
            "or a duplicate row holding the same fixture)")

    # A check that did not run is not a clean check.
    if facts.get("unpriced_check_dead"):
        hits.append("the unpriced-fixture check DID NOT RUN — no evidence "
                    "either way, not a clean result")

    if facts.get("duplicate_fixture_refusals"):
        hits.append(
            f"{facts['duplicate_fixture_refusals']} pick(s) refused because two "
            "ROWS were one FIXTURE — s5.9 acting, not the ordinary per-match cap")

    if facts.get("picks_run_guard_declined"):
        hits.append("the odds refresh DECLINED — closing-line capture stopped "
                    "for this run; every pending pick will be rejected as late")

    # A report that stops carrying the experiment block is silent otherwise:
    # the message still sends, it just says less. Absence is the finding.
    if facts.get("report_sent") and not facts.get("experiment_record_built"):
        hits.append("a report was sent WITHOUT the experiment record block — "
                    "the paper series silently stopped being reported")

    if facts.get("tracebacks"):
        hits.append(f"{facts['tracebacks']} traceback(s) in the log")
    if facts.get("account_suspended"):
        hits.append("API-Football reported the account suspended")
    if facts.get("telegram_failed"):
        hits.append(f"{facts['telegram_failed']} alert(s) failed to deliver")

    # ---- DEL-3, three assertions, each on a DIFFERENT failure shape -------
    #
    # A lost middle chunk and a stream that stopped early are not the same
    # event and do not present the same way. One leaves a hole and ends
    # normally; the other ends early and leaves no hole. A single assertion
    # would catch whichever was written first and miss the other — which is
    # precisely how the defect survived: `_send_chunked` returned the LAST
    # chunk's Message, so it was blind to holes and blind to nothing else.
    for d in (facts.get("report_deliveries") or []):
        # 1. A HOLE. Parts failed; the report reached a reader incomplete.
        if d["failed"]:
            hits.append(
                f"the {d['report']} was delivered INCOMPLETE — part(s) "
                f"{d['failed']} of {d['chunks']} failed to send; a reader sees "
                f"a report that ends normally")
        # 2. AN EARLY END. No terminator, so the stream stopped before the
        #    last part and nothing in it says so.
        elif not d["terminator"]:
            hits.append(
                f"the {d['report']} carries NO TERMINATOR — the final part "
                f"never sent, so the message a reader has is truncated with "
                f"nothing marking the cut")
        # 3. ARITHMETIC. sent + failed must equal chunks. A mismatch with an
        #    empty `failed` list means a part went missing WITHOUT being
        #    recorded, which is the original defect returning by another route.
        if d["sent"] + len(d["failed"]) != d["chunks"]:
            hits.append(
                f"the {d['report']}'s parts do not add up: {d['sent']} sent + "
                f"{len(d['failed'])} failed != {d['chunks']} chunks — a part "
                f"went missing without being recorded")
    if facts.get("steps_failed"):
        hits.append(f"core step(s) reported failure: {facts['steps_failed'][-1]}")
    return hits


def discovery_summary(facts: Dict[str, object]) -> str:
    """`disc[fs=N fdo=N af=N]` — per source, never an aggregate.

    Printed on every daily-picks row and carried into the ledger note. The
    total is the number that hid Flashscore's death from 2026-05-30 to
    2026-08-26; only the per-source split makes a silent substitution visible.
    Absent (rather than 0) is shown as `-`, because "did not report" and
    "reported nothing" are different facts.

    Shown as `created/matched` since 2026-09-09. A bare creation count read
    identically for a healthy source that matched everything and a dead one
    that returned nothing — it cried wolf on every daily-picks run for five
    days, which trains the reader to skip the very line that would catch the
    88-day failure this assertion exists for.
    """
    keys = (("fs", "src_flashscore_fixtures", "src_flashscore_matched"),
            ("fdo", "src_footballdataorg_fixtures", "src_footballdataorg_matched"),
            ("af", "src_apifootball_fixtures", "src_apifootball_matched"))
    if not any(k in facts for _, k, _m in keys):
        return ""
    parts = []
    for label, ckey, mkey in keys:
        c = facts[ckey] if ckey in facts else "-"
        m = facts[mkey] if mkey in facts else "-"
        parts.append(f"{label}={c}c/{m}m")
    return "disc[" + " ".join(parts) + "]"


#: A log shorter than this carries no step output — a run that is still
#: executing, or whose log could not be fetched. Not a threshold on quality:
#: a real daily-picks log is ~600k and the smallest real capture log ~10k.
_MIN_AUDITABLE_LOG_BYTES = 200


def verdict(facts: Dict[str, object], hits: List[str],
            log: str = None) -> str:
    """CLEAN means the checks ran and found nothing. It cannot mean nothing ran.

    THIRD INSTANCE OF THE SAME COLLAPSE IN THIS TOOL. `[]` versus `None` in the
    odds path, 429-with-credits versus 429-with-zero, and now an empty result
    versus an unmeasured one — all three returned the same value for "found
    nothing" and "could not look".

    Found 2026-09-13: run 34745992077 (daily-picks, 09-13 07:44) had a
    ZERO-BYTE cached log and this function returned CLEAN. No assertion can fire
    against an empty file, so `hits` was empty, so the run scored clean. A
    verdict from no evidence.

        No evidence is not a clean verdict.

    `UNAUDITABLE` is named and counted so it appears in the ledger as its own
    row rather than inflating the CLEAN count — which is exactly how nine
    DEGRADED runs went unnoticed in the pass this tool was built to replace.
    """
    # A run still executing has no verdict to give. Its log is empty because
    # `gh run view --log` returns nothing until completion, NOT because
    # anything failed — found 2026-09-13 on run 34745992077, which was 20
    # minutes old and mid-step-15 when it was audited and reported UNAUDITABLE.
    # Checked BEFORE the log test so an in-flight run is never mistaken for a
    # missing one: both look identical from the log alone, and only the run
    # metadata separates them.
    if str(facts.get("run_status") or "").lower() in ("in_progress", "queued",
                                                      "waiting", "pending",
                                                      "requested"):
        return "IN_PROGRESS"
    if log is not None and len(log.strip()) < _MIN_AUDITABLE_LOG_BYTES:
        return "UNAUDITABLE"
    if facts.get("steps_failed") or facts.get("tracebacks"):
        return "BROKEN"
    return "DEGRADED" if hits else "CLEAN"


# ────────────────────────────────────────────────────────────────────── main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unaudited", action="store_true",
                    help="only runs with no ledger row")
    ap.add_argument("--since")
    ap.add_argument("--until")
    ap.add_argument("--run", help="one run id")
    ap.add_argument("--limit", type=int, default=250)
    a = ap.parse_args()

    if a.run:
        # Fetch the run's real status: --run used to synthesise a row with no
        # status at all, so an in-flight run audited this way could never
        # report IN_PROGRESS. The single-run path is the one used to chase an
        # anomaly, which is exactly when the distinction matters most.
        _st = ""
        try:
            _st = json.loads(_sh("gh", "run", "view", a.run, "--json",
                                 "status") or "{}").get("status", "")
        except Exception:
            pass
        runs = [{"databaseId": int(a.run), "workflow": "?", "status": _st,
                 "startedAt": "", "conclusion": "?"}]
        listing_warning = None
    else:
        listing = list_runs(a.since, a.until, a.limit)
        # CONSULTED, not merely available. The defect this exists for was never
        # that the truncation was unknowable — it was that nothing asked.
        listing_warning = listing.warning()
        runs = list(listing)
        if a.unaudited:
            done = audited_run_ids()
            runs = [r for r in runs if str(r["databaseId"]) not in done]

    if listing_warning:
        print(f"!! {listing_warning}\n")

    if not runs:
        # "No runs" is the reading most changed by a truncated query, so the
        # warning is repeated rather than assumed to have been read above.
        print("No runs to audit."
              + ("  (SEE THE TRUNCATION WARNING — this may be a cut-off "
                 "query, not an empty window.)" if listing_warning else ""))
        return 0

    by_wf: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    print(f"{'run':<12} {'workflow':<14} {'started':<17} {'verdict':<10} findings")
    print("-" * 100)
    seen_days: set = set()
    for r in runs:
        rid = str(r["databaseId"])
        log = fetch_log(rid)
        facts = extract(log)
        # Run metadata, not log-derived: the only evidence that
        # separates "still running" from "log unavailable".
        facts["run_status"] = r.get("status")
        # Only the DAY'S FIRST run of a workflow exercises discovery from cold.
        # A same-day re-run legitimately finds no NEW fixtures, because the
        # first run already added them — so applying the per-source check to
        # every run fires on 2026-03-03, a day discovery was working fine.
        _day = (r["workflow"], (r.get("startedAt") or "")[:10])
        facts["is_first_run_of_day"] = _day not in seen_days
        seen_days.add(_day)
        hits = assertions(facts, by_wf[r["workflow"]])
        by_wf[r["workflow"]].append(facts)
        v = verdict(facts, hits, log)
        # STAGE 19 item 2: per-source discovery figures are printed on EVERY
        # daily-picks row, verdict or not, and belong in the ledger note.
        # The AGGREGATE is the number that lied for three months: a healthy
        # total hid a dead source for 88 days. A reader must not have to
        # reconstruct which source produced what.
        disc = discovery_summary(facts)
        print(f"{rid:<12} {r['workflow']:<14} {(r.get('startedAt') or '')[:16]:<17} "
              f"{v:<10} {((disc + '  ') if disc else '') + '; '.join(hits)}"[:170])
        for h in hits[1:]:
            print(f"{'':<56} {h[:60]}")
    if listing_warning:
        # Printed AGAIN under the table: the number a reader carries away is
        # the one at the bottom of a long listing, and that is the number the
        # truncation qualifies.
        print(f"\n!! {listing_warning}")
        print(f"!! {len(runs)} run(s) listed above is a LOWER BOUND.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
