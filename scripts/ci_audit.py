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


def audited_run_ids() -> set:
    """Run ids already carrying a verdict in the ledger."""
    if not LEDGER.exists():
        return set()
    return set(re.findall(r"^\|\s*(\d{9,})\s*\|", LEDGER.read_text(
        encoding="utf-8"), re.M))


def list_runs(since: Optional[str], until: Optional[str],
              limit: int = 250) -> List[dict]:
    """`limit` is per workflow and must outrun the busiest one.

    closing-lines fires roughly every two hours, so 40 covered five days and
    silently dropped everything older — which read as "no closing-lines runs in
    the window" rather than as a truncated query.
    """
    out = []
    for wf in WORKFLOWS:
        raw = _sh("gh", "run", "list", "--workflow", wf, "--limit", str(limit),
                  "--json", "databaseId,startedAt,conclusion,event,status")
        try:
            rows = json.loads(raw or "[]")
        except json.JSONDecodeError:
            rows = []
        for r in rows:
            day = (r.get("startedAt") or "")[:10]
            if since and day < since:
                continue
            if until and day > until:
                continue
            r["workflow"] = wf.replace(".yml", "")
            out.append(r)
    return sorted(out, key=lambda r: r.get("startedAt") or "")


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
    # Stage 19. The audit was blind to a day that discovered nothing: 2026-08-26
    # analysed 0 fixtures against a card of six real matches and was flagged
    # only for the API-Football suspension. It surfaced because Niki looked at a
    # football calendar.
    #
    # Keyed on the scraper's OWN warning, which already excludes
    # `off_season_leagues` — so a genuinely dormant league does not fire it, and
    # the assertion needs no threshold of its own to get wrong.
    "fixtures_zero_active": r"returned 0 fixtures for \S+ . expected .1 for active season",
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
                   "no_fixtures_at_all"):
            f[key] = len(ms)
        elif key == "injuries_saved":
            f["injuries_saved"] = int(ms[-1][0])
            f["injury_fixtures"] = int(ms[-1][1])
        else:
            try:
                f[key] = int(ms[-1] if isinstance(ms[-1], str) else ms[-1][0])
            except (TypeError, ValueError):
                pass
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
        for key, label in (("src_flashscore_fixtures", "Flashscore fixtures"),
                           ("src_footballdataorg_fixtures", "football-data.org fixtures"),
                           ("src_apifootball_fixtures", "API-Football fixtures")):
            if key in facts and (facts.get(key) or 0) == 0 and produced_recently(key):
                hits.append(
                    f"{label} = 0 while other sources still produce — this "
                    f"source produced within the last {LOOKBACK_RUNS} runs")

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

    if facts.get("tracebacks"):
        hits.append(f"{facts['tracebacks']} traceback(s) in the log")
    if facts.get("account_suspended"):
        hits.append("API-Football reported the account suspended")
    if facts.get("telegram_failed"):
        hits.append(f"{facts['telegram_failed']} alert(s) failed to deliver")
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
    """
    keys = (("fs", "src_flashscore_fixtures"),
            ("fdo", "src_footballdataorg_fixtures"),
            ("af", "src_apifootball_fixtures"))
    if not any(k in facts for _, k in keys):
        return ""
    parts = [f"{label}={facts[k] if k in facts else '-'}" for label, k in keys]
    return "disc[" + " ".join(parts) + "]"


def verdict(facts: Dict[str, object], hits: List[str]) -> str:
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
        runs = [{"databaseId": int(a.run), "workflow": "?",
                 "startedAt": "", "conclusion": "?"}]
    else:
        runs = list_runs(a.since, a.until, a.limit)
        if a.unaudited:
            done = audited_run_ids()
            runs = [r for r in runs if str(r["databaseId"]) not in done]

    if not runs:
        print("No runs to audit.")
        return 0

    by_wf: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    print(f"{'run':<12} {'workflow':<14} {'started':<17} {'verdict':<10} findings")
    print("-" * 100)
    seen_days: set = set()
    for r in runs:
        rid = str(r["databaseId"])
        facts = extract(fetch_log(rid))
        # Only the DAY'S FIRST run of a workflow exercises discovery from cold.
        # A same-day re-run legitimately finds no NEW fixtures, because the
        # first run already added them — so applying the per-source check to
        # every run fires on 2026-03-03, a day discovery was working fine.
        _day = (r["workflow"], (r.get("startedAt") or "")[:10])
        facts["is_first_run_of_day"] = _day not in seen_days
        seen_days.add(_day)
        hits = assertions(facts, by_wf[r["workflow"]])
        by_wf[r["workflow"]].append(facts)
        v = verdict(facts, hits)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
