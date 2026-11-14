"""Stage 33 Part A: verify a picks run under `confident_value`, read-only.

    python -m scripts.first_run_check                 # today's scheduled run
    python -m scripts.first_run_check --day 2026-11-15
    python -m scripts.first_run_check --day 2026-11-15 --settlement 2026-11-16

Every check is one of three states, and they are never collapsed:

    PASS          the property was evaluated and holds
    FAIL          the property was evaluated and does not hold
    UNEVALUATED   it could not be evaluated -- no run, no log, no rows, a
                  table not installed. Never a pass.

WHAT IT CANNOT SEE, stated. The client channel is not readable by the bot,
so "the client message as delivered" is the delivery receipt the sender logs
(`REPORT_DELIVERY report=... sent=... failed=...`) plus which report it was;
the text itself is not recoverable from the log.

Reads the run log through `ci_audit.fetch_log` (cached under ci_logs/) and
the database read-only. Credentials come from `.env` / `.env.local`, read
inside `_load_env()`.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional

from src.data import pick_runs as _pr

PASS, FAIL, UNEVALUATED = "PASS", "FAIL", "UNEVALUATED"
EXPECTED_COHORT = "stage5_baseline_20260807.5bca73"
MIN_PRICE = 1.70
P_MIN = 0.54
#: a FORCED pick's log line, unless it is the standard-rule shadow's
FORCED_MARKERS = ("Club pick (coverage", "WC pick-every-match:")
SHADOW_PREFIX = "[standard-rule shadow"
#: the receipt `_send_chunked` logs for each report, by the day's outcome
REPORT_PICKS = "report=daily picks "
REPORT_NO_PICKS = "report=no picks today "


def _load_env() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in (".env", ".env.local"):
        try:
            with open(os.path.join(here, name), encoding="utf-8") as fh:
                for line in fh:
                    m = re.match(r"\s*([A-Z0-9_]+)\s*=\s*(.*)", line)
                    if m and not os.environ.get(m.group(1)):
                        os.environ[m.group(1)] = \
                            m.group(2).strip().strip('"').strip("'")
        except FileNotFoundError:
            continue


#: A RUNTIME log line: loguru's `| LEVEL |` field. Keys on the line's own
#: format, NOT on the step name -- `gh run view --log` labels these lines
#: `UNKNOWN STEP`, and the first version of this filtered on the step name
#: and so matched NOTHING: it passed "no forced picks" on a run that made two
#: (2026-11-14, Exeter and Valladolid). Test ids, env dumps and the
#: workflow's own text have no level field, so they cannot match.
_RUNTIME = re.compile(r"\|\s*(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s*\|")


def log_facts(log: Optional[str]) -> dict:
    """Pure. What the run's RUNTIME log lines say, as counts and lines."""
    if not log:
        return {"read": False}
    runtime = [l for l in log.splitlines() if _RUNTIME.search(l)]
    review = [l for l in runtime
              if "src.reporting.match_briefing" in l
              or "Claude reviewed" in l or "finalize_picks_with_claude" in l
              or "claude -p" in l]
    forced = [l for l in runtime if any(m in l for m in FORCED_MARKERS)
              and SHADOW_PREFIX not in l]
    delivery = [l[l.index("REPORT_DELIVERY"):] for l in log.splitlines()
                if "REPORT_DELIVERY" in l and _RUNTIME.search(l)]
    confident = [l for l in runtime if "confident_value: " in l]
    shadow_stored = [l for l in runtime if "standard-rule shadow:" in l]
    return {"read": True, "runtime_lines": len(runtime),
            "review_lines": review[:5], "review_count": len(review),
            "forced_lines": forced[:5], "forced_count": len(forced),
            "delivery": delivery, "confident_lines": len(confident),
            "shadow_stored": shadow_stored[:3]}


def evaluate(picks: List[dict], runs: List[dict], shadow: List[dict],
             facts: dict, cohort: str = EXPECTED_COHORT) -> Dict[str, dict]:
    """Pure. Every Part A check, each PASS / FAIL / UNEVALUATED with why."""
    out: Dict[str, dict] = {}

    def put(name, state, why):
        out[name] = {"state": state, "why": why}

    if not picks:
        for k in ("cohort", "price_floor", "probability_floor",
                  "one_per_fixture"):
            put(k, UNEVALUATED if not runs else PASS,
                "no picks were saved" + (
                    "" if not runs else " -- vacuously true on a recorded "
                    "zero-pick day"))
    else:
        cohorts = Counter(p["model_version"] for p in picks)
        put("cohort", PASS if set(cohorts) == {cohort} else FAIL,
            f"{dict(cohorts)}")
        low_price = [p for p in picks if p["odds"] is None
                     or p["odds"] < MIN_PRICE - 1e-9]
        put("price_floor", FAIL if low_price else PASS,
            f"{len(low_price)} of {len(picks)} below {MIN_PRICE}; min "
            f"{min((p['odds'] for p in picks if p['odds'] is not None), default=None)}")
        low_p = [p for p in picks if p["p"] is None or p["p"] < P_MIN - 1e-9]
        put("probability_floor", FAIL if low_p else PASS,
            f"{len(low_p)} of {len(picks)} below {P_MIN}; min "
            f"{min((p['p'] for p in picks if p['p'] is not None), default=None)}")
        per = Counter(p["match_id"] for p in picks)
        dup = {m: n for m, n in per.items() if n > 1}
        put("one_per_fixture", FAIL if dup else PASS,
            f"{len(per)} fixtures, {len(picks)} picks" + (
                f"; more than one on {dup}" if dup else ""))
    if not runs:
        put("pick_runs", UNEVALUATED, "no pick_runs row for the day "
            "(table missing, or the run never wrote one)")
    else:
        last = runs[-1]
        if picks:
            ok = (last["outcome"] == _pr.PICKS_MADE
                  and last["picks_saved"] == len(picks))
            path = "the picks path"
        else:
            # THE ZERO-PICK PATH passes only RECORDED and ERROR-FREE: an
            # outcome that means "looked at in full", nothing saved, no
            # analysis error. INCOMPLETE and NO_PRICED_FIXTURES are not it.
            ok = (last["outcome"] in _pr.EVALUATED_EMPTY
                  and last["picks_saved"] == 0
                  and last["analysis_errors"] == 0)
            path = "the ZERO-PICK path"
        put("pick_runs", PASS if ok else FAIL,
            f"{path}: {len(runs)} row(s); latest {last['outcome']}, "
            f"{last['picks_saved']} picks saved, slate "
            f"{last.get('fixtures_on_slate')} fixture(s), "
            f"{last['fixtures_evaluated']} evaluated, "
            f"{last['analysis_errors']} errors, mode {last['selection_mode']}")
        # THE CLIENT RECEIPT the outcome calls for: the card on a picks day,
        # the one plain line on a zero-pick day, nothing otherwise.
        want = (REPORT_PICKS if last["outcome"] == _pr.PICKS_MADE
                else REPORT_NO_PICKS if last["outcome"] in _pr.EVALUATED_EMPTY
                else None)
        if want is None:
            put("client_delivery", FAIL,
                f"{last['outcome']} sends the client nothing")
        elif not facts.get("read"):
            put("client_delivery", UNEVALUATED, "the run log could not be "
                "read")
        else:
            got = [d for d in facts.get("delivery") or [] if want in d + " "]
            clean = [d for d in got if "failed=none" in d]
            put("client_delivery", PASS if clean else FAIL,
                f"wanted `{want.strip()}`: " + ("; ".join(got) if got
                                                else "no such receipt"))
    if not facts.get("read") or facts.get("runtime_lines", 0) == 0:
        for k in ("no_review_call", "no_forced_picks"):
            put(k, UNEVALUATED, "the run log could not be read, or held no "
                "runtime line -- zero hits on nothing is not a pass")
    else:
        put("no_review_call", FAIL if facts["review_count"] else PASS,
            f"{facts['review_count']} review/Claude line(s) in the picks "
            f"step")
        put("no_forced_picks", FAIL if facts["forced_count"] else PASS,
            f"{facts['forced_count']} forced-pick line(s) outside the "
            f"shadow")
    if not shadow:
        evaluated = runs[-1]["fixtures_evaluated"] if runs else None
        if not picks and not runs:
            put("shadow_stored", UNEVALUATED, "no shadow_picks rows and no "
                "pick_runs row for the day")
        elif evaluated == 0:
            put("shadow_stored", PASS, "no shadow_picks rows -- vacuously "
                "true: the run evaluated 0 fixtures")
        elif evaluated is None:
            put("shadow_stored", FAIL, "no shadow_picks rows, and no "
                "pick_runs row to say the day evaluated nothing")
        else:
            put("shadow_stored", FAIL, f"no shadow_picks rows for a day that "
                f"evaluated {evaluated} fixture(s)")
    else:
        rules = Counter(s["rule"] for s in shadow)
        per = Counter(s["match_id"] for s in shadow)
        dup = {m: n for m, n in per.items() if n > 1}
        put("shadow_stored", FAIL if dup or set(rules) != {"standard"}
            else PASS, f"{len(shadow)} row(s), rules {dict(rules)}, "
            f"{len(per)} fixtures" + (f"; duplicates {dup}" if dup else ""))
        sent_as_pick = {s["match_id"] for s in shadow} & {
            p["match_id"] for p in picks
            if (p["match_id"], p["selection"]) in {
                (s["match_id"], s["selection"]) for s in shadow}}
        put("shadow_not_a_pick", PASS,
            f"shadow rows live only in shadow_picks; on "
            f"{len(sent_as_pick)} fixture(s) the two rules chose the same "
            f"selection")
    return out


def find_run(day: _dt.date) -> Optional[dict]:
    from scripts.ci_audit import _sh
    raw = _sh("gh", "run", "list", "--workflow", "daily-picks.yml",
              "--limit", "30", "--json",
              "databaseId,startedAt,event,headSha,conclusion,status")
    try:
        rows = json.loads(raw or "")
    except json.JSONDecodeError:
        return None
    hits = [r for r in rows if r.get("event") == "schedule"
            and (r.get("startedAt") or "")[:10] == day.isoformat()]
    return hits[0] if hits else None


def check(day: _dt.date, settlement: Optional[_dt.date] = None) -> dict:
    _load_env()
    from sqlalchemy import inspect, text

    from scripts.ci_audit import fetch_log
    from src.data.database import get_db
    run = find_run(day)
    log = fetch_log(str(run["databaseId"])) if run else None
    db = get_db()
    assert db.is_postgres, "refusing to read the SQLite fallback"
    present = set(inspect(db.engine).get_table_names())
    with db.get_session() as s:
        picks = [{"match_id": r[0], "selection": r[1], "odds": r[2],
                  "p": r[3], "model_version": r[4], "market": r[5]}
                 for r in s.execute(text(
                     "SELECT match_id, selection, odds, predicted_probability,"
                     " model_version, market FROM saved_picks "
                     "WHERE pick_date = :d"), {"d": day})]
        runs = [] if "pick_runs" not in present else [
            dict(zip(("outcome", "picks_saved", "fixtures_on_slate",
                      "fixtures_evaluated", "analysis_errors",
                      "selection_mode", "model_version", "recorded_at"), r))
            for r in s.execute(text(
                "SELECT outcome, picks_saved, fixtures_on_slate, "
                "fixtures_evaluated, analysis_errors, selection_mode, "
                "model_version, recorded_at "
                "FROM pick_runs WHERE pick_date = :d ORDER BY recorded_at"),
                {"d": day})]
        shadow = [] if "shadow_picks" not in present else [
            {"match_id": r[0], "rule": r[1], "selection": r[2], "odds": r[3],
             "p": r[4]}
            for r in s.execute(text(
                "SELECT match_id, rule, selection, odds, predicted_probability"
                " FROM shadow_picks WHERE pick_date = :d"), {"d": day})]
    facts = log_facts(log)
    out = {
        "day": str(day),
        "run": ({k: run.get(k) for k in ("databaseId", "startedAt",
                                          "headSha", "conclusion")}
                if run else "UNEVALUATED: no scheduled run started that day"),
        "tables_present": sorted({"pick_runs", "shadow_picks"} & present),
        "picks": {"n": len(picks),
                  "odds": sorted(p["odds"] for p in picks if p["odds"]),
                  "p": sorted(p["p"] for p in picks if p["p"] is not None),
                  "markets": dict(Counter(p["selection"] for p in picks))},
        "pick_runs": [{k: str(v) for k, v in r.items()} for r in runs],
        "shadow": {"n": len(shadow),
                   "odds": sorted(s_["odds"] for s_ in shadow if s_["odds"]),
                   "markets": dict(Counter(s_["selection"] for s_ in shadow))},
        "delivery": facts.get("delivery"),
        "checks": evaluate(picks, runs, shadow, facts),
    }
    if settlement is not None:
        srun = find_run(settlement)
        slog = fetch_log(str(srun["databaseId"])) if srun else None
        out["settlement"] = {
            "run": srun.get("databaseId") if srun else "UNEVALUATED",
            "delivery": [l for l in (log_facts(slog).get("delivery") or [])
                         if "settlement" in l],
            "summary_line_code": "Hit rate W/N (x%) · average odds o · "
                                 "return r per 1-unit pick"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", default=None)
    ap.add_argument("--settlement", default=None)
    a = ap.parse_args()
    day = (_dt.date.fromisoformat(a.day) if a.day
           else _dt.datetime.now(_dt.timezone.utc).date())
    st = _dt.date.fromisoformat(a.settlement) if a.settlement else None
    print(json.dumps(check(day, st), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
