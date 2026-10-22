"""Was a model actually written, and how old is the newest one? — 2026-10-22.

WHY THIS EXISTS. On 2026-10-22 the "ML training step killed by CI timeout"
Telegram alert was removed as run status, and the removal left a hole that was
named out loud rather than papered over:

  * the retrain step carries `continue-on-error: true`, so a failed retrain
    does NOT redden the run and GitHub notifies nobody;
  * `_ml_models_stale()` repairs a SINGLE failure on the next run;
  * a PERSISTENT failure had no detector at all. Staleness was
    `logger.info`-only, and the ensemble degrades quietly to Poisson + Elo.

THIS DETECTOR IS ON THE OUTCOME, NOT THE STEP. It does not ask whether a step
exited zero; it asks whether an artefact exists, what timestamp it carries,
and which run first produced that timestamp. A step that succeeds while
writing nothing is the shape the removed alert could never have caught.

IT IS CONTENT, NOT RUN STATUS. "The model has not been retrained since X" is a
fact about the system that no run's conclusion reports, so under the rule
established on 2026-10-22 it goes to Telegram. It also carries a different
number every day it stays broken -- the age -- so it cannot become the
repeated identical alarm that `pipeline_loss_alert` was written to replace.

THRESHOLD, DERIVED FROM THE CADENCE RATHER THAN CHOSEN. A retrain is attempted
on the first run after the model passes `models.ml_retrain_days` (default 3),
so in healthy operation the maximum age is 3 days plus one scheduling gap.

    `measured` 2026-10-22 over n=183 gaps between consecutive SCHEDULED
    daily-picks runs (2026-05-22 .. 2026-10-22):
        p50 0.97  p90 1.02  p95 1.04  p98 1.28  p99 1.95  max 9.18 days

    threshold = ceil(ml_retrain_days + p99 gap) = ceil(3 + 1.95) = 5 days

The p100 gap of 9.18 days is the 2026-06-01 outage, not a cadence: during an
outage nothing runs at all and `pipeline_loss_alert` already fires, so this
threshold deliberately does not stretch to cover it. p95 and p99 both ceil to
5, so the figure does not depend on which quantile is picked -- worth saying,
because elsewhere in this project that choice is still open.
"""

from __future__ import annotations

import datetime as _dt
import json as _json
import math as _math
import os as _os
import pathlib as _pathlib
import pickle as _pickle
import sys as _sys
from typing import Dict, List, Optional, Tuple

MODELS_DIR = _pathlib.Path("data/models")

#: The artefacts the ensemble loads. Both, because one of them going missing
#: while the other is written is exactly the partial failure a step-outcome
#: check cannot see.
ARTEFACTS = ("ml_models.pkl", "goals_model.pkl")

#: Where this script remembers which run first OBSERVED each `trained_at`.
#: The pickle does not record its writing run and adding a key to it would
#: change the signed bytes for a reporting nicety, so the run is recorded
#: BESIDE the artefact instead. The freshness check runs in the same job as
#: `--train`, so "first observed" is the writing run whenever the timestamp
#: moved.
SIDECAR = MODELS_DIR / "retrain_provenance.json"

#: `models.ml_retrain_days`'s default. Read from config when available.
DEFAULT_RETRAIN_DAYS = 3

#: `measured` 2026-10-22, n=183 scheduled-run gaps. See the module docstring.
SCHEDULING_GAP_P99_DAYS = 1.95
SCHEDULING_GAP_PROVENANCE = (
    "p99 of n=183 gaps between consecutive SCHEDULED daily-picks runs, "
    "measured 2026-10-22 over 2026-05-22..2026-10-22; p50 0.97, p95 1.04, "
    "p99 1.95, max 9.18 (the 2026-06-01 outage, deliberately not covered)")


def stale_threshold_days(retrain_days: int = DEFAULT_RETRAIN_DAYS) -> int:
    """The age at which a model is a finding. DERIVED, not chosen."""
    return int(_math.ceil(retrain_days + SCHEDULING_GAP_P99_DAYS))


def _retrain_days() -> int:
    """`models.ml_retrain_days` from config, or the documented default.

    THE CONFIG IS OPTIONAL HERE ON PURPOSE. This script must still answer in a
    job that never built a config, and a missing config is not a reason to
    report a model as fresh.
    """
    try:
        import yaml
        for name in ("config/config.yaml", "config/config.example.yaml"):
            p = _pathlib.Path(name)
            if p.is_file():
                cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                v = (cfg.get("models") or {}).get("ml_retrain_days")
                if isinstance(v, int) and v > 0:
                    return v
    except Exception:                                      # noqa: BLE001
        pass
    return DEFAULT_RETRAIN_DAYS


def read_artefacts(models_dir: Optional[_pathlib.Path] = None) -> List[dict]:
    """One row per artefact: `{name, present, trained_at, error}`.

    `None` RESOLVED AT CALL TIME, not `= MODELS_DIR` in the signature. A
    default argument is bound when the function is DEFINED, so the first
    version could not be redirected at all -- the tests pointed
    `mf.MODELS_DIR` at a temp directory and the function went on reading the
    real one, reporting the 205-day working copy from inside every fixture.
    A default that cannot be substituted makes a function untestable, and it
    silently ignores any later reassignment in production too.

    THREE STATES, KEPT APART. An artefact that is ABSENT is not an artefact
    that is OLD, and neither is one that is present but unreadable -- "we
    could not look" is not "we looked and it is fine".
    """
    models_dir = MODELS_DIR if models_dir is None else models_dir
    out = []
    for name in ARTEFACTS:
        p = models_dir / name
        row = {"name": name, "present": p.is_file(), "trained_at": None,
               "error": None}
        if not row["present"]:
            row["error"] = "artefact ABSENT - no model has been written here"
            out.append(row)
            continue
        try:
            state = _pickle.loads(p.read_bytes())
            ta = state.get("trained_at") if isinstance(state, dict) else None
            if not ta:
                row["error"] = ("artefact present but carries NO trained_at, "
                                "so its age is UNKNOWN, not zero")
            else:
                row["trained_at"] = str(ta)
        except Exception as exc:                           # noqa: BLE001
            row["error"] = (f"artefact present but UNREADABLE "
                            f"({type(exc).__name__}) - its age is UNKNOWN")
        out.append(row)
    return out


def _as_aware(ts: str) -> Optional[_dt.datetime]:
    try:
        d = _dt.datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)


def record_provenance(rows: List[dict], run_id: str,
                      sidecar: Optional[_pathlib.Path] = None) -> dict:
    """Remember which run first saw each `trained_at`; return what is known.

    Merged and rewritten atomically, the same shape as the briefings-sent
    marker, because two jobs could in principle observe the same cache.
    """
    sidecar = SIDECAR if sidecar is None else sidecar
    try:
        known = _json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(known, dict):
            known = {}
    except Exception:                                      # noqa: BLE001
        known = {}
    changed = False
    for r in rows:
        ta = r.get("trained_at")
        if not ta:
            continue
        entry = known.get(r["name"])
        if not isinstance(entry, dict) or entry.get("trained_at") != ta:
            known[r["name"]] = {
                "trained_at": ta,
                "first_seen_run": run_id or "UNKNOWN (no GITHUB_RUN_ID)",
                "first_seen_at": _dt.datetime.now(
                    _dt.timezone.utc).isoformat(timespec="seconds"),
            }
            changed = True
    if changed:
        try:
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            tmp = sidecar.with_suffix(".tmp")
            tmp.write_text(_json.dumps(known, indent=2), encoding="utf-8")
            _os.replace(tmp, sidecar)
        except OSError as exc:                             # pragma: no cover
            print(f"could not record retrain provenance: {exc}")
    return known


def assess(rows: List[dict], known: dict, now: _dt.datetime,
           retrain_days: int = DEFAULT_RETRAIN_DAYS) -> dict:
    """`{'state', 'threshold_days', 'oldest_age_days', 'findings', 'lines'}`.

    `state` is one of FRESH / STALE / UNKNOWN. UNKNOWN is a finding too: an
    artefact we could not read is not a fresh one.
    """
    threshold = stale_threshold_days(retrain_days)
    findings, lines = [], []
    ages = []
    for r in rows:
        prov = known.get(r["name"]) or {}
        run = prov.get("first_seen_run", "UNKNOWN")
        if r["error"]:
            findings.append(f"{r['name']}: {r['error']}")
            lines.append(f"  {r['name']:16s} UNKNOWN - {r['error']}")
            continue
        ta = _as_aware(r["trained_at"])
        if ta is None:
            findings.append(f"{r['name']}: trained_at "
                            f"{r['trained_at']!r} is unparseable, so the age "
                            f"is UNKNOWN")
            lines.append(f"  {r['name']:16s} UNKNOWN - unparseable stamp "
                         f"{r['trained_at']!r}")
            continue
        age = (now - ta).total_seconds() / 86400.0
        ages.append(age)
        mark = "STALE" if age > threshold else "fresh"
        lines.append(f"  {r['name']:16s} {mark:5s} age {age:5.1f}d  "
                     f"trained_at {r['trained_at'][:19]}  written by run "
                     f"{run}")
        if age > threshold:
            findings.append(
                f"{r['name']} is {age:.1f} days old against a threshold of "
                f"{threshold} - trained_at {r['trained_at'][:19]}, written by "
                f"run {run}")
    if findings:
        state = "STALE" if ages and max(ages) > threshold else "UNKNOWN"
    else:
        state = "FRESH"
    return {"state": state, "threshold_days": threshold,
            "oldest_age_days": max(ages) if ages else None,
            "findings": findings, "lines": lines}


def format_alert(a: dict, retrain_days: int) -> str:
    """The Telegram body. CONTENT: what the system is, not how a run ended."""
    # PLAIN ASCII, DELIBERATELY. The first version opened with an emoji and
    # `print(body)` raised UnicodeEncodeError on a cp1251 console -- a
    # detector that dies on its own output is worse than no detector.
    # `pipeline_loss_alert` carries no emoji either, for the same reason.
    head = ("THE MODEL HAS NOT BEEN RETRAINED."
            if a["state"] == "STALE"
            else "THE MODEL'S AGE IS UNKNOWN.")
    age = (f"{a['oldest_age_days']:.1f} days"
           if a["oldest_age_days"] is not None else "UNKNOWN")
    return chr(10).join([
        head,
        f"Oldest artefact: {age}, against a threshold of "
        f"{a['threshold_days']} days.",
        "",
        "THE CONSEQUENCE: the 1X2 and goals ML components are loaded from a "
        "stale fit, or not loaded at all. The ensemble still produces picks "
        "from Poisson + Elo, so NOTHING LOOKS BROKEN - that is why this is "
        "reported rather than left to a run's colour.",
        "",
        *a["lines"],
        "",
        f"THRESHOLD IS DERIVED, not chosen: ceil(ml_retrain_days="
        f"{retrain_days} + {SCHEDULING_GAP_P99_DAYS} scheduling gap) = "
        f"{a['threshold_days']} days.",
        f"  gap provenance: {SCHEDULING_GAP_PROVENANCE}",
        "",
        "CLEARING ACTION: run `python -m src.agent.betting_agent --train "
        "--force` and read the retrain step's log. The retrain step carries "
        "continue-on-error, so a failure there does NOT redden the run and "
        "no notification is sent for it.",
    ])


def main(argv: Optional[List[str]] = None) -> int:
    """`0` fresh, `1` a finding. Never raises on a missing model directory."""
    argv = list(_sys.argv[1:] if argv is None else argv)
    quiet = "--quiet" in argv
    retrain_days = _retrain_days()
    rows = read_artefacts()
    known = record_provenance(rows, _os.environ.get("GITHUB_RUN_ID", ""))
    a = assess(rows, known, _dt.datetime.now(_dt.timezone.utc), retrain_days)

    print(f"model_freshness[{a['state']}] threshold={a['threshold_days']}d "
          f"(ml_retrain_days={retrain_days} + p99 gap "
          f"{SCHEDULING_GAP_P99_DAYS}d)")
    for line in a["lines"]:
        print(line)
    if a["state"] == "FRESH":
        # SILENT ON SUCCESS. The whole point of the rule it was written under.
        print("no finding: every artefact is within the derived threshold.")
        return 0

    body = format_alert(a, retrain_days)
    print(body)
    if not quiet:
        try:
            from src.reporting.alert_delivery import deliver_alert
            result = deliver_alert(body)
            print(f"model-freshness alert delivered={result.ok} "
                  f"attempts={result.attempts}")
        except Exception as exc:                           # noqa: BLE001
            # A delivery failure must not hide the finding.
            print(f"model-freshness alert NOT delivered ({type(exc).__name__})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
