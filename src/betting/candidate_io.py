"""Serialise candidate recommendations between a shard and the collect step.

Why this exists
---------------
Analysis is per-fixture and embarrassingly parallel; the phase that follows is
not. Ranking, the per-match cap, correlation filtering and — critically — the
daily exposure cap (``betting.max_total_kelly_pct``) are properties of the whole
day's slate. Four shards each applying a 40% cap would stake 160%.

So shards emit *candidates* and a single collect step runs the portfolio phase
over the union. This module is the wire format between them: a plain JSON file
per shard, written to the workspace and passed between jobs as an artifact.

The format is deliberately dumb — a dict per recommendation, one schema version,
no pickles. It crosses a process and a job boundary, so it has to be
inspectable, diffable in a failed-run artifact, and safe to load from a file
another job produced.
"""

from __future__ import annotations

import json
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from src.betting.value_calculator import BetRecommendation
from src.utils.logger import get_logger, utcnow

logger = get_logger()

SCHEMA_VERSION = 1

# The one field that is not a JSON scalar.
_DATETIME_FIELDS = {"match_date"}


def _encode(rec: BetRecommendation) -> dict:
    out = {}
    for f in fields(rec):
        value = getattr(rec, f.name)
        if f.name in _DATETIME_FIELDS and isinstance(value, datetime):
            value = value.isoformat()
        out[f.name] = value
    return out


def _decode(row: dict) -> BetRecommendation:
    known = {f.name for f in fields(BetRecommendation)}
    # Ignore unknown keys rather than raising: a shard running a slightly older
    # revision than the collector must degrade, not abort the day.
    kwargs = {k: v for k, v in row.items() if k in known}
    for name in _DATETIME_FIELDS:
        raw = kwargs.get(name)
        if isinstance(raw, str):
            try:
                kwargs[name] = datetime.fromisoformat(raw)
            except ValueError:
                kwargs[name] = None
    return BetRecommendation(**kwargs)


def write_candidates(path, recommendations: Iterable[BetRecommendation],
                     shard: int = None, shards: int = None) -> Path:
    """Write one shard's candidates. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "shard": shard,
        "shards": shards,
        "written_at": utcnow().isoformat(),
        "recommendations": [_encode(r) for r in recommendations],
    }
    # Atomic: the collect step may start reading while a slower shard is still
    # writing its file into a shared directory.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(path)
    logger.info(
        f"Wrote {len(payload['recommendations'])} candidate(s) to {path}"
        + (f" (shard {shard}/{shards})" if shards else "")
    )
    return path


def read_candidates(paths: Iterable) -> List[BetRecommendation]:
    """Load and concatenate candidates from every shard file.

    A missing or unreadable file is logged and skipped rather than fatal: losing
    one shard should cost that shard's picks, not the entire day.
    """
    out: List[BetRecommendation] = []
    for raw in paths:
        p = Path(raw)
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.warning(f"Candidate file missing, skipping: {p}")
            continue
        except Exception as e:
            logger.error(f"Candidate file unreadable ({p}): {e} — skipping")
            continue

        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            logger.error(
                f"Candidate file {p} has schema {version}, expected "
                f"{SCHEMA_VERSION} — skipping (shard/collector version skew?)"
            )
            continue

        rows = payload.get("recommendations", [])
        out.extend(_decode(r) for r in rows)
        logger.info(
            f"Loaded {len(rows)} candidate(s) from {p} "
            f"(shard {payload.get('shard')}/{payload.get('shards')})"
        )
    return out


def discover(directory) -> List[Path]:
    """All candidate files in a directory, in deterministic shard order.

    Ordering matters: the portfolio phase sorts by EV x confidence, and ties
    would otherwise resolve by whichever shard's file the filesystem listed
    first, making the day's picks depend on directory iteration order.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("candidates-*.json"))
