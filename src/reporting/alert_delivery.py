"""The one alert delivery path. Stdlib only — no third-party imports, ever.

Stage 14, DEL-1.

WHY THIS EXISTS
---------------
Two observed failures, different causes, different code paths, identical class:

    2026-08-11  run 31482430418   alert built, undelivered — HTTP 400
    2026-08-23  run 32646469497   "Failed to send Telegram message: Timed out"

The first was fixed by adding `scripts/ci_alert.py` — a second sender to the
same last hop. The class survived the remedy, because the second failure landed
in the OTHER sender, `TelegramNotifier._send_message`, which had no retry and
whose entire surface was a `logger.error` its caller discarded.

An alert whose only channel can fail silently is not an alert. It is a log line
with ambition.

WHY IT IS STDLIB-ONLY, AND WHY THAT IS THE WHOLE POINT
------------------------------------------------------
`ci_alert.py` documented its own duplication as deliberate: `telegram_bot.py`
imports `python-telegram-bot`, and the closing-lines and paper-trading-report
workflows install a minimal dependency set without it, so importing the notifier
there would fail or force a full install.

That reasoning is correct, and it rejected consolidating in one direction only.
Consolidating the OTHER way works: a stdlib-only module can be imported by both,
because `src/reporting/__init__.py` is a docstring and nothing else. Keep it
that way — an import of `python-telegram-bot` here silently breaks three
workflows.

WHAT A DELIVERY GUARANTEE MEANS HERE
------------------------------------
1. **Retry with backoff** on transient failures — timeouts, DNS, 5xx. NOT on a
   4xx, which is a permanent error that retrying only delays.
2. **A surface that does not depend on Telegram**: `::error::` annotations and
   the GitHub step summary, both visible in the run without opening a log.
3. **The outcome is returned, not just logged**, so "alert fired" and "alert
   arrived" stop being the same line.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not fail the workflow step. `ci_alert.py` argued this and is right: an
alert is an observability channel, not the thing under test. A delivery failure
must never turn a red run green — and must never turn a green run red either,
or the channel becomes a source of false failures. The annotation is the
surface; the exit code stays 0.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

API = "https://api.telegram.org/bot{token}/sendMessage"
TIMEOUT = 10

#: Attempts, not retries: 1 initial + 2 more.
MAX_ATTEMPTS = 3
#: Seconds before attempt 2 and 3. Short — an alert that arrives late is still
#: an alert, but a CI step blocked for a minute on a dead channel is a cost.
BACKOFF = (2.0, 5.0)


@dataclass
class DeliveryResult:
    """What actually happened. Returned so callers can act on it."""

    ok: bool
    attempts: int
    detail: str = ""
    surfaced: bool = False

    def __bool__(self) -> bool:
        return self.ok


def _annotate(msg: str) -> bool:
    """GitHub Actions annotation + step summary. Visible without the log.

    Returns whether a Telegram-independent surface was actually written.
    """
    print(f"::error::{msg}")
    surfaced = True
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(f"\n> **ALERT NOT DELIVERED** — {msg}\n")
        except Exception:
            # The annotation already printed; a summary failure is not fatal.
            pass
    return surfaced


def _post(token: str, chat_id: str, text: str):
    """(ok, detail, transient). Content-Type is explicit — that was the 08-11 bug.

    `urlopen` defaults an unlabelled body to form-encoding, so Telegram parsed
    the JSON as form data, found no `text` field, and answered HTTP 400.
    """
    req = urllib.request.Request(
        API.format(token=token),
        data=json.dumps({"chat_id": chat_id, "text": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return True, {"status": resp.status}, False
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode() or "{}")
        except Exception:
            body = {}
        # 5xx is Telegram having a bad moment; 4xx is us being wrong.
        return False, (body or {"error_code": e.code}), e.code >= 500
    except Exception as e:
        # Timeout, DNS, connection reset — the 08-23 class. Always transient.
        return False, {"description": f"{type(e).__name__}: {e}"}, True


def deliver_alert(
    text: str,
    *,
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> DeliveryResult:
    """Deliver one alert, or say loudly and specifically why it did not arrive."""
    token = (token if token is not None
             else os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
    chat_id = (chat_id if chat_id is not None
               else os.environ.get("TELEGRAM_CHAT_ID", "")).strip()

    if not token or not chat_id:
        # Never print which is missing beyond the name — no values.
        surfaced = _annotate(
            "alert NOT sent: TELEGRAM_BOT_TOKEN and/or TELEGRAM_CHAT_ID is "
            "not set on this step. The step needs both in its `env:` block.")
        return DeliveryResult(False, 0, "not configured", surfaced)

    detail: dict = {}
    attempt = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        ok, detail, transient = _post(token, chat_id, text)
        if ok:
            return DeliveryResult(True, attempt, "delivered")

        migrated = (detail.get("parameters") or {}).get("migrate_to_chat_id")
        if migrated:
            # A group upgraded to a supergroup and its id changed. Retry against
            # the id Telegram supplied so today's alert still arrives, and say
            # loudly that the stored secret is stale. Never write the new id
            # anywhere: rotating a secret is a human decision, and inventing one
            # would hide a configuration problem instead of surfacing it.
            _annotate(
                "TELEGRAM_CHAT_ID is STALE: the group became a supergroup and "
                "its id changed. Retrying with the id Telegram returned. "
                "ACTION REQUIRED: update the secret — this is a stopgap.")
            ok2, detail2, _ = _post(token, str(migrated), text)
            if ok2:
                return DeliveryResult(True, attempt + 1, "delivered after migration")
            detail = detail2

        if not transient or attempt == MAX_ATTEMPTS:
            break
        sleep(BACKOFF[min(attempt - 1, len(BACKOFF) - 1)])

    desc = detail.get("description") or detail
    # The real count, not MAX_ATTEMPTS. A permanent 4xx breaks after one try,
    # and a result that reported three would be lying about the one thing this
    # function exists to record.
    surfaced = _annotate(
        f"alert NOT delivered after {attempt} attempt(s). "
        f"Telegram said: {desc}")
    return DeliveryResult(False, attempt, str(desc), surfaced)
