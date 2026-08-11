"""Send one CI failure alert to Telegram. Stdlib only.

    python -m scripts.ci_alert "message text"

Stage 12.1, Defect 2.

WHY THIS EXISTS
---------------
The smoke test found CI correctly detecting a failed test step, trying to
notify, and getting ``HTTP Error 400`` — which the workflow then swallowed with
``print(f"Telegram alert failed: {e}")``. The alert that exists to prevent a
"green but broken" run was itself broken, and silently.

THE ACTUAL CAUSE, reproduced against the live API::

    urllib.request.urlopen(url, data=json.dumps({...}).encode())
    -> HTTP 400: "Bad Request: message text is empty"

``urlopen`` defaults an unlabelled body to
``application/x-www-form-urlencoded``. Telegram parsed the JSON as form data,
found no ``text`` field, and said so. The chat id was never the problem — the
one call site that DID set ``Content-Type`` was the only correct one, and it
happened to be the step with no ``env:`` block, so it never ran to prove it.

Hence ``_post`` sets the header explicitly. That one line is the fix.

Five near-identical inline implementations had accumulated across three
workflows, three of them missing the header and one missing its secrets.
Duplicated error handling is how that goes unnoticed for months.

WHY NOT REUSE TelegramNotifier
------------------------------
``src/reporting/telegram_bot.py`` already handles the group→supergroup
migration properly, and reusing it would be the obvious move. It cannot be used
here: it imports ``python-telegram-bot``, and the closing-lines and
paper-trading-report workflows deliberately install a minimal dependency set
(no browser, no ML, no telegram package) so they stay fast and cheap. Importing
it would either fail or force those jobs to install the full stack to send one
message.

So this is a deliberate second implementation, kept to stdlib and to the one
behaviour that matters: report what actually went wrong.

THE MIGRATION CASE (defensive, not the bug that was fixed)
----------------------------------------------------------
Verified 2026-08-11: the stored ``TELEGRAM_CHAT_ID`` is CURRENT — ``getChat``
resolves it to a live supergroup, and the app's own notifier sent through it in
CI without ever taking its migration branch. No secret rotation is needed.

The handling below is kept anyway because the app already carries the same
logic (``telegram_bot.py``), a future migration is plausible, and the failure
would otherwise look exactly like the bug above. When a Telegram group is
upgraded to a supergroup its chat id changes and the API answers a send to the
old id with::

    400 {"ok": false,
         "description": "Bad Request: group chat was upgraded to a supergroup chat",
         "parameters": {"migrate_to_chat_id": -100...}}

This retries against the id Telegram supplies, so today's alert still arrives —
and says loudly that the stored secret is stale. It does NOT write the new id
anywhere: rotating a GitHub secret is a human decision, and silently inventing a
chat id would hide a configuration problem rather than surface it.

EXIT CODE
---------
Always 0. A failure to deliver an alert must never turn a red run green, but it
must also never turn a green run red — the alert is an observability channel,
not the thing under test. Delivery failures are printed as ``::error::``
annotations so GitHub surfaces them in the run summary regardless.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

API = "https://api.telegram.org/bot{token}/sendMessage"
TIMEOUT = 10


def _annotate(msg: str) -> None:
    """GitHub Actions error annotation — visible without opening the log."""
    print(f"::error::{msg}")


def _post(token: str, chat_id: str, text: str):
    """(ok, detail). `detail` carries Telegram's own description on failure."""
    req = urllib.request.Request(
        API.format(token=token),
        data=json.dumps({"chat_id": chat_id, "text": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return True, resp.status
    except urllib.error.HTTPError as e:
        # The body is the whole point: urllib's str(e) is just "HTTP Error 400:
        # Bad Request", which is what made the original failure undiagnosable.
        try:
            body = json.loads(e.read().decode() or "{}")
        except Exception:
            body = {}
        return False, body or {"error_code": e.code}
    except Exception as e:                      # network, DNS, timeout
        return False, {"description": f"{type(e).__name__}: {e}"}


def send(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        # Never print which one is missing beyond the name — no values.
        _annotate("CI alert NOT sent: TELEGRAM_BOT_TOKEN and/or "
                  "TELEGRAM_CHAT_ID is not set on this step. The alert step "
                  "needs both in its `env:` block.")
        return False

    ok, detail = _post(token, chat_id, text)
    if ok:
        print("CI alert sent to Telegram.")
        return True

    desc = (detail or {}).get("description", "")
    migrated = ((detail or {}).get("parameters") or {}).get("migrate_to_chat_id")

    if migrated:
        _annotate(
            "TELEGRAM_CHAT_ID is STALE: the group was upgraded to a supergroup "
            "and its id changed. Retrying with the id Telegram returned. "
            "ACTION REQUIRED: update the TELEGRAM_CHAT_ID GitHub secret — this "
            "retry is a stopgap, not a fix.")
        ok2, detail2 = _post(token, str(migrated), text)
        if ok2:
            print("CI alert sent to the migrated chat id.")
            return True
        _annotate(f"CI alert failed after migration retry: "
                  f"{(detail2 or {}).get('description', detail2)}")
        return False

    _annotate(f"CI alert NOT delivered. Telegram said: {desc or detail}")
    return False


def main() -> int:
    text = " ".join(sys.argv[1:]).strip() or os.environ.get("ALERT_TEXT", "").strip()
    if not text:
        _annotate("scripts.ci_alert called with no message.")
        return 0
    print(text)          # always in the log, delivered or not
    send(text)
    return 0             # see EXIT CODE above


if __name__ == "__main__":
    raise SystemExit(main())
