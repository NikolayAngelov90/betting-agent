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

import os
import sys



def send(text: str) -> bool:
    """Delegates to the one delivery path. Kept as a thin wrapper because three
    workflows invoke this module by name.

    Stage 14 (DEL-1): the retry, backoff, migration handling and the
    Telegram-independent surface all moved to
    `src.reporting.alert_delivery` — stdlib only, so the minimal-dependency
    workflows can still import it. The dependency argument in this module's
    header rejected consolidating toward TelegramNotifier; consolidating toward
    stdlib was always available and is what happened.
    """
    from src.reporting.alert_delivery import deliver_alert

    result = deliver_alert(text)
    if result.ok:
        print(f"CI alert delivered (attempt {result.attempts}).")
    return result.ok


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
