"""Stage 14 (DEL-1): delivery moved to `src.reporting.alert_delivery`, which
both `ci_alert` and `TelegramNotifier.send_alert` now use. These tests kept
every assertion and only follow the code — the patch target is the shared
module. `test_alert_delivery.py` adds retry/backoff and the two historical
replays on top.

Stage 12.1, Defect 2 — the CI alert must not fail silently.

The defect this replaces: workflows sent Telegram alerts with inline `urllib`
and `except Exception as e: print(...)`.

Reproduced against the live API: `urlopen(url, data=json_bytes)` with no
`Content-Type` header is sent as `application/x-www-form-urlencoded`, so
Telegram finds no `text` field and answers `400 "message text is empty"`. The
body explaining that was discarded by the bare except, so the alert meant to
stop a "green but broken" run vanished into the log. The chat id was never
involved — `getChat` resolves it to a live supergroup.

No network in these tests — `urllib.request.urlopen` is replaced throughout.
"""

import io
import json
import urllib.error

import pytest

from scripts import ci_alert
from src.reporting import alert_delivery as _ad


@pytest.fixture(autouse=True)
def _secrets(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100999")


def _ok(*_a, **_k):
    class _R:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
    return _R()


def _http_error(code, payload):
    def _raise(*_a, **_k):
        raise urllib.error.HTTPError(
            "https://api.telegram.org", code, "Bad Request", {},
            io.BytesIO(json.dumps(payload).encode()))
    return _raise


def test_happy_path_sends_once(monkeypatch):
    calls = []
    monkeypatch.setattr(_ad.urllib.request, "urlopen",
                        lambda req, **kw: calls.append(req) or _ok())
    assert ci_alert.send("boom") is True
    assert len(calls) == 1


def test_missing_secrets_is_reported_not_swallowed(monkeypatch, capsys):
    """The dead `Notify Telegram on failure` step had no env: block at all and
    printed nothing useful. A missing secret must be an annotation."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    called = []
    monkeypatch.setattr(_ad.urllib.request, "urlopen",
                        lambda *a, **k: called.append(1) or _ok())

    assert ci_alert.send("boom") is False
    out = capsys.readouterr().out
    assert "::error::" in out
    assert "TELEGRAM_BOT_TOKEN" in out
    assert not called, "attempted a send with no token"


def test_request_is_labelled_as_json(monkeypatch):
    """THE regression test for the actual bug.

    Without this header urllib sends the JSON body as form data and Telegram
    replies `400 "message text is empty"` — which is precisely what CI hit.
    """
    seen = {}

    def fake(req, **kw):
        seen["ctype"] = req.headers.get("Content-type") or req.headers.get("Content-Type")
        seen["body"] = json.loads(req.data.decode())
        return _ok()

    monkeypatch.setattr(_ad.urllib.request, "urlopen", fake)
    assert ci_alert.send("hello") is True
    assert seen["ctype"] == "application/json", (
        f"body not labelled as JSON (got {seen['ctype']!r}) — Telegram will "
        f"parse it as form data and reject it as empty")
    assert seen["body"]["text"] == "hello"


def test_empty_text_400_is_surfaced(monkeypatch, capsys):
    """The exact production error string must reach the log, not be swallowed."""
    monkeypatch.setattr(_ad.urllib.request, "urlopen", _http_error(
        400, {"ok": False, "error_code": 400,
              "description": "Bad Request: message text is empty"}))
    assert ci_alert.send("boom") is False
    assert "message text is empty" in capsys.readouterr().out


def test_supergroup_migration_retries_with_the_new_id(monkeypatch, capsys):
    """The exact production failure: 400 + migrate_to_chat_id."""
    seen = []

    def fake(req, **kw):
        body = json.loads(req.data.decode())
        seen.append(str(body["chat_id"]))
        if body["chat_id"] == "-100999":
            raise urllib.error.HTTPError(
                "u", 400, "Bad Request", {}, io.BytesIO(json.dumps({
                    "ok": False, "error_code": 400,
                    "description": "Bad Request: group chat was upgraded to a "
                                   "supergroup chat",
                    "parameters": {"migrate_to_chat_id": -1001234567890},
                }).encode()))
        return _ok()

    monkeypatch.setattr(_ad.urllib.request, "urlopen", fake)
    assert ci_alert.send("boom") is True
    assert seen == ["-100999", "-1001234567890"], seen

    out = capsys.readouterr().out
    assert "::error::" in out
    assert "STALE" in out
    assert "ACTION REQUIRED" in out, (
        "a stale secret must be flagged for a human, not silently worked around")


def test_migration_id_is_never_persisted(monkeypatch, tmp_path):
    """Retrying is a stopgap. Writing the new id anywhere would hide a
    configuration problem behind a self-healing illusion."""
    import inspect

    src = inspect.getsource(ci_alert)
    # Specific enough not to trip on `urlopen(` — the point is persistence,
    # not any occurrence of the substring "open".
    for forbidden in ("write_text(", "GITHUB_ENV", "GITHUB_OUTPUT",
                      "gh secret", "os.environ[", "setenv", "json.dump("):
        assert forbidden not in src, (
            f"ci_alert persists state via {forbidden!r}")


def test_other_400s_surface_telegrams_own_description(monkeypatch, capsys):
    monkeypatch.setattr(_ad.urllib.request, "urlopen", _http_error(
        400, {"ok": False, "description": "Bad Request: chat not found"}))
    assert ci_alert.send("boom") is False
    out = capsys.readouterr().out
    assert "chat not found" in out, (
        "the API's explanation was discarded — this is the original defect")


def test_network_failure_is_reported(monkeypatch, capsys):
    def boom(*a, **k):
        raise OSError("connection reset")
    monkeypatch.setattr(_ad.urllib.request, "urlopen", boom)
    assert ci_alert.send("boom") is False
    assert "connection reset" in capsys.readouterr().out


def test_exit_code_is_always_zero(monkeypatch):
    """An undeliverable alert must not turn a green run red, nor a red run
    green — it is an observability channel, not the thing under test."""
    monkeypatch.setattr(_ad.urllib.request, "urlopen",
                        _http_error(400, {"description": "nope"}))
    monkeypatch.setattr(ci_alert.sys, "argv", ["ci_alert", "a failure"])
    assert ci_alert.main() == 0


def test_message_is_always_printed_even_when_undelivered(monkeypatch, capsys):
    monkeypatch.setattr(_ad.urllib.request, "urlopen",
                        _http_error(400, {"description": "nope"}))
    monkeypatch.setattr(ci_alert.sys, "argv", ["ci_alert", "SENTINEL-TEXT"])
    ci_alert.main()
    assert "SENTINEL-TEXT" in capsys.readouterr().out


def test_no_secret_value_is_ever_printed(monkeypatch, capsys):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "SUPERSECRET-TOKEN")
    monkeypatch.setattr(_ad.urllib.request, "urlopen",
                        _http_error(400, {"description": "nope"}))
    ci_alert.send("boom")
    assert "SUPERSECRET-TOKEN" not in capsys.readouterr().out


# ═══════════════════ the workflows must use it, and not mask failures

def test_every_workflow_alert_goes_through_this_script():
    import pathlib

    import yaml

    for f in pathlib.Path(".github/workflows").glob("*.yml"):
        text = f.read_text(encoding="utf-8")
        assert "api.telegram.org" not in text, (
            f"{f.name} still contains an inline Telegram call")
        wf = yaml.safe_load(text)
        for job in wf.get("jobs", {}).values():
            for st in job.get("steps", []):
                if "scripts.ci_alert" in (st.get("run", "") or ""):
                    env = st.get("env", {}) or {}
                    assert "TELEGRAM_BOT_TOKEN" in env and "TELEGRAM_CHAT_ID" in env, (
                        f"{f.name} / {st.get('name')} does not pass the secrets")


def test_a_failing_test_suite_fails_the_workflow():
    """The masking effect Stage 12 found: `continue-on-error` on the tests step
    left a failing suite green AND the alert undelivered."""
    import pathlib

    import yaml

    wf = yaml.safe_load(
        pathlib.Path(".github/workflows/daily-picks.yml").read_text(encoding="utf-8"))
    steps = [s for j in wf["jobs"].values() for s in j.get("steps", [])]
    tests = [s for s in steps if s.get("id") == "tests"]
    assert tests, "the daily-picks workflow no longer runs the test suite"
    assert not tests[0].get("continue-on-error", False), (
        "continue-on-error is back on the tests step — a failing suite would "
        "leave the workflow green")
