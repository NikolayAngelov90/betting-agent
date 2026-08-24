"""Stage 14, DEL-1 — the delivery guarantee, and both historical failures replayed.

An alert whose only channel can fail silently is not an alert. It is a log line
with ambition.

Two observed failures, different causes, different code paths, identical class:

    2026-08-11  run 31482430418   alert built, undelivered — HTTP 400
    2026-08-23  run 32646469497   "Failed to send Telegram message: Timed out"

The first was "fixed" by adding a second sender to the same last hop. The class
survived the remedy, because the second failure landed in the other sender.
Both are replayed below against the consolidated path.
"""

import json
import urllib.error
import urllib.request

import pytest

from src.reporting import alert_delivery as ad


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch):
    monkeypatch.setattr(ad, "TIMEOUT", 0.01)


def _fake_http_error(code, body):
    return urllib.error.HTTPError(
        "https://api.telegram.org", code, "err", {},
        __import__("io").BytesIO(json.dumps(body).encode()))


# ─────────────────────────────────────────────── the two historical failures

def test_replay_2026_08_11_http_400_empty_text(monkeypatch, capsys):
    """The Content-Type bug: urlopen form-encoded the JSON, so `text` was absent.

    The original symptom was an alert that was built, sent, rejected, and then
    swallowed by `print(f"Telegram alert failed: {e}")`. What must now happen is
    that the failure reaches a surface that does not depend on Telegram.
    """
    sent = {}

    def fake_urlopen(req, timeout=None):
        sent["content_type"] = req.headers.get("Content-type")
        raise _fake_http_error(400, {
            "ok": False, "description": "Bad Request: message text is empty"})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = ad.deliver_alert("boom", token="t", chat_id="c", sleep=lambda _: None)

    assert sent["content_type"] == "application/json", (
        "the header whose absence caused the 2026-08-11 failure is gone again")
    assert not result.ok
    assert result.surfaced, "a 400 did not reach a Telegram-independent surface"
    assert "::error::" in capsys.readouterr().out
    assert result.attempts == 1, (
        "a 4xx was retried — it is a permanent error and retrying only delays "
        "the report")


def test_replay_2026_08_23_timeout(monkeypatch, capsys):
    """The timeout: no retry existed, and the whole surface was a logger.error.

    Now it must retry with backoff and then surface. This is the failure that
    proved a second sender does not fix a delivery class.
    """
    calls = {"n": 0}
    slept = []

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise TimeoutError("Timed out")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = ad.deliver_alert("boom", token="t", chat_id="c", sleep=slept.append)

    assert calls["n"] == ad.MAX_ATTEMPTS, (
        f"a transient failure was tried {calls['n']}x, expected "
        f"{ad.MAX_ATTEMPTS} — no retry means the 08-23 failure recurs exactly")
    assert slept == list(ad.BACKOFF), "retries did not back off"
    assert not result.ok and result.surfaced
    assert "::error::" in capsys.readouterr().out


def test_a_transient_failure_that_later_succeeds_is_delivered(monkeypatch):
    """The point of retrying: most timeouts are one bad moment."""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("Timed out")

        class R:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return R()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = ad.deliver_alert("x", token="t", chat_id="c", sleep=lambda _: None)
    assert result.ok and result.attempts == 2


def test_5xx_retries_but_4xx_does_not(monkeypatch):
    """Telegram having a bad moment is transient; us being wrong is not."""
    for code, expected in ((503, ad.MAX_ATTEMPTS), (400, 1)):
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None, _c=code):
            calls["n"] += 1
            raise _fake_http_error(_c, {"description": "x"})

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        ad.deliver_alert("x", token="t", chat_id="c", sleep=lambda _: None)
        assert calls["n"] == expected, f"HTTP {code} retried {calls['n']}x"


# ───────────────────────────────────────────────────── the outcome is returned

def test_the_outcome_is_returned_not_only_logged():
    """"alert fired" and "alert arrived" must stop being one line in a log."""
    r = ad.deliver_alert("x", token="", chat_id="", sleep=lambda _: None)
    assert isinstance(r, ad.DeliveryResult)
    assert not r.ok and not r
    assert r.detail == "not configured"


def test_missing_config_surfaces_without_leaking_values(monkeypatch, capsys):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    r = ad.deliver_alert("x", sleep=lambda _: None)
    out = capsys.readouterr().out
    assert not r.ok and r.surfaced
    assert "TELEGRAM_BOT_TOKEN" in out and "::error::" in out


def test_step_summary_is_written_when_github_provides_one(monkeypatch, tmp_path):
    """The surface that survives Telegram being down AND the log being long."""
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(TimeoutError("x")))
    ad.deliver_alert("boom", token="t", chat_id="c", sleep=lambda _: None)
    assert "ALERT NOT DELIVERED" in summary.read_text(encoding="utf-8")


# ────────────────────────────────────────────────────────── one path, guarded

def test_there_is_exactly_one_alert_sender():
    """The habit, guarded — instance 8 would be a third sender.

    `451fe3f` fixed a delivery failure by ADDING a sender. This test fails if
    that happens again: any module that builds a sendMessage request itself,
    rather than calling `deliver_alert`, is a parallel path to the same last hop.
    """
    import pathlib
    import re

    senders = []
    for path in (list(pathlib.Path("src").rglob("*.py"))
                 + list(pathlib.Path("scripts").rglob("*.py"))):
        text = path.read_text(encoding="utf-8")
        if path.name == "alert_delivery.py":
            continue
        for n, line in enumerate(text.splitlines(), 1):
            code = line.split("#", 1)[0]
            # Scoped to REQUEST CONSTRUCTION, not to mentions of the domain.
            # `--telegram-setup` prints a getUpdates URL as help text; that is
            # documentation, not a sender, and a guard that cannot tell the
            # difference gets switched off.
            if re.search(r"(urlopen|Request|requests\.(post|get)|httpx)",
                         code) and "telegram" in code.lower():
                senders.append(f"{path}:{n}  {line.strip()[:60]}")

    assert not senders, (
        "a second alert sender exists. DEL-1: the 2026-08-11 failure was "
        "'fixed' by adding a sender to the same last hop, and the class "
        "survived because the next failure landed in the other one. Route it "
        "through src.reporting.alert_delivery.deliver_alert instead:\n  "
        + "\n  ".join(senders))


def test_the_delivery_module_stays_stdlib_only():
    """Three workflows install a minimal dependency set without python-telegram-bot.

    `ci_alert.py` documented why it could not import TelegramNotifier. This
    module is the consolidation in the other direction, and it only works while
    it imports nothing but stdlib.
    """
    import ast
    import pathlib
    import sys

    tree = ast.parse(pathlib.Path(
        "src/reporting/alert_delivery.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])

    third_party = imported - set(sys.stdlib_module_names) - {"src"}
    assert not third_party, (
        f"alert_delivery imports {sorted(third_party)} — that breaks the "
        "closing-lines and paper-trading-report workflows, which install no "
        "third-party packages to send a message")


def test_send_alert_routes_through_the_shared_path():
    """The agent's alert path is where the 08-23 timeout landed."""
    import inspect

    from src.reporting.telegram_bot import TelegramNotifier

    src = inspect.getsource(TelegramNotifier.send_alert)
    assert "deliver_alert" in src, (
        "send_alert no longer uses the guaranteed path — this is exactly the "
        "code that timed out on 2026-08-23 with no retry and no surface")
