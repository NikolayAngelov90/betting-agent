"""DEL-3. A multi-chunk report that loses a middle chunk must not look complete.

    A DELIVERY IS NOT A SEQUENCE UNTIL SOMETHING CAN TELL THAT A PART IS
    MISSING.

The defect, observed in production on 2026-09-13 (run 34745992077):
`send 08:53:09, FAIL 08:53:14, send 08:53:17`. `_send_chunked` discarded every
chunk's return value and returned the LAST chunk's Message, so the caller got a
truthy object and the report read as delivered. The hole was in the middle,
where nothing looks at.

THREE SURFACES, BECAUSE THERE ARE TWO FAILURE SHAPES AND THEY ARE OPPOSITES:

    lost MIDDLE chunk   stream ends normally, content has a gap
                        -> caught by the in-stream failure marker
    lost LAST chunk     stream ends early, content has no gap
                        -> caught by the missing terminator

A single assertion catches one and is blind to the other, which is how the
original defect survived being "the sender works".
"""

import asyncio

from src.reporting.telegram_bot import ChunkOutcome, TelegramNotifier


class _FakeMsg:
    def __init__(self, text):
        self.text = text
        self.message_id = abs(hash(text)) % 100000


class _Bot:
    """Records what was sent, and fails the parts it is told to fail."""

    def __init__(self, fail_parts=(), fail_times=99):
        self.sent = []
        self.fail_parts = set(fail_parts)
        self.fail_times = fail_times
        self._n = 0

    async def send_message(self, chat_id=None, text=None, parse_mode=None):
        self._n += 1
        for p in self.fail_parts:
            if f"({p}/" in text and self._n <= self.fail_times:
                raise RuntimeError("Timed out")
        self.sent.append(text)
        return _FakeMsg(text)


def _notifier(bot):
    n = TelegramNotifier.__new__(TelegramNotifier)
    n.config = None
    n.enabled = True
    n.bot_token = "t"
    n.chat_id = "1"
    n._bot = bot
    n._last_send_error = ""
    return n


def _long(parts=4):
    """A message that splits into roughly `parts` chunks."""
    return "\n\n".join(["x" * 3000 for _ in range(parts)])


# ── the split itself ──────────────────────────────────────────────────────

def test_a_short_message_is_one_part_and_gets_no_markers():
    bot = _Bot()
    n = _notifier(bot)
    asyncio.run(n._send_chunked("short", report="daily picks"))
    assert bot.sent == ["short"], (
        "a single-part report was decorated — markers and terminators exist to "
        "make a SEQUENCE checkable, and one message is not a sequence")


def test_a_long_message_splits_and_every_part_is_numbered():
    bot = _Bot()
    n = _notifier(bot)
    asyncio.run(n._send_chunked(_long(4), report="daily picks"))
    assert len(bot.sent) >= 2
    total = len(bot.sent)
    for i, text in enumerate(bot.sent, 1):
        assert f"({i}/{total})" in text, (
            f"part {i} carries no position marker — the reader cannot see a "
            f"gap without one")


def test_the_last_part_carries_a_terminator():
    bot = _Bot()
    n = _notifier(bot)
    asyncio.run(n._send_chunked(_long(4), report="daily picks"))
    assert "end of daily picks" in bot.sent[-1], (
        "no terminator on the final part — a stream that stops early is then "
        "indistinguishable from one that finished")
    assert "end of daily picks" not in bot.sent[0]


# ── THE REGRESSION: a lost MIDDLE chunk ───────────────────────────────────

def test_a_lost_middle_chunk_announces_itself_in_the_stream():
    """THE 2026-09-13 FAILURE. The stream still ends normally.

    This is the case the old sender could not see: the final chunk succeeds,
    `last_msg` is truthy, and the report reads as complete.
    """
    bot = _Bot(fail_parts=[2])
    n = _notifier(bot)
    asyncio.run(n._send_chunked(_long(4), report="daily picks"))
    blob = "\n".join(bot.sent)
    assert "FAILED TO SEND" in blob and "INCOMPLETE" in blob, (
        "a middle part failed and nothing in the delivered stream says so — "
        "this is exactly the corrupt-but-complete-looking report DEL-3 exists "
        "for")


def test_a_lost_middle_chunk_is_RECORDED_not_merely_survived():
    bot = _Bot(fail_parts=[2])
    n = _notifier(bot)
    sink = []
    from src.utils.logger import get_logger
    h = get_logger().add(lambda m: sink.append(str(m)), level="INFO")
    try:
        asyncio.run(n._send_chunked(_long(4), report="daily picks"))
    finally:
        get_logger().remove(h)
    line = [s for s in sink if "REPORT_DELIVERY" in s]
    assert line, "no REPORT_DELIVERY line — the failure is inferable, not recorded"
    assert "failed=2" in line[0], line[0]
    assert "chunks=" in line[0] and "sent=" in line[0]


def test_a_failing_part_is_RETRIED_before_being_called_lost():
    """DEL-1 decided 3 attempts with backoff; this reuses that policy."""
    bot = _Bot(fail_parts=[2], fail_times=1)   # fails once, then succeeds
    n = _notifier(bot)
    asyncio.run(n._send_chunked(_long(4), report="daily picks"))
    blob = "\n".join(bot.sent)
    assert "FAILED TO SEND" not in blob, (
        "a transient failure was reported as a lost part — the retry policy "
        "is not being applied")


# ── the OTHER shape: a stream that ends early ─────────────────────────────

def test_a_lost_LAST_chunk_leaves_no_terminator():
    bot = _Bot(fail_parts=[4])
    n = _notifier(bot)
    sink = []
    from src.utils.logger import get_logger
    h = get_logger().add(lambda m: sink.append(str(m)), level="INFO")
    try:
        asyncio.run(n._send_chunked(_long(4), report="daily picks"))
    finally:
        get_logger().remove(h)
    line = [s for s in sink if "REPORT_DELIVERY" in s][0]
    assert "terminator=no" in line, (
        "the final part failed but the record claims a terminator — the "
        "early-end shape is invisible")


# ── the third state, again ────────────────────────────────────────────────

def test_not_configured_is_NOT_a_failed_send():
    """`attempted` exists so a disabled bot is not counted as a broken one."""
    n = _notifier(None)
    n.enabled = False
    out = asyncio.run(n._send_one("anything"))
    assert isinstance(out, ChunkOutcome)
    assert out.ok is False and out.attempted is False, (
        "an unconfigured Telegram reported as an attempted-and-failed send — "
        "that is the [] vs None collapse in a fourth place")


def test_an_attempted_send_that_fails_says_so():
    bot = _Bot(fail_parts=[1])
    n = _notifier(bot)
    out = asyncio.run(n._send_one("(1/2) body"))
    assert out.ok is False and out.attempted is True
    assert out.attempts >= 2, "no retry was attempted"
    assert "Timed out" in out.detail


def test_send_message_still_returns_a_message_for_existing_callers():
    """`send_performance_report` pins `sent.message_id`; do not break it."""
    bot = _Bot()
    n = _notifier(bot)
    got = asyncio.run(n._send_chunked(_long(3), report="performance report"))
    assert got is not None and hasattr(got, "message_id")
