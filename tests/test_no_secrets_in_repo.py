"""Stage 13 Step 1d — no secret-shaped literals in tracked files.

Two live API keys sat in a tracked `.mcp.json` for 188 days on a public
repository, and the production Telegram bot token was hardcoded in
`scripts/settle_feb15.py` — still live when it was found. Rotation kills an
exposed key; this test stops the next one being committed.

Same shape as the bulk-delete ban and the is_fixture filter guard: scan the
source, fail on the call/value shape, name the offender. Only TRACKED files are
scanned — gitignored trees (`mcp-servers/`) never reach the remote.

Never print a matched value. Offenders are reported masked.
"""

import re
import subprocess

import pytest

# High-confidence standalone shapes: these are secrets wherever they appear.
STANDALONE = [
    ("anthropic key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("telegram bot token", re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b")),
    ("aws access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
]

# A secret-shaped VALUE assigned to a key-shaped FIELD. 32-hex alone is not a
# secret (The Odds API uses 32-hex event ids in fixture data), so it only counts
# when it is the value of something named like a credential.
KEY_FIELD = re.compile(
    r"""(?ix)
    ["']?\b\w*(?:api[_-]?key|apikey|secret|token|password|passwd|access[_-]?key)\w*\b["']?
    \s*[:=]\s*
    ["']([A-Za-z0-9_\-]{20,})["']
    """
)

# Values that are obviously not secrets.
PLACEHOLDER = re.compile(
    r"(?i)^(\$\{.+\}|<.+>|your[_-].*|change[_-]?me|xxx+|\.\.\.|placeholder|"
    r"dummy|example|test[_-]?key|fake.*|redacted|none|null)$"
)

CONN_STRING = re.compile(
    r"(?:postgres(?:ql)?|mysql|mongodb)(?:\+\w+)?://[^\s:@/]+:([^\s@/]{4,})@")


def _mask(value):
    return value[:6] + "\u2026" + value[-6:] if len(value) > 14 else "\u2026"


def _tracked_text_files():
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    for name in out.stdout.splitlines():
        if not name:
            continue
        path = __import__("pathlib").Path(name)
        if not path.is_file():
            continue
        try:
            yield name, path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue


def test_no_secret_shaped_literals_in_tracked_files():
    offenders = []
    for name, text in _tracked_text_files():
        if name.endswith("test_no_secrets_in_repo.py"):
            continue                      # this file describes the shapes
        for lineno, line in enumerate(text.splitlines(), 1):
            for label, pat in STANDALONE:
                for m in pat.finditer(line):
                    offenders.append(
                        f"{name}:{lineno}: {label} {_mask(m.group(0))}")
            for m in KEY_FIELD.finditer(line):
                value = m.group(1)
                if PLACEHOLDER.match(value):
                    continue
                offenders.append(
                    f"{name}:{lineno}: credential-shaped value {_mask(value)}")
            for m in CONN_STRING.finditer(line):
                pw = m.group(1)
                if PLACEHOLDER.match(pw) or pw in {"pass", "password"}:
                    continue
                offenders.append(
                    f"{name}:{lineno}: connection-string password {_mask(pw)}")

    assert not offenders, (
        "secret-shaped literal(s) in tracked files — rotate the credential "
        "FIRST, then replace it with an environment lookup:\n  "
        + "\n  ".join(offenders))


@pytest.mark.parametrize("sample,should_flag", [
    ('bot_token = "1234567890:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"', True),
    ('API_KEY = "deadbeefcafe0123456789abcdef0123"', True),
    ('"ODDS_API_KEY": "${ODDS_API_KEY}"', False),
    ('url: postgresql://user:pass@host/db?sslmode=require', False),
    ('"id": "0123456789abcdef0123456789abcdef"', False),
])
def test_patterns_discriminate(sample, should_flag):
    """The guard must catch real shapes and ignore placeholders and event ids.

    The last case matters: The Odds API returns 32-hex event ids in fixture
    data, so a bare 32-hex string cannot be treated as a secret.
    """
    flagged = any(p.search(sample) for _, p in STANDALONE)
    for m in KEY_FIELD.finditer(sample):
        if not PLACEHOLDER.match(m.group(1)):
            flagged = True
    for m in CONN_STRING.finditer(sample):
        if not PLACEHOLDER.match(m.group(1)) and m.group(1) not in {"pass", "password"}:
            flagged = True
    assert flagged is should_flag
