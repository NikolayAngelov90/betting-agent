"""Stage 13 Step 1d — no secret-shaped literals in tracked files.

Two live API keys sat in a tracked `.mcp.json` for 188 days on a public
repository, and the production Telegram bot token was hardcoded in
`scripts/settle_feb15.py` — still live when it was found. Rotation kills an
exposed key; this test stops the next one being committed.

Same shape as the bulk-delete ban and the is_fixture filter guard: scan the
source, fail on the call/value shape, name the offender. Tracked AND untracked
files are scanned — anything not gitignored will reach the remote, and scoping
on staging made the answer depend on when the suite ran (see
`_tracked_text_files`). Gitignored trees (`mcp-servers/`) stay out.

Never print a matched value. Offenders are reported masked.
"""

import os
import re
import subprocess

import pytest

# High-confidence standalone shapes: these are secrets wherever they appear.
STANDALONE = [
    ("anthropic key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("telegram bot token", re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b")),
    ("aws access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    # ── THE TELEGRAM CHAT ID, added 2026-11-04. ───────────────────────────
    #
    # NOT because it is a credential. IT ADDRESSES A CHANNEL OTHER PEOPLE
    # READ, so publishing it lets anyone who also holds the bot token post
    # into the chat Niki shares with clients — which is why it is a
    # repository secret in the first place.
    #
    # It joins this list because a payload drive printed it to a terminal on
    # 2026-11-04 and the remedy offered was "I will not print it again".
    #
    #     A REMEDY THAT MUST BE REMEMBERED IS NOT A REMEDY.
    #
    # Fourth application of that rule, and the first where the thing to
    # remember was a single line of output rather than a code change.
    #
    # FALSE-POSITIVE RATE, MEASURED OVER THE REPOSITORY BEFORE WIRING:
    # 2 matches in 1 of 875 tracked+untracked text files = 0.11%. Both are
    # `-100` followed by an ascending digit run, in the supergroup-migration
    # fixture, and confirmed NOT the real id by comparison against the
    # environment. They are handled by `_is_placeholder_digits` rather than
    # by exempting the file, so a REAL id pasted into that same file would
    # still be caught.
    ("telegram chat id", re.compile(r"-100\d{10,13}\b")),
]


#: Labels whose pattern is SHAPE ONLY, so a digit-run placeholder rule is the
#: only way to tell a fixture from a real value.
#:
#: SCOPED, because applying it to every STANDALONE pattern broke the
#: bot-token case: that fixture is `1234567890:AAAA...`, whose digits are also
#: an ascending run, and the guard stopped flagging a token it is meant to
#: catch. A token carries 35 further characters of entropy; a chat id carries
#: none, which is exactly why only it needs this.
SHAPE_ONLY = {"telegram chat id"}


def _is_placeholder_digits(value: str) -> bool:
    """Is this digit run obviously a human-typed stand-in?

    A chat id has no checksum and no distinguishing alphabet — only its
    shape — so the `PLACEHOLDER` word list below cannot help it. Sequential
    and single-repeated runs are the two forms this repository's fixtures
    use, and without this rule every such fixture is a permanent offender
    and the guard gets switched off.
    """
    digits = "".join(c for c in value if c.isdigit())
    if value.startswith("-100"):
        digits = digits[3:]                       # drop the 100 prefix
    if len(digits) < 6:
        return True
    if len(set(digits)) == 1:
        return True                               # 0000000000
    # MODULO 10, because the commonest hand-typed run is `1234567890` and its
    # last step is 9 -> 0. A plain `+1` test missed exactly that, which is the
    # fixture already in this repository.
    #
    # A real id whose digits ascend mod 10 end to end is a 1-in-10^9 shape, so
    # the widening costs nothing measurable.
    return all((ord(b) - ord(a)) % 10 == 1 for a, b in zip(digits, digits[1:]))

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
    """Every file that WILL reach the remote — tracked and untracked alike.

    `--others --exclude-standard` is the whole point and it was added on
    2026-09-20 after a plain `git ls-files` cost a day of picks.

    THE SCOPE WAS DERIVED FROM STAGING, SO THE ANSWER DEPENDED ON WHEN IT RAN.
    A dummy `api_key` literal was written into a new script, the suite was run,
    **1072 passed**, the file was committed, and CI then ran the same suite at
    the same commit and reported **1 failed, 1071 passed**. The file was
    untracked locally and tracked in CI; `git add` changed the test's input
    set, not the code.

        The count was 1072 both times. An unchanged number reads as
        confirmation, which is worse than a number that moves.

    Scanning untracked-but-not-ignored files makes the scope depend on what
    exists rather than on what has been staged, so the check gives the same
    answer before and after `git add`. Gitignored trees (`mcp-servers/`) stay
    out — `--exclude-standard` honours .gitignore, and those never reach the
    remote. This is a superset of the old scope: nothing previously scanned is
    dropped.
    """
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
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
                    # A shape-only secret needs a placeholder rule, or every
                    # fixture using that shape is a permanent offender.
                    if (label in SHAPE_ONLY
                            and _is_placeholder_digits(m.group(0))):
                        continue
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
    flagged = any(
        any(not (lab in SHAPE_ONLY and _is_placeholder_digits(m.group(0)))
            for m in p.finditer(sample))
        for lab, p in STANDALONE)
    for m in KEY_FIELD.finditer(sample):
        if not PLACEHOLDER.match(m.group(1)):
            flagged = True
    for m in CONN_STRING.finditer(sample):
        if not PLACEHOLDER.match(m.group(1)) and m.group(1) not in {"pass", "password"}:
            flagged = True
    assert flagged is should_flag


# ════════ THE TELEGRAM CHAT ID — its own controls ══════════════════════════
#
# The value is never written into this file. The known-wrong case uses a
# SYNTHETIC id of the same shape whose digits are neither sequential nor
# repeated, so it exercises the pattern without becoming the exposure it
# guards against.

def _chat_id_from_env_file() -> str:
    """The id, from the environment or from `.env`, NEVER printed.

    `conftest` strips credentials from the environment, so reading only
    `os.environ` made both history checks SKIP — and a skipped control proves
    nothing. This reads `.env` directly because the guard's whole job is to
    confirm the value is absent everywhere else.
    """
    import os
    import pathlib
    v = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
    if v:
        return v
    p = pathlib.Path(".env")
    if not p.exists():
        return ""
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip().startswith("TELEGRAM_CHAT_ID"):
            _, _, val = line.partition("=")
            return val.strip().strip("'\"")
    return ""


_SYNTH = "-1009481726354"        # same shape, not the real id, not sequential


def test_KNOWN_WRONG_the_payload_drive_output_IS_REFUSED():
    """The exact line that printed the id on 2026-11-04 must be caught.

    It read:  `  chat_id = <id>   (client id = <id>)`
    """
    line = f"  chat_id = {_SYNTH}   (client id = {_SYNTH})"
    hits = [m.group(0) for lab, p in STANDALONE for m in p.finditer(line)
            if not (lab in SHAPE_ONLY and _is_placeholder_digits(m.group(0)))]
    assert hits, (
        "the payload-drive output line is NOT refused by the guard, so the "
        "only thing stopping a repeat is remembering not to print it")
    assert len(hits) == 2, f"expected both occurrences, got {hits!r}"


def test_the_MASKED_FORM_PASSES():
    """The remedy must be usable, or it will be bypassed.

    `_mask` style output and a truncated form are what a report should carry.
    """
    for ok in (f"client id = {_SYNTH[:4]}…{_SYNTH[-3:]}",
               "chat_id = <redacted>",
               "chat_id = ${{ secrets.TELEGRAM_CHAT_ID }}",
               'TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}'):
        hits = [m.group(0) for lab, p in STANDALONE for m in p.finditer(ok)
                if not (lab in SHAPE_ONLY
                        and _is_placeholder_digits(m.group(0)))]
        assert not hits, f"the masked form {ok!r} was refused: {hits!r}"


@pytest.mark.parametrize("sample,should_flag", [
    # unrelated numbers of similar shape must NOT fire
    ("match_id = -1001", False),                 # too short
    ("total = -100000000000", False),            # single repeated digit
    ("fixture = -1001234567890", False),         # ascending: the migration fixture
    ("epoch_ms = 1762300000000", False),         # no -100 prefix
    ("delta = -100.25", False),                  # not a digit run
    ("goal_diff = -1005", False),
    ("chat = -1009481726354", True),             # the real shape
])
def test_the_CHAT_ID_PATTERN_DOES_NOT_FIRE_ON_UNRELATED_NUMBERS(sample, should_flag):
    """The 0.11% false-positive rate measured before wiring, as cases."""
    hits = [m.group(0) for lab, p in STANDALONE for m in p.finditer(sample)
            if not (lab in SHAPE_ONLY and _is_placeholder_digits(m.group(0)))]
    assert bool(hits) is should_flag, (
        f"{sample!r} -> flagged={bool(hits)}, expected {should_flag}")


def test_the_REAL_ID_APPEARS_IN_NO_COMMIT_measured_over_history():
    """MEASURED OVER HISTORY, not asserted over the working tree.

    `git log -S` searches every commit's content, so this answers "was it
    ever committed and later removed", which a working-tree scan cannot.
    The value is read from the environment and never printed.
    """
    import os
    cid = _chat_id_from_env_file()
    if not cid:
        pytest.skip("no TELEGRAM_CHAT_ID available to search for")
    r = subprocess.run(["git", "log", "--all", "--oneline", f"-S{cid}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"git log unavailable: {r.stderr[:80]}")
    commits = [ln for ln in r.stdout.splitlines() if ln.strip()]
    assert not commits, (
        f"the chat id appears in {len(commits)} commit(s) in history. "
        f"Rotating the CHAT is not possible, so the bot token must be "
        f"rotated and the chat recreated. Commit subjects are withheld here "
        f"because they would identify the blobs.")


def test_MEMORY_AND_DOCS_CARRY_NO_SECRET_SHAPED_LITERAL():
    """EVERY pattern, over docs/ and the memory directory. Added 2026-11-04.

    `_tracked_text_files` scans the repository, and the memory directory is
    OUTSIDE it — so six live credentials sat in
    `memory/github_token.md` in plaintext, in a file that was not even
    indexed in `MEMORY.md`. The repository guard could never have seen them.

    MEMORY IS PLAINTEXT AT REST AND IS LOADED INTO CONTEXT EVERY SESSION, so
    a credential there is published to every future conversation. That is a
    wider exposure than a commit, not a narrower one.

    Offenders are reported masked, and the connection-string password rule is
    applied here too: a 16-character password is shorter than the opaque-run
    heuristics and was missed once by a hand-rolled redaction.
    """
    import pathlib
    roots = [pathlib.Path("docs"),
             pathlib.Path(os.path.expanduser(
                 "~/.claude/projects/c--Users-nikit-Desktop-betting-agent/memory"))]
    offenders = []
    scanned = 0
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix in {".db", ".pyc", ".png", ".jpg"}:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            scanned += 1
            for lineno, line in enumerate(text.splitlines(), 1):
                for label, pat in STANDALONE:
                    for m in pat.finditer(line):
                        if (label in SHAPE_ONLY
                                and _is_placeholder_digits(m.group(0))):
                            continue
                        offenders.append(
                            f"{p.name}:{lineno}: {label} {_mask(m.group(0))}")
                for m in KEY_FIELD.finditer(line):
                    if PLACEHOLDER.match(m.group(1)):
                        continue
                    offenders.append(
                        f"{p.name}:{lineno}: credential-shaped "
                        f"{_mask(m.group(1))}")
                for m in CONN_STRING.finditer(line):
                    pw = m.group(1)
                    if PLACEHOLDER.match(pw) or pw in {"pass", "password"}:
                        continue
                    offenders.append(
                        f"{p.name}:{lineno}: connection-string password "
                        f"{_mask(pw)}")
    # THE HAYSTACK IS ANCHORED BY ITS OWN SIZE. A zero-hit scan over zero
    # files passes for the wrong reason, and the memory directory is outside
    # the repository so a moved path would silently empty it.
    assert scanned >= 20, (
        f"only {scanned} file(s) scanned; docs/ and the memory directory "
        f"should hold far more, so this zero-hit pass is not evidence")
    assert not offenders, (
        "secret-shaped literal(s) in docs/ or the memory directory — ROTATE "
        "each one FIRST, then replace it with a pointer to where it lives:"
        + "".join(chr(10) + "  " + o for o in offenders))


def test_the_REAL_ID_IS_IN_NO_LEDGER_MEMORY_OR_FIXTURE():
    """The three places a value like this gets copied to by habit."""
    import os
    import pathlib
    cid = _chat_id_from_env_file()
    if not cid:
        pytest.skip("no TELEGRAM_CHAT_ID available to search for")
    roots = [pathlib.Path("docs"), pathlib.Path("tests"),
             pathlib.Path(os.path.expanduser(
                 "~/.claude/projects/c--Users-nikit-Desktop-betting-agent/memory"))]
    found = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix in {".db", ".pyc"}:
                continue
            try:
                if cid in p.read_text(encoding="utf-8", errors="ignore"):
                    found.append(str(p))
            except Exception:
                continue
    assert not found, f"the chat id is written in: {found}"
