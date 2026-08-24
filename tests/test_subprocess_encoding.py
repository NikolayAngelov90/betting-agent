"""Stage 14 — every subprocess capture decodes UTF-8 explicitly.

THE RULE, promoted from three incidents to a guard:

    A subprocess capture must never rely on the platform's default codec.

WHY IT IS PINNED RATHER THAN REMEMBERED
---------------------------------------
`subprocess.run(..., text=True)` decodes with the locale encoding. On a Windows
console that is cp1251. A byte outside it does not raise where you are standing
— it kills the reader thread, and `.stdout` comes back as **None**.

That failure mode is uniquely bad here: None reads as an empty result, an empty
result reads as "nothing found", and "nothing found" reads as a clean finding.
An audit tool that cannot decode a log reports a healthy pipeline.

Three instances in Stage 14, and two were in the tooling doing the measuring
rather than in the code under test:

1. `ci_alert` printing an emoji to a cp1251 console — crashed the DEL-2 test
   harness and looked like a missing annotation
2. the DEL-2 harness itself, reporting "no annotation" for that reason
3. `scripts/ci_audit.py`, whose reader thread died on a CI log and wrote None to
   a file — surfacing as a TypeError only because the next call was write_text

CI is Linux/UTF-8 and would never have shown any of them.

SCOPE, deliberately narrow. This pins subprocess captures, where the failure
produces None. `read_text()` and `open()` without an encoding are the same class
and raise UnicodeDecodeError instead — noisy, immediate, and not silent — so
they are documented in the guard-design notes and not pinned here. A guard that
flags 28 harmless sites gets switched off.
"""

import pathlib
import re

VIOLATION = re.compile(r"(subprocess\.run|check_output)\(")


def _sources():
    for base in ("src", "scripts", "tests"):
        for path in pathlib.Path(base).rglob("*.py"):
            if path.name == "test_subprocess_encoding.py":
                continue
            yield path


def test_every_subprocess_capture_declares_utf8():
    """Any capture within 6 lines of `text=True` must also set an encoding."""
    offenders = []
    for path in _sources():
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if not VIOLATION.search(line.split("#", 1)[0]):
                continue
            window = "\n".join(lines[i:i + 7])
            captures_text = ("text=True" in window
                             or "universal_newlines=True" in window)
            if captures_text and "encoding=" not in window:
                offenders.append(f"{path}:{i + 1}: {line.strip()[:70]}")

    assert not offenders, (
        "subprocess capture decoding with the platform codec.\n\n"
        "On a non-UTF-8 console this kills the reader thread and .stdout "
        "becomes None — which reads as an empty result, which reads as a clean "
        "finding. Pass encoding=\"utf-8\", errors=\"replace\".\n\n"
        + "\n  ".join(offenders))
