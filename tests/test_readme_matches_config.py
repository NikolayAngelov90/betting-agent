"""The README states the selection rule's values; this keeps them true.

README.md is the first thing a stranger reads on a public repository, and the
one it replaced described a system that no longer existed: a Claude pick
review, value betting on model edge, a 1.50 floor. The values a reader would
act on -- the selection mode, the price floor, the probability floor and the
blend weights -- are read here from BOTH files and compared, so a config
change that is not carried into the README fails the suite instead of going
stale silently.

Every comparison below sits beside a known-wrong control: the same comparator
run on a README with one value changed must report it.

No network, no database, no clock: two committed files are read.
"""
import pathlib
import re

import pytest
import yaml

# NOT `from tests.conftest import anchored`: conftest imports
# `src.data.odds_quota`, which makes this file capture-critical by the import
# graph and puts it in the gate -- where a stale README would stop a picks
# run. The same sentinel check is written inline below instead.

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CONFIG = ROOT / "config" / "config.example.yaml"

SECTION = "## How picks are made now"

#: README table label -> config weight family.
TABLE_FAMILIES = {
    "home / draw / away": "1X2",
    "over / under": "over_under",
    "both teams to score": "btts",
    "team goals": "team_goals",
}
#: Column order of the README weight table.
TABLE_COLUMNS = ("market", "elo", "poisson")


def _section(text: str) -> str:
    start = text.index(SECTION)
    nxt = text.find("\n## ", start + len(SECTION))
    return text[start:] if nxt < 0 else text[start:nxt]


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _num(cell: str):
    cell = cell.strip()
    return None if cell in ("—", "-", "") else float(cell)


def mismatches(readme_text: str, cfg: dict) -> list:
    """Every disagreement between the README's stated rule and the config.

    Empty means they agree. A README that no longer states a value at all is a
    mismatch too: a value that cannot be found cannot be checked, and an
    unchecked value is how the previous README went stale.
    """
    out = []
    betting = cfg["betting"]
    mode = betting["selection_mode"]
    cv = betting["confident_value"]
    sec = _section(readme_text)

    # 1. The yaml block, compared whole.
    blocks = [b for b in re.findall(r"```yaml\n(.*?)```", sec, flags=re.S)
              if b.lstrip().startswith("betting:")]
    if len(blocks) != 1:
        out.append(f"expected ONE yaml block starting `betting:` in the "
                   f"section, found {len(blocks)}")
    else:
        stated = yaml.safe_load(blocks[0])["betting"]
        if stated.get("selection_mode") != mode:
            out.append(f"yaml selection_mode {stated.get('selection_mode')!r}"
                       f" != config {mode!r}")
        if stated.get("confident_value") != cv:
            out.append(f"yaml confident_value {stated.get('confident_value')!r}"
                       f" != config {cv!r}")

    # 2. The two floors as the prose states them.
    for label, pattern, want in (
            ("price floor", r"taken price is \*\*at least ([0-9.]+)\*\*",
             cv["min_price"]),
            ("p_min", r"blended probability is \*\*at least ([0-9.]+)\*\*",
             cv["p_min"])):
        found = re.findall(pattern, sec)
        if len(found) != 1:
            out.append(f"{label}: stated {len(found)} times in the prose, "
                       f"expected once")
        elif float(found[0]) != float(want):
            out.append(f"{label}: README {found[0]} != config {want}")

    # 3. The weight table, row by row.
    for label, family in TABLE_FAMILIES.items():
        # The table sits inside a numbered list item, so it is indented.
        rows = re.findall(rf"^[ \t]*\| {re.escape(label)} \|(.*)\|[ \t]*$",
                          sec, flags=re.M)
        if len(rows) != 1:
            out.append(f"weight table row {label!r}: found {len(rows)}")
            continue
        cells = [_num(c) for c in rows[0].split("|")]
        stated = {col: v for col, v in zip(TABLE_COLUMNS, cells)
                  if v is not None}
        want = {k: float(v) for k, v in cv["weights"][family].items()}
        if stated != want:
            out.append(f"weight row {label!r}: README {stated} != config "
                       f"{want}")

    # 4. The market's share as the prose summarises it.
    shares = [round(100 * float(w["market"]))
              for w in cv["weights"].values()]
    want_range = f"{min(shares)}–{max(shares)}%"
    if f"**{want_range}**" not in sec:
        out.append(f"market share range: README does not state "
                   f"**{want_range}**")

    # 5. The mode named in prose.
    if f"`{mode}`" not in readme_text:
        out.append(f"the README never names the mode `{mode}`")
    return out


def test_the_README_states_the_DEPLOYED_selection_values():
    # An empty list of problems is a claim only if the haystack is the README
    # that states the rule. `mismatches` also reports every value it cannot
    # find, so a section that vanished cannot read as agreement.
    text = README.read_text(encoding="utf-8")
    assert SECTION in text, f"README.md has no {SECTION!r} section"
    assert "```yaml\nbetting:" in text, "README.md states no rule block"
    problems = mismatches(text, _config())
    assert problems == [], "\n".join(problems)


def test_the_README_says_PAPER_and_NO_LIVE_WAGER_in_its_first_paragraph():
    text = README.read_text(encoding="utf-8")
    first = text.split("\n\n")[1]
    assert "paper trading mode" in first
    assert "no live" in first and "wager has ever been placed" in first
    assert _config()["betting"]["paper_trading_mode"] is True


# ── KNOWN-WRONG CONTROLS: the comparator must report each single change. ────

_EDITS = [
    ("min_price: 1.70", "min_price: 1.65", "yaml confident_value"),
    ("p_min: 0.54", "p_min: 0.56", "yaml confident_value"),
    ("selection_mode: confident_value", "selection_mode: standard",
     "yaml selection_mode"),
    ("taken price is **at least 1.70**", "taken price is **at least 1.65**",
     "price floor"),
    ("blended probability is **at least 0.54**",
     "blended probability is **at least 0.52**", "p_min"),
    ("| home / draw / away | 0.94 | 0.04 | 0.02 |",
     "| home / draw / away | 0.93 | 0.06 | 0.01 |", "weight row"),
    ("| team goals | 0.97 | — | 0.03 |", "| team goals | 0.97 | — | 0.00 |",
     "weight row"),
    ("**94–100%**", "**93–100%**", "market share range"),
    ("taken price is **at least 1.70**", "taken price is high enough",
     "price floor"),
]


@pytest.mark.parametrize("old, new, expect", _EDITS,
                         ids=[e[2] + ":" + e[1][:24] for e in _EDITS])
def test_KNOWN_WRONG_a_single_changed_value_is_REPORTED(old, new, expect):
    text = README.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"control is stale: {old!r} not found once"
    problems = mismatches(text.replace(old, new), _config())
    assert any(p.startswith(expect) for p in problems), problems


# ── WITHHELD FIGURES. 2026-11-15, the operator's decision. ─────────────────
#
# Two market-wide results were measured on API-Football and Flashscore data,
# whose terms do not permit publication. The README carries their arithmetic
# and a withholding note, never their figures, until the re-run registered in
# docs/findings-inventory.md section 5. A withholding that must be remembered
# is not a withholding, so it is checked here.

#: Figures from the band test, its correction and the cost measurement.
#: "0.94" alone is NOT listed: it is also the deployed 1X2 market weight.
WITHHELD = ("52,737", "52737", "10,700", "8,830", "4,039", "−0.94", "-0.94",
            "0.94 points", "0.94-point", "−1.92", "-1.92", "6.40",
            "3.70 points", "3.70pp", "1.628")


def withheld_in(text: str) -> list:
    return [w for w in WITHHELD if w in text]


def test_no_WITHHELD_figure_is_in_the_README():
    text = README.read_text(encoding="utf-8")
    # the anchors: both withholding notes are where the figures would be
    assert "**Its figures, and those of its correction, are withheld**" in text
    assert "**The figures are withheld**" in text
    assert withheld_in(text) == []


@pytest.mark.parametrize("planted", [
    "**6.40% relative**", "52,737 price legs", "win 0.94 points less often",
    "(interval −1.92 to −0.02)"])
def test_KNOWN_WRONG_a_WITHHELD_figure_put_back_is_FOUND(planted):
    text = README.read_text(encoding="utf-8") + "\n" + planted + "\n"
    assert withheld_in(text), f"{planted!r} was not detected"


def test_KNOWN_WRONG_a_CONFIG_change_not_carried_into_the_README_is_REPORTED():
    cfg = _config()
    cfg["betting"]["confident_value"]["min_price"] = 1.75
    problems = mismatches(README.read_text(encoding="utf-8"), cfg)
    assert any(p.startswith("price floor") for p in problems), problems
    assert any(p.startswith("yaml confident_value") for p in problems)
