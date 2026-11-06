"""The schedule-margin acknowledgement: RETIRED 2026-11-06, predicate kept.

WHAT HAPPENED. The entry acknowledged a standing weekday breach at
`measured_at=20.665799601890512` with a floor 11.32 minutes under it, and its
registered clearing action was "the parallel-regime picks series reaching
n=10 and FULL_CARD_PICKS_MAX being re-derived from it".

    THAT ACTION WAS TAKEN. The series reached n=12, the bound was re-derived
    off the parallel regime (max 88.1 -> 40.35, mean+3sd 107.0 -> 64.59), and
    the weekend schedule margin moved from -7m to pooled_binding +53m against
    a 30-minute threshold. There is no breach left to defer.

Its review date had also passed (2026-11-04), and the two `DRIVEN` tests
below had been red since -- the expiry working, not a bug. It is retired for
the first reason; a merely-expired acknowledgement would have to be cleared or
deleted anyway.

WHY THIS FILE DID NOT SHRINK TO "the register is empty".

    A predicate with an empty register returns `{}` for every input, so every
    "it refuses" assertion would pass FOR THE WRONG REASON.

That is the matcher-with-zero-hits shape, and it would quietly delete the
guarantees this file exists for. So each behaviour is now driven against a
SYNTHETIC entry, and `test_the_PREDICATE_CAN_SAY_YES` is the vacuity guard
that proves the driving works at all.
"""
import datetime as dt
import pathlib

import pytest

import scripts.ci_audit as ci

SRC = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")

#: The retired entry's exact shape and figures, kept here to DRIVE the
#: predicate. It is deliberately the real retired values: a fixture invented
#: from nothing would stop testing the thing that shipped.
SYNTHETIC = {
    "since": "2026-10-22",
    "measured_at": 20.665799601890512,
    "binding": 20.665799601890512 - 11.32,
    "jitter": 11.32,
    "why": "a standing breach, retained here to drive the predicate",
    "action": ("re-derive FULL_CARD_PICKS_MAX once the parallel series "
               "reaches n=10"),
    "review": "2026-11-04",
}


def _with_register(monkeypatch, register):
    monkeypatch.setattr(ci, "ACKNOWLEDGED_MARGIN_BREACH", register)


# ── THE RETIREMENT ITSELF ──────────────────────────────────────────────────

def test_the_REGISTER_IS_EMPTY_which_is_the_correct_state():
    assert ci.ACKNOWLEDGED_MARGIN_BREACH == {}, (
        f"something is deferred again: {ci.ACKNOWLEDGED_MARGIN_BREACH!r}. An "
        f"entry here must carry its own review date and its clearing action.")


def test_the_RETIREMENT_RECORDS_WHY_IT_WENT_and_WHAT_CLEARED_IT():
    """An acknowledgement removed on its expiry alone leaves the condition
    unexplained. This one expired AND its action is blocked, and the record
    has to carry both or the next reader re-acknowledges it."""
    i = SRC.index("ACKNOWLEDGED_MARGIN_BREACH: dict = {}")
    head = SRC[max(0, i - 3000):i]
    # WHITESPACE-NORMALISED: these phrases wrap across comment line breaks,
    # and a raw substring search finds nothing -- which would report "no
    # longer written down" about text that is right there.
    # CASE-INSENSITIVE: the record uses capitals for emphasis, and a
    # case-sensitive search for a phrase that is written in caps reports
    # "no longer written down" about text that is right there.
    flat = " ".join(head.replace("#", " ").split()).lower()
    for frag in ("retired 2026-11-06", "review date passed",
                 "its action was then taken", "88.1 -> 40.35",
                 "-7m -> +35m", "-6.93", "27.6-minute",
                 "absorbing a deterioration",
                 "empty register is the correct state",
                 "byte-identical"):
        assert frag in flat, f"the retirement does not record {frag!r}"


def test_the_PICKS_BOUND_WAS_SUBSTITUTED_FORWARD():
    """The clearing action is a number in the source, so check the number."""
    assert ci.FULL_CARD_PICKS_MAX == pytest.approx(40.35), (
        f"the live picks bound is {ci.FULL_CARD_PICKS_MAX}, not the 40.35 "
        f"the acknowledgement was retired on. If it was reverted, the "
        f"schedule breach is back and the register must carry it again")
    assert ci.FULL_CARD_PICKS_3SD == pytest.approx(64.59)
    assert ci.FULL_CARD_PICKS_MAX < ci.FULL_CARD_PICKS_3SD, (
        "max above mean+3sd means the two were swapped")


def test_the_DECISION_RECORD_KEPT_THE_FIGURES_THE_DECISION_WAS_MADE_ON():
    """FORWARD ONLY. The s5.19 argument keeps its own numbers.

    `tests/test_decision_record_is_not_a_live_constant.py` holds the
    separation in both directions and the swap control; this asserts the
    record exists and still DISAGREES with the live constant, because an
    agreeing pair makes that whole file vacuous.
    """
    rec = ci.PICKS_BOUND_AT_S519_DECISION
    assert rec["max"] == 88.1 and rec["mean_plus_3sd"] == 107.0
    assert rec["n"] == 4 and rec["measured"] == "2026-10-01"
    assert "SERIAL" in rec["regime"]
    assert rec["max"] != ci.FULL_CARD_PICKS_MAX, (
        "the record and the live constant agree, so the separation between "
        "them cannot be verified")
    sup = rec["superseded_by"]
    assert sup["n"] == 12 and sup["max"] == pytest.approx(
        ci.FULL_CARD_PICKS_MAX)
    k = SRC.index("FULL_CARD_PICKS_MAX = 40.35")
    flat = " ".join(SRC[max(0, k - 2200):k].replace("#", " ").split()).lower()
    for frag in ("parallel", "all-card", "n=12", "2026-11-06", "not pooled",
                 "cccb0e6"):
        assert frag in flat, f"the substitution does not state {frag!r}"


# ── THE VACUITY GUARD ──────────────────────────────────────────────────────

def test_the_PREDICATE_CAN_SAY_YES(monkeypatch):
    """Without this, every refusal below passes because the register is empty.

    An empty register makes `margin_breach_is_acknowledged` return `{}` for
    all input, which satisfies every negative assertion in this file for
    entirely the wrong reason.
    """
    _with_register(monkeypatch, {"weekday": SYNTHETIC})

    real_date = ci._dt.date

    class _Before(real_date):
        @classmethod
        def today(cls):
            return real_date(2026, 11, 1)

    monkeypatch.setattr(ci._dt, "date", _Before)
    got = ci.margin_breach_is_acknowledged("weekday", SYNTHETIC["measured_at"])
    assert got, ("the predicate cannot return a covering entry even when one "
                 "is present and in date, so nothing else here is verified")


# ── deterioration, not level: DRIVEN against the synthetic entry ──────────

def _in_date(monkeypatch):
    _with_register(monkeypatch, {"weekday": SYNTHETIC})
    real_date = ci._dt.date

    class _Before(real_date):
        @classmethod
        def today(cls):
            return real_date(2026, 11, 1)

    monkeypatch.setattr(ci._dt, "date", _Before)


def test_DRIVEN_it_does_NOT_alarm_at_the_acknowledged_level(monkeypatch):
    _in_date(monkeypatch)
    assert ci.margin_breach_is_acknowledged("weekday",
                                            SYNTHETIC["measured_at"])
    assert ci.margin_breach_is_acknowledged("weekday", SYNTHETIC["binding"]), (
        "the floor itself is not covered, so the comparison is not >=")


def test_DRIVEN_it_DOES_alarm_on_a_DETERIORATION_past_the_floor(monkeypatch):
    _in_date(monkeypatch)
    worse = SYNTHETIC["binding"] - 0.01
    assert ci.margin_breach_is_acknowledged("weekday", worse) == {}, (
        f"a binding of {worse} is worse than the acknowledged floor "
        f"{SYNTHETIC['binding']} and was still absorbed")


def test_DRIVEN_an_IMPROVEMENT_is_covered_too(monkeypatch):
    """The acknowledgement is a floor on how bad it may get, not a window."""
    _in_date(monkeypatch)
    assert ci.margin_breach_is_acknowledged("weekday", 29.0)


def test_DRIVEN_the_EXPIRY_is_COMPARED_and_not_merely_STORED(monkeypatch):
    """A deferral with a date nothing reads is not a deferral."""
    _with_register(monkeypatch, {"weekday": SYNTHETIC})
    real_date = ci._dt.date

    class _After(real_date):
        @classmethod
        def today(cls):
            return real_date(2026, 11, 5)

    monkeypatch.setattr(ci._dt, "date", _After)
    assert ci.margin_breach_is_acknowledged(
        "weekday", SYNTHETIC["measured_at"]) == {}, (
        "the acknowledgement outlived its own review date")


# ── the figures stay DERIVED, in the source ───────────────────────────────

def test_the_JITTER_is_DERIVED_IN_THE_SOURCE_with_its_measurements():
    """A band whose width is a bare literal is a threshold hard-coded from a
    small sample. The five single-point moves it came from must stay beside
    it, and the excluded sixth must stay excluded FOR A REASON."""
    assert ci.ACKNOWLEDGED_MARGIN_BREACH_JITTER == 11.32
    i = SRC.index("ACKNOWLEDGED_MARGIN_BREACH_JITTER = ")
    head = SRC[max(0, i - 2600):i]
    for move in ("+5.53", "+2.79", "-11.32", "-2.24", "+1.77", "-97.91"):
        assert move in head, (
            f"the single-point move {move} is no longer recorded beside the "
            f"band derived from it")
    assert "cluster" in head, (
        "the excluded move must be excluded FOR A STATED REASON")


def test_the_BIMODALITY_IS_STILL_RECORDED_WITH_BOTH_CLUSTERS():
    """`mean+3sd` over a bimodal series measures between-cluster spread, not
    tail risk. The claim is only checkable if both clusters are written down,
    and only means anything if the cron did NOT change between them.

    It moved out of the retired entry's `why` and into the source commentary;
    this asserts it survived the move rather than being deleted with it.
    """
    assert "255.9-304.9" in SRC and "128.4-144.1" in SRC, (
        "the two clusters are no longer recorded anywhere in ci_audit")
    i = SRC.index("255.9-304.9")
    near = SRC[max(0, i - 600):i + 600]
    assert "cron did NOT" in near or "UNCHANGED CRON" in near, (
        "without this a reader cannot tell the bimodality from an ordinary "
        "regime change -- and a regime change would justify SPLITTING the "
        "series instead of qualifying it")


def test_the_ACKNOWLEDGED_BRANCH_still_prints_the_READING_FLOOR_and_CLOCK():
    """Kept because the branch is kept: the next deferral must be a data
    entry, and it inherits this printing."""
    i = SRC.index("ACKNOWLEDGED since {_ack['since']}")
    block = SRC[i:i + 1600]
    assert "reading now {_margin['binding']:+.2f}" in block, (
        "the CURRENT reading is not printed, so the headroom is invisible")
    assert "FLOOR of {_ack['binding']:+.2f}" in block, (
        "the floor is printed without being called a floor")
    assert "_ack['jitter']" in block, "the band's width is not shown"
    assert "{_ack['why']}" in block, "the why is not printed in full"
    assert "{_ack['action']}" in block, "the clearing action is not printed"
    assert "[:88]" not in SRC, "the truncation is back"


# ── KNOWN-WRONG CONTROLS ──────────────────────────────────────────────────

def test_KNOWN_WRONG_the_WEEKEND_margin_is_NOT_acknowledged(monkeypatch):
    """Acknowledging a day type that does not breach would be a standing
    absorber waiting for a real breach to arrive under it."""
    _in_date(monkeypatch)
    assert "weekend" not in ci.ACKNOWLEDGED_MARGIN_BREACH
    assert ci.margin_breach_is_acknowledged("weekend", 5.0) == {}


def test_KNOWN_WRONG_an_UNPARSEABLE_review_date_absorbs_NOTHING(monkeypatch):
    _with_register(monkeypatch,
                   {"weekday": dict(SYNTHETIC, review="not-a-date")})
    assert ci.margin_breach_is_acknowledged(
        "weekday", SYNTHETIC["measured_at"]) == {}


def test_KNOWN_WRONG_an_entry_with_NO_review_absorbs_NOTHING(monkeypatch):
    entry = {k: v for k, v in SYNTHETIC.items() if k != "review"}
    _with_register(monkeypatch, {"weekday": entry})
    assert ci.margin_breach_is_acknowledged(
        "weekday", SYNTHETIC["measured_at"]) == {}, (
        "an undated acknowledgement is a suppression")


def test_KNOWN_WRONG_a_day_type_with_no_entry_is_not_covered():
    assert ci.margin_breach_is_acknowledged("", -500.0) == {}
    assert ci.margin_breach_is_acknowledged("holiday", -500.0) == {}


def test_KNOWN_WRONG_a_ROUNDED_figure_refuses_the_real_reading(monkeypatch):
    """Recorded twice already, and it caught me a third time on 2026-11-06:
    I pinned the structural acknowledgement at the DISPLAYED -30.1 and the
    predicate refused the live -39.31."""
    # `round(binding)` goes DOWN here, which is the SAFE direction; the
    # dangerous one is a floor rounded UP, so the reading it was measured
    # from now sits below its own floor. That is the 2026-10-21 case: `19.0`
    # stored for a reading of 18.89232917906861.
    _with_register(monkeypatch,
                   {"weekday": dict(SYNTHETIC,
                                    binding=float(round(
                                        SYNTHETIC["measured_at"])))})
    real_date = ci._dt.date

    class _Before(real_date):
        @classmethod
        def today(cls):
            return real_date(2026, 11, 1)

    monkeypatch.setattr(ci._dt, "date", _Before)
    rounded = float(round(SYNTHETIC["measured_at"]))
    assert rounded > SYNTHETIC["measured_at"], (
        "this control needs the rounding to go UP to be the dangerous case")
    assert ci.margin_breach_is_acknowledged(
        "weekday", SYNTHETIC["measured_at"]) == {}, (
        "a floor stored at a rounded value absorbed the very reading it was "
        "measured from")
