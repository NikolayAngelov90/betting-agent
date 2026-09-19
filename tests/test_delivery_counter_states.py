"""A failed ATTEMPT is not a LOST message — and the counter predated the retry.

    DEL-3 added retry on 2026-09-17. On 09-19 one chunk of five timed out, the
    retry recovered it, all five were delivered, and `ci_audit` reported
    "1 alert(s) failed to deliver".

THE COUNTER PREDATED THE MECHANISM IT COUNTS. Third instance here, after `af=0`
and `fixtures_zero_active` — and both of those cried wolf for days before anyone
acted. This one had a single data point and a known cause, which is the cheapest
moment it will ever be fixed.

THREE STATES, KEPT APART RATHER THAN SUMMED:

    LOST         a part the retry did not recover, or a DEL-1 alert that
                 exhausted its attempts. The only one that alarms.
    RECOVERED    an attempt failed and a later attempt succeeded.
    UNEXPLAINED  a raw failure no REPORT_DELIVERY or ALERT line accounts for.
                 Reported as UNKNOWN — never silently as either.

The third state is the point. A log from before DEL-3 carries no sequence
record, so whether its failure was recovered is genuinely unknowable, and the
audit must say that rather than pick the flattering reading.
"""

import scripts.ci_audit as ci


def _f(log):
    return ci.extract(log)


RD = ("REPORT_DELIVERY report=daily picks chunks={c} sent={s} failed={f} "
      "terminator={t} attempts={a}")


def test_a_recovered_timeout_is_not_a_loss():
    """THE 09-19 CASE. attempts > chunks means a retry succeeded."""
    log = ("Failed to send Telegram message: Timed out\n"
           + RD.format(c=5, s=5, f="none", t="yes", a=6) + "\n")
    f = _f(log)
    assert f["telegram_attempts_failed"] == 1
    assert f["telegram_recovered"] == 1
    assert f["telegram_lost"] == 0
    assert f["telegram_unexplained"] == 0
    assert not [h for h in ci.assertions(f, []) if "LOST" in h or "UNEXPLAINED" in h]


def test_an_unrecovered_part_IS_a_loss_and_alarms():
    log = ("Failed to send Telegram message: Timed out\n"
           + RD.format(c=5, s=4, f="3", t="yes", a=7) + "\n")
    f = _f(log)
    assert f["telegram_lost"] == 1
    hits = ci.assertions(f, [])
    assert any("LOST" in h for h in hits), hits


def test_a_pre_DEL3_failure_is_UNEXPLAINED_not_assumed_either_way():
    """THE 09-13 CASE, replayed. No sequence record exists for it.

    The ledger knows 09-13 lost a middle chunk; the LOG does not carry that,
    and the audit must not invent it. Unknown is the honest reading and it is
    still a finding — silence would be the flattering one.
    """
    f = _f("Failed to send Telegram message: Timed out\n")
    assert f["telegram_attempts_failed"] == 1
    assert f["telegram_lost"] == 0
    assert f["telegram_recovered"] == 0
    assert f["telegram_unexplained"] == 1
    hits = ci.assertions(f, [])
    assert any("UNEXPLAINED" in h for h in hits), hits


def test_a_DEL1_alert_that_exhausted_its_attempts_is_LOST():
    f = _f("ALERT NOT DELIVERED after 3 attempt(s): timeout — surfaced to CI: True\n")
    assert f["telegram_lost"] == 1
    assert any("LOST" in h for h in ci.assertions(f, []))


def test_a_clean_run_produces_no_delivery_finding():
    log = (RD.format(c=1, s=1, f="none", t="yes", a=1) + "\n"
           + RD.format(c=2, s=2, f="none", t="yes", a=2) + "\n")
    f = _f(log)
    assert f["telegram_lost"] == 0 and f["telegram_unexplained"] == 0
    assert not [h for h in ci.assertions(f, []) if "LOST" in h or "UNEXPLAINED" in h]


def test_the_derived_counts_come_AFTER_their_inputs():
    """The first version of this block was computed before the raw counter.

    It read `telegram_attempts_failed` from a dict that did not carry it yet
    and reported 0 recovered — a derived value computed ahead of its inputs is
    a zero that looks like a measurement. Pinned by source order, because the
    symptom is a plausible number rather than an error.
    """
    import inspect
    src = inspect.getsource(ci.extract)
    assert (src.index('f["telegram_attempts_failed"]')
            < src.index('f["telegram_recovered"]')), (
        "the reconcile runs before the counter it reads — it will silently "
        "report 0 recovered")
