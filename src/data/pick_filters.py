"""The two pick predicates, defined once.

Stage 13 (s5.3). These lived inside `betting_agent` as module-level functions,
which meant every other module that needed them either imported the agent or
wrote its own filter by hand. `match_briefing` did the latter — and got it
wrong, in a way three audits missed.

The two questions are orthogonal and must stay separately answerable:

    live_only()       did this WAGER HAPPEN? gates the money record.
    valid_evidence()  does this observation say anything true ABOUT THE MODEL?
                      gates everything that learns or measures.

A learner needs both. The settled ROI record needs only the first — the money
was real whatever informed it.
"""

from sqlalchemy import and_ as _and
from sqlalchemy import or_ as _or

from src.data.models import SavedPick


def live_only():
    """Real wagers only — paper picks and superseded picks excluded.

    Paper picks are measurement-only: letting them into anything that adjusts
    the model would let the frozen experiment retrain its own subject.
    Superseded picks (`disposition` set) were never placed at all.
    """
    return _and(
        _or(SavedPick.is_paper.is_(False), SavedPick.is_paper.is_(None)),
        SavedPick.disposition.is_(None),
    )


def valid_evidence():
    """Observations that say something true about the model that made them.

    A pick whose features were computed from the wrong club is a real wager at
    a real price on a real fixture — it belongs in the ROI record — but the
    model was never shown that match, so its outcome teaches nothing about the
    model and must not be learned from or measured.
    """
    return SavedPick.evidence_status.is_(None)
