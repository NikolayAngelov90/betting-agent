"""The cohort pins, stated once.

Six hardcoded `"s5.2"` literals across five test files was one fact duplicated
six ways: the next bump edits five and misses one, and the miss is silent
because a passing test looks the same as a correct one.

**These values are stated independently of `src/` on purpose.** A pin that
imports the value it is pinning can never fail — it would be a vacuous guard
built into the very mechanism whose job is to fail when a cohort changes. The
literals below must be edited by a human, deliberately, as the act of
authorising a cohort break. That edit is the authorisation.

When you bump these, you are asserting: the change was intended, its effect on
the fingerprint is fully attributable to the edits you made, and picks before
and after are not one cohort.

History:
    s5.2  Stage 8   over/under cross pairs, post-Claude correlation re-check,
                    normalized dedup key
    s5.3  Stage 13  Part B team-identity gate (name + country) and Part C
                    one-pick-per-match with a single ranking definition;
                    also covers a training-data correction, which the
                    fingerprint itself does not cover
"""

#: Must equal src.models.model_version.CODE_REVISION.
CODE_REVISION_PIN = "s5.3"

#: Must equal model_version(config.example.yaml).
FROZEN_MODEL_VERSION = "stage5_baseline_20260807.098437"

#: The previous cohort, kept so a reader can see what moved and when.
PREVIOUS_CODE_REVISION = "s5.2"
PREVIOUS_MODEL_VERSION = "stage5_baseline_20260807.485823"


#: How many completed-match queries are exempt from the training-exclusion
#: filter. Pinned for the same reason CODE_REVISION is: an exemption marker
#: travels with the code, which beats a central list that drifts — but the
#: failure mode inverts. Someone silences the guard by pasting the comment.
#:
#: Shape alone cannot catch that. A count can: adding a seventh exemption
#: requires editing this literal, which is a deliberate act a reviewer sees.
#: Unobserved growth is where this stage kept finding defects.
TRAINING_EXCLUSION_EXEMPTIONS = 6

#: Evidence-gate exemptions: get_stats, the cold-streak alert, and
#: _reset_stale_ml_calibration. Pinned for the same reason as the
#: training-exclusion count.
EVIDENCE_GATE_EXEMPTIONS = 4
