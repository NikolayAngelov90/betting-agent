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
    s5.9  2026-08-31  the per-match cap keys on FIXTURE identity, not ROW
                    identity. s5.3 established one-pick-per-match; two rows
                    for one fixture defeated it silently, measured on
                    2026-08-30 (Deportivo v Valencia, two picks). 835 pairs
                    match the provable branch; exactly 2 second-picks would
                    have been refused out of 1,458.
    s5.11 2026-09-11  the merge's survivors were invisible to the duplicate
                    check. flashscore._get_or_create_team scanned
                    filter_by(league=<scraped>) and every s5.10 survivor carries
                    league IS NULL, so same_team_strict was never called against
                    it. 26 of 129 merged rows were re-created within a day.
                    Scan now covers league == scraped OR league IS NULL.
    s5.10 2026-09-10  Stage 22 team identity repair. 2 wrong provider ids
                    cleared, 42 rows absorbed into 40 provable components,
                    86 unresolved rows merged into evidenced twins;
                    teams 1561 -> 1433. Fixtures whose both participants
                    resolve: 79.86% -> 89.18% since 2026-08-01, 83.38% ->
                    94.26% over 365 days. SELECTION-AFFECTING: it changes
                    which fixtures resolve and therefore which are picked,
                    and Elo/Poisson key on team_id, so a club whose history
                    was split now trains as one club.
"""

#: Must equal src.models.model_version.CODE_REVISION.
CODE_REVISION_PIN = "s5.11"

#: Must equal model_version(config.example.yaml).
FROZEN_MODEL_VERSION = "stage5_baseline_20260807.32df36"

#: The previous cohort, kept so a reader can see what moved and when.
PREVIOUS_CODE_REVISION = "s5.10"
PREVIOUS_MODEL_VERSION = "stage5_baseline_20260807.dfd410"


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
