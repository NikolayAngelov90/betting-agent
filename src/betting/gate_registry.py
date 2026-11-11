"""Registry of every outcome-derived betting gate, with provenance and status.

Stage 3, Phase 4. The 2026-08-07 audit found ~15 rules in the pick pipeline whose
only justification was a code comment citing a settled-pick sample of 30-100 bets,
chosen by inspecting the outcomes those same bets produced. A permutation test on
the league-level ROI spread returned p = 0.407, and no market, agreement level or
review action reached p < 0.15 against its break-even rate — so those samples
could not support the rules built on them.

Rather than delete the rules outright, each one is declared here with:

* what it removes,
* the evidence originally cited for it,
* the walk-forward verdict from ``scripts/validate_gates.py``,
* whether it is enabled in production.

Two categories are treated differently, because they are different kinds of claim:

**A-priori risk constraints** (odds range, stake caps, divergence sanity, per-match
and per-league caps) are statements about variance and bankroll, not about which
bets win. They need no outcome evidence and stay on.

**Empirical edge claims** ("market X loses", "cohort Y underperforms") are exactly
what walk-forward validation is for. Every one of them was re-tested on a holdout
window it was never fitted on (train 2026-02-28..04-26, validate ..06-17, holdout
..08-05). None survived: six returned INSUFFICIENT EVIDENCE, and six could not be
tested at all because the gate was already live, so the cohort it excludes has no
holdout picks — an object lesson in why an active filter cannot be validated from
production data.

They are therefore **disabled by default**. Set the flag in
``betting.gates.<name>`` to re-enable one, and record new evidence here when you do.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.utils.logger import get_logger

logger = get_logger()


@dataclass(frozen=True)
class Gate:
    """One declared gate."""

    name: str
    removes: str
    kind: str                  # "risk" (a-priori) or "edge" (outcome-derived)
    original_evidence: str
    holdout_verdict: str
    holdout_detail: str = ""
    default_enabled: bool = False
    #: Overall holdout ROI change from applying this gate, in percentage points.
    #: Negative means the gate COST money on data it was not fitted on.
    holdout_roi_delta_pp: Optional[float] = None


#: Verdict vocabulary, so the strings cannot drift.
KEEP = "KEEP — cohort underperforms out of sample (bootstrap CI < 0)"
INSUFFICIENT = "INSUFFICIENT EVIDENCE — bootstrap CI spans zero"
UNTESTABLE = "UNTESTABLE — gate already live, so no holdout picks in the cohort"
APRIORI = "N/A — risk constraint, not an edge claim"


REGISTRY: Dict[str, Gate] = {g.name: g for g in [
    # ---------------------------------------------------------------- risk
    Gate(
        name="min_odds",
        removes="selections priced below betting.min_odds (1.50)",
        kind="risk",
        original_evidence="a-priori: short prices leave no margin for model error",
        holdout_verdict=APRIORI,
        default_enabled=True,
    ),
    Gate(
        name="max_odds",
        removes="selections priced above betting.max_odds (10.0)",
        kind="risk",
        original_evidence="a-priori: variance control",
        holdout_verdict=APRIORI,
        default_enabled=True,
    ),
    Gate(
        name="divergence_sanity",
        removes="picks where model probability exceeds 2x the market's",
        kind="risk",
        original_evidence="a-priori: a 2x disagreement indicates a data fault, "
                          "not an edge",
        holdout_verdict=APRIORI,
        default_enabled=True,
    ),
    Gate(
        name="min_kelly_stake",
        removes="picks whose Kelly stake is below betting.min_kelly_stake",
        kind="risk",
        original_evidence="a-priori: sub-0.5% stakes are not worth the variance",
        holdout_verdict=APRIORI,
        default_enabled=True,
    ),

    # ----------------------------------------------------------- edge claims
    Gate(
        name="exclude_over_3_5",
        removes="Over 3.5 Goals picks",
        kind="edge",
        original_evidence="config comment: 'proven loser in settled data: 38% win, "
                          "-14% ROI over 32 picks'",
        holdout_verdict=UNTESTABLE,
        holdout_detail="n=14 in train at -54.3% ROI; 0 picks in the holdout "
                       "because the exclusion was already live. Binomial test on "
                       "the full 33-pick history vs break-even: p = 0.299.",
        holdout_roi_delta_pp=0.0,
        default_enabled=False,
    ),
    Gate(
        name="exclude_under_2_5",
        removes="Under 2.5 Goals picks",
        kind="edge",
        original_evidence="settled history cited at -32% ROI (n=11)",
        holdout_verdict=UNTESTABLE,
        holdout_detail="n=11 lifetime. Far below any usable sample size.",
        holdout_roi_delta_pp=0.0,
        default_enabled=False,
    ),
    Gate(
        name="exclude_under_3_5",
        removes="Under 3.5 Goals picks",
        kind="edge",
        original_evidence="settled history cited at -16% ROI (n=45)",
        holdout_verdict=UNTESTABLE,
        holdout_detail="4 holdout picks, bootstrap CI [-100%, +26.7%].",
        holdout_roi_delta_pp=+0.91,
        default_enabled=False,
    ),
    Gate(
        name="club_btts_yes_ban",
        removes="BTTS Yes as a club forced pick",
        kind="edge",
        original_evidence="'worst club selection (33% win, -42% ROI since 6/11 on "
                          "top of 44% at the WC)' — 32 picks",
        holdout_verdict=INSUFFICIENT,
        holdout_detail="holdout cohort n=11, ROI -20.6%, CI [-69.4%, +28.9%]. "
                       "Lifetime binomial vs break-even: p = 0.582.",
        holdout_roi_delta_pp=+0.91,
        default_enabled=False,
    ),
    Gate(
        name="club_pick_min_ev",
        removes="club forced picks whose model EV is below -5%",
        kind="edge",
        original_evidence="'41 club forced picks since 2026-07-08: EV >= -5% won "
                          "61.5% (+8.5% ROI); below -5% won 53.6% (~-13% ROI)'",
        holdout_verdict=INSUFFICIENT,
        holdout_detail="the excluded cohort returned +4.3% in the holdout "
                       "(n=74, CI [-14.6%, +21.6%]); applying the gate cost "
                       "1.95pp of overall holdout ROI.",
        holdout_roi_delta_pp=-1.95,
        default_enabled=False,
    ),
    Gate(
        name="club_pick_min_blend",
        removes="club forced picks whose 50/50 model+market blend is below 55%",
        kind="edge",
        original_evidence="'a 46%-conf forced pick lost 2026-07-14; that profile "
                          "wins ~37% historically'",
        holdout_verdict=INSUFFICIENT,
        holdout_detail="excluded cohort +0.4% in the holdout (n=90, "
                       "CI [-22.0%, +22.0%]); gate cost 0.46pp.",
        holdout_roi_delta_pp=-0.46,
        default_enabled=False,
    ),
    Gate(
        name="wc_mismatch_routing",
        removes="BTTS and underdog markets on strong-favourite fixtures, "
                "re-routing to the favourite's goal lines",
        kind="edge",
        original_evidence="four World Cup routs (France 3-0 Iraq, Canada 6-0 "
                          "Qatar, Brazil 3-0 Haiti, Spain 4-0 Saudi)",
        holdout_verdict=UNTESTABLE,
        holdout_detail="the World Cup ended 2026-07-19 and was removed from the "
                       "league config; the rule can no longer fire or be tested.",
        default_enabled=False,
    ),
    Gate(
        name="split_agreement_low_conf",
        removes="picks where models are split AND probability is below "
                "min_confidence",
        kind="edge",
        original_evidence="'this combination typically indicates miscalibration'",
        holdout_verdict=INSUFFICIENT,
        holdout_detail="split picks returned -14.4% in the holdout (n=55, "
                       "CI [-39.3%, +11.2%]) — the most promising of the edge "
                       "gates, but still not significant.",
        holdout_roi_delta_pp=+3.82,
        default_enabled=False,
    ),
]}


def is_enabled(config, name: str) -> bool:
    """Whether a declared gate is active.

    Resolution order: ``betting.gates.<name>`` in config, else the registry
    default. Unknown names are refused loudly rather than silently treated as
    disabled — a typo in a gate name should not quietly change behaviour.
    """
    gate = REGISTRY.get(name)
    if gate is None:
        raise KeyError(
            f"Unknown gate {name!r}. Declare it in gate_registry.REGISTRY with "
            f"its provenance and holdout verdict before using it."
        )
    try:
        configured = config.get(f"betting.gates.{name}", None)
    except Exception:
        configured = None
    return gate.default_enabled if configured is None else bool(configured)


def describe(name: str) -> str:
    """One-paragraph provenance for logs and reports."""
    g = REGISTRY[name]
    lines = [
        f"{g.name} [{g.kind}] — removes {g.removes}",
        f"  original evidence : {g.original_evidence}",
        f"  holdout verdict   : {g.holdout_verdict}",
    ]
    if g.holdout_detail:
        lines.append(f"  detail            : {g.holdout_detail}")
    if g.holdout_roi_delta_pp is not None:
        lines.append(
            f"  holdout ROI effect: {g.holdout_roi_delta_pp:+.2f}pp "
            f"({'cost' if g.holdout_roi_delta_pp < 0 else 'gain'})")
    lines.append(f"  default           : {'ON' if g.default_enabled else 'OFF'}")
    return "\n".join(lines)


def summary() -> str:
    """Registry overview for `--stats` / audit output."""
    risk = [g for g in REGISTRY.values() if g.kind == "risk"]
    edge = [g for g in REGISTRY.values() if g.kind == "edge"]
    on = [g.name for g in edge if g.default_enabled]
    enabled_note = (
        ", ".join(on) if on
        else "none — none survived walk-forward validation"
    )
    return (
        f"Gate registry: {len(risk)} a-priori risk constraints (always on), "
        f"{len(edge)} outcome-derived edge claims, of which {len(on)} are enabled "
        f"by default ({enabled_note})."
    )
