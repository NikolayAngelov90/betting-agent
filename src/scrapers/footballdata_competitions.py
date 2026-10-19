"""football-data.org's covered competitions — ONE definition, no dependencies.

WHY THIS MODULE EXISTS. The map lived in `footballdataorg_scraper`, which
imports `httpx` at module scope. `scripts/ci_audit.py` is **stdlib-only at
import by design** ("No pip install" in `ci-audit.yml`), and it now needs the
same map to report which leagues the day-ahead reference could have covered.

The two wrong answers were:

  * **duplicate the map** into the deadline module — THE HABIT, one rule with
    two definitions, free to drift the day a competition is added
  * **import the scraper** from the audit — pulls `httpx` into a job that
    installs nothing, so the audit dies at import

So the map moves here and both import it. This module must import NOTHING but
the standard library, and `test_the_competition_module_has_no_dependencies`
asserts that.
"""

from __future__ import annotations

#: football-data.org competition code -> our internal league key.
#:
#: FREE TIER, `measured` 2026-10-19 against `config/config.yaml`'s 30
#: configured leagues: **9 are covered, 21 are not** (the tenth entry, `WC`,
#: is `world/*` and is excluded from every kickoff extremum by the phantom
#: rule, so it is not usable for a deadline and is not in the 30).
#:
#: THE COVERAGE IS THE WHOLE PROBLEM. A minimum over 9 of 30 leagues is an
#: UPPER BOUND on the true minimum, which errs LATE — the unsafe direction.
#: Never substitute it for the conservative constant; combine with `min`.
COMPETITION_MAP = {
    "PL":  "england/premier-league",
    "ELC": "england/championship",
    "PD":  "spain/laliga",
    "BL1": "germany/bundesliga",
    "SA":  "italy/serie-a",
    "FL1": "france/ligue-1",
    "CL":  "europe/champions-league",
    "DED": "netherlands/eredivisie",
    "PPL": "portugal/primeira-liga",
    # International — national team competitions
    "WC":  "world/fifa-world-cup",
}

#: Statuses that mean "will not be played as listed". Only `status` can express
#: these — a kickoff time cannot — so it stays authoritative for exactly this.
STATUS_NOT_PLAYABLE = frozenset(
    {"POSTPONED", "CANCELLED", "SUSPENDED", "AWARDED"})

#: Statuses that mean "not yet played". Anything outside both sets is decided
#: by `utcDate`, never by guessing what the status meant.
STATUS_UPCOMING = frozenset({"SCHEDULED", "TIMED"})

#: Leagues the reference can supply a KICKOFF for. `world/*` is excluded
#: because every kickoff extremum in this project excludes it.
REFERENCE_LEAGUES = frozenset(
    v for v in COMPETITION_MAP.values() if not v.startswith("world/"))

#: `measured` 2026-10-19: |REFERENCE_LEAGUES & configured| / |configured|,
#: where configured is `scraping.flashscore_leagues` in `config/config.yaml`.
#: Re-derived by `test_the_COVERAGE_FRACTION_is_re_derived_from_the_config`, so
#: it cannot drift into a stale literal the way the `LIVE` cron label did.
REFERENCE_COVERAGE_MEASURED = (9, 30)
