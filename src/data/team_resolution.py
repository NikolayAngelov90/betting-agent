"""ONE resolution function, called by every site that needs a team row.

Stage 23. Three creation paths existed, with three different matching regimes
and three different blind spots, and s5.10's merge survivors were invisible to
all three for three different reasons:

    flashscore      exact name, then same_team_strict scoped `league = scraped`
                    -> blind to every row whose league differs or is NULL
    apifootball     provider id, then `Team.league.notin_(national_leagues)`
                    -> `NULL NOT IN (...)` is NULL, so blind to NULL-league rows
    footballdataorg exact name, then a prefix guess, then create
                    -> calls no comparator at all

**And the patch for one was itself blind in the reverse direction**:
`Team.league == None` is never true in SQL, so s5.11 — written to fix a
NULL-blindness — fixed only "survivor NULL, scraped concrete" and left
"scraped NULL, survivor concrete". A patch written against the path in front of
it produces the next blind spot. Hence one function, not a fourth patch.

WHAT IT COST, measured before this was built:

    44 resurrections in 3 days, 100% of them EXACT matches to a name s5.10
    removed; 28 shared-provider-id duplicate components regrown where the merge
    left ZERO; 3 unpriced-fixture alarms a day, because a row created without a
    provider id carries no odds.

THE ORDER, and why each step is where it is:

    1. PROVIDER ID      proof. Not a string, not a guess.
    2. FORMER NAME      exact. The names a merge removed, recorded when it
                        removed them. This is the step that stops the decay.
    3. EXACT NAME       scoped by identity partition, never by league.
    4. same_team_strict over candidates chosen WITHOUT a league filter.
    5. CREATE           only when every step above refuses.

**Step 2 is exact by design.** The resurrections are exact — measured, 44 of 44,
no diacritic or punctuation variance — so an exact match closes them without a
comparator. Using `team_names_similar` here was considered and refused: unioning
it was measured on 2026-09-13 and produced 38 new matches of which roughly half
are absurd (`ac milan` == `manchester utd`, `cremonese` == `usa`), because it is
a RATIO and the raw-versus-aliased cross product invents comparisons neither
pure form performs. **An exact lookup has no ratio, no cross product and no
deleted-token hazard**, so none of the 75 alias rulings are needed for it.

**Step 2 refuses on ambiguity**, exactly as s5.10 refused the two `Sporting
Clube` rows rather than guessing between `Sporting CP` and `Braga`. The
`team_former_names` primary key makes a name resolve to one club or fail.

LEAGUE IS METADATA ABOUT A ROW, NOT ABOUT A CLUB. No step filters by it. The
identity partition (national teams versus clubs) is a real boundary and is
preserved; `league` is not, and treating it as one is what created this stage.
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import text

from src.data.models import Team
from src.utils.logger import get_logger
from src.utils.team_names import same_team_strict

logger = get_logger()


def _partition_filter(query, league: Optional[str]):
    """Restrict a lookup to the fixture's identity space.

    National teams and clubs live in disjoint identity spaces. Without this,
    fuzzy matching crossed them: the USA national team matched the club
    "Lausanne" and a World Cup fixture was priced with Swiss club history.

    `notin_` is NULL-unsafe — `NULL NOT IN (...)` is NULL, not TRUE — so a
    NULL-league row would vanish from every club lookup. That is one of the
    three blind spots this module exists to remove, so the NULL case is spelled
    out rather than left to SQL's three-valued logic.
    """
    from src.models.poisson_model import NATIONAL_TEAM_LEAGUES

    nat = list(NATIONAL_TEAM_LEAGUES)
    if league in NATIONAL_TEAM_LEAGUES:
        return query.filter(Team.league.in_(nat))
    return query.filter(
        (Team.league.is_(None)) | (Team.league.notin_(nat)))


def lookup_former_name(session, name: str) -> Optional[int]:
    """team_id a REMOVED name now resolves to, or None.

    Fails open: if the table does not exist yet (migration 010 unapplied) this
    returns None and the caller falls through to its remaining steps. The
    pipeline degrades to pre-Stage-23 behaviour rather than breaking — the
    resurrections come back, which is the observable cost.
    """
    try:
        row = session.execute(
            text("SELECT team_id FROM team_former_names WHERE name = :n"),
            {"n": name}).fetchone()
        return int(row[0]) if row else None
    except Exception as e:                       # pragma: no cover - env-dependent
        logger.debug(f"team_former_names unavailable ({e}) — skipping step 2")
        return None


def record_former_name(session, name: str, team_id: int,
                       revision: str = None) -> None:
    """Called BY A MERGE, at the moment it removes a name.

    THE LESSON THIS ENFORCES: an operation that removes an identifier must
    record it, or it leaves behind the reason the identifier existed. s5.10 did
    not, and 44 rows came back in three days.
    """
    try:
        session.execute(text("""
            INSERT INTO team_former_names (name, team_id, source, revision)
            VALUES (:n, :t, 'merge', :r)
            ON CONFLICT (name) DO NOTHING
        """), {"n": name, "t": team_id, "r": revision})
    except Exception as e:                       # pragma: no cover
        logger.warning(f"could not record former name {name!r}: {e}")


def _conflicts(candidate, provider_id) -> bool:
    """True when a candidate is PROVABLY a different club.

    Two rows carrying DIFFERENT provider ids are different clubs by the
    provider's own assertion, whatever their names say. `Arsenal` (England) and
    `Arsenal` (Argentina) both exist and are not the same club.

    Caught by `test_exact_name_but_different_api_id_creates_new_team`, which is
    the pin left by the 2026-07-22 dedup work: name-first matching split 53
    clubs, and id-first matching is what stopped it. A name match must never
    override a provider-id mismatch.
    """
    return (provider_id is not None
            and candidate.apifootball_team_id is not None
            and candidate.apifootball_team_id != provider_id)


def _country_conflict(candidate, country) -> bool:
    """True when two rows name DIFFERENT real countries.

    THE SAME CHECK THE AF-ID GATE USES, not a second one. It refused
    `Rapid Vienna` against `Rapid Bucuresti` and `Pau FC` against `St. Pauli`,
    and it is deliberately shaped to FAIL OPEN: refuse only when both sides name
    a real country and they differ.

    That shape is not timidity. `teams.country` records where a club was FIRST
    SEEN rather than where it plays — Levski Sofia is stored as "Europe" because
    a Conference League tie created it — so refusing on a missing or continental
    value would reject legitimate fixtures wholesale. 44% of rows carry a real
    country; the rest fall through.

    APPLIED HERE 2026-09-13 to close an asymmetry, not to fix an observed
    failure: the AF-id gate refused a cross-country match while this name path
    permitted one. Measured before applying — of the 25 name-path joins
    `resolve_team` could currently make, it refuses ZERO. It costs nothing today
    and stops `Arsenal` (England) from ever matching `Arsenal FC` (Argentina),
    which was otherwise left to be discovered by biting.
    """
    from src.scrapers.apifootball_scraper import is_a_real_country

    return (is_a_real_country(country)
            and is_a_real_country(candidate.country)
            and str(country).strip().lower()
            != str(candidate.country).strip().lower())


def _record(name: str, team, step: str, league: Optional[str]) -> None:
    """ONE structured line per resolution. THE INCOMING NAME BESIDE THE ROW.

    WHY THIS EXISTS, and why it is worth a line on every resolution: until now
    the database stored only the row a name RESOLVED TO, never the name that
    came in. So "how many resurrection attempts were there" could not be asked
    — this card or any future one — and on 2026-09-14/15 both the creations and
    the step-2 interceptions were ZERO. **Zero attempts and zero failures are
    the same observation**, and the p = 0.004 on the creation drop rested on
    fixtures-created being a fair exposure measure, which that ambiguity is
    precisely the evidence against.

    With this line the next card answers the two separately:

        attempts    = lines whose `name` is in `team_former_names`
        intercepted = those with `step=former_name`
        residual    = those with `step=create`

    It also closes a gap named in Stage 24's registration: only step 2
    announced itself, so an attribution for steps 1, 3 and 4 had to be argued
    from `league IS NULL` in the data rather than read from a log.

    DEBUG, because it fires once per team per scrape (~90-180 a run) and
    DEBUG is confirmed to reach CI logs. It carries no PII and no secret.
    """
    logger.debug(
        f"TEAM_RESOLVE name={name!r} step={step} "
        f"team={getattr(team, 'id', None)} "
        f"resolved={getattr(team, 'name', None)!r} league={league!r}")


def resolve_team(session, name: str, *, league: str = None,
                 provider_id: int = None, country: str = None,
                 create: bool = True):
    """Resolve a scraped team name to a Team row. THE single entry point.

    Returns the Team, or None when `create=False` and nothing matched.
    """
    # 1. PROVIDER ID — proof, when present.
    if provider_id:
        t = (_partition_filter(session.query(Team), league)
             .filter(Team.apifootball_team_id == provider_id)
             .order_by(Team.id).first())
        if t:
            _record(name, t, "provider_id", league)
            return t

    # 2. FORMER NAME — the step that stops the decay.
    tid = lookup_former_name(session, name)
    if tid is not None:
        t = session.get(Team, tid)
        if t is not None:
            logger.debug(
                f"resolve_team: {name!r} is a FORMER NAME of team {t.id} "
                f"({t.name!r}) — not creating a duplicate")
            if provider_id and not t.apifootball_team_id:
                t.apifootball_team_id = provider_id
            _record(name, t, "former_name", league)
            return t

    # 3. EXACT CURRENT NAME, partition-scoped, never league-scoped.
    t = (_partition_filter(session.query(Team), league)
         .filter(Team.name == name)
         .order_by(Team.id).first())
    if t and (_conflicts(t, provider_id) or _country_conflict(t, country)):
        t = None          # provably a different club: provider id or country
    if t:
        if provider_id and not t.apifootball_team_id:
            t.apifootball_team_id = provider_id
        _record(name, t, "exact_name", league)
        return t

    # 4. same_team_strict over candidates chosen WITHOUT a league filter.
    for cand_id, cand_name in _partition_filter(
            session.query(Team.id, Team.name), league).order_by(Team.id):
        if same_team_strict(name, cand_name):
            t = session.get(Team, cand_id)
            if t is not None and (_conflicts(t, provider_id)
                                  or _country_conflict(t, country)):
                continue
            if provider_id and t is not None and not t.apifootball_team_id:
                t.apifootball_team_id = provider_id
            _record(name, t, "strict", league)
            return t

    # 5. CREATE.
    if not create:
        _record(name, None, "no_match", league)
        return None
    t = Team(name=name, league=league)
    if country:
        t.country = country
    if provider_id:
        t.apifootball_team_id = provider_id
    session.add(t)
    session.flush()
    _record(name, t, "create", league)
    return t
