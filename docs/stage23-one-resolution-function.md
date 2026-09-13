# STAGE 23 — ONE RESOLUTION FUNCTION, AND ITS EFFECT REGISTERED FIRST

**Registered 2026-09-13, BEFORE the change is written.**

---

## THE PREDICTION, fixed now so the result cannot be read selectively

`resolve_team()` closes two things and deliberately does not close a third.

| cause | closed by Stage 23? | measured 09-12 | measured 09-13 | two-day |
| --- | --- | --- | --- | --- |
| **SQL NULL-blindness** — `Team.league == None` never true, `NULL NOT IN (...)` is NULL | **YES** | 4 | 3 | **7** |
| **`footballdataorg` calls no comparator at all** | **YES** | — | — | — |
| **alias-needed** — `same_team_strict` refuses even when the pair IS compared | **NO — out of scope** | 14 | 5 | **19** |
| genuinely new club | n/a | 0 | 2 | 2 |
| **total new unresolved rows** | | **18** | **10** | **28** |

### SUPERSEDED 2026-09-13 — see the re-registration below

### What was predicted (registration 1)

> **The resurrection rate FALLS TO APPROXIMATELY THE ALIAS-NEEDED COUNT — a
> mean of 9.5/day against the current ~14/day — and does NOT reach zero.**

**Stated as a composition, because the daily total tracks card size and a raw
count would confound the two:**

| | prediction |
| --- | --- |
| SQL-null-blind rows per day | **0** |
| alias-needed rows per day | **unchanged**, ~9.5 |
| new-club rows per day | unaffected |
| share of the total removed | **~25%** (7 of 28 over the two measured days) |

### What each outcome would mean, fixed in advance

| outcome | reading |
| --- | --- |
| **falls to ≈ alias-needed** | the model is right: two causes closed, one left, and the 75 rulings are the remaining work |
| **falls FURTHER than alias-needed** | something else was contributing that the three-way classification did not separate — **learned cheaply**, and the classification needs re-deriving |
| **falls SHORT — SQL-null-blind rows persist** | **THERE IS A FOURTH CREATION PATH.** Three were found by looking at three; a fourth is not remote. |

**This costs nothing and it has paid every time it has been done here** — the
four schedule predictions, H5's registration, the identity gate's replay, the
branch-2 measurement. **The failure mode it prevents is reading a drop as
success without knowing which cause it removed.**

---

## THE CHANGE

**Three creation paths, three matching regimes, three blind spots — and the
repair for one was itself blind in the reverse direction.**

| path | matching today | blind to |
| --- | --- | --- |
| `flashscore._get_or_create_team` | exact → strict scan, `league == scraped OR league IS NULL` | rows whose league is CONCRETE and different |
| `apifootball._get_or_create_team_id` | provider id → `Team.league.notin_(nat_list)` | **every NULL-league row** |
| `footballdataorg` | exact → prefix → create | **everything** |

> **`Team.league == None` is never true in SQL, so s5.11 — the repair for a
> NULL-blindness — was itself NULL-blind in the other direction.** A patch
> written against the path in front of it produces the next blind spot. That is
> why this is one function and not a fourth patch.

```
resolve_team(session, name, *, league=None, provider_id=None) -> Team
```

1. **provider id** when present — the only identity that is not a string;
2. **exact name**, scoped by the identity partition (national vs club), **never
   by league**;
3. **`same_team_strict`** over candidates selected **without any league
   filter** — league is metadata about a row, not about a club;
4. **create** only when every step refuses.

**The enforcement is what makes this a stage rather than a refactor:** a test
that scans `src/` for `Team(` construction outside this function and fails — the
same move as `test_no_test_writes_prod_state` and
`test_overround_band_is_one_definition`. **A fourth creation site must fail the
suite, not become a fourth blind spot.**

## OUT OF SCOPE, stated so the remedy is not reported as whole

* **the 75 alias rulings** — blocked on classification, and the union shortcut
  was measured and refused;
* **the exact-name collapse** — unsized, and its remedy is *splitting* rows,
  which is harder than merging;
* **cohort** — selection-affecting, so it takes a bump. s5.11 carries picks.

## THE MEASURED DAILY COST — added 2026-09-13

| per day | |
| --- | --- |
| SQL-null-blind rows | **3-4** |
| **unpriced-fixture ALARMS** | **3** |

**A row created without a provider id carries no odds, so the unpriced alarms
are DOWNSTREAM of this defect rather than a separate problem.**
`Freiburg vs M'gladbach` and `Getafe vs Dep. A Coruna` both name rows involved
in it.

> **3 unpriced fixtures/day plus 3-4 SQL-null rows/day, compounding, against a
> merge whose benefit was already measured as decaying.** That is the case - not
> a tidy-up of three code paths, but a measured daily loss of priced fixtures in
> a pipeline whose whole output is priced fixtures.

*Registered 2026-09-13, before the code.*

---

# RE-REGISTERED 2026-09-13, BEFORE THE CODE LANDED

**Registration 1 is kept above with the reason it was superseded: it assumed the
remaining ~75% needed 75 alias rulings. It does not. The resurrections are
EXACT — measured 44 of 44, no diacritic, punctuation or token variance — so an
exact former-name lookup closes them with no comparator, no ratio, no cross
product and none of the rulings.**

## The premise, verified before the design

| | |
| --- | --- |
| resurrections since s5.10 | **44** |
| **EXACT match to a merged-away name** | **44 (100%)** |
| differing by diacritics | 0 |
| differing by punctuation or case | 0 |
| not a merged name at all | 0 |

**The "5 survivor-unidentified" residual in registration 1 was an artefact of my
generated alias table's normalisation, not a real class.**

## The new prediction

| | registration 1 | **registration 2** |
| --- | --- | --- |
| SQL-null-blind rows/day | 0 | **0** |
| alias-needed rows/day | unchanged, ~9.5 | **0** |
| new-club rows/day | unaffected | unaffected |
| **share of resurrections removed** | ~25% | **~100%** |

**If the rate does not fall to ~0, the residual is a mechanism none of the five
steps covers** — and a fifth path after four were found by looking at four.

## WHAT WAS FOUND WHILE BUILDING, and it is the registration paying for itself

**The enforcement test found a FOURTH creation path before shipping:
`src/scrapers/historical_loader.py`.** Registration 1 said a shortfall would
mean "there is a fourth creation path". There was — and the test found it rather
than a resurrection finding it a week later.

**Also found: 28 shared-provider-id duplicate components have regrown, where
s5.10 left ZERO.** `af=193` holds `582 PEC Zwolle || 1784 PEC Zwolle` — an exact
name and an exact provider id on two rows. The resurrected rows are acquiring
provider ids and becoming full duplicate clubs with split history.

> **That is a SECOND repair this stage does not perform.** `resolve_team()` stops
> new ones; the 28 that exist need a merge, and it must record its removed names
> this time. Out of scope, recorded, and the reason the decay measurement must
> continue after this lands.
