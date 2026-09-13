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

### What is predicted

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

*Registered 2026-09-13, before the code.*
