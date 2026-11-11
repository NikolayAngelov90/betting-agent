# The Absence Sweep — Registration

**Registered 2026-11-05. NOT PERFORMED.** This exists so the sweep is a stage
with a shape rather than a recurring surprise. It is scoped to run *after* the
market-mix stage; nothing in it is time-sensitive.

> **Every place the code concludes "nothing" must be able to distinguish
> "could not look".** For a guard that proves an absence, those two must never
> share an output — and for a security guard, a false clean is the worst
> available direction.

---

## WHY THIS IS A CLASS AND NOT TEN BUGS

Ten instances have been found one at a time, each by paying for it. They are
one defect wearing ten syntaxes: **a read that fails is converted into a value
that means "nothing there", and the conversion is invisible at the call site.**

| # | syntax | the instance, with its date | what it cost |
|---|---|---|---|
| 1 | `or ""` | `ci_audit._sh` returns `r.stdout or ""` (one site, **15 `_sh` callers, 10 of them `gh`**) | a 401, a 403 and a rate limit all read as zero rows; still live — see **the registered residual** below |
| 2 | `or "[]"` | `json.loads(o.stdout or "[]")` in the two live `gh` callers | unreadable read as `n=0`; fixed 2026-11-05 |
| 3 | `.stdout` is `None` | `text=True` decoding a CI log with the cp1251 console codec killed the reader thread | three instances in Stage 14; the DEL-2 harness and `ci_alert`'s emoji were the others |
| 4 | `pytest.skip` on an unreadable input | three sites; a skip and a pass are indistinguishable in a green run | a consumer could be broken AND green; fixed 2026-11-05 |
| 5 | a matcher with zero hits | **39 of 116** matcher sites unguarded | four incidents; `anchored`/`must_match` cover two of the three shapes |
| 6 | a regex that cannot match | `\b` written through a non-raw string became a BACKSPACE byte | the pattern matched nothing and `assert not offenders` passed |
| 7 | an empty selection | a third `pytest_collection_modifyitems` made the marker hook dead, so `-m capture_critical` selected **nothing, with no error** | the gate ran zero tests |
| 8 | `any([0, 0])` | a cron is MINUTES PAST MIDNIGHT, so `0 0` is `0` and `any` is `False` | `missing_scheduled_runs()` returned `UNEVALUATED` on **every call for a day**, silently, killing the detector built for the 422 window |
| 9 | a silent SQLite fallback | an empty SQLite file answers every query with a plausible zero | **three confident wrong answers**; now refused unless `BETTING_AGENT_ALLOW_SQLITE=1` |
| 10 | `git log` under a platform codec | the chat-id history guard captured with `text=True` and no encoding | **the guard written to prove a credential absent would have reported clean because it could not read**; fixed 2026-11-05 |
| **11** | **a size limit that truncates instead of erroring** | `MEMORY.md` is loaded into every session with a **200-line cap**. It reached 202 lines on 2026-11-07 and the loader **dropped the tail and carried on** — one line of user preferences vanished from context with no error, no exception and nothing in the file to show it | **the preference line was absent from the session that was acting on the file**; found only because the host emitted a warning that happened to be read. Fixed by moving ~90 lines of reference detail to `project-reference.md`, taking the index to 101 lines |

Instances 2, 4 and 10 were fixed on the day this class was named. **1 is
registered and deliberately not fixed** (below). 3, 6, 7, 8 and 9 have guards;
5 is partially guarded and carries its own count.

---

## 1. THE PREDICATE

**Not a string match on a name, and not a claim to decide reachability.**
`scripts/falsy_zero_scan.py` is the precedent and its reasoning carries over:
whether a quantity can legitimately be zero depends on what produces it, so a
scanner that *decided* that would itself be the confident-wrong-answer shape.

The sweep's scan is **AST-based, with the anchor asserted**, and it flags a
site where *both* of these hold:

1. **a read that can fail** — `subprocess.run`/`check_output`/`Popen`,
   `json.loads`, `re.search`/`match`/`findall`, a DB query, a file read, an
   env lookup, or a call into one of this project's own readers (`_sh`,
   `fetch_log`, `load_observations`, …); **and**
2. **a conversion of that failure into a value** — `or <falsy literal>`,
   a bare `except` returning a default, `.get(k, <default>)` on a result that
   was read rather than constructed, `pytest.skip`, `return []`/`{}`/`0`/
   `None` inside the failure branch, or truthiness on a numeric quantity.

Every hit is **triaged by hand** into:

* `REACHABLE` — fixed, with a known-wrong control;
* `UNREACHABLE` — listed **with its reason**, which is a claim someone can
  later falsify;
* `BY DESIGN` — the default is correct because the value is a collection and
  empty genuinely means nothing (`if not rows:`), or because the third state
  is already reported separately.

**Deliberately not flagged**, carried over from `falsy_zero_scan`: `x = x or {}`
on a collection, and `if not rows:` on a list. For a collection, empty and
absent are the same fact. For a *number*, a *read*, or an *absence proof*,
they are not.

---

### INSTANCE 11 IS NOT A SITE IN THIS REPOSITORY, AND THAT IS THE POINT

Every other instance is a line of our code that we can guard. This one is a
**host limit on a file we write**: the truncation happens outside the
repository, so there is no predicate here that can catch it and no test that
can fail on it.

> **The class is about the SHAPE, not the syntax.** "Content disappears and
> nothing errors" is the same defect whether it is `or ""` swallowing a 401
> or a line cap swallowing a preference. Registering it here is the only
> control available: the next person to add an index line knows the cap
> exists and knows what exceeding it does.

**THE RULE IT YIELDS**, stated so it survives this session: `MEMORY.md` is an
INDEX — one line per memory, under ~200 characters each, and reference detail
belongs in a topic file. A growing index is not a neutral act; past the cap it
**silently deletes its own tail**, and the entries nearest the end are the
ones most recently added.

## 2. THE THREE OUTCOMES EVERY SUCH SITE MUST HAVE

A site that can conclude "nothing" must be able to return three states, and
the caller must be able to tell them apart:

| state | meaning | may it be silent? |
|---|---|---|
| **PRESENT** | looked, and here is the value — **including zero** | yes |
| **ABSENT** | looked, and there is genuinely nothing; `n=0` is a measurement | yes, if reported as a measurement |
| **UNEVALUATED** | **could not look** | **never** |

Three rules follow, and they are what the sweep asserts:

1. **`UNEVALUATED` is reported, never returned as a value.** It may be a
   raised `AssertionError` led by the word, a `state` field, or a printed
   alarm — but it may not be `0`, `[]`, `""`, `None` or a skip.
2. **`ABSENT` and `UNEVALUATED` never share an output.** This is the whole
   rule for a guard that proves an absence. Instance 10 is the worst case in
   the table because the output they shared was *clean*.
3. **A read reports its own failure.** `returncode` **and** `stderr`, not
   `stdout`. A reader that discards why it failed forces every caller to
   re-derive it, and they will all re-derive it the same wrong way.

---

## 3. THE KNOWN-WRONG CONTROLS, ALREADY AVAILABLE

The sweep does not need to invent its controls: each instance is its own
known-wrong input, and nine of the ten are reproducible without live
infrastructure.

| instance | known-wrong input | required outcome |
|---|---|---|
| 1 `or ""` | stub a `gh` call returning rc=1 with an auth stderr | `UNEVALUATED` naming rc and stderr |
| 2 `or "[]"` | stdout `""` with rc 0 | `UNEVALUATED`, not `n=0` |
| 3 codec | a byte sequence invalid in cp1251 | decoded with replacement, never `None` |
| 4 skip | a dead token | reported; **not skipped and not passed** |
| **11 truncation** | a file one line over the cap | **the limit reports which lines were dropped, or the file is kept under it**; silence plus a shorter file is the defect |
| 5 matcher | a haystack missing its anchor | `UNEVALUATED, not clear` |
| 6 regex | the pattern against a **planted positive** | ≥1 hit, else the pattern is refused |
| 7 selection | a marker expression selecting zero node ids | refuse the run |
| 8 `any` | every quantity set to `0` | `0` is PRESENT, not ABSENT |
| 9 SQLite | `DATABASE_URL` unset | refuse, as it already does |
| 10 `git log` | a commit subject with a non-cp1251 byte | the search still reads it |

Controls 4 and 10 shipped on 2026-11-05 and both are driven, not asserted:
runs `42022239292` (positive) and `42022279541` (known-wrong).

---

## 4. THE SWEEP'S OWN SCAN IS SUBJECT TO ITS OWN RULE

Stated because this project has already paid for the exception twice —
`pnc_scan`'s `KNOWN_INSTANCES` table read as the evidence it was scanning, and
a matcher's own zero hits certified a clean tree.

* **Zero findings is `UNEVALUATED` until the predicate is proved against a
  planted positive.** The scan ships with planted examples for each of the two
  clauses in §1, checked at load, the way every `ci_audit` PATTERN already is
  (35/35).
* **The triage register is not evidence.** A site listed `UNREACHABLE` must
  carry a reason, and the scan must not treat its own register as a reason to
  stop looking.
* **The scan reports its own denominator.** "39 of 116" is a finding; "39" is
  a number without a population.
* **Its node-id effect on the capture gate is measured, not counted** — a new
  shared module must not recruit tests into the gate. Four occurrences so far;
  the fourth, 2026-11-05, was caught by `FOUR_LOSSES` rather than by a card.

---

## THE REGISTERED RESIDUAL — instance 1, deliberately not fixed

`ci_audit._sh` returns `r.stdout or ""`. **15 call sites, 10 of them `gh`.**

It is the reason a token that is **present but dead** still reaches the
`pytest.skip` in `test_cron_split_by_daytype`: the lag series is built through
`_sh`, so a 401 arrives as an empty series, and an empty series with a token
set is indistinguishable from one that is refilling. The two callers that were
moved to `tests/gh_support.gh_json_or_unevaluated` **do** catch it; this one
cannot until `_sh` reports its own returncode.

**It is not fixed here because the fix is not local.** Making `_sh` return a
result object changes 15 call sites, several of which already have their own
third-state handling (`_unevaluated`, `PROVISIONAL_VERDICTS`,
`cluster_state`), and merging two third-state vocabularies in the same change
as the sweep would make both unverifiable. It is instance 1 of the class and
belongs in the sweep, with the controls in §3.

## WHAT THIS DOES NOT DO

Adopts nothing, changes no config, moves no threshold, spends no credits, and
touches no registered statistic. ρ=0.42, σ, `TARGET_N`, the floor and the
168-credit reservation are unrelated and untouched. `CODE_REVISION` stays
`s5.19` — nothing here is a cohort event.
