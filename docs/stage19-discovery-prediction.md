# Stage 19 — Discovery Repair: Pre-Registered Prediction

**Written and committed BEFORE the 2026-08-27 09:37 UTC `daily-picks` run.**
At the time of writing, the most recent run of any workflow is
`closing-lines` at 2026-08-27T04:30Z; `daily-picks` has not fired.

The discipline is Stage 17's and Stage 13's: a repair that produces "some
fixtures" is not proven. A repair that produces **the number written down before
looking** is.

## The real card for 2026-08-27

MEASURED 2026-08-27 from two independent public sources, before the run.

**Flashscore fixtures pages, loaded directly via camoufox** (rows counted with
the repaired `.event__match` selector):

| league | rows on page | fixtures dated 27.08. |
| --- | --- | --- |
| **spain/laliga** | 112 | **2** — `21:30 Celta Vigo v Osasuna`, `22:00 Barcelona v Ath Bilbao` |
| england/premier-league | 120 | 0 |
| italy/serie-a | 120 | 0 |
| germany/bundesliga | 117 | 0 |
| france/ligue-1 | 116 | 0 |
| portugal/primeira-liga | 110 | 0 |
| netherlands/eredivisie | 111 | 0 |
| **europe/champions-league** | **0** | **0** |

**football-data.org API, same date:**

```
2 matches total
  PD TIMED 2026-08-27T18:30:00Z  RC Celta de Vigo vs CA Osasuna
  PD TIMED 2026-08-27T19:00:00Z  FC Barcelona vs Athletic Club
```

**The two sources agree exactly.** Same two fixtures, same competition, and
football-data.org's `status` is a clean `TIMED` on both — the malformed value
seen on 2026-08-26 has not recurred.

## THE PREDICTION

1. **Exactly 2 fixtures discovered for 2026-08-27**, both `spain/laliga`:
   Celta Vigo v Osasuna and Barcelona v Athletic Bilbao.
2. **Their kickoffs are whole-minute times near 18:30 and 19:00 UTC** — not
   `now()` stamps. Concretely: `EXTRACT(SECOND FROM match_date) = 0` for both.
   This is the phantom-regression check.
3. **`europe/champions-league` yields 0 and that is CORRECT.** Its Flashscore
   fixtures page returns **zero rows even under camoufox with the repaired
   selector**, so UCL is a *separate, still-open* problem. A UCL absence
   tomorrow must NOT be scored against the selector repair, and must NOT be
   rounded up into success either.
4. **The per-source assertion stays silent for any source that produced**, and
   fires for any that produced within 7 runs and now returns zero.
5. **`model_version` on any pick is `stage5_baseline_20260807.b16ec7`** (`s5.5`).

Leagues outside the eight sampled could in principle add fixtures, so the strict
claim is: **exactly 2 within the sampled set and within football-data.org's nine
competitions**, and any extra must be verifiable against that league's own
calendar before it counts as discovery rather than as a new phantom.

## What counts as what — so the declaration cannot be rounded up

| outcome | definition |
| --- | --- |
| **DISCOVERY RESTORED** | Both LaLiga fixtures present, whole-minute kickoffs, **and Flashscore itself reports a non-zero fixture count for `spain/laliga`** |
| **PARTIAL — football-data.org only** | The 2 fixtures present, but Flashscore still logs `0 fixtures for spain/laliga`. Discovery works; **the selector repair is unproven in CI** and the system is again running on one source |
| **PARTIAL — Flashscore only** | Flashscore returns the fixtures but football-data.org still adds 0. The 2a repair is unproven |
| **NOT RESTORED** | 0 fixtures discovered |
| **REGRESSION** | Fixtures discovered with `EXTRACT(SECOND FROM match_date) <> 0` — the phantom defect has returned under a new guise |

**The largest named risk, stated in advance.** I reproduced and fixed the
selector under **camoufox**; CI runs **Chrome/Selenium under Xvfb**
(`_get_driver: Xvfb detected — running Chrome in headed mode`). The fix is
class-based and should be driver-independent, but the page Chrome is served may
differ from the page camoufox was served. **If Flashscore still reports zero in
CI while camoufox sees 112 rows, that difference is the finding**, and it is a
bot-detection question rather than a selector one.

**A second risk:** `spain/laliga` must be in `_important` to be scraped for
fixtures at all — the circularity recorded in Stage 19 Part B and deliberately
not fixed. If no LaLiga fixture is already known, the league may never be
attempted, and **the repair would go untested rather than fail**. That outcome
is `UNTESTED`, not `NOT RESTORED`, and it would make the circularity the next
thing to fix.

---

*Committed before the run. Stage 19, 2026-08-27.*

---

## ADDENDUM — the measurement instrument, identified before the run

*A registered prediction whose instrument is unidentified is only half
registered. This resolves it, and it INVERTS the risk registered above.*

### Which browser fetches `/fixtures/`

**Chrome/Selenium under Xvfb. Not camoufox.**

`_scrape_fixtures_page` (flashscore_scraper.py:813) calls `self._get_driver()`,
which is the Selenium path (`:122`), and the 2026-08-26 log carries its own
signature: `_get_driver:163 - Xvfb detected — running Chrome in headed mode
(Cloudflare bypass)`. Camoufox (`_cf_browser`, `:290`) serves other paths.

So my reproduction used a **different browser** from CI, exactly as feared.

### But the log settles what the reproduction could not — and in the good direction

`_scrape_fixtures_page` wraps its fetch in `try/except`, and on any failure logs
`Fixtures page failed (<Type>) for <url> — retrying once`.

**That warning appears ZERO times in the 2026-08-26 log.** No exception was
raised, so the `WebDriverWait(driver, 45)` for `.event__match` **was satisfied**
— and `spain/laliga` took **13.0s**, nowhere near a 45s timeout.

> **CI's Chrome fetched and rendered the fixtures page successfully.
> Bot detection is RULED OUT for the domestic leagues.**

### Which means the zero has a complete, different explanation

```python
cutoff = datetime.now() + timedelta(days=1)        # computed BEFORE the loop
...
if match_data["match_date"] > cutoff: continue     # "skip far-future fixtures"
```

The old fixtures-path default was `datetime.now() + timedelta(days=1)`,
evaluated **inside** the loop — microseconds later than `cutoff`. So
`default > cutoff` is **True** (demonstrated: it is), and **every fixture whose
kickoff failed to parse was silently skipped as "far-future".**

**This corrects the Part B framing twice over:**

1. The selector rename **alone would not have zeroed fixtures** — the
   `.event__match` fallback at `:830` already existed and would have found the
   rows. It took *both* halves of the 2026 redesign.
2. The binding cause of `Scraped 0 fixtures` was **the date default colliding
   with the far-future cutoff**, not the selector. The same broken parser
   produced *phantoms* on the results path (default `now()`, no cutoff) and
   *zeros* on the fixtures path (default `now()+1d`, cutoff). **One cause, two
   opposite symptoms, which is why they never looked related.**

### Consequence for tomorrow's declaration

**The registered risk is withdrawn.** A zero tomorrow is **not** bot detection
and **not** "unproven" — Chrome demonstrably reaches the page. A zero would mean
the repair failed, and the outcomes above should be read with that settled:

- `PARTIAL — football-data.org only` now carries a **stronger** implication than
  written above: Flashscore reaching the page but still returning zero would
  mean the parse or cutoff logic is still wrong, which is directly testable.
- The `UNTESTED` outcome (spain/laliga never entering `_important`) is unchanged
  and remains the one result that would prove nothing either way.

*Addendum committed before the run, 2026-08-27.*

---

## PRECONDITION CHECKED — and it was failing, so it was fixed

*Checked before the run rather than registered as a risk. It was not remote: it
was the actual state.*

### What `_important` is, exactly

```python
_today_leagues  = leagues with Match.is_fixture == True AND match_date within TODAY
_pending_leagues = leagues with any SavedPick whose result IS NULL
_important      = _today_leagues | _pending_leagues
```

`_today_leagues` has a **one-day window** and can only be filled by a fixture
scrape. `_pending_leagues` has **no window** but drains as picks settle.

### Evaluated against production, 2026-08-27

| | |
| --- | --- |
| `_today_leagues` | **0 leagues** — no `is_fixture=True` row exists for today |
| `_pending_leagues` | **2** — `europe/champions-league`, `portugal/primeira-liga` |
| **`spain/laliga` in `_important`** | **NO**, by neither path |

**The prediction's outcome would have been `UNTESTED`.** Flashscore would have
been asked for two leagues — one that returns zero rows even under camoufox, and
one with no fixtures today — and never for the league carrying the two fixtures
the whole test rests on.

### What would have put it back, and why that was not good enough

`sync_fixtures` (`betting_agent.py:418`) runs **before** `_important` is computed
(`:458–492`), and `_ensure_fixture` creates rows with `is_fixture=True`
(`footballdataorg_scraper.py:705`). So football-data.org adding the two LaLiga
fixtures **would** have pulled `spain/laliga` into `_today_leagues` in time.

**That is a rescue, not a guarantee, and it depends on the exact source that
failed yesterday.** An instrument whose availability is conditional on the thing
it is measuring is not an instrument.

### Fixed: the circularity is removed (`s5.6`)

`_fixture_leagues` no longer filters on `_important`. Every configured league is
attempted, priority leagues first, bounded by the existing
`_FIXTURES_BUDGET_S = 300` — the same guard the **results** loop has always used
while iterating all 30 leagues. The two paths are now symmetric instead of one
being silently narrower than the other.

`spain/laliga` is **2nd in config order**, so at the measured ~12s/league it is
reached within the first minute.

### The precondition now reads

> **`spain/laliga` WILL be attempted for fixtures, unconditionally, regardless of
> what football-data.org does.**

Which means `UNTESTED` is off the table, and **tomorrow's zero means exactly one
thing: the repair failed.** That is the position this stage has been working
toward, and it is now reached rather than hoped for.

*Precondition checked and fixed before the run, 2026-08-27.*
