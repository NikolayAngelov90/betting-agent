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
