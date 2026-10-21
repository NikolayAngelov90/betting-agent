# Workflow inventory — 2026-10-20

A workflow is a producer. The same three outcomes apply as to any producer: a
**live consumer**, a **registered question with a date**, or it runs against
something **closed**.

**SIX AT THE TIME OF WRITING; SEVEN SINCE 2026-10-21.** The directive asked
for seven and `.github/workflows/` held six, stated rather than padded. A
seventh was then created deliberately: **`reminders.yml`**, which took the
calendar and reminder assertions out of `daily-picks` so that one run
carries one fact. It is registered in the audit's `WORKFLOWS` in the same
commit that created it — `ci-audit.yml` went a month unregistered and eight
of its red runs survived unrecorded in the gap.

| **reminders** (added 2026-10-21) | `workflow_run` on daily-picks `completed`, `0 13 * * *` backstop, dispatch | — | ~2 (est.) | — | none | `logs/reminders.txt` → artifact | **Niki — Telegram when a decision is due**, worded so it cannot be mistaken for a lost card | is a self-expiring guard due? **OPEN** | **KEEP** |

The row above is appended rather than folded into the table, because every
figure in that table is `measured` over 2026-09-20…2026-10-20 and this
workflow did not exist then. Its cost is `assumed`, from
`paper-trading-report`'s 0.8 min/run plus a dependency install.

**Query window:** 2026-09-20 … 2026-10-20 (30 days) for run counts and runner
minutes; TheOddsAPI figures are month-keyed and therefore October-to-date.
All figures `measured` 2026-10-20 unless marked.

---

## The six

| workflow | triggers (and which have fired) | runs / ok / fail / canc | min per run | 30-day min | provider calls per run | writes | **who reads it** | question | disposition |
|---|---|---|---|---|---|---|---|---|---|
| **daily-picks** | `0 0 * * 1-5`, `0 0 * * 6,0`, dispatch — all fired (31 sched, 4 dispatch) | **35 / 25 / 10 / 0** | **41.2** | **1441.6 (48%)** | api-football **83 in run 39358701217** (2 date, 35 odds, 16 xG, 30 injury; ~62/day is the 30-day mean over smaller cards), theoddsapi via refresh, Flashscore ~60 pages, Anthropic 1/fixture | `matches`, `odds`, `odds_snapshots`, `saved_picks`, `predictions`, `teams` | **Niki — pushed to Telegram as the daily card** | the product. **OPEN** | **KEEP** |
| **closing-lines** | `47 10 * * *`, `17 11,13,15,17,19,21,23 * * *`, dispatch — schedule only (123) | 123 / 121 / 2 / 0 | 0.8 | 103.9 | theoddsapi (shared pool) | `saved_picks.closing_odds`, `odds_snapshots` | `get_stats()` → genuine `avg_clv`; the CLV experiment | CLV coverage. **OPEN** (coverage still low) | **KEEP** |
| **ci-audit** | `0 12 * * *`, `workflow_run` ×3, dispatch — all fired (26 / 157 / 2) | **185 / 106 / 79 / 0** | 1.4 | 264.8 | none | nothing persistent; emits `::error::` + Telegram alert | **Niki — Telegram alert on a finding**; me — the ledger | is the pipeline producing cards and clearing its margins? **OPEN** | **KEEP** |
| **paper-trading-report** | `47 10 * * *`, dispatch — schedule only (31) | 31 / 31 / 0 / 0 | **0.8** | **23.7 (cheapest)** | none measured (holds `ODDS_API_KEY` but spends nothing) | `logs/paper-trading-report.txt` → **GitHub artifact**, 30-day retention | **nobody is notified.** Telegram fires **only on failure**. A person reads it only by opening the run and downloading the artifact | the model's realised record. **SETTLED** (no-information, four routes) but the record is still *monitored* | **→ NIKI** |
| **fixture-horizon-probe** | dispatch only since 2026-10-18; previously `0 18 * * *` (7 sched, 2 dispatch, 1 push) | 10 / 2 / **8** / 0 | 4.1 | 40.7 | Flashscore 30 pages per fetching run | **nothing** — proved by a row census with a positive control | **nothing.** Zero deadlines in 8 fetching runs | tomorrow's earliest kickoff. **CLOSED by arithmetic** | **DELETE** |
| **h1-collect** | dispatch only since 2026-10-20; previously `0 9 * * 2,3,4` (4 sched, 1 dispatch) | 5 / 4 / 0 / **1** | **221.3** | **1106.7 (37%)** | theoddsapi ~4 credits per round, 3 rounds | `odds_snapshots` | `h1_analysis` — which reports **NO DATA** at n=4 | H1 price-trajectory effect. **CLOSED 2026-10-17**, power 0.073 | **DELETE**, re-triage **2026-11-01** |

### The two disabled producers were 38% of all runner time

`h1-collect` 1106.7 + `fixture-horizon-probe` 40.7 = **1147.4 of 2981.3
minutes (38.5%)** spent on two producers that answered a closed question and a
question closed by arithmetic. `h1-collect` alone ran **221.3 minutes per run**
— the most expensive thing in the repository — five times, for n=4 and no
analysis.

### The three contested ones, resolved by measurement

**`h1-collect` → DELETE, re-triage 2026-11-01.** Its header read *"NO CRON YET
— `workflow_dispatch` ONLY, DELIBERATELY"* while `0 9 * * 2,3,4` was live from
`d8df667`. **Corrected in place.** A header contradicting the file it sits in
defeated three separate audits of that trigger, mine included, and the cron
then fired three days past the window and spent **8 TheOddsAPI credits** from
the pricing pool. It is `DELETE` rather than `DECLARE` because it is in no
register: not `AWAITING_CONSUMER`, not a registered question. The 2026-11-01
date is the re-run *decision*, not a deferral of this workflow — a re-run needs
a fresh pre-registration, and `workflow_dispatch` is retained to execute it.

**`fixture-horizon-probe` → DELETE.** Zero deadlines in 8 fetching runs. Its
one surviving purpose was 30-league coverage against the reference's 9 — and
**nothing reads it**, for two independent reasons measured since:

1. the reference binds on **0 of 834** days, because the deadline constant went
   below the earliest covered kickoff (10:00);
2. the deadline is now anchored to **our own observed minimum** over all 30
   configured leagues, so the coverage advantage the probe existed to supply is
   already in the figure that binds.

A repaired probe could still bind on a day when *tomorrow's* 30-league card is
earlier than the historical minimum — but it cannot be repaired without
`verify_offset`'s deferred fix, and it has never produced one deadline.

**`paper-trading-report` → NIKI DECIDES.** Cost is trivial: **0.8 min/run,
23.7 min per 30 days, zero provider calls** — the cheapest workflow here. Its
last report (2026-10-20) said:

```
PAPER TRADING REPORT — last 120 days, 1610 picks
  settled            : 1534
  win rate           : 58.0%
  flat ROI           : -4.67%  95% CI [-8.8%, -0.5%]
  Brier (model)      : 0.2466   (n=1534)
  Brier (market)     : 0.2433   (n=479)  <- the number the model must beat
  KEEP   n=788  win=58.1%  ROI=-6.4%  95% CI [-12.0%, -0.7%]
```

The content is live and informative — the ROI confidence interval now
**excludes zero on the upside**, so the loss is statistically distinguishable
from break-even. But **the delivery is broken**: on success the report goes to
an artifact nobody is told about; Telegram fires only on failure. So its
consumer is "Niki, if he opens the run" — a pull with no notification, and per
the rule that a named person counts while "possibly useful" does not, this is
**undelivered output**, not a read.

> **The decision is Niki's, and it is not keep-or-kill.** It is: *deliver it*
> (send the six-line summary to Telegram, as the daily card and the settlement
> report already are) **or** *stop producing it*. Keeping a daily artifact
> nobody is notified about is the third option, and it is the one currently in
> force by default.

---

## Interactions — where this month's failures actually were

### Concurrency: every workflow guards itself, nothing guards the set

All six declare a concurrency group, each **named after itself**, all with
`cancel-in-progress: false`. So two runs of the *same* workflow never overlap,
and **any two different workflows may run concurrently without restriction.**
There is no cross-workflow serialisation anywhere.

### The cron collision

`closing-lines` and `paper-trading-report` both fire at **`47 10 * * *`** —
the same minute, every day. Different groups, so both start. Measured
consequence today: none observable (the report spends no provider calls), but
they contend for nothing only by luck of what the report does.

### Dependency without a dependency trigger

| the dependency | how it is expressed | risk |
|---|---|---|
| `ci-audit` ← daily-picks, closing-lines, paper-trading-report | **`workflow_run`** — correct, and the only one | — |
| `closing-lines` ← daily-picks having written picks | **a clock** (`47 10`, then 2-hourly) | a failed or late `daily-picks` leaves nothing to capture; the CLV gap is silent |
| `paper-trading-report` ← daily-picks having written picks | **a clock** (`47 10`) | reports on a stale set and still says "1610 picks" |
| `h1-collect` ← daily-picks having written the card | **a clock** (`0 9`) | **this is the measured instance**: three H1 runs in the 10-13…10-17 outage succeeded, spent 0 credits, wrote 0 rows and reported `RUNNING_AND_FINE` |

> **A nominal gap cannot order two jobs that are both late.** Three of the four
> dependencies are expressed as clocks, and the one real ordering failure this
> month came from exactly that.

### Shared budget — who can starve whom

Four of six carry `ODDS_API_KEY` and draw on the **same TheOddsAPI pool**:
`daily-picks`, `closing-lines`, `h1-collect`, `paper-trading-report`. October
to date: **334 of 450 used, 16 remaining.** H1's post-window run took **8** of
those, from `normal_ceiling` because its reservation was 0 — a closed
experiment starving the pricing pool. `api-football` is shared by
`daily-picks` alone (1,739 calls / 30 days, ~62/day against a 100/day limit).

### Same table, same cache key

* `odds_snapshots` is written by **`daily-picks`**, **`closing-lines`** and
  **`h1-collect`** — three producers, one table, no per-producer tag. That is
  why "20,971 rows since 10-18" could not be attributed to H1 without the
  `theoddsapi:h1` credit counter.
* `data/models/` is restored and saved by **`daily-picks`** only, via
  `actions/cache` — and finding 13 in the register notes it survives *only*
  there.
* `config/config.yaml` is created by **every** workflow from
  `config.example.yaml`, which is why the example is the only one that ships.

### Failing silently removes a later one's input

`daily-picks` fails → no picks → `closing-lines` captures nothing → CLV
coverage gap → `paper-trading-report` reports an unchanged set → `ci-audit`
reports the daily-picks failure but **not** the downstream starvation. The
2026-10-13…10-18 outage is the measured instance: six cards lost, and the H1
job reporting healthy throughout.

---

## Daily totals across all six — never measured before

| | measured |
|---|---|
| **runner minutes** | **99.4 / day** (2,981 over 30 days) — 48% daily-picks, 37% h1-collect (now disabled), 9% ci-audit |
| **api-football calls** | **~62 / day** mean (1,739 / 28 day-rows) against a 100/day free limit; **83 on a 33-fixture day** (run 39358701217), so the limit binds on a big card |
| **TheOddsAPI credits** | **334 / month to date**, 16 remaining of 450 |
| **Flashscore page loads** | **~60 / day** (30 fixtures + 30 results in daily-picks); was 90 while the probe ran |
| **Open-Meteo calls** | **0** — was 33–58/day until the producer was deleted 2026-10-19 |
| **Anthropic calls** | **1 per reviewed fixture**, 13–65/day measured over three runs |
| **Telegram messages** | **~4.6 / day**: 1 daily card + 1 settlement + ~2.6 audit alerts (79 failures / 30 days) + failure alerts |
| **GitHub runs** | **13 / day** (389 / 30 days) |

**After the two disables**, the projected daily runner minutes fall from 99.4
to **~61** — a 38% reduction, and the single largest operational saving
available without touching the product.
