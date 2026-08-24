Audit every CI run of this project that does not yet have a verdict in `docs/ci-audit-ledger.md`, and append one row per run.

This is the daily version of the Stage 13 Part A pass, which took a day by hand and found **1 BROKEN and 9 DEGRADED runs that GitHub had reported as `success`**. Read that sentence before starting: `conclusion: success` is not evidence here. Every core step in `daily-picks.yml` carries `continue-on-error: true`, so a run is green whenever the *runner* survived, which is not the same as the pipeline having worked.

## Run the mechanical half first

```
python -m scripts.ci_audit --unaudited
```

It finds unaudited runs across all three workflows, fetches and caches the **full log of every step**, extracts counts, applies the assertions, and prints a provisional verdict per run.

It was validated by re-auditing 2026-08-11 → 2026-08-13 and reproducing the manual pass exactly: 27 runs, 1 BROKEN, 9 DEGRADED, 17 CLEAN, zero disagreements. If you change its patterns, re-run that window and check it still reproduces before trusting it on anything new:

```
python -m scripts.ci_audit --since 2026-08-11 --until 2026-08-13
```

## Then do the half a script cannot

The script counts. You decide what a count means. Specifically:

**Is a zero a defect or a quiet day?** The assertions are self-calibrating — they fire when a unit that produced data within the last 7 runs produces none — but they cannot tell a genuinely thin card from a broken scraper. Check the fixture count before calling a low pick count a fault. Stage 13 established both directions of this error: alerting on every zero trains people to ignore alerts, and treating every zero as normal is how nine DEGRADED runs went unnoticed.

**Read the Telegram output as evidence.** Part A audited 27 runs and never looked at what the system *said*, while the pipeline had been announcing an API suspension daily in the channel built for it. For each run, establish what was sent, and whether anything that should have been sent was not. `alert NOT delivered` and `Failed to send Telegram message` both mean the detection worked and the delivery did not.

**Look for the failure that gets quieter as it gets total.** `API-Football update complete (1 requests used)` is what a completely dead integration looks like: the first call is refused, a flag suppresses every subsequent call *before* it logs, and the run ends green. A small number where a large one belongs is the signal, not an error line.

**Check what the run should have produced against what it did.** Picks saved vs fixtures available. `pick_observations` vs 2 × picks saved. Credits claimed vs closing lines captured. Briefing decisions made vs applied — a decision computed and then discarded leaves the model's original pick in place and a gap in the KEEP/CHANGE record.

## Verdicts

* `CLEAN` — did what it was supposed to, and the counts agree
* `DEGRADED` — completed, but a stage silently did less than it should
* `BROKEN` — a core step failed, or a traceback reached the log
* `UNTESTABLE — reason` — the run could not exercise the thing in question. This is a first-class verdict, not a failure to reach one. A gate that never ran is not a gate that passed.

## Append to the ledger

One row per run, in the existing table format:

```
| run_id | workflow | started | conclusion | steps failed | audited on | verdict | notes |
```

Notes carry **counts, not adjectives** — "injuries 0 from 30 fixtures", not "injury coverage poor". A future reader needs the number to decide whether the situation changed.

If something new surfaces that is not about this run — a defect, a pattern, a claim that turns out to be wrong — add a ledger entry for it and **do not pursue it**. That is how Stage 13 nearly failed to land.

## Hard rules

1. **Read-only.** No code, config, schema, migration, workflow or production-data changes. No commits beyond the ledger row.
2. **Do not trigger a workflow** to produce evidence, and do not spend Odds API or API-Football credits.
3. **Do not mark a run CLEAN because nothing looked wrong.** Mark it CLEAN because the counts you checked agreed with what the run should have produced. Say which you checked.
4. If a run cannot be judged from its log, say so and record `UNTESTABLE` with the reason.
