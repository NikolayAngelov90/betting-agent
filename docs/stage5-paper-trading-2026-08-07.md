# Stage 5 — Production Hardening, Paper Trading & Prospective CLV Experiment

Executed against `docs/adter-stage4-prompt.md`.

**483 tests passing, 0 regressions** (457 after Stage 4, +26).
**Nothing committed, nothing deployed.** Migrations were applied to production —
see the deviation note below.

### Deviation from Phase 24, on instruction

Phase 24 says *"Do NOT apply production migration automatically… Do not execute
destructive production operations yourself."* You instructed me to apply all
unapplied migrations via the Supabase MCP, so I did. The operations were
additive only (nullable columns and one partial index), every migration is
idempotent, and each has a rollback file. Production row counts are unchanged:
**1,026 picks before, 1,026 after; 1,018 settled before and after.**

Everything else in Phase 24 is honoured: no commit, no deploy, and the exact
manual steps are in section 11.

---

## 1. Objective

Build a trustworthy prospective experiment that can finally determine whether
the system has genuine value information beyond the bookmaker market — without
touching the model.

The model is now frozen and treated as an experimental subject. No parameter was
retuned, no algorithm added, no threshold optimised.

---

## 2. Changes made

| File | Change |
|---|---|
| `src/models/model_version.py` | **NEW.** Computed version identifier: label + date + BLAKE2s fingerprint of the 23 settings that actually change a prediction. |
| `migrations/005_stage5_experiment_metadata.{sql,rollback.sql}` | **NEW.** `model_version`, `closing_capture_status`, `closing_bookmaker_count`, `closing_fair_probability`, `pre_claude_ev`, partial index. |
| `src/data/models.py` | Those five columns on `SavedPick`. |
| `scripts/capture_closing_lines.py` | **Rewritten.** Status codes, retry with backoff, `market_spec` validation, per-(match, bookmaker, market) overround gating, `--stats`, cost instrumentation. |
| `scripts/paper_trading_report.py` | **NEW.** Volume / pricing / CLV / outcomes / edge calibration / Claude isolation / checkpoints / health checks. |
| `src/agent/betting_agent.py` | Stamps `model_version`, `closing_capture_status`, `pre_claude_ev` on every saved pick. |
| `scripts/{capture_closing_lines,run_baseline,run_clean_baseline}.py` | **Safety fix:** `load_dotenv()` moved out of module import into `main()`. |
| `tests/test_closing_capture.py` | **NEW**, 26 tests. |

### A safety bug I introduced in Stage 4 and fixed here

`capture_closing_lines.py` called `load_dotenv()` at module import. `conftest.py`
deliberately pops `DATABASE_URL` so no test can reach production — and importing
the module put it straight back. A SQLite unit test then ran against the **live
production database**, caught only by an `IntegrityError` on a colliding primary
key.

I verified production immediately: **0 test rows, 0 closing odds written, 1,026
picks unchanged.** The insert rolled back before anything persisted.

Fixed in all three scripts, with two regression tests: one AST check that no
script calls `load_dotenv()` at module level, and one that importing the capture
script leaves `DATABASE_URL` unset. This is the same hazard the project already
knew about; the guard now enforces it mechanically.

---

## 3. Database migration

Applied via the Supabase MCP to project `nhlurscyrlvpjzapmqcr` (betting-agent,
eu-central-1, PostgreSQL 17.6).

### Pre-flight verification (Phase 2)

| Check | Result |
|---|---|
| Idempotent? | Yes — `ADD COLUMN IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS` throughout |
| Nullable changes safe? | Yes — every column nullable; `is_paper`/`closing_capture_status` carry defaults, which PG 11+ applies without a table rewrite |
| Existing data preserved? | Yes — 1,026 rows before and after |
| Destructive operations? | None — grep for DROP/DELETE/TRUNCATE/UPDATE returned nothing outside comments |
| Indexes appropriate? | One partial index on the capture job's exact predicate; indexes only unresolved rows |
| Supabase-compatible? | Yes — PG 17.6, plain DDL, no extensions |

### Migrations applied

| Version | Name | Adds |
|---|---|---|
| `20260806094511` | history_mirror_and_indexes | *(pre-existing)* |
| `20260806110205` | concurrency_safety | *(pre-existing)* |
| `20260806110230` | harden_api_budget_and_trigger | *(pre-existing)* |
| **new** | `closing_line_value` (003) | `closing_odds`, `closing_odds_captured_at` |
| **new** | `prospective_measurement` (004) | `market_probability`, `market_books`, `is_paper` |
| **new** | `stage5_experiment_metadata` (005) | `model_version`, `closing_capture_status`, `closing_bookmaker_count`, `closing_fair_probability`, `pre_claude_ev`, `ix_saved_picks_closing_pending` |

### Post-migration verification

All ten columns present and nullable. Row counts unchanged. Index created.
Supabase performance advisor reports one pre-existing INFO lint on `api_budget`,
unrelated to this change.

```
rows_total 1026 | rows_settled 1018 | with_closing 0
pending_capture 1026 | paper_picks 0 | versioned 0 | new_index_present 1
```

This also resolves the Stage 4 blocker: `session.query(SavedPick)` no longer
fails with `UndefinedColumn`.

---

## 4. Paper trading architecture

`betting.paper_trading_mode` (default `false`). When on, picks are generated,
saved, settled and measured exactly as before, but flagged `is_paper = true` and
excluded from the live ROI record.

Every pick now stores, per Phase 4:

| Field | Source |
|---|---|
| fixture, league, kickoff, market, selection | existing |
| odds taken, opening odds | existing |
| prediction timestamp | `created_at` / `pick_date` |
| model probability | `predicted_probability` |
| **market probability + book count** | `market_probability`, `market_books` (Stage 4) |
| predicted edge | derivable: `predicted_probability − market_probability` |
| predicted EV, confidence | `expected_value`, `confidence` |
| **model version** | `model_version` (Stage 5) |
| Claude decision | `review_action`, `review_reason`, `model_*`, `pre_claude_ev` |

**Immutability:** after prediction, only `closing_*`, `result`, `actual_*`,
`settled_at` and the Claude review fields are written. `model_market`,
`model_selection`, `model_probability` and `pre_claude_ev` are snapshotted at
save time and never overwritten — the Claude review updates the *live*
selection, not the pre-review record.

### Model version (Phase 1)

Current value: **`stage5_baseline_20260807.ac04cc`**

Computed, not hard-coded, from 23 tracked settings plus a manual `CODE_REVISION`.
A literal string drifts: across Stages 1–4 the blend weight moved 0.40→0.60→0.80,
the half-life 180→540, rho −0.13→0, the de-vig rule changed and six gates were
switched off — and nothing recorded any of it, so March and August picks were
pooled as one system. Untracked churn (league lists, tokens, log levels) is
deliberately excluded so it cannot make every prediction look like a new model.

---

## 5. Closing line capture

`scripts/capture_closing_lines.py`, rewritten against Phases 5, 7 and 10.

| Requirement | Implementation |
|---|---|
| Timing | `betting.clv_capture_window_minutes` (default 90), overridable with `--within-minutes` |
| Idempotency | Only `closing_capture_status = 'pending'` rows are considered; a captured price is never overwritten |
| Retry | 3 attempts with exponential backoff; the work is idempotent so a retry cannot double-capture |
| Missing odds | `status = 'missing'`. **Never invents a price** |
| Late capture | `status = 'late'`, `closing_odds` left NULL, excluded from CLV |
| Wrong market | Resolution goes through `market_spec`; a two-way price can never attach to a 1X2 pick |
| Wrong selection | `SELECTION_SPEC` carries an explicit leg index per selection |

**Two real defects the dry run surfaced**, both fixed:

1. `Double Chance 1X` had no mapping — one production pick would have been
   permanently uncapturable. A test now asserts every selection the value
   calculator can emit is mappable.
2. The odds query pulled **104,117 rows** (all markets on all matches). Now
   restricted to the market types the pending picks actually need and to
   non-excluded bookmakers.

Verified against production read-only: 1,025 past picks correctly classified
`late`, 1 capturable pick found and priced from 2 books.

---

## 6. CLV calculation

`src/evaluation/clv.py`. Formula:

```
price_clv = taken_odds / closing_odds − 1        ← primary
prob_clv  = 1/taken_odds − 1/closing_odds
fair_clv  = closing_fair / taken_fair − 1        ← optional, margin-free
```

### Phase 9 — why `price_clv` is primary

Three candidates were considered and the choice is deliberate:

- **Raw decimal price movement (chosen).** It needs only the two numbers a real
  bettor observes, it is scale-free (a 10% better price is +10% at 1.21 or at
  11.0 — tested), and it requires no assumption about the rest of the market. It
  is the only one computable for **every** market, including overlapping ones.
- **Implied probability movement.** Stored as `prob_clv` for diagnostics. It
  compounds more naturally across bets but is not scale-free, so a fixed
  probability delta means something very different at 1.2 than at 10.0.
- **De-vigged probability movement.** Theoretically the cleanest — immune to the
  bookmaker's margin changing between capture and close — but it needs the
  *whole* market priced at both timestamps, which is often unavailable, and it
  is undefined for overlapping markets such as double chance. Stored as
  `fair_clv` when computable.

**Raw prices are always retained** (`odds`, `closing_odds`,
`closing_fair_probability`, `closing_bookmaker_count`), so any of these can be
recomputed later under a different definition.

### Phase 10 — validation applies to closing odds too

A closing price is accepted only when the market structure is valid, the odds are
valid, the timestamp is valid, the mapping is valid, and the overround is
plausible — gated **per (match, bookmaker, market)**, never by discarding whole
bookmakers. Tested with the real corrupt Bet365 book from match 49032: it is
excluded from the closing consensus while Pinnacle and 1xBet contribute.

---

## 7. Odds API usage — **the finding that blocks CLV**

The Odds API charges **1 credit per region per market per request**. The scraper
requests `regions=eu` (1) and `markets=h2h,totals` (2) → **2 credits per league
request**. One request returns all upcoming fixtures for that league, and only
leagues with fixtures that day are requested.

| scenario | credits/month | free tier 500 |
|---|---|---|
| current: 1 run/day × 27 mapped leagues | **1,620** | **3.2× over** |
| 2 runs/day | 3,240 | 6.5× over |

Usage looks low today (468/500 remaining) only because most configured leagues
are **off-season** — the top-5 European leagues start mid-August. **When the
season starts, the existing single daily run alone will exhaust the free tier in
about nine days.**

### The consequence for CLV

`capture_closing_lines.py` makes **zero API calls** — it reads odds already in
the database. But `clv.validate_pair` rejects a capture taken more than 180
minutes before kickoff, because that is a pre-match snapshot, not a close.

The single CI run is at **09:37 UTC**; club matches kick off in the evening. The
stored odds are therefore **8–10 hours stale at kickoff**, and every capture
would be rejected.

> **As scheduled today, the system cannot produce a single valid closing line.**

### The fix, within the free quota

Refresh only the leagues with fixtures kicking off in the **next 90 minutes**,
then capture. On a typical evening that is 2–5 leagues:

| leagues imminent | credits/run | 1 run/day → month |
|---|---|---|
| 2 | 4 | 120 |
| 3 | 6 | 180 |
| 5 | 10 | 300 |

That fits inside 500/month **provided the full daily refresh is also reduced**.
`TheOddsScraper.update()` already skips leagues without fixtures; it needs a
kickoff-window filter in `_get_today_fixtures` — a small, contained change.

**I did not implement it.** It changes odds-ingestion behaviour, which is outside
"do not touch the model" but squarely inside "do not introduce a new egress or
quota problem without your say-so". It is recommendation #1 in section 11.

---

## 8. Supabase usage impact

Measured on a full-history dry run (`--within-minutes 2880`, 1,026 picks):

| metric | value |
|---|---|
| DB queries per capture run | **2** (+1 write) regardless of fixture count |
| odds rows read (full history) | 103,859 (was 104,117 before the market filter) |
| wall time | 5.6 s |

In production the 90-minute window covers a handful of fixtures. At the measured
~101 odds rows per match, a 10-fixture window reads ~1,000 rows ≈ 40 KB. No N+1,
no `SELECT *`, every query column-projected. The new partial index covers the
capture job's exact predicate and shrinks as captures complete.

The paper-trading report is one joined, column-projected query.

---

## 9. Data quality safeguards (Phase 20)

`python -m scripts.paper_trading_report --health-only` exits non-zero on any
alert, so it can gate a CI step. Checks:

- closing capture coverage below 80%
- captured prices timestamped after kickoff
- captured prices with no capture timestamp
- picks marked captured but carrying no closing price
- picks still `pending` more than a day after kickoff
- picks unsettled more than 2 days after kickoff

Health checks are scoped to picks carrying a `model_version` — the 1,026
pre-Stage-5 picks pre-date closing capture entirely and would otherwise pin
coverage at 0% forever. A permanently-red check is one nobody reads.

Duplicate closing records are structurally impossible: the capture query selects
only `closing_capture_status = 'pending'`, and there is one closing price per
pick row rather than a separate table.

---

## 10. Tests

**483 passing, 0 regressions.** New in Stage 5: **26**, in
`tests/test_closing_capture.py`.

Phase 8's ten required cases: favourable movement, unfavourable movement,
unchanged odds, missing close, invalid timestamp (after kickoff and far before),
bookmaker mismatch, market mismatch, selection mismatch, different decimal odds
scales, multiple bookmakers. Plus:

- `test_clv_is_not_model_edge` — maintained; asserts by source inspection that
  the CLV computation cannot reference a model probability
- `test_full_lifecycle_predict_capture_settle_clv` — Phase 3, end to end on
  SQLite: create prediction → store odds → capture → settle → CLV, with a
  corrupt book present and correctly excluded
- `test_capture_is_idempotent`, `test_late_capture_is_marked_not_captured`,
  `test_missing_price_is_recorded_not_invented`
- `test_every_tradeable_selection_is_mappable` — caught the Double Chance gap
- `test_scripts_do_not_load_dotenv_at_import` — the production-safety regression

---

## 11. Exact manual production steps

**1. Migrations — already applied.** Nothing to do. To verify:

```sql
SELECT column_name FROM information_schema.columns
WHERE table_name='saved_picks'
  AND column_name IN ('closing_odds','closing_capture_status','model_version');
-- expect 3 rows
```

Rollback if ever needed (reverse order, each is safe and documented):
```bash
psql "$DATABASE_URL" -f migrations/005_stage5_experiment_metadata.rollback.sql
psql "$DATABASE_URL" -f migrations/004_prospective_measurement.rollback.sql
psql "$DATABASE_URL" -f migrations/003_closing_line_value.rollback.sql
```

**2. Enable paper trading** — edit `config/config.yaml`:
```yaml
betting:
  paper_trading_mode: true
```
Expected: picks continue to appear, now flagged `is_paper = true`.

**3. Fix the odds-freshness problem** (required before any CLV exists). Add a
kickoff-window filter to `TheOddsScraper._get_today_fixtures`, then schedule:

```yaml
# .github/workflows/closing-lines.yml
on:
  schedule:
    - cron: '5 17,18,19,20,21 * * *'   # hourly across European evening kickoffs
jobs:
  capture:
    steps:
      - run: python -m src.agent.betting_agent --refresh-odds --imminent-only
      - run: python -m scripts.capture_closing_lines
```

**Timezone:** all cron expressions are UTC and all stored timestamps are UTC
(`utcnow()` throughout). Bulgaria is EET/EEST (UTC+2/+3) and switches on the last
Sundays of March and October, so a fixed UTC cron drifts by an hour relative to
local kickoff times twice a year. Kickoff comparisons use stored UTC datetimes,
never local offsets, so correctness is unaffected — only the *convenience* of
the schedule shifts. Widening the cron to several hourly runs absorbs it.

**4. Verify the first capture:**
```bash
python -m scripts.capture_closing_lines --dry-run
python -m scripts.capture_closing_lines --stats
```
Expected after a real run: `clv_coverage_rate` above 0%, and captured picks
showing a `closing_odds` with a capture timestamp inside 180 minutes of kickoff.

**5. Monitor:**
```bash
python -m scripts.paper_trading_report --days 30
python -m scripts.paper_trading_report --health-only   # exits 1 on any alert
```

---

## 12. First 100-pick evaluation plan

**Purpose: data quality only. No model decision.**

Confirm: `clv_coverage_rate` ≥ 80%; zero captures after kickoff; zero captured
rows without a price; every pick carries a `model_version`; capture lead times
cluster inside the configured window; `closing_bookmaker_count` ≥ 2 for the
large majority.

Report the CLV distribution with a bootstrap CI, but **draw no conclusion** — at
n=100 the CI on mean CLV will span several percentage points.

Stop-and-fix triggers: coverage below 80%, any market/selection mismatch, any
`late` capture that should have been in-window.

## 13. 200-pick evaluation plan

**Purpose: initial CLV signal.**

Compute mean and median CLV with a bootstrap CI, and % beating the close, split
by market (1X2 / O/U 2.5 / BTTS) and by odds bucket — reported separately, never
pooled, since Stage 4 found no market where the model helps.

Also: log-loss and Brier of the model versus the stored `market_probability` on
the same picks, with a paired bootstrap; and the edge-calibration table
(0–5%, 5–10%, 10–15%, 15–20%, 20%+) comparing predicted edge against realised
rate — the direct prospective test of the +8.60pp overstatement Stage 4 measured.

Still **no production change**. A positive CLV at n=200 is not proof.

## 14. 500-pick evaluation plan

**Purpose: meaningful model-vs-market evaluation.**

The full counterfactual, all on identical picks: market-only, model-only, the
production blend, and what the system actually chose. Paired bootstrap on
per-pick log-loss for each against market-only; permutation test on the CLV
difference between the model's high-edge and low-edge cohorts; per-market
breakdown; league analysis **only** where a league has enough picks to say
anything.

Decision rule, fixed now:

- Persistent positive CLV whose 95% CI excludes zero **and** better log-loss than
  market-only → the model has earned its weight; consider raising it.
- CLV CI spanning zero → keep paper trading. Collect more.
- Persistent negative CLV → set `bookmaker_blend_weight: 1.0` and run as a
  market-consensus price-shopper.

---

## 15. What we intentionally did NOT change

- **The model.** No Elo or Poisson retuning, no new algorithms, no new feature
  families, no threshold optimisation, no blend change. `bookmaker_blend_weight`
  stays at 0.80.
- **`min_expected_value` / `min_confidence`.** Stage 4 showed both are worthless
  as value filters, but removing them triples pick volume. Left as volume
  controls pending the prospective data.
- **The six disabled betting gates.** Still off, still declared in
  `gate_registry`.
- **Understat.** Not integrated — the scraper payload remains unverified (Phase 19).
- **Odds-ingestion scheduling.** Diagnosed and costed, not implemented — see §7.
- **Historical pick statuses.** The 1,026 pre-Stage-5 picks were left `pending`
  rather than bulk-updated to `late`. They pre-date the capture system; the
  report excludes them by scope instead.

---

## Current state

```
CLV coverage
  picks considered   : 1026
  valid CLV pairs    : 0
  clv_coverage_rate  : 0.0%
  capture status     : pending 1026

model_version        : stage5_baseline_20260807.ac04cc
checkpoints          : 100 / 200 / 500 valid closing lines — all 0
```

---

# FINAL DECISION

### Is the system ready for real-money betting?

# NO

The evidence has not changed since Stage 4, and Stage 5 did not attempt to change
it — it built the instrument that could.

What is now true that was not before: migrations are applied, every prediction
will carry the configuration that produced it, closing capture is
production-safe and validated against the same corruption gate that protects the
features, CLV is defined and tested and cannot be faked by a model-edge
substitute, and the health checks will say so loudly when the data goes wrong.

What is still true: **`clv_coverage_rate` is 0%**. Not one valid closing line
exists. The first checkpoint needs 100 and the decision checkpoint needs 500.
And section 7 found that on the current schedule the system would produce **zero**
valid closing lines however long it ran, because the odds are 8–10 hours stale at
kickoff — a scheduling and quota problem that must be fixed before the experiment
can even begin collecting.

The system should remain **PAPER TRADING ONLY** until genuine prospective CLV and
model-vs-market evidence has been collected.

A simpler profitable system is preferable to a sophisticated unproven one — and
on everything measured across Stages 1–5, the simpler system here is the
bookmaker consensus.
