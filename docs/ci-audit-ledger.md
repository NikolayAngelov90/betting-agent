# CI Audit Ledger

One row per workflow run. `verdict` is:

* **CLEAN** — every step did what it was supposed to do.
* **DEGRADED** — the run completed but something silently did less than intended.
* **BROKEN** — a step failed or produced wrong output.

**`conclusion: success` is not evidence.** Every core step in `daily-picks.yml`
except `Run tests` carries `continue-on-error: true`, so the job is green
whenever the runner survived. Verdicts below come from reading each log, not
from the conclusion field.

Logs are saved under `ci_logs/run_<id>/full.log` (gitignored, per the existing
repo convention — re-fetch with `gh run view <id> --log`).

Audited by `.claude/commands/daily-ci-audit.md`.

---

## 2026-08-11 → 2026-08-13 (first pass, Stage 13 Part A)

| run_id | workflow | started (UTC) | conclusion | steps failed | audited on | verdict | notes |
|---|---|---|---|---|---|---|---|
| 31482430418 | daily-picks | 2026-08-11 10:29 | success | **tests (1 failed)** | 2026-08-14 | **BROKEN** | `test_ev_threshold_calibration_ignores_paper_picks` failed; masked by `continue-on-error`; alert built but undelivered (HTTP 400). Both fixed in `451fe3f`. 13 picks, 26 obs, review 13/13. Injuries **0**. Odds API **0 rows**. |
| 31486892830 | paper-report | 2026-08-11 11:30 | success | — | 2026-08-14 | CLEAN | 700 picks considered, 0 valid CLV pairs. Correct identity printed. |
| 31488781816 | closing-lines | 2026-08-11 11:55 | success | — | 2026-08-14 | CLEAN | `no pending picks kick off in the next 120 minutes`. No claim. |
| 31501920283 | closing-lines | 2026-08-11 14:30 | success | — | 2026-08-14 | **DEGRADED** | 4 credits claimed, **0 odds rows**, 0 captured / 4 missing. |
| 31510913965 | closing-lines | 2026-08-11 16:09 | success | — | 2026-08-14 | **DEGRADED** | 4 credits, **0 rows**, 0 captured / 5 missing. |
| 31520932810 | closing-lines | 2026-08-11 18:05 | success | — | 2026-08-14 | **DEGRADED** | 2 credits, **0 rows**, 0 captured / 4 missing. |
| 31531085995 | closing-lines | 2026-08-11 20:04 | success | — | 2026-08-14 | CLEAN | nothing to do. |
| 31540633532 | closing-lines | 2026-08-11 22:02 | success | — | 2026-08-14 | CLEAN | nothing to do. |
| 31548445580 | closing-lines | 2026-08-11 23:57 | success | — | 2026-08-14 | CLEAN | nothing to do. |
| 31588427891 | daily-picks | 2026-08-12 10:40 | success | — | 2026-08-14 | **DEGRADED** | **1 briefing decision discarded** (match 49423, `NotNullViolation` on `pick_observations.pick_id`). 4 picks on 3 matches. Injuries **0**. Odds API **0 rows**. 651 tests pass. |
| 31592087184 | paper-report | 2026-08-12 11:29 | success | — | 2026-08-14 | CLEAN | 691 considered, 0 valid CLV pairs. |
| 31594120566 | closing-lines | 2026-08-12 11:56 | success | — | 2026-08-14 | CLEAN | nothing to do. |
| 31607264276 | closing-lines | 2026-08-12 14:31 | success | — | 2026-08-14 | **DEGRADED** | 2 credits, **0 rows**, 0 captured / 4 missing. |
| 31616029877 | closing-lines | 2026-08-12 16:07 | success | — | 2026-08-14 | CLEAN | nothing to do. |
| 31625875088 | closing-lines | 2026-08-12 18:05 | success | — | 2026-08-14 | CLEAN | nothing to do. |
| 31635799575 | closing-lines | 2026-08-12 20:04 | success | — | 2026-08-14 | CLEAN | nothing to do. |
| 31645053895 | closing-lines | 2026-08-12 22:00 | success | — | 2026-08-14 | CLEAN | nothing to do. |
| 31652680706 | closing-lines | 2026-08-12 23:56 | success | — | 2026-08-14 | CLEAN | nothing to do. |
| 31692176305 | daily-picks | 2026-08-13 10:40 | success | — | 2026-08-14 | **DEGRADED** | **5 briefing decisions discarded** (49458, 49460, 49468, 49485, 49486). 45 picks on 37 matches. Review 37/37. Injuries **0** from 30 fixtures. Odds API **0 rows**. 12 Flashscore results pages timed out. Contains the CSKA-Sofia wrong-fixture pick (Part B). |
| 31695565691 | paper-report | 2026-08-13 11:27 | success | — | 2026-08-14 | CLEAN | 726 considered, 0 valid CLV pairs. |
| 31697782489 | closing-lines | 2026-08-13 11:57 | success | — | 2026-08-14 | CLEAN | nothing to do. |
| 31710772556 | closing-lines | 2026-08-13 14:32 | success | — | 2026-08-14 | **DEGRADED** | 2 credits, **0 rows**, 0 captured / 5 missing. |
| 31719136797 | closing-lines | 2026-08-13 16:08 | success | — | 2026-08-14 | **DEGRADED** | 4 credits, **0 rows**, 0 captured / **26 missing**. |
| 31729025375 | closing-lines | 2026-08-13 18:05 | success | — | 2026-08-14 | **DEGRADED** | 4 credits, **0 rows**, 0 captured / 14 missing. |
| 31738584323 | closing-lines | 2026-08-13 19:59 | success | — | 2026-08-14 | CLEAN | nothing to do. |
| 31748152343 | closing-lines | 2026-08-13 22:00 | success | — | 2026-08-14 | CLEAN | nothing to do. |
| 31755593408 | closing-lines | 2026-08-13 23:55 | success | — | 2026-08-14 | CLEAN | nothing to do. |

**Totals for this pass:** 27 runs audited — 1 BROKEN, 9 DEGRADED, 17 CLEAN.

> **Annotated 2026-08-24.** This was "every run since the Stage 12 deployment
> boundary", which is accurate and is not what a reader takes from it. It was
> never every run. `scripts/ci_audit.py --unaudited` reports **334 runs with no
> ledger verdict, back to 2026-02-24** — 268 CLEAN, 59 DEGRADED, 7 BROKEN.
> Read the 27 as a window, not a census.
**0 of 27 were flagged by CI.** Every one reported `conclusion: success`.

---

## THE HABIT — read this before the individual findings

Every defect below is an instance of one habit:

> **When something needs to be used somewhere else, a second copy appears.**
> The copy is always the site that drifts.
>
> **And the corrective is not reachability.** The first framing of this habit
> ended "...instead of the first being made reachable", which implies making it
> reachable prevents the copy. Instance 6 disproves that outright. The only
> thing that has ever prevented a second definition in this repository is **a
> test that fails when one appears** — which is exactly what separates
> `team_names.py` from the three predicates s5.3 protects.

| # | The shared thing | What appeared instead | The defect it caused |
| --- | --- | --- | --- |
| 1 | Telegram delivery | `scripts/ci_alert.py`, a second sender | DEL-1: fix built a second path to the same last hop; the class survived it |
| 2 | `match_history._base_filter()` | `feature_engineer.py:173`, hand-copied `is_fixture == False` | contamination stayed in form/H2H/rolling goals |
| 3 | `_live_only()` | `match_briefing`'s own filters | EXP-1: paper outcomes reached the KEEP/CHANGE prompt |
| 4 | one ranking | three orderings that disagreed | the per-match survivor was not the best pick |
| 5 | one cohort literal | six copies of `"s5.2"` across five files | the next bump edits five and misses one |
| 6 | `src/utils/team_names.py` | five more name matchers | the wrong-fixture class this stage exists for |

### Instance 6 is the one that should change how the corrective is stated

The obvious lesson is "make the first one reachable". **That is not sufficient,
and instance 6 proves it.**

`src/utils/team_names.py` already existed as the canonical home. It is already
imported by `betting_agent`, `match_briefing`, `apifootball_scraper` and
`flashscore_scraper`. `flashscore_scraper.py:1703` even carries a migration note
telling future callers to import it directly.

Copies appeared anyway:

* `theodds_scraper.py:109` — its docstring says outright *"Reuse
  FlashscoreScraper's fuzzy match logic **without importing the class**"*
* `footballdataorg_scraper.py:260` — its own `_names_match`
* `apifootball_scraper.py:1402` — its own `_names_similar`, in a file that
  **imports the canonical utility 266 lines later**

The reachable version was reachable, documented, and already in use in the same
file. The copy appeared regardless. So reachability is necessary and does not
prevent this; **only a guard that fails when a second definition appears does.**

### And this stage committed the habit while fixing it

`names_share_an_anchor` was added at `apifootball_scraper.py:440` — a
general-purpose name predicate placed inside a scraper rather than in
`team_names.py` with the others. It is not a straight duplicate (it is
deliberately an anchor test, not a similarity ratio — see Part B), but it is the
same habit: a name-matching function that belongs in the shared module, put
somewhere else because that was where it was needed.

Recorded because the stage that named the habit is not exempt from it, and
because a pattern nobody catches themselves committing is a pattern nobody
believes.

### What the three guards actually are

Named this way, the guards built in s5.3 stop looking like three separate good
ideas:

| guard | one definition it protects |
| --- | --- |
| `test_training_exclusion.py` | the exclusion predicate |
| `test_valid_evidence_gate.py` | `live_only()` / `valid_evidence()` |
| `test_no_secrets_in_repo.py` | credentials belong in the environment, once |

All three enforce the same corrective: **one definition, and a test that fails
when a second appears.** `team_names.py` never had one, which is why it has five
copies despite being the canonical home.

### The search strategy this gives Stage 14

Not a to-do list — a query: **find the places where the same idea is spelled
twice.** Start with the ones already visible:

* five name matchers vs `src/utils/team_names.py`
* two senders vs one delivery guarantee (DEL-1)
* `_normalise` / `_norm` / `_tokens` duplicated across scrapers
* any predicate a guard does not yet pin

Each is a candidate for the same treatment: consolidate, then guard the
consolidation. The guard is the half that lasts.


## Incidents (Stage 13, recorded 2026-08-23)

### SEC-1 — Two live API keys public for 188 days (CRITICAL, remediated)

`.mcp.json` is tracked and pushed to a public repository. It carried
`API_FOOTBALL_KEY` (`efdc87…cb3c4a`) and `ODDS_API_KEY` (`309170…c81e83`) in
plaintext from **02de2b0, 2026-02-16** until **2026-08-23** — 188 days.

Both keys are now **dead**, verified: Odds API returns `401 INVALID_KEY`;
API-Football returns `errors.token: Error/Missing application key`. Rotation was
performed by the operator before this entry was written.

Remediation: both values replaced with `${ODDS_API_KEY}` / `${API_FOOTBALL_KEY}`
so the file stays versioned and readable while the secrets come from the
environment.

**Relationship to the API-Football suspension: strong hypothesis, not
established cause.** A publicly exposed key is harvested by automated scanners
within hours, and third-party traffic on it produces exactly the shared-address
and abusive-request-pattern conditions the provider lists as suspension grounds.
This outranks the multi-account explanation and reframes the support request
from "I had several accounts" to "my key was leaked and used by others". It
remains unproven from here.

**The fix is not the safety. The current file is clean; the history is not.**
A reader who opens `.mcp.json` today sees `${ODDS_API_KEY}` and would
reasonably conclude nothing was ever exposed. Both values remain in every
commit from 02de2b0 onward, and in every fork, clone and scraper cache that
took a copy during those 188 days. **Rotation is what made these keys safe —
this commit only stopped the bleeding.** Rotation performed by the operator on
2026-08-23, before the remediation commit, and verified dead by probe:
Odds API `401 INVALID_KEY`, API-Football `errors.token: Missing application
key`.

**Git history deliberately NOT rewritten.** Rotation is what kills an exposed
key. A rewrite unpublishes nothing — forks, clones and scrapers already hold the
values — and it would break every existing checkout for a benefit rotation has
already delivered. The literals remain in history by decision, not oversight.

### SEC-2 — Production Telegram bot token hardcoded and LIVE (CRITICAL, open)

`scripts/settle_feb15.py:117` hardcoded the production Telegram bot token,
byte-identical to `TELEGRAM_BOT_TOKEN` in `.env`, tracked and public since the
file was added. Line 118 carried the chat id beside it.

**Still live when found** — `getMe` returned HTTP 200 for bot `@na_bets_bot`.
Anyone with the public repository can post to the picks channel as the bot and
read group updates.

**The fix is not the safety.** Line 117 now reads
`os.environ.get("TELEGRAM_BOT_TOKEN")`. That does not mean the token was
never exposed — it sat in this file, tracked and public, from the day it was
added until 2026-08-23, and it remains in the history and in every copy taken
during that period. **Only the BotFather rotation makes it safe.** Rotated
**2026-08-23 10:00 UTC (13:00 Europe/Sofia)** via BotFather, and verified dead
by probing the old value retrieved from history: `401 Unauthorized`.

Remediation in code: both values replaced with `os.environ.get(...)`.
**Rotation is the operator's action and was outstanding at the time of writing**
— the code fix does not revoke the token.

### SEC-3 — Secret-shaped literals elsewhere

Repo-wide scan of **693 tracked files** (32-hex, `sk-ant-`, bearer tokens,
connection strings with embedded passwords, Telegram token shape, AWS keys):

| Finding | File | Tracked? | Status |
| --- | --- | --- | --- |
| Telegram bot token | `scripts/settle_feb15.py:117` | yes | **live** — see SEC-2 |
| `postgresql://user:pass@host/db` | `config/config.example.yaml:77` | yes | placeholder, benign |
| 4 × 32-hex | `mcp-servers/.../mocks_live/nba_games_live.json` | **no** (gitignored) | Odds API *event ids*, not keys |
| 2 × 32-hex | `mcp-servers/wagyu_mcp_hackathon/old/docs/` | **no** (gitignored) | unverified; vendored third-party, not probed |

`mcp-servers/` has **0 tracked files** (`.gitignore:60`), so nothing under it was
ever published. The two doc literals were not tested for liveness: they appear to
belong to a vendored third-party project, and probing another party's
credentials is not appropriate.

Guard added: `tests/test_no_secrets_in_repo.py`, failing when any *tracked* file
carries a secret-shaped literal in a credential-shaped field. Verified
non-vacuous against a probe file. A bare 32-hex is deliberately **not** treated
as a secret — The Odds API returns 32-hex event ids in fixture data.

### OPS-1 — API-Football account suspended (CRITICAL, open)

Window **opens 2026-08-19 10:10:28 UTC** (first `errors.access`, after 24
requests had succeeded that run); **still open** at the last audited run,
2026-08-22. Not quota-driven: daily usage ran 51–65 against a 100/day limit, and
the refusal arrived at 25 requests.

| Date | Suspended | AF requests | Injuries |
| --- | --- | --- | --- |
| 08-11 → 08-18 | no | 51–65 | varies (0–98) |
| 08-19 | **yes** | 25 | 0 from 4 |
| 08-20 / 21 / 22 | yes | **1** | — |

**39 picks** were generated inside the window (08-19: 4, 08-20: 1, 08-21: 3,
08-22: 31), all paper, all carrying `stage5_baseline_20260807.485823` —
indistinguishable by fingerprint from picks made while the integration was alive,
because `model_version` tracks configuration, not data availability.

**Deliberately NOT stamped with `evidence_status`.** Exclusion would rest on the
premise that the outage materially changed model inputs, and that is unmeasured:
xG was already inert (2.2% coverage against a 0.35 threshold), injuries reach
only the Claude review prompt, and fixtures still arrive from Flashscore and
football-data.org. Only the odds contribution could matter, via book coverage in
the de-vigging consensus gate. `evidence_status` is write-once, so exclusion can
be applied later but never undone. The window is recorded here as the fact;
membership is derivable from `pick_date`. **Open follow-up:** measure per-market
book coverage for those 39 picks against surrounding days.

### A1 — blast radius annotated, NOT re-investigated

Stage 13 measured six matches that lost their review verdict (49423 on 08-12;
49458, 49460, 49468, 49485, 49486 on 08-13) and treated that as the population.

> **Annotated 2026-08-24.** Six is a **floor bounded by three days of saved
> logs**, not a count. The daily audit finds the same `Could not apply` line in
> **June** — before Stage 10 introduced the relationship that caused A1. So
> either that log line has a second cause, or the timeline is wrong. Both are
> possible and **neither has been investigated.**
>
> This is the same correction as the Telstar blast radius, which was also
> bounded by log retention rather than by the defect: 1 of 65 fixtures, where 65
> was simply how many creation lines the saved logs held.
>
> Whoever opens the 334-run backlog should start here. And ask first, before
> anything else: **of the 7 BROKEN runs, is any of them a run whose failure
> still affects current data?**

### OBS-1 — The alert that fires daily and changes nothing

The suspension alert (`betting_agent.py:611`) posted to Telegram on 08-20, 08-21
and 08-22. Part A audited 27 runs and read none of the system's own outbound
messages. `daily-ci-audit.md` gains Telegram output as a first-class evidence
source, and D3 gains a recorded acknowledgement path — a repeating alert must
escalate rather than repeat.

### OBS-2 — A breaker that also breaks observability

On the first refusal `_quota_exhausted = True` suppresses every later `_api_get`
**before** it logs. A completely dead integration therefore presents as one ERROR
line and `1 requests used`; the failure gets quieter as it gets total. Audited
across all integrations — the shape is **unique to API-Football** (4 set-sites:
506, 518, 548, 564). The Odds API, Flashscore and football-data.org carry only
the shared `CircuitBreaker`, which logs every transition and self-heals after
60s. `85167b4` (`_claude_code_exhausted`) is graceful degradation to the paid
API, not this defect. Fix: keep the short-circuit, count what it suppresses, and
report the count at end of run.

### OBS-3 — Prune log truncates at ten names

`ml_models.py:182` logs `dropped[:10]` with `...` for the rest. 31 features were
pruned; whether `xg_for_diff` / `xg_against_diff` are among them is therefore
unanswerable from logs. This is what forced the `s5.3` deferral to be argued on
both branches. One-line fix, Stage 14.

### DOC-1 — `--refresh-odds` documented but never implemented

`scripts/capture_closing_lines.py` documents the flag in its module docstring;
argparse defines only `--within-minutes`, `--dry-run`, `--stats`. Docstring fix
only — it explains nothing about D1, whose cause lies in
`TheOddsScraper.refresh_imminent`.

### ENV-1 — A suspended key in `.env`

On 2026-08-14 a key belonging to an already-suspended API-Football account was
placed in `.env` at the operator's instruction, and three diagnostic calls were
made from the operator's own network. `.env` is gitignored and local; the
observable event is the requests, not the file. Production was suspended five
days later. **A correlated event with a plausible mechanism, cause unproven** —
and now the weaker of two hypotheses beside SEC-1.

The reason it is recorded at all: a suspended key returns HTTP 200 with an empty
response and no exception, so a local run would have produced zero rows
indistinguishable from a code defect. That signature has now appeared three times
in this stage.

### A4 — Flashscore result-ingestion gaps (promoted out of deferred)

12 Flashscore results pages timed out on 2026-08-13, four in top-five leagues.
Deferred on condition that a recurrence promotes it. Pick **1134** (`Under 3.5`,
match 49470, KI Klaksvik vs Lech Poznan) reached a saved pick whose match result
was never ingested — a recurrence *with consequence*. Promoted: needs a root
cause and a D3 assertion. Not in the `s5.3` break; it touches neither predictions
nor selection.

### Audit coverage gap

Runs from **2026-08-14 to 2026-08-22** were not in the original 27-run pass.
08-19 through 08-22 have since been read for OPS-1; the remainder still need a
full A2/A3 pass.

### SEC-4 — The guard's own fixtures carried real credential prefixes

The first version of `tests/test_no_secrets_in_repo.py` used discriminator
samples built on the real six-character prefixes of the exposed credentials
(`309170…`, `858806…`, and a real Odds API event id). The tails were invented, so
none was a working credential — but a test that hardcodes fragments of the secret
it exists to detect is a slower version of the same leak, and it would have been
committed by the very step that removed the originals.

Caught before anything was pushed and amended to fully synthetic values sharing
no prefix with any real credential. Recorded because the near-miss is the
instructive part: the scan found the exposures, and the scan's own fixtures
nearly reintroduced them.

### HYG-1 — Completed one-off scripts deleted

`scripts/settle_feb15.py` — docstring "One-time script to settle Feb 15 2026
picks" — was still tracked six months later and was the origin of SEC-2. The code
fix removed the literal; it did not remove the reason the file was there.

Audited all 13 scripts. Six were completed one-offs with zero inbound references
(every apparent reference was a self-citation in the file's own usage docstring):

| Deleted | Why it is finished |
| --- | --- |
| `settle_feb15.py` | one-time settlement for a single day in February |
| `migrate_to_neon.py` | SQLite → Neon; Neon is retired |
| `migrate_to_supabase.py` | Neon → Supabase; the migration completed |
| `merge_old_neon_to_supabase.py` | one-time merge of the old Neon data |
| `import_mcp_odds.py` | imported MCP JSON dumps into Neon; not in any pipeline |
| `sync_db.py` | synced the SQLite CI database that the Neon move eliminated |

Seven durable tools remain: `capture_closing_lines`, `ci_alert`,
`paper_trading_report`, `refresh_and_capture`, `run_baseline`,
`run_clean_baseline`, `simulate_odds_quota`. All are recoverable from history;
none is referenced by a workflow that would break.

`simulate_odds_quota.py` is the borderline case and was **kept**: it is a Stage 6
artefact, but quota planning is live work while OPS-1 and D1 are open.

**Why this is not housekeeping.** Three of this stage's defects share one shape —
the vacuous cascade test, the 39 FK-violating fixtures, and this token. Each was
code nothing forced anyone to re-read. A one-off that lingers is a file nobody
reviews, and that is the condition under which a live credential survives six
months and a 693-file scan only finds it because someone went looking.

### HYG-2 — Which survivors would fail silently

`simulate_odds_quota.py` was kept because quota planning is live work and the
README documents it. But being documented is not what makes a file safe: a
script listed in a table and never executed has exactly the shape this stage
keeps finding — code nothing forces anyone to re-read.

Mapped all seven survivors. "Imported" counts a real `from scripts.X import`,
not a mention in a docstring or a string literal; the two differ sharply.

| Script | Imported by tests | Workflow | Appears in logs |
| --- | --- | --- | --- |
| `paper_trading_report` | **14** | 1 | 3 |
| `capture_closing_lines` | **7** | 0 (runs via `refresh_and_capture`) | 21 |
| `refresh_and_capture` | 0 | **1** | 21 |
| `ci_alert` | 0 | **3** | 11 |
| `run_baseline` | **0** | **0** | **0** |
| `run_clean_baseline` | **0** | **0** | **0** |
| `simulate_odds_quota` | **0** | **0** | **0** |

Three answer no to all three, and the important one is **`run_baseline.py`**:
it is the evidence bar every model parameter change must clear, and nothing
exercises it. If it broke, the failure would surface at exactly the moment it is
needed — the next time someone tries to justify a model change — and would look
like the change being unjustifiable rather than the harness being broken.

`run_clean_baseline.py` has the same profile against the clean dataset.
`simulate_odds_quota.py` is the least consequential of the three but the most
obviously idle.

Remedy for all three: a smoke test that imports the module and runs its argument
parser, so a change that breaks them fails the suite instead of failing silently
in six months. Same instinct as the structural guards, applied to files rather
than to call shapes. **Cohort-neutral; deliberately NOT inside the s5.3 break.**


### SEC-5 — Key identity, recorded so a rotation is distinguishable from a leak

Nothing in the ledger recorded *which* key a period's spending belonged to.
Fingerprints below are SHA-256 prefixes; a 128-bit secret is not recoverable
from one, and the values themselves appear nowhere.

| Credential | Fingerprint (post-rotation, 2026-08-23) |
| --- | --- |
| `ODDS_API_KEY` | `sha256:2b98923b9944` |
| `API_FOOTBALL_KEY` | `sha256:4b8f1e8b118e` |

The pre-rotation keys are deliberately not fingerprinted: they are published in
git history in full, so a fingerprint would add nothing and invite the mistake
of treating one as a safe stand-in for the other.

### OPS-2 — The Odds API ledger was NOT desynchronised by the rotation

Expected: a rotated key restarts the provider's counter while the ledger holds
the accumulated period total, so `reconcile()` — which only ever raises — would
refuse valid claims for the rest of the month.

Measured instead, from one free `/v4/sports` probe on the rotated key
(`x-requests-last: 0` confirms it was not billed):

```
HTTP 200
x-requests-remaining: 151
x-requests-last:      0
x-requests-used:      349
```

**The provider reports 349 used — exactly what the ledger holds.** The Odds API
quota is per ACCOUNT, not per key: a rotated key inherits the account's usage.
There is no divergence, the August row is correct, and no correction is needed.
`reconcile()`'s guarantee was never violated.

The probe also answers the second-order question it was chosen for: the key is
valid (HTTP 200), so the rotated value is at least correct in `.env.local`.
Whether the GitHub secret was updated remains unconfirmed and unconfirmable from
logs while the budget blocks all requests — no call, no 401.

**What is true is a different, smaller thing.** Free tier is 500
(151 + 349). The repo's own `odds_api.monthly_credit_budget` is 400 with a
50-credit safety margin, so 350 are spendable and 349 are spent: **1 credit for
the last nine days of August, while 151 sit unused at the provider.** That is a
deliberate conservative budget nearly consumed, not a defect — and it is a
config decision, not a cohort one (`odds_api.*` is absent from `TRACKED_KEYS`).

Sequencing note: this does not cost nine days of closing lines, because D1 is
unfixed and the capture writes zero rows whether or not credits exist. The two
failures are stacked, not additive. But the budget must be right before D1's fix
can be demonstrated, or there will be nothing to demonstrate it with.

Adding key identity as a ledger dimension was considered and is **not built**:
the desync it would have fixed does not exist, and a per-key dimension would
misrepresent a per-account quota. Recorded here so the reasoning survives.

### ENV-2 — A retired Neon connection string in `.env.local`

`.env.local` carries a `DATABASE_URL` pointing at
`ep-bold-field-al1me8dx-pooler.c-3.eu-central-1.aws.neon.tech`, with
credentials, while production runs on Supabase
(`aws-1-eu-central-1.pooler.supabase.com`, per `.env`).

Three questions, answered rather than assumed:

**Which file wins?** Neither, because **nothing loads `.env.local`.** Every
`load_dotenv()` in the repo either names `.env` explicitly
(`betting_agent.py:4473`) or calls the bare form, which python-dotenv resolves
to `.env` — it does not read `.env.local` by convention. Zero Python files in
the repo reference the filename at all.

**Can any path reach Neon?** No. The pointer is unreachable by code, and the
credential is dead independently: connecting returns
`password authentication failed for user 'neondb_owner'`. Doubly inert.

**Does the database still hold data?** Unknown, and not determinable with a dead
credential. The endpoint still resolves and completes a Postgres handshake, so
**the Neon project still exists at the provider** — an unretired asset, even if
this repo cannot reach it.

Severity is therefore LOW, not the "quiet writes to the wrong target" this repo
already has a scar for — the two AST regression tests guarding import-time
`load_dotenv()` remain the relevant protection, and they are unaffected.

Actions: the dead credential should come out of `.env.local` (trivial, local,
no code depends on it). Decommissioning the Neon project is Niki's call and is
recorded here rather than taken — a dead credential proves nothing about whether
the data behind it is still wanted.

### EXP-1 — The paper/live isolation was absent on the review path (CRITICAL)

`src/reporting/match_briefing.py` contained **zero** references to `is_paper`
and never called the live-only predicate. Two functions read settled outcomes:

| Function | Line | What it produced |
| --- | --- | --- |
| `_recent_selection_stats` | 902 | per-selection win rates |
| `_recent_review_stats` | 936 | KEEP-vs-CHANGE win rates |

Both are injected into the KEEP/CHANGE decision prompt (lines 1146–1149). So
paper-pick outcomes were computed into statistics, shown to Claude, and used to
choose the pick that the **FINAL series then measures**.

**This falsified the README's central isolation claim as written.** The
experiment could not retrain the MODEL — that part held. It had been informing
the REVIEW, which produces one of the two series the experiment exists to
measure. Corrected in README with the original claim kept visible.

A third site surfaced with them: `probability_calibration.fit_from_db` excluded
paper picks correctly (its docstring explains why) but had **no evidence gate**,
so it would have calibrated on all three corrupt-feature picks.

**Root cause, and it is a pattern rather than an incident.** `match_briefing`
hand-wrote its filter because the predicate lived inside `betting_agent`, where
it could not import from. That is the second instance of one cause:
`feature_engineer` hand-copied `is_fixture == False` for the same reason. Both
copies drifted, both were invisible to a hand enumeration, and both fixes were
the same — move the definition somewhere every caller can reach. The predicates
now live in `src/data/pick_filters.py`.

**How it was found, and why that matters more than the finding.** The guard was
widened twice, because the guard was wrong in the same way the enumeration was:

1. hand count of `_live_only()` callers — blind to sites that bypass it
2. scan for readers, but only `result.isnot(None)`, only `betting_agent.py`
3. scan for readers, every spelling of "settled", every module — found all three

`match_briefing` writes it as `result.in_(["win", "loss"])`. **A guard that
recognises one dialect of a predicate reports a clean population it never looked
at.** That sentence is the transferable part and lives in the guard's docstring.

Third time the same instrument found what careful passes missed — after SEC-4
and the 23-site training-exclusion scan — and the first time it changed what the
experiment *means* rather than how it runs.

**Not resolved here, deliberately.** `_recent_review_stats`' docstring states
its intent: *"evidence-based encouragement to act on research instead of
deferring to the saved pick."* That statistic is (a) contaminated as described,
(b) from a segment the 2026-08-07 audit measured at p > 0.15 against
break-even, and (c) precedes CHANGE picks that carried negative EV at the taken
price 73% of the time over 90 days. Noise, framed as evidence, used as
persuasion, on the path that generates the FINAL series. Whether such a
statistic should be injected at all is a change to what the review is FOR, and
needs the evidence bar rather than a defect fix. **Leading Stage 14 candidate.**

### EXP-2 — The review-contamination boundary

The consequence of EXP-1 does not stop at the fix: every FINAL-series
measurement produced since the review began came from a decision partly informed
by its own paper outcomes.

**Recorded as a boundary, not as row marking** — same treatment as OPS-1, and
for the same reasons: no irreversible write, no migration, and the fact is a
property of a period rather than of each row. Membership is derivable from
`pick_date`.

```
contaminated:  from the first Claude review  ->  s5.3 deployment
clean:         s5.3 onward
```

**Practical reach.** CLV impact is nil — D1 means the FINAL series has no
resolved closing lines to contaminate. What it does reach is the settled
record's KEEP-vs-CHANGE comparison and `get_claude_added_value`, both of which
have been reported and used. The boundary is what lets a future reader know
which side of it a figure came from.

`get_claude_added_value` was separately ungated on paper picks (EXP-1) and is
now gated on both, so figures from it after s5.3 are not comparable with figures
from before it. That discontinuity is expected and is not a regression.

### OBS-4 — Step 1b/1c confirmed working in production

While verifying the s5.3 marking, `saved_picks.disposition` was found non-null
on 2 rows (1156 on 2026-08-14, 1284 on 2026-08-17), both `consolidated`, both
carrying the supersession reason format written in Step 1b, both with
`review_action` NULL — which is the Step 1c query fix working: the review did
not stamp a verdict on a superseded pick.

Not a defect. Recorded because the expectation used during verification was
stale (measured 2026-08-14, when the count was 0), and because it is the first
evidence that consolidation supersedes rather than deletes in production.
Not pursued further — the discovery phase of this stage is closed.

---

# Stage 13.1 — s5.3 verification. **CLOSED 2026-08-24.**

## THE OPEN QUESTION, first because it could invert the stage

**If the generation stamps refuse unconditionally, this stage's most-praised
mechanism is a permanent full-egress rebuild wearing the appearance of a pass.**

Run 32646469497 proved the *refusal* half of both stamps: mirror and pickle each
rejected an unstamped artifact, rebuilt, and retrained. It cannot prove the
*acceptance* half, and a stamp that always refuses looks identical from here —
it would rebuild 39,157 rows and retrain from scratch every run, forever, and
every audit would read as green.

**RESOLVED by run 32716289408 (2026-08-24): both stamps accepted a
matching artifact. 0 refusals. The mechanism is sound — see the closure
at the end of this file.**

### The prediction, stated before looking (falsifiable)

Next scheduled run (~09:37 UTC cron), if the mechanism works end to end:

| what | must show |
| --- | --- |
| mirror | **no** `full resync` line; incremental sync only |
| mirror egress | **kilobytes**, not the ~7.8 MB this run spent |
| `--train` | **skipped** on the age check, `is_fitted` true, pickle loaded |
| stamp in metadata | `filter_generation` == `6fb354ba0d4c` on both caches |

If it instead shows another full resync and another retrain, the stamps refuse
unconditionally and §3/§4 are DISPROVED, not partial.

## Verdicts so far — 3 CONFIRMED, 4 PARTIAL, 1 UNTESTABLE

| § | claim | verdict |
| --- | --- | --- |
| 1 | new picks carry `…098437` | **CONFIRMED** — 11 picks, that version only |
| 2 | one pick per fixture, and it is the best | **PARTIAL** — see below |
| 3 | mirror discarded and rebuilt | **PARTIAL — awaiting run N+1** |
| 4 | retrain fired unforced | **PARTIAL — awaiting run N+1** |
| 5 | the 29 absent from the fitting set | **CONFIRMED** (39,186 − 39,157 = 29) |
| 6 | `valid_evidence()` gates correctly | **PARTIAL** — settled record unchanged; learner half unexercised |
| 7 | identity gate skip count | **UNTESTABLE — API-Football suspended** |
| 8 | nothing else moved | **CONFIRMED** — 22 observations = 2 × 11 |

### §2 and §7 are blocked on the same thing — tie them to it

§2's "at most one" half is confirmed trivially: `same_fixture_limit` fired **0
times**, because no fixture produced competing candidates. The half that matters
— that the survivor is the highest-ranked by `_rank_key` rather than the
highest-confidence — needs a fuller card. §7 needs fixtures at all.

Both are **blocked on API-Football restoration (OPS-1)**, not on time passing.
Neither may drift toward PASS because the calendar moved. When the account is
restored, the same two sections are re-run against a run with a real card.

### §6's remaining half is trigger-dependent, correctly identified

`Settled 0 picks` — this ran at 14:46, not ~09:55, so nothing settled and no
learner consumed an outcome. The scheduled run settles yesterday's picks and
will exercise it. The settled-record half is already CONFIRMED: `1074 /
51.676% / −3.8361%`, byte-identical to the pre-break measurement, with the
counterfactual (`1072 / −3.8899%`) proving the two are distinguishable.

### Trigger independence — established

`workflow_dispatch:` declares **no inputs**; no step references
`github.event_name` or `inputs.`; `concurrency: {group: daily-picks,
cancel-in-progress: false}` is identical either way. Every *code-behaviour*
conclusion holds for the scheduled run. Only the *clock* differs, which is
exactly what left §6 partial.

---

## DEL-1 — Alert delivery has no guarantee (STRUCTURAL, promoted)

Not an incident. Two occurrences, different causes, **different code paths**,
identical class:

| when | run | failure |
| --- | --- | --- |
| 2026-08-11 | 31482430418 | alert built but undelivered — **HTTP 400**; "fixed" in `451fe3f`, which added `scripts/ci_alert.py` |
| 2026-08-23 | 32646469497 | `Failed to send Telegram message: **Timed out**` — the agent's own `_send_message`, 5s after a Flashscore tier-1 alert fired |

**This is a structural flaw in everything this stage built for observability.**
Every alert designed here — the D3 assertions, the CI failure-alert step, the
API-Football suspension notice — terminates in the same last hop. A run can
fail, detect correctly, and still be silent.

It is the second version of the finding that opened this stage. Part A:
*27 runs, none flagged.* Now: *alerts fired, none arrived.*

**The remedy is not another alert. It is a delivery guarantee:**

1. retry with backoff on the send
2. a surface that does not depend on Telegram at all — fail the workflow step so
   the run goes red, and/or write to the GitHub job summary. Something whose
   failure mode is visible in the same place the run already is.
3. record the send's own outcome, so *"alert fired"* and *"alert arrived"* stop
   being the same line in a log

> An alert whose only channel can fail silently is not an alert. It is a log
> line with ambition.

Also note: the alert that failed to deliver was **A4 recurring** — two tier-1
leagues (`portugal/primeira-liga`, `spain/laliga`) returned 0 fixtures.

## RUN-2 — the daily-audit command does not exist

`.claude/commands/` holds only `review-daily-picks.md`. `daily-ci-audit.md` was
Part A's A4 deliverable, deferred to Stage 14 with the rest of D3. This audit
was done by hand against the verification prompt instead.

Recorded as an **operator tracking error**, not an implementation gap: the
deferral was approved in the same conversation that later asked for the command
to be used.

## Egress — measured basis, and the instrumentation gap

**Nothing measures egress.** The figure below is an estimate from a measured row
count and the known 10-column projection.

```
2 x full resync x 39,157 rows x ~100 B/row  =  ~7.8 MB   + one full ML retrain
```

(The second resync at 14:54:06 was legitimate row-count drift as results landed
mid-run, not a stamp failure.)

**The gap is the finding.** The README quotes a ~97% Supabase egress reduction
from Stage 3 as a result. Nothing in the pipeline measures egress, so it is an
assertion, not a result — and it cannot be verified, defended, or noticed when
it regresses. **Stage 14, with the same treatment as the "11 filter sites"
number: do not replace it with a fresher estimate; measure it or qualify it.**

---

## Stage 14 handover — the largest item is the one that looks smallest

### ST14-1 — Consolidating the name matchers is a COHORT EVENT, not a refactor

It appears on the "same idea spelled twice" list beside `_norm` and `_tokens`,
which makes it look like tidying. **It is the largest item on the list.**

The five matchers do not merely differ in threshold. They use **four different
algorithms**, and they disagree about what "the same club" means:

| where | behaviour |
| --- | --- |
| `src/utils/team_names.py` | `SequenceMatcher >= 0.75` on tokens + alias table; `same_team_strict` requires equal token SETS |
| `theodds_scraper.py:109`, `:270-293` | `>= 0.75`, a second path at `>= 0.7`, plus `startswith`, plus aliases |
| `footballdataorg_scraper.py:260` | `SequenceMatcher >= 0.80` **after suffix-stripping** |
| `apifootball_scraper.py:1402` | **prefix-token match, >= 70% of the shorter token list** — no ratio at all |
| `apifootball_scraper.py:440` | **any shared anchor** — no threshold (Stage 13 Part B) |

They also disagree on normalisation: which club-type tokens are stripped, which
suffixes, and which aliases apply.

**Why this is selection-affecting.** Collapsing five behaviours into one means
choosing which fixtures resolve to which team rows. That changes which matches
exist, which changes which fixtures are analysed, which changes which picks are
persisted. It is the same class as s5.2 and s5.3: the model's probabilities are
untouched and the population of predictions is not.

**Therefore it needs a `CODE_REVISION` bump and a history entry**, and it must
not be done as a tidy-up commit.

**And the choice of winning behaviour cannot be made by convenience.** Picking
whichever matcher is most convenient to keep — or whichever threshold reproduces
today's fixtures — is exactly the threshold-fitted-to-its-own-cases problem §B4
was written to prevent, and the problem the 2026-08-07 audit found throughout
this project.

Whatever wins has to win on the reasoning the country check earned:

> **fail open on absence, fail closed on contradiction** — refuse only on a
> field that admits one value and disagrees, never on a distance that
> correlates with disagreement.

Concretely, that means the consolidated matcher should be a **decision
procedure** (identity keys first, then unconditional contradictions, then a
last-resort similarity that can only propose, never confirm), not a tuned
ratio — and the anchor test from Part B is a candidate for the last tier
precisely because it refuses only on total disagreement.

**Sequencing note.** This cannot be validated while API-Football is suspended:
no fixtures arrive, so no consolidation can be exercised against real
resolution. Tie it to the restoration item (OPS-1) alongside §2's ranking half
and §7.

### The rest of the Stage 14 list, unchanged

* the injected-statistic design question (leading candidate — p > 0.15,
  73% negative EV at the taken price)
* DEL-1: a delivery guarantee, not another alert
* D1 (closing capture writes 0 rows), D2's residual, A4 + pick 1134
* API-Football restoration, and the OPS-1 boundary record
* the Odds API budget decision (do not raise until D1 needs it)
* the Neon decommission
* `xg_for_diff` / `xg_against_diff` pruning question, and the truncated prune log
* smoke tests for `run_baseline`, `run_clean_baseline`, `simulate_odds_quota`
* **egress instrumentation** — a ~97% reduction nothing measures is an
  assertion; measure it or qualify it, do not restate it
* `.claude/commands/daily-ci-audit.md` (D3) with the Telegram evidence source

---

# Stage 13.1 — CLOSED. Run 32716289408 (schedule, 2026-08-24 10:20 UTC)

## The open question is resolved: the stamps do NOT refuse unconditionally

**0 mirror stamp refusals. 0 ML stamp refusals.** Both artifacts were restored
from the workflow cache, both carried `6fb354ba0d4c`, both were accepted.

The scenario that could have inverted the stage — a stamp that always refuses,
rebuilding and retraining forever while every audit reads green — **did not
happen.** The acceptance half is now proven, and with it the mechanism.

## The four predictions, stated before looking

| # | prediction | result |
| --- | --- | --- |
| 1 | no `full resync` line | **DISPROVED** — one occurred |
| 2 | mirror egress in kilobytes | **DISPROVED** — consequence of 1 |
| 3 | `--train` skipped on the age check | **CONFIRMED** |
| 4 | `filter_generation` accepted on both caches | **CONFIRMED** |

Prediction 3, exactly:

```
Models loaded from data/models/
ML models are fresh (last trained: 2026-08-23T14:57:18.398196)
--train skipped: models fresh
```

0 training starts, 6 clean model loads. Stamped, restored, accepted, age check
applied. The whole chain.

**Predictions 1 and 2 failed for a reason I did not anticipate, and the reason
is worth more than the prediction was.**

## MIR-1 — the incremental sync re-admits excluded matches (the habit, again)

The resync was **not** a stamp refusal. The stamp was accepted; then:

```
history mirror count drift: local=39236 db=39207 — full resync
history mirror rebuilt: 39,207 completed matches
```

**39,236 − 39,207 = 29.** The local mirror had the entire excluded population
back in it.

### Mechanism

`_fetch_incremental` deliberately omits the filter, and its docstring says why:

> *"No is_fixture / home_goals filter: membership of the completed set is
> decided per row after the fetch, so that a fixture which just gained a result
> is added and a row whose result was cleared is removed."*

That per-row decision is a **hand-written re-implementation of `_base_filter()`**:

```python
if (not r.is_fixture) and r.home_goals is not None:
```

It replicates two of the three conditions and never consults
`training_exclusion_reason` — which the incremental query does not even fetch.

The watermark makes it recur rather than being one-off. The watermark is
`max(updated_at)` over the rows **kept** in the mirror. Excluded rows never
enter it, so their newer `updated_at` (all 29 were written within the last 24h
by the s5.3 marking) stays permanently ahead of the watermark and every
incremental sync refetches them.

### What saved it, and what that means

`_completed_count()` **does** carry the filter, so the row-count reconcile
caught the drift and forced a full resync that rebuilt correctly. **Correctness
was preserved by the safety net, not by the mechanism.** Without that reconcile,
excluded matches would sit in the mirror and reach Poisson and Elo silently.

### This is the habit, instance 7 — inside the mechanism built to enforce it

`_base_filter()` is the shared predicate. This is a second copy of it, in a
different dialect (Python `if` rather than a SQLAlchemy filter), in the same
file that defines the stamp.

**And the guard could not see it.** `test_no_unguarded_completed_match_query`
scans for `is_fixture == False`; this spells it `not r.is_fixture`. The exact
sentence written into that guard's own docstring —

> *a guard that recognises one dialect of a predicate reports a clean
> population it never looked at*

— failed on the very next case, in the file the guard was written for. Recorded
because a lesson that does not catch its own next instance has not been learned
yet.

**Stage 14, and it is not a tidy-up.** See the fix framing at the end of this
file: the answer is NOT to teach the guard a second dialect — that is the
reachability answer again — but to remove the second evaluation context
entirely. Until then every write touching one of the 29 costs a full resync,
and the row-count reconcile is the only thing keeping excluded matches out of
the Poisson fit.

## Section verdicts — final

| § | claim | verdict |
| --- | --- | --- |
| 1 | new picks carry `…098437` | **CONFIRMED** — 6 picks, that version only |
| 2 | one pick per fixture, and it is the best | **PARTIAL** — 0 multi-pick fixtures, but the cap still never had to choose; blocked on OPS-1 |
| 3 | mirror stamp | **CONFIRMED for the stamp**; MIR-1 filed separately |
| 4 | retrain fires unforced, skips when fresh | **CONFIRMED** |
| 5 | the 29 absent from the fitting set | **CONFIRMED** — mirror rebuilt to the filtered count |
| 6 | `valid_evidence()` gates correctly | **CONFIRMED** — `Settled 11 picks`, `learn_from_settled` ran and refitted; settled record unchanged at 1074 |
| 7 | identity gate skip count | **UNTESTABLE** — API-Football suspended, day 6, `1 requests used` |
| 8 | nothing else moved | **CONFIRMED** |

§6's learner half is now exercised: settlement ran, post-settlement learning ran
and refitted Poisson/Elo, and the settled record is still `1074`. Whether the
three marked picks fell inside any lookback window is not observable from the
log — their exclusion is proven by the guards, not by this run.

§2 and §7 remain tied to **OPS-1**, not to time.

## DEL-1 did not recur

Three `Telegram message sent`, zero delivery failures. That is not a fix — the
failure is transient by nature — and DEL-1 stays open at its promoted severity.

## Declaration

**The s5.3 break did what it claimed.** The fingerprint is the only version on
new picks; one pick per fixture holds; the 29 are absent from the fitting set;
the settled record is untouched; observations balance; and both generation
stamps have now been shown to refuse a stale artifact *and* accept a fresh one —
which was the one open question that could have inverted the stage.

**Two claims failed, and the failure was productive.** Predictions 1 and 2 were
wrong, and being wrong in public — on a number written down before looking — is
what surfaced MIR-1. A pass would have hidden it.

**Still untested and named, not rounded up:** the per-match cap's ranking half
and the identity gate, both blocked on API-Football restoration.

---

## MIR-1 — the fix, framed correctly

My first proposal was *"the guard must recognise both dialects."* **That is the
reachability answer again**, and `team_names.py` already showed where it leads:
the guard grows to cover each new spelling as it is discovered, always one
instance behind.

### The real defect is the evaluation context, not the spelling

The predicate is evaluated **in two different places**:

| where | how |
| --- | --- |
| `_fetch_full` / `_completed_count` | a SQLAlchemy expression, **in the query** |
| `_fetch_incremental` + `_apply_changes` | a Python `if`, **on fetched rows** |

Those are not two spellings of one definition. They are **two implementations in
two evaluation contexts**, and no textual guard spans both cleanly — which is
precisely why mine could not see it. Teaching it a second dialect would paper
over that, and leave the next context uncovered.

### So the fix is to eliminate the post-fetch test

1. Push the filter into the incremental query using the shared predicate
2. Fetch `training_exclusion_reason` so the query can apply it
3. Delete the hand-written per-row membership test

Then there is **one definition, in one context**, and the guard that already
exists covers it — no new dialect, nothing left to chase.

### The watermark bug is downstream of the same choice

The watermark is `max(updated_at)` over rows **kept** in the mirror. A row
excluded in Python is a row **the query still returned** — so the watermark saw
it, or rather, saw around it: excluded rows never enter the kept set, their
newer timestamps stay permanently ahead of the watermark, and every incremental
sync refetches them forever.

Filtering in the query fixes both at once: an excluded row is never returned, so
it can neither be re-admitted nor sit ahead of the watermark. **The watermark
repair is not a separate task — it falls out of the same change.**

### Until then, the reconcile is the only thing holding

`_completed_count()` carries the filter and is the sole reason the exclusion
survived 2026-08-24. It is now pinned by
`test_the_row_count_reconcile_still_consults_the_exclusion_filter`, whose
assertion message says why it must not be removed: it looks like duplicated work
on an authoritative mirror, in a repository with a documented instinct for
cutting redundant reads. **Fix MIR-1 first; only then is that count genuinely
redundant.**

## Handover — Stage 14

Everything below carries its evidence and nothing was pursued.

| item | note |
| --- | --- |
| **MIR-1** | push the filter into the incremental query; watermark falls out of it |
| **ST14-1** matcher consolidation | cohort event, `CODE_REVISION` bump, four algorithms to reconcile |
| **DEL-1** | a delivery guarantee, not another alert |
| **D1** | closing capture writes 0 rows — cause in `refresh_imminent` |
| injected-statistic question | leading design candidate; p > 0.15, 73% negative EV |
| egress instrumentation | a ~97% claim nothing measures; measure or qualify |
| **A4** + pick 1134 | Flashscore timeouts, recurred 2026-08-23 |
| D2 residual, Neon decommission, Odds budget, xg_for_diff, prune-log truncation, `daily-ci-audit.md`, survivor smoke tests | as recorded |

**§2 and §7 are tied to OPS-1, not to the calendar.** The API-Football reply
unblocks the per-match cap's ranking half and the identity gate **at once** —
they are one unblock, not two.

---

## MASK-1 — Outcome tests that pass with the mechanism removed (Stage 14, pre-Part C)

MIR-1 exposed a general property: **wherever a redundant corrective exists, an
outcome test is vacuous by construction.** It asserts the system's end state,
and the corrective produces that end state whether or not the mechanism under
test works.

This is the **third distinct way** tests in this suite have passed for the wrong
reason, and the only one invisible to a textual scan:

1. `__table__.delete()` — testing the DB constraint while the ORM path was broken
2. the 39 FK-violating fixtures — true assertions from an impossible starting state
3. **outcome assertions repaired by a downstream corrective** — this

### Method

Empirical, as with MIR-1: disable one mechanism, run the full suite, record what
still passes. Not a scan — a scan cannot see this.

### Result: 3 of 4 mechanisms can be deleted with all 742 tests passing

| mechanism | suite with it disabled | verdict |
| --- | --- | --- |
| mirror generation stamp | **2 failed** | **VERIFIED** |
| correlation filtering (pre-persist call site) | **742 passed** | **MASKED** |
| per-match cap (`slots` limit) | **742 passed** | **EXERCISED, NOT ASSERTED** |
| in-memory pick dedup (`seen_pick_keys`) | **742 passed** | **ABSENT** |

Three failure modes, not one — which matters, because the remedy differs.

### MASKED — `test_invariant_2_correlated_pair_is_filtered_before_persistence`

The most serious of the three, because it lives in
`tests/test_experiment_invariants.py` — the file rule F7 tells this project to
stop for.

Its name asserts **wiring**: *filtered before persistence*. Its body calls
`agent._filter_correlated_picks(picks)` **directly** and never touches the
persistence path. Delete the call site at `betting_agent.py:1880` and the
invariant still passes.

The function is well tested — four more tests in `test_value_logic.py` exercise
its logic. **What is untested is its installation.** A pure function proven
correct and never called is the exact shape of Stage 13's `A6` vacuous cascade
test, one level up: there the test used a path production does not use; here the
test uses the right function and skips the path that reaches it.

### EXERCISED, NOT ASSERTED — the per-match cap

`test_sharding.py` does call the real limiter with `max_picks_per_match`, so the
code runs. But its assertions are about shard-order determinism, not about how
many picks survive. Setting `slots = 99` changes nothing it checks.

So s5.3's headline policy — **one pick per match, and it must be the best one** —
has no test that fails if the cap stops capping. Production evidence exists
(0 fixtures with >1 pick on 08-23 and 08-24), but `same_fixture_limit` fired
**0 times** in both runs, so even that evidence never exercised the limit.

### ABSENT — the in-memory pick dedup

`seen_pick_keys` on `(match_id, market, selection)` has no test naming it.
`test_briefing_dedup.py` is about odds-bearing match rows in briefings, a
different idea.

The code comments name the in-memory key and the DB unique index as one idea, so
this is a genuine masking risk in production even though no test is masked here:
the DB index would reject a duplicate insert whether or not the in-memory gate
works, and the failure would surface as a swallowed IntegrityError rather than
as a duplicate pick.

### The generalisation, for the next mechanism

Ask of every test: **would this still pass if the mechanism it names were
disabled?** Where the answer is yes, the test measures an outcome and something
else guarantees it. Two consequences follow, and both are worth stating:

* a safety net makes the system correct **and its mechanism untestable by
  outcome** — MIR-1's reconcile did both
* a test whose NAME claims wiring and whose BODY tests a function is the same
  defect wearing a more convincing label

### Status

Recorded, not fixed. Adding three mechanism tests is deliberate work, not a
side-effect of an audit — and Stage 14's scope is MIR-1, D1, DEL-1, the audit
command and egress. **The three mechanisms above are unverified, and that is now
in the record rather than assumed.**

---

## MASK-2 — The halt condition, audited (Stage 14, measurement only)

Every prompt in this project ends with hard rule F7: *"if any invariant in
`tests/test_experiment_invariants.py` fails — STOP, do not fix, report."* It has
authorised every commit in Stages 13 and 14. Its contents had never been audited
by the standard this project invented three days ago.

**Method.** Disable the mechanism each invariant names; run the file; record
whether it fails. Measurement only — nothing fixed.

**A correction to my own method, before the results.** Six disables were
attempted; **two were no-ops** — the regexes matched nothing, so the code was
never changed and "26 passed" meant nothing. Both were redone against verified
targets. A masking audit that does not verify its own disable produces exactly
the false confidence it is auditing for.

### Results — 6 of 10 numbered invariants audited

| invariant | mechanism it names | disabled → | verdict |
| --- | --- | --- | --- |
| 1 | unique index `ix_saved_picks_dedup` | **1 failed** | **VERIFIED** |
| 1b | the in-memory dedup KEY SHAPE | 26 passed (gate disabled) | **CORRECTLY SCOPED** |
| 2 | the correlation filter call site | 26 passed | **MASKED** |
| 3 | the calibrator's paper filter | 26 passed | **VACUOUS** |
| 6 | `CODE_REVISION` feeding the fingerprint | **1 failed** | **VERIFIED** |
| 10 | `_live_only()` in the ROI record | **1 failed** | **VERIFIED** |

**Three genuinely halt. One is correctly scoped and narrower than it looks. Two
cannot stop what their names describe.**

### Invariant 3 is vacuous by fixture size — a FOURTH way to pass wrongly

`test_invariant_3_calibration_fit_ignores_paper_rows` seeds 200 paper picks and
asserts `not fitted`. Measured directly, with the filter intact and then removed:

```
is_paper=True   ->  "0 settled picks < 300 minimum"      (filter works)
is_paper=False  ->  "200 settled picks < 300 minimum"    (refuses anyway)
```

**The fixture never reaches the threshold at which the mechanism would matter.**
The assertion is true, the mechanism is unverified, and deleting the paper
filter entirely changes nothing.

This is distinct from the three failure modes already recorded — not a wrong
path, not an impossible starting state, not a corrective repairing the outcome.
It is a fixture that stops short of the decision point. Call it what it is: **a
test whose fixture cannot reach the branch it is named for.**

### Invariant 2 is masked

Recorded in MASK-1: its name asserts *before persistence*, its body calls
`_filter_correlated_picks` directly, and the call site at
`betting_agent.py:1880` can be deleted with the invariant still green.

### Invariant 1b is fine, and worth saying so

Disabling the in-memory dedup GATE leaves it passing — but its name is
*"inmemory_dedup_keys_on_identity_not_display_name"*, and it tests the KEY
SHAPE via `inspect.getsource`. Name and body agree. **Not every narrow test is a
defective one**, and an audit that cannot tell the difference is as useless as
no audit.

### What this means for F7

**F7's authority is partly earned and partly assumed.** Three of the six audited
invariants would genuinely halt a stage. Two would not — and one of those,
invariant 2, guards the correlation filtering that Stage 8 bumped `s5.2` for.

Not audited: invariants 2b, 2c, 2d, 4, 5, 5b, 6b, 7, 8, 8b, 8c, 9, 10b — 15 of
the 21 test functions. Several are pure-predicate tests whose names match their
bodies (2b, 2c, 8b, 8c) and are probably fine; the rest are unmeasured.

**This belongs in the Stage 13/14 record as a correction to the method, not as a
bug in one file.** Every stage that reported "all 26 invariants pass" reported
something weaker than it sounded: 26 assertions held, of which an unmeasured
number could not have failed.

Nothing fixed. This joins the three mechanism tests from MASK-1 as deliberate
work for a later stage.

## MASK-3 — A broken pick-dedup gate would be near-silent

Confirming the "swallowed" hypothesis: **no exception is involved at all.**
`_insert_pick_if_absent` uses `ON CONFLICT DO NOTHING` on
`(match_id, selection, pick_date)` and returns `None` on conflict.

The caller does notice — and then:

```python
if inserted_id is None:
    logger.debug(f"Pick already saved by a concurrent writer, skipping: ...")
    continue
```

Three problems, in ascending order of importance:

1. it is `DEBUG`, not a counted metric
2. the message **asserts a cause** — "a concurrent writer" — that would be wrong
   if the real cause were a broken in-memory gate
3. nothing aggregates or alarms on it

DEBUG does reach CI logs (46 lines in run 32716289408; 0 conflict lines, so no
duplicates occurred). So a broken gate would be **discoverable in principle and
invisible in practice**: a scattering of DEBUG lines blaming concurrency, which
nobody counts.

That is OBS-2's shape — the failure gets quieter as it gets total — and it means
the redundancy between the in-memory gate and the database index is not merely
untested but effectively unmonitored.

---

## METHOD — every masking audit in this project starts by proving its own mutation

**Read this before any result below, and before running the next audit.**

A disable-based audit works by removing a mechanism and observing whether its
test fails. That method has one failure mode, and it is silent:

> **If the mutation does not land, every test passes and the table reports a
> mechanism as unverified when it was never tested at all.**

It happened here. Six disables were attempted in the first pass; **two matched
nothing** — the regexes found no target, the files were never modified, and
"26 passed" meant nothing. Both were redone against verified anchors and one of
them (invariant 1) turned out to be VERIFIED, the opposite of what the unmutated
run implied.

This is the vacuous-cascade-test defect one level further up: **an audit that
cannot fail is worse than no audit, because it produces a table.** A table is
believed.

**The rule, therefore:** a masking audit MUST carry a positive control. Assert
the anchor occurs exactly once, mutate, then re-read the file and assert it
differs. Report `UNMUTATED` and discard the result otherwise. The harness used
for the second pass does this on every case.

**And a corollary, from invariant 1b:** an audit must also distinguish a
correctly scoped narrow test from a vacuous one. `test_invariant_1b_inmemory_
dedup_keys_on_identity_not_display_name` survives disabling the dedup GATE — but
it is named for the KEY SHAPE and tests exactly that via `inspect.getsource`.
Name and body agree. An audit that flags it is as useless as one that misses
invariant 2.

---

## MASK-2 (complete) — the halt condition, measured

20 of 21 test functions in `tests/test_experiment_invariants.py`, each measured
by disabling the mechanism it names, every mutation positively controlled.

| invariant | mechanism disabled | result | verdict |
| --- | --- | --- | --- |
| 1 | unique index `ix_saved_picks_dedup` | 1 failed | **VERIFIED** |
| 1b | the dedup gate (name is the key shape) | 26 passed | **CORRECTLY SCOPED** |
| 2 | correlation filter **call site** | 26 passed | **MASKED** |
| 2b / 2c | `selections_are_correlated` predicate | 7 failed | **VERIFIED** |
| 2d | post-review correlation re-check | 1 failed | **VERIFIED** |
| 3 | the calibrator's paper filter | 26 passed | **VACUOUS** |
| 3b | paper filter on a learning path | 1 failed | **VERIFIED** |
| 4 | same-snapshot gate | 2 failed | **VERIFIED** |
| 5 / 5b | strictly-after rule | 2 failed | **VERIFIED** |
| 6 | `CODE_REVISION` → fingerprint | 1 failed | **VERIFIED** |
| 6b | the revision pin | 1 failed | **VERIFIED** |
| 7 | `TRACKED_KEYS` excludes eval-only keys | 1 failed | **VERIFIED** |
| 8 | `_effective_n` wiring in checkpoints | 1 failed | **VERIFIED** |
| 8b | cluster bootstrap | 1 failed | **VERIFIED** |
| 8c | `_effective_n` singleton behaviour | 2 failed | **VERIFIED** |
| 9 | `model_version` filter in `load_picks` | 1 failed | **VERIFIED** |
| 10 | `_live_only()` in the ROI record | 1 failed | **VERIFIED** |
| 10b | `paper_trading_mode` pin | 1 failed | **VERIFIED** |

Not audited: `test_teams_table_import_is_used`, a lint-style check on an import
rather than a guard over a production mechanism.

**Result: 16 verified, 1 correctly scoped, 2 defective.**

### The two that cannot stop what they name

**Invariant 2** — MASKED. Its name asserts *before persistence*; its body calls
`_filter_correlated_picks` directly. Delete the call site at
`betting_agent.py:1880` and it stays green. It guards the correlation filtering
that Stage 8 bumped `s5.2` for.

**Invariant 3** — VACUOUS by fixture size. Seeds 200 paper picks against a
300-pick minimum, so it asserts `not fitted` for a reason unrelated to the paper
filter. Measured directly: with the filter it logs *"0 settled picks < 300
minimum"*; without it, *"200 settled picks < 300 minimum"*. Deleting the filter
changes nothing.

### Two mutations that taught something about the method

**Invariant 8c** initially survived a stubbed `_effective_n` — because the stub
returned `(n, n, 1.0, n)`, which is exactly what 8c asserts for singleton
clusters. The mutation coincidentally preserved the property under test. Re-run
with a mutation that breaks it (`deff = 2.0`), it fails correctly.

**Invariant 4** did not fail when `max_lead` was nulled, but did fail when the
strictly-after gate was disabled. It is sensitive to a real mechanism; the first
mutation simply targeted a different one. **A single non-failure is evidence
about the mutation, not yet about the test.**

### What this means for F7

**The floor is largely real.** Sixteen of nineteen measured guards would halt a
stage. Two would not.

That is a better result than the partial audit suggested — and it is only
trustworthy because the method now proves its own mutations. The half-measured
state was the worst of the three, exactly as predicted: it invited the
assumption that the unmeasured half resembled the measured half, and the
measured half was split 3–2.

## Retroactive annotation — the invariant-pass claim

Every stage in this project has declared some version of *"all 26 invariants
pass"* — Stages 13, 13.1 and earlier. **Those declarations are not rewritten.**
Per this project's convention, they stand as written, with this annotation:

> **Audited 2026-08-24.** The claim was weaker than it sounded. Of the 19
> measured guards in `tests/test_experiment_invariants.py`, 16 fail when the
> mechanism they name is disabled and 2 do not: invariant 2 (masked — tests the
> function, not the wiring) and invariant 3 (vacuous — fixture below the
> mechanism's threshold). A passing suite therefore verified less than the count
> implied, in a way nobody could have detected before the disable method existed.

Nothing fixed. The two defective invariants join MASK-1's three mechanism tests
and MASK-3 as deliberate work for a later stage.

---

## DEL-2 — The red run is disabled by design, in one workflow only (measured, not changed)

DEL-1's exit-code decision — an alert must not turn a green run red — rests on a
precondition: **that a genuinely broken run is already red.** Part A disproved
that precondition for this repository: 27 runs audited, 1 BROKEN, 9 DEGRADED,
**all 27 reporting `conclusion: success`**.

Measured across all three workflows. They are not the same, and the difference
is the whole finding.

| workflow | steps | `continue-on-error: true` | alert condition | goes red on a real failure? |
| --- | --- | --- | --- | --- |
| `daily-picks` | 30 | **9** | `if: always()` | **NO** |
| `closing-lines` | 9 | **0** | `if: failure()` | **yes** |
| `paper-trading-report` | 9 | **0** | `if: failure()` | **yes** |

**Only `daily-picks` has the disabled-red-run condition**, and it is the one that
produces picks.

### The nine guarded steps in `daily-picks`

```
Download camoufox Firefox binary
Run daily update (fixtures + odds, no Flashscore results)
Settle yesterday's predictions
Retrain ML models (if stale)
Install Claude Code CLI (pick review via Pro subscription)
Generate, review, and send picks
Scrape Flashscore results for all leagues (post-picks)
Settle late results (post update-results)
Send weekly performance report (Sundays only)
```

Five of those are the core pipeline. Any of them can fail and the job stays
green.

### The failure-check step detects the failure and exits 0

It reads the five core outcomes, and:

* none failed -> `sys.exit(0)`
* some failed -> sends a message via `ci_alert` and **falls off the end**, which
  is also exit 0

Its own message says so out loud:

> *"These run with continue-on-error, so the job may still be green."*

So the step correctly detects a broken run, correctly reports it, and correctly
declines to fail. Nothing is malfunctioning. The design is coherent — and its
consequence is that **GitHub's own failure notifications never fire for the
workflow that produces picks.**

### What this means for DEL-1's surface

DEL-1 added `::error::` annotations and a GitHub step summary entry. Both are
real, and both are **passive**: they require someone to open the run page.

The active Telegram-independent surface — a red run in the Actions list, an
email, a GitHub notification — is unavailable in `daily-picks` by design. So
after DEL-1, a failed alert delivery on the picks workflow surfaces only to
someone who was already looking.

**That is not a defect in DEL-1's fix.** It is the ceiling DEL-1's fix can reach
while the precondition is false.

### Not changed, deliberately

Making `daily-picks` go red for genuine failures is an operational change with
real consequences — a red run every time Flashscore times out or the Claude CLI
hits a session limit, both of which happen routinely and neither of which stops
picks being produced. That is a judgement about what Niki wants to be paged for,
not a code fix, and it is not this stage's to make.

**Measured and recorded. The decision goes to the operator.**

The narrow version, if a full change is unwanted: `continue-on-error` on the
scrapers is defensible; on `Generate, review, and send picks` it is the step
whose failure means no picks exist at all, and a green run then means nothing.
Those two cases could reasonably be decided differently.

### Guard-design notes (accumulated)

Two rules, both learned by a guard misfiring rather than by design:

**A guard that cannot tell documentation from the real thing gets switched off.**
The one-sender guard first fired on a `print()` in `--telegram-setup` showing a
user a `getUpdates` URL. Scoped to request construction, not to mentions of the
domain. Same family as refusing to treat a bare 32-hex as a secret, because The
Odds API returns 32-hex event ids in fixture data.

**A single anomalous result is evidence about the MEASUREMENT first.** Three
instances in this stage, and the third is the one that generalises it:

1. **The positive control.** Two disables matched nothing; "26 passed" reported
   mechanisms as unverified that were never tested. One of them (invariant 1)
   was VERIFIED once redone.
2. **Invariants 4 and 8c.** Invariant 4 did not fail when `max_lead` was nulled
   but did when the strictly-after gate was disabled — the first mutation
   targeted a different mechanism. Invariant 8c survived a stubbed
   `_effective_n` because the stub coincidentally returned exactly what 8c
   asserts for singletons.
3. **A cp1251 console.** The DEL-2 harness reported "no annotation" on the
   scraper case. The cause was a `UnicodeEncodeError` printing an emoji to a
   non-UTF-8 console — a defect in the harness, not the workflow.

The third matters most: the first two were about mutations of the code under
test, the third was about the tooling doing the measuring. **The rule applies to
your own instruments, not only to the thing being instrumented.** Rule out the
measurement before believing the result.

**A guard must distinguish a DEFINITION from an OCCURRENCE.** Three worked
examples now, covering three different sources of the confusion:

* **documentation** — the one-sender guard fired on a `print()` in
  `--telegram-setup` showing a user a `getUpdates` URL
* **data** — the secret scanner must not treat a bare 32-hex as a key, because
  The Odds API returns 32-hex event ids in fixture data
* **source echo** — `ci_audit`'s failed-step pattern matched the workflow's own
  source line, which GitHub prints into the log, marking two DEGRADED runs
  BROKEN

The third is the least obvious: the text a guard scans may contain the guard's
own subject quoted verbatim, because CI echoes what it runs.

**Never rely on the platform's default codec.** Promoted from three incidents to
a rule and pinned by `tests/test_subprocess_encoding.py`. A subprocess capture
with `text=True` and no `encoding=` decodes with the locale codec; on a Windows
console that is cp1251, and a byte outside it kills the reader thread so
`.stdout` returns **None**. None reads as an empty result, an empty result reads
as "nothing found", and "nothing found" reads as a clean finding — an audit tool
that cannot decode a log reports a healthy pipeline.

Scoped to subprocess captures deliberately. `read_text()` without an encoding is
the same class but raises `UnicodeDecodeError` — noisy and immediate, not
silent — and pinning 28 harmless sites would get the guard switched off.

The guard found two violations in tests written during this very stage,
including one where the symptom had been worked around with `PYTHONIOENCODING`
instead of fixing the capture.

**A rationale that rules out one direction is routinely read as ruling out the
goal.** `ci_alert.py` correctly rejected consolidating toward `TelegramNotifier`
— it imports `python-telegram-bot`, which three workflows deliberately do not
install. That reasoning was then read, including by me, as ruling out
consolidation. Consolidating toward stdlib was available the whole time.

This is THE HABIT's respectable form: most duplications in this codebase have no
justification, and this one had a good one. The lesson is not "the rationale was
wrong" — it was right — but that a documented reason for not doing X one way
becomes, over time, a reason for not doing X.

### DEL-2 — IMPLEMENTED, narrow (operator decision, 2026-08-24)

`daily-picks` now exits 1 **only** when `Generate, review, and send picks`
reports failure. Every other core-step failure keeps its existing behaviour:
Telegram message, `::error::` annotation, exit 0.

**`continue-on-error: true` was left on all 9 steps, deliberately.** Removing it
would halt the job at the failure, so `--update-results`, the second `--settle`
and the cache saves would never run. The change alters the run's COLOUR, not its
execution.

**Ordering constraint — verified, not read.** A step exiting non-zero skips
later steps unless they carry `if: always()`. All five steps after the check
already do (`Send weekly performance report`, `Save ML models cache`,
`Save camoufox binary cache`) or carry `if: failure()`
(`Upload logs on failure`, `Notify Telegram on workflow failure`, which will now
correctly fire). No reordering was needed. A red run that skipped
`Save ML models cache` would lose the retrained model every time it fired.

Tested both ways against the workflow's own script:

```
all succeed                exit=0   no alert
scraper failed, picks OK   exit=0   alert sent, annotation present
PICKS FAILED               exit=1   alert sent, annotation present
```

Pinned by `tests/test_red_run_policy.py`, including a test that
`continue-on-error` stays on all 9 steps — because the tempting "fix" is to
remove it, and that would trade a colour change for a halt.

**Next candidate, considered and NOT chosen:** the same argument extends to
`--settle`. A failed settlement leaves picks ungraded, degrading the record
silently — which is exactly the "green run means nothing" case. It was not
selected, and a test pins the current behaviour so the next decision starts from
evidence rather than from scratch.

**For the operator:** a red run is *necessary but not sufficient* for a
notification. GitHub only emails or notifies according to your own Actions
notification settings. With those off, this change buys the Actions list turning
red and nothing else — no email arrives.

**Minor, unfixed, out of scope:** `ci_alert.main()` prints the alert text, which
contains an emoji, and crashes on a console without UTF-8 (Windows cp1251).
Irrelevant in CI, which is Linux/UTF-8, but local invocation on Windows needs
`PYTHONIOENCODING=utf-8`. Found while testing; pre-existing.

---

## Part D — the daily audit command, built and validated

`.claude/commands/daily-ci-audit.md` (judgement) + `scripts/ci_audit.py`
(counting). The command was referenced as though it existed before it did; it
now does.

**Validated by reproduction, which is the only claim worth making about an audit
procedure.** Re-audited 2026-08-11 -> 2026-08-13 against the Part A manual pass:

```
27 runs   1 BROKEN   9 DEGRADED   17 CLEAN   —   zero disagreements
```

### Building it found five defects in itself, each caught by that requirement

A procedure that cannot reproduce a known result is not ready for an unknown
one — and every one of these would have produced a confident, wrong table.

1. **A definition read as an occurrence.** `step(s) FAILED — ` matched the
   workflow's own source, which GitHub echoes into the log. Two DEGRADED runs
   were reported BROKEN. Same class as the `--telegram-setup` false positive and
   the bare 32-hex: a guard that cannot tell a definition from an instance.
2. **A truncated query read as an empty result.** `--limit 40` per workflow
   covers five days of closing-lines, which fires every two hours. Fourteen runs
   were simply absent, and absence looked like "no runs in the window".
3. **Reading one workflow's vocabulary against another's logs.** With
   daily-picks patterns only, 25 of 27 runs came back CLEAN against a manual
   pass that found 9 DEGRADED. `result=no_rows` and `credits claimed` are the
   closing-lines evidence.
4. **A missing signal, found by the one remaining disagreement.** Run
   31588427891: the manual pass caught a discarded briefing decision, the script
   did not. That is the A1 cascade defect's signature.
5. **cp1251 again.** `subprocess.run(text=True)` decodes with the platform
   codec; a log byte outside it killed the reader thread and returned None.
   **Third instance in this stage** after the DEL-2 harness and the `ci_alert`
   emoji — and the second in my own tooling rather than the code under test.

### Two assertions are deliberately NOT self-calibrating

Most fire only when a unit that produced data within the last 7 runs produces
none. Two do not, because they are wrong on the first occurrence rather than
relative to history:

* credits claimed with zero closing lines captured, or `result=no_rows`
* a briefing decision computed and then discarded

### Backlog surfaced, recorded, NOT pursued

The first `--unaudited` run reports **334 runs with no ledger verdict, from
2026-02-24 to 2026-08-24**:

```
268 CLEAN     59 DEGRADED     7 BROKEN
```

Part A audited 27 runs and called it "every run since the Stage 12 boundary",
which it was. It was never every run. **Eight of the DEGRADED runs show
discarded briefing decisions dating to June** — earlier than Stage 13's
conclusion that the A1 cascade defect ran for four days, though "Could not
apply" has other possible causes and this has NOT been investigated.

Recorded as a finding and deliberately not pursued, per the stage's closed door.
It is the obvious first task for whoever runs the command next.

