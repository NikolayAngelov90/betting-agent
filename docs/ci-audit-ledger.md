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


---

## D1 — PREMISE FALSIFIED. The experiment is measuring.

Stage 14 opened with: *"the experiment has never collected a single valid
closing line... 0 valid CLV pairs... The fault is in request construction,
inside `refresh_imminent`."*

**That was true when Stage 13 measured it and is not true now.** No fix was
made, because there is no defect of the described shape.

### What today's paper-trading report says (run 32720148367, 2026-08-24)

```
captured           57
valid CLV pairs  : 57

MODEL  (model_market / model_selection)     CLV mean -0.509%  95% CI [-1.4%, +0.3%]
FINAL  (market / selection, post-review)    CLV mean -0.470%  95% CI [-1.2%, +0.3%]
```

Both series resolved independently, with cluster-bootstrapped intervals. This is
the measurement the whole project exists to make, and it exists.

### The refresh path works, and has for months

Across all saved logs:

```
result=ok       65     result=no_rows  17     result=skipped  41
refresh_imminent: 1339 / 1323 / 1289 / 204 / 179 odds rows written
```

`no_rows` is real but is the minority case, not the rule. B1's inference — *both
counts zero means an empty event list, so the fault is request construction* —
was drawn from the DEGRADED runs and generalised to all of them.

### The actual timeline, from `closing_capture_status`

| window | captured | missing | late |
| --- | --- | --- | --- |
| 08-10 → 08-13 | **0** | 65 | 1 |
| 08-14 → 08-22 | **57** | 111 | 16 |
| 08-23 → | 0 | 8 | 3 |

**The turning point is 2026-08-14**, and the stop on 08-22 is explained: the
August budget reached 349/400 with 1 credit spendable, and every closing-lines
run since requests `0 league(s) = 0 credits`. Capture did not break — it ran out
of money, exactly as OPS-2 predicted it would.

The three reports that said *"0 valid CLV pairs"* were **correct for their
dates**. Nothing has been collected before 08-14 and nothing since 08-22.

### What has NOT been established

**Why it began working on 2026-08-14.** The commits that day are the Stage 13
Step 1 fixes (ORM cascade, consolidation, FK pragma) and none of them touch the
capture path. So the change was not ours, and until the cause is known it cannot
be relied on to continue. **This is the open question, and it is the one that
matters** — a measurement that started by accident can stop by accident.

### The real remaining defect is COVERAGE, not capture

In the window where capture worked: **57 captured, 111 missing, 16 late** — about
31%. The README already concedes ~36% of markets (Team Goals, BTTS, Double
Chance) sit outside the `h2h` + `totals` refresh, which accounts for part of it
and not all.

`late` is its own question: a price observed at or after kickoff is excluded, so
16 captures were thrown away for arriving too late relative to the run schedule.

### Status

**D1 is not `FIX DEPLOYED — UNDEMONSTRATED`. It is `PREMISE FALSIFIED — NO FIX
MADE`.** The stage's purpose was overtaken by events between Stage 13's
measurement and Stage 14's execution, which is the same staleness that made my
own "44 unsettled picks" reading wrong nine days after I took it.

Recorded, not pursued: why capture began on 08-14; whether the ~31% coverage is
explained by the market gap alone; and why 16 captures arrived late.

---

## D1 — the open question CLOSES. It did not start by accident.

One query settled it. The 65 `missing` picks in 2026-08-10 → 08-13, by
competition:

| competition | picks |
| --- | --- |
| europe/europa-conference-league | 36 |
| europe/europa-league | 16 |
| europe/champions-league | 10 |
| bulgaria, portugal, sweden (domestic) | 3 |

**62 of 65 — 95.4% — are UEFA competitions.** And the mirror image is exact:
of the 57 captured from 08-14 onward, **57 are domestic and 0 are UEFA.**

The measurement did not begin by accident on 2026-08-14. **It began when the
domestic season resumed.** Capture had nothing to capture while the card was
European qualifiers, because the provider does not price them.

**Stage 12.2 was right and its conclusion was overruled on a reasonable-sounding
argument that turned out to be a category error.** The objection was that
qualifiers could not explain zero across ~700 picks — true, but the ~700 are
picks *considered by the report*, while capture concerns only picks with a
fixture inside the coming capture window. Those were the qualifiers. Same shape,
same dates and same cause as D2's injury coverage, which was reclassified for
exactly this reason and then not connected to this.

## Coverage decomposed, not averaged

The headline "31%" conflates two different things. In 08-14 → 08-22:

| | picks |
| --- | --- |
| captured | **57** (100% domestic, 100% inside `h2h`+`totals`) |
| missing — **outside** `h2h`+`totals` | **78** (structural) |
| missing — **inside** `h2h`+`totals` | **33** (genuinely not captured) |
| late | 16 |

**Of the 90 picks the refresh could even price, 57 were captured — 63%, not
31%.** The other 78 were never capturable: Double Chance, BTTS, Draw No Bet,
Team Goals sit outside the two markets the refresh requests.

### `late` is a start-time problem, not a density problem

All 16 late picks kick off at **11:00 or 11:30 UTC**. The first closing-lines
run of the day is ~11:57. Every one of them had already kicked off before the
first capture attempt of the day existed.

A denser 2-hourly schedule would not help. **An earlier first run would** — the
window is not wrong, its start is. Recorded, not fixed.

## Budget raised 400 → 450, and the ledger honours it

Changed in **both** `config/config.yaml` and `config/config.example.yaml` — the
example is what CI deploys, and the two drifting is its own defect class.

**Verified rather than recalled:** `odds_api` appears **0 times** in
`TRACKED_KEYS`, and `model_version` is unchanged at
`stage5_baseline_20260807.098437`, still matching the pin. No cohort break, no
`CODE_REVISION` bump.

**Verified the ledger acts on it**, because a budget raised in config that the
ledger ignores looks identical to no change:

```
before:  349/400 used,  1 spendable, =  0 more league request(s)
after:   349/450 used, 51 spendable, = 25 more league request(s)
```

## The pattern this exposes — and it recurs on 1 September

August burned **349 credits in 24 days: ~14.5/day, ~450/month** against 400
spendable. **Raising to 450 buys eight days, not the pattern.** September will
exhaust around the 27th, October the same.

The README's `measured: 212 credits/month` is **not a measurement** — it and the
`212 cr at 88% vs 340 cr at 85%` comparison both come from
`scripts/simulate_odds_quota.py`, whose `simulate()` replays the selection rules
over historical picks. Corrected in README, relabelled SIMULATED with the real
burn rate stated beside it, rather than replaced with a fresher estimate — a
simulation and a measurement are different kinds of claim, and swapping the
number would preserve the error. Same treatment as "11 filter sites".

**Structural consequence, and the next stage's opening question.** At 57
observations per 9 days against a 500-observation target, and a budget that
affords roughly three weeks of every month, the question is no longer *"is it
measuring"* — it is **"how many days a month can it afford to?"**

---

# STAGE 15 — THE COVERAGE/CREDIT FRONTIER

One question: **what does each additional valid closing observation cost, and
which gains are cheapest?**

## Part A — the arithmetic

Measured 2026-08-25, re-derived rather than carried forward from Stage 14 (the
57 quoted there was nine days old and is now 61):

| series | observations | distinct fixtures | design effect |
| --- | --- | --- | --- |
| FINAL (post-review) | 61 | 61 | 1.00 |
| **MODEL (frozen model's own selection)** | **46** | 46 | 1.00 |

**MODEL is the binding series.** It is smaller because the review changes the
pick in 27 of 61 cases (44%), and when it does, the captured close prices the
FINAL selection — the model's own selection resolves only if it happens to sit
in the same requested market. **500 − 46 = 454 observations still required.**

**Why deff is exactly 1.00, which is not luck.** Every captured fixture carries
exactly one observation because Stage 13 imposed one pick per match. Before that
rule, multiple picks on one fixture would have been correlated within it and the
effective n would have been strictly below the nominal n. The clean design
effect is a *consequence of a Stage 13 decision*, and it holds only while that
rule holds — if `max_picks_per_match` is ever raised, the bootstrap must cluster
by fixture and 500 nominal will stop meaning 500 effective.

## Part B — every lever, priced against MODEL

Marginal rate, measured directly: 73 request outcomes in 2026-08-10 → 08-24 at
2 credits each = 146 credits, producing 46 MODEL observations = **3.17 credits
per MODEL observation**.

| lever | Δ credits/mo | MODEL gained | cr / MODEL obs |
| --- | --- | --- | --- |
| **L2** stop requesting leagues the provider does not price | **−17** | 0 | **negative — free** |
| **L4** move the 21:17/23:17 windows earlier | +0 (reallocated) | ~14/mo | ~1.4 |
| **L1** add an earlier run | +20 | ~14/mo | ~1.4 |
| **L6b** `min_interval` 180 → 110 | +54 | ~16/mo | ~3.3 |
| **L6a** stop pick-time writes suppressing the first close | +16 | ~5/mo | ~3.3 |
| **L3a** add `team_totals` | +72 | ~9/mo | ~8 |
| **L3b** add `double_chance` / `draw_no_bet` | +72 each | **0** | **infinite** |
| **L5** cut pick-time pricing | −213 max | negative | **disqualified** |

### L5 — pick-time pricing is not waste, it is the product

61% of the budget (213 credits/month) had never been examined. It is not waste.
Of **66 league-days priced at pick time, 2 produced no pick at all** — 3%. The
other 64 did exactly what they were paid to do. The 27 that produced a pick but
no closing observation are a *capture* failure, not a pricing one: the credit
bought the pick, which is the thing the experiment is about.

So the L2 question, asked of the larger consumer, returns almost nothing. Any
larger cut means pricing fewer books or markets, which changes the blend and
therefore the model's probability — **a cohort event, out of scope**, and now
recorded with its arithmetic rather than left as an open suspicion.

**The consequence is the important part.** 213 of 450 credits is committed
before capture gets any. The capture budget is not "whatever is left over" — it
is **237/month, fixed**, unless the cohort is broken or the tier is paid.

### L6 — structural AND a defect, and they are different halves

11 picks kick off at hour 12; 0 captured. But **7 of the 11 are outside
`h2h`+`totals`** (Double Chance ×2, BTTS ×2, Team Goals ×2, Draw No Bet ×1), so
the anomaly is worth **at most 4** observations, not 11. That alone demoted it.

Then the refresh plan gave the cause: every candidate league was skipped as
`refreshed within the last 180 min`. Across all cached logs that is **35
league-requests, the largest single skip reason** — ahead of `not mapped` (21)
and `no pending pick` (7). It splits cleanly by the hour of the run:

- **8 suppressions at 11:00** — `daily-picks` fires at 09:37 and writes odds at
  pick time. Those rows are the `taken_at` rows. **A `taken_at` row can never
  satisfy the strictly-after rule**, so treating it as "recently refreshed"
  guarantees `missing`. This half is a genuine defect.
- **27 suppressions at 15:00/17:00** — the cron is `17 11,13,15,17,19,21,23`,
  every **120** minutes, and the interval is **180**. Consecutive runs cannot
  both fire for the same league. **Every second closing-lines run is a no-op by
  construction.**

**The two 180s are different quantities that share a number.**
`clv.DEFAULT_MAX_CAPTURE_LEAD = 180 min` is the maximum gap between the close
and *kickoff*. `odds_refresh_min_interval_minutes = 180` is the minimum gap
between two *fetches*. The config comment reasons from the first to justify the
second — but "fresh enough" for CLV means *close to kickoff*, not *recently
fetched*. A price pulled 170 minutes ago for a fixture kicking off in 10 minutes
is at the validity edge, not comfortably inside it.

The same comment records `2-hourly runs, 256 credits/month, 96% coverage` as
the chosen operating point. That is **`simulate_odds_quota.py` output, the third
time this simulator's numbers have been recorded as measurement** (after the
README's `212 credits/month` and its `212 cr at 88%` comparison). Measured
coverage is **63% of capturable picks**, not 96%.

**Neither half of L6 qualifies for Part D**: lifting a skip *spends* credits.
L6 is a coverage-for-credits trade at ~3.3 cr/obs, i.e. an operator's decision.

### A defect found in the instrumentation itself

`refresh_imminent` derived its per-league `result=ok|no_rows` attribution line
from `written`, the **batch total**. A batch where one league returned rows and
three returned nothing logged four `result=ok` lines. **Every `no_rows` figure
ever read out of these logs — including Stage 15's own — is a lower bound.**
Fixed in Part D, because L2 cannot be implemented on a signal that is wrong.

## Part C — recommendation

**The experiment reaches 500 MODEL observations around March 2027, and that is
the optimistic case.** 454 needed ÷ (237 capture credits ÷ 3.17 per
observation ≈ 75/month) = **6.1 months**. This is the number the project has
been circling since paper trading began, and it is the first time it has
existed.

It is optimistic because it assumes the full 450 is available every month.
August spent 349 in 24 days — **~436/month against a 450 cap.** The cap is
already binding. At August's *realised* capture rate, which stopped on the 22nd
when the money ran out, the answer is 46/month and **June 2027**.

| scenario | MODEL obs/month | 500 reached |
| --- | --- | --- |
| realised August (budget exhausted on the 22nd) | 46 | June 2027 |
| **full 450 spent, no changes** | **75** | **March 2027** |
| + L2 (free) | 80 | mid-February 2027 |
| + L2 + L4 (free, reallocation only) | ~90 | late January 2027 |
| + L6b (spends 54 more) | — | not affordable inside 450 |

**Recommended, in order:** L2 (implemented, Part D — it is free). Then **L4**,
which is the best lever on the board and costs nothing: it does not add a run,
it *moves* the 21:17 and 23:17 windows earlier, into the hours where fixtures
actually kick off. Every `late` pick measured kicks off at 11:00 or 11:30 UTC,
before the day's first capture attempt exists. **L4 is a scheduling change with
zero credit cost and it is the single highest-value action available.**

**Do not buy L3b at any price. A lever that moves the non-binding series is not
a lever.** `double_chance` and `draw_no_bet` add FINAL observations and **zero**
MODEL observations, and MODEL is the series that is short. FINAL is already at
61 and is not the constraint. The same test disqualifies any future proposal
that widens post-review coverage without widening the frozen model's own.

**On the free tier the checkpoint is not reachable before 2027.** No combination
of free levers gets there in 2026; the cheapest levers are exhausted at ~90
observations/month, and 454 will not fit in four months at that rate. Reaching
it sooner requires either paid credits or breaking the cohort — **both are
Niki's decisions, not this stage's.**

**Not implemented, by the stage's own rule:** L1, L3a, L4, L6a, L6b all cost
credits or change the schedule. L4 in particular is *recommended and left
undone* — it is a one-line cron change, but it is an operating decision about
when the system runs, and Stage 15 was scoped to measure, not to re-time.

## Part D — L2 implemented, and nothing else

`src/scrapers/barren_leagues.py`. A league that returns an empty event list
**three runs running** is not priced by the provider; the request cannot produce
an observation, so declining it cannot lose one. Strictly negative credits,
exactly zero coverage.

**Not a blocklist, deliberately.** A hard-coded UEFA set would have excluded
`france/ligue-1`, `netherlands/eredivisie`, `spain/laliga` and
`portugal/primeira-liga` — each returned `no_rows` exactly once in this window
and each is plainly priced. And coverage *changes*: the provider does not price
Conference League qualifiers in July and does price the group stage in
September. So the exclusion is earned (3 consecutive), **expires** (10 days),
must be **re-earned** after expiry, and is **cleared** by one success.

**Verified by replay, not by assertion.** The 73 real request outcomes from
2026-08-10 → 08-24 were replayed through the cache:

```
requests avoided       : 4  = 8 credits   (~17/month at this rate)
of which WRONG (was ok): 0
```

**Smaller than Part B first priced it (−26/month), and the correction stands.**
Two reasons: the threshold of 3 is deliberately conservative, and the
attribution defect above means the logged `no_rows` count it replays is itself a
lower bound. With attribution fixed, the cache will exclude sooner and save
more — but that is a prediction, and it is written here as one.

The load-bearing guard is `test_the_exclusion_expires`. An exclusion that never
lifts encodes a fact that expires and never learns it was wrong — which is worse
than the 17 credits it saves. 769 passed.

### The lever was a no-op when first written, and nothing would have said so

`closing-lines.yml` **had no cache step of any kind.** Every run starts from a
fresh checkout, so the record would have been written, consulted once inside the
same process, and thrown away. The exclusion threshold is three *consecutive*
empty fetches — unreachable when nothing survives a single run.

**The failure mode is the one this ledger keeps recording.** It would have
reported as "L2 implemented, 0 credits saved", which is indistinguishable from
"the provider started pricing everything". A lever that measures as working
while doing nothing is worse than an absent lever, because the next stage
prices its successor against a saving that never happened.

Fixed with the restore/save pair already used for `data/briefings_sent.json`
(run-id key + `restore-keys` prefix), and pinned by
`test_the_workflow_persists_the_record_across_runs`, which fails if either half
is removed. The record is gitignored: committing it would ship one machine's
observations as every deployment's permanent exclusions.

Two designs were tried and rejected first, and the reasons are worth keeping:

- **Derive it from the `odds` table, no new state.** Fails: the UEFA
  competitions carry thousands of odds rows (Bet365, 1xBet, Pinnacle) from
  API-Football, and there is no source column, so "has this league received
  odds" cannot distinguish the provider that declined to price it from the one
  that did.
- **A new table in Neon.** Correct and durable, and rejected as disproportionate
  for a 17-credit lever inside a stage scoped to measurement.

### Instrumentation added, as approved

`OddsApiQuota.claim_requests` now emits one structured line per claim:

```
CREDITS_CLAIMED account=<workflow> credits=N requests=N asked=N month=YYYY-MM
```

The 213/144 split this stage's entire frontier rests on was **reconstructed by
inference over CI logs and reconciled to within 2.3%** — it was never measured.
Next stage reads it off the logs instead of arguing for it.

---

**STAGE 15 — FRONTIER MEASURED.**

454 MODEL observations remain; the capture budget is 237 credits/month against a
measured 3.17 credits per MODEL observation; the checkpoint arrives around
**March 2027**, and no combination of free levers brings it into 2026.

---

# STAGE 15 (continued) — L4 SHIPPED, AND PART B's RANKING WAS WRONG

## The correction that matters: L3b is not worthless, and market expansion is not cheap

Part B asserted: *"`double_chance` and `draw_no_bet` add FINAL observations and
**zero** MODEL observations."* **That is false.** Uncaptured picks since
2026-08-14, grouped by the FROZEN MODEL's own market:

| `model_market` | uncaptured |
| --- | --- |
| Team Goals | 25 |
| **Draw No Bet** | **24** |
| **Double Chance** | **22** |
| Over 2.5 | 20 |
| BTTS | 16 |
| Under 3.5 | 15 |
| Under 2.5 | 12 |
| 1X2 | 5 |

**46 uncaptured MODEL observations sit in exactly the two markets I declared
worthless for MODEL.**

**How the error was made, because the shape recurs.** The claim came from
measuring the market mix of *captured* picks, where Double Chance and Draw No
Bet are 0 by construction — they cannot be captured, which is the entire point.
**Selection on the outcome.** Asking "what do our observations look like?" and
concluding "therefore nothing else exists" is the same move as reading a
survivor's traits off the survivors. The corrective is to measure the
*population*, not the sample the process already filtered.

### But correcting it does NOT make market expansion cheap

The second half of the correction cancels most of the first. Credits are
`requests × regions × markets`, so **adding a market raises the cost of every
request**, not just the ones that yield a new observation:

| configuration | credits | MODEL obs | **obs per credit** |
| --- | --- | --- | --- |
| current (h2h + totals) | 72 × 2 = 144 | 46 | **0.319** |
| + `draw_no_bet` | 72 × 3 = 216 | 70 | **0.324** |
| + `draw_no_bet` + `double_chance` | 72 × 4 = 288 | 92 | 0.319 |
| + all four missing markets | 72 × 6 = 432 | 133 | 0.308 |

**The frontier is nearly FLAT at ~0.32 MODEL observations per credit.** Adding
Draw No Bet is a 1.6% improvement; adding everything is a 3% *degradation*.

## The finding this stage actually produced

**Almost every lever prices at the same rate, and the rate is set by the budget,
not by the schedule.** Both of Part B's headline recommendations were wrong in
opposite directions and land in the same place:

- L3b was priced at **infinity** and is really **~0.32** — the average.
- L4 was priced at **1.4 cr/obs** and is really **~5** — worse than average.

**Only L2 beats the frontier**, because it removes requests that return nothing
rather than adding requests that return something. That asymmetry is the whole
lesson: on a flat frontier the only free move is deleting waste.

**March 2027 stands, and is now better supported than when it was a guess.** It
is not movable by re-timing, re-intervalling or re-marketing. It is a function
of 237 capture credits per month and ~0.32 observations per credit. **Moving it
requires more money or a smaller target — nothing else on the board touches it.**

## Where the shortfall actually lives

Uncaptured picks since 2026-08-14, by whether the refresh could price them:

| | picks | captured |
| --- | --- | --- |
| in requested markets (`h2h` + `totals`) | 116 | **61 (53%)** |
| outside them (DC 28, DNB 24, TG 19, BTTS 14) | 85 | **0** |

**55 of 140 uncaptured picks (39%) are schedule or budget losses; 85 (61%)
require a market that is not requested.** An earlier reading of this table said
10% — that was a **defect in the query, not in the data**: `saved_picks.market`
uses a display taxonomy (`Over 2.5`, `Under 3.5`), not the odds table's
(`over_under`), so `market IN ('1X2','over_under')` silently matched no totals
picks at all. Two vocabularies for one concept, which is THE HABIT in its
purest form and was found here by a number that looked implausible.

## L4 — shipped as approved, and inert until L6a is fixed

Added `47 10 * * *`; **all seven existing windows kept.** The correction was
right and my reasoning had the error the operator identified: moving is free in
credits and **not free in seasonal robustness**, and only the first was in the
model. The 21:17/23:17 runs claim nothing against an *August* kickoff
distribution; winter fixture lists carry late kickoffs, so what looked like dead
weight is unpriced insurance against a one-month, one-season sample.

**Not 09:17, and not 10:17.** MEASURED: picks for early kickoffs are written
**10:07 → 10:32 UTC** (daily-picks fires 09:37, then runs Claude's per-match
review before saving). At 09:17, **0 of 54** early-kickoff observations exist
yet; at 10:17, **8 of 54**. A window scheduled before its input exists finds no
pending picks and claims no credits — it would have measured as *"added, 0
credits, 0 observations"*, indistinguishable from a lever that does not work.
**10:47** clears the last observed write by 15 minutes and precedes an 11:00
kickoff by 13.

**AND IT CANNOT PRODUCE AN OBSERVATION UNTIL L6a IS FIXED.** daily-picks writes
odds at ~10:30. A 10:47 refresh is 17 minutes later, so `min_interval: 180`
skips every candidate league — the same suppression L6a describes. **L4 and L6a
target the same picks, and L4 is downstream of L6a.** Shipped anyway because it
costs nothing precisely because it is suppressed, and it activates the moment
L6a lands. Recorded so nobody reads its zero as a verdict on the schedule.

## L6a and L6b, priced

| lever | Δ credits/mo | recoverable in-market picks | cr / MODEL obs |
| --- | --- | --- | --- |
| **L6a** pick-time write resets the interval clock | +34 | 2 per 11 days | **~8.5** |
| **L6b** `min_interval` 180 → 120 | +116 | 12 per 11 days | **~4.6** |

**L6a is a defect and is still not worth buying yet.** The operator's reasoning
is exactly right — the interval clock is reset by a write that can *never* serve
as a close, so the skip guarantees a lost observation rather than deferring one.
But the picks it guarantees losing are overwhelmingly in markets the refresh
does not request: at kickoff hours 11–12, **27 uncaptured picks, 2 of them in
`h2h`+`totals`**. Fixing the bug recovers 2 per 11 days at ~8.5 credits each,
the most expensive lever measured. **It should still be fixed** — a clock reset
by a non-qualifying write is wrong independent of its yield, and its yield rises
the moment market coverage widens. It is a correctness fix whose value is
currently held down by a *different* constraint.

**L6b is the second-cheapest lever, and it was never on the board.** Aligning
the interval to the cron recovers ~12 in-market picks per 11 days for +116
credits/month. At ~4.6 cr/MODEL obs it is worse than the 3.13 average and
better than everything except L2 — but **there is no headroom to buy it**: the
cap is already binding at ~436/month against 450.

## PROVENANCE — a convention, not a third correction

`256 credits/month, 96% coverage` in `config.example.yaml` is relabelled
**SIMULATED (`scripts/simulate_odds_quota.py`, pre-2026-08)** with the measured
figure stated beside it — relabelled, not restated, the same treatment as the
README's `212 credits/month` and the `~97% egress reduction`. The two numbers
were never the same quantity: the simulation counted *pick* coverage, the 63%
counts *closing-line capture*. **That they were ever compared is precisely what
untagged numbers cause.**

### The review rule

> **Every quantitative claim in a config comment, docstring or README carries
> its provenance — MEASURED, SIMULATED or ASSUMED — and its date. An untagged
> number is the defect; specific wrong values are only its instances.**

**A mechanical guard was prototyped and deliberately NOT shipped.** Scanning
config, README, `src/` and `scripts/` for claim-shaped figures (`N credits/month`,
`N requests/day`, `N% coverage|win|ROI`) found **40 figures, 27 untagged** — and
several of the 27 are vendor specifications (`free tier, 100 requests/day`)
where the convention does not naturally apply. That is the profile the project
already ruled unacceptable in `test_subprocess_encoding.py`: *a guard that flags
28 harmless sites gets switched off.* Stated once as a review rule, per the
operator's own fallback, and left to code review.

## Two measurement defects that deserve their own line

### 1. Every `no_rows` figure ever read from these logs is a lower bound

`refresh_imminent` derived per-league `result=ok|no_rows` from `written`, the
**batch total**: a batch where one league returned rows and three returned
nothing logged four `result=ok` lines. Fixed in Part D.

**This touches a conclusion already acted upon.** The `no_rows` counts were part
of the evidence used to overrule Stage 12.2, and they were systematically
*understated* — the true rate of empty responses was higher than the number that
argument was built on. The overrule was later falsified on independent grounds
(D1: capture began when the domestic season resumed), so the conclusion did not
survive on this evidence either way. But **the reasoning had a silent bias in
it, and nothing in the pipeline would have surfaced that.**

### 2. L2 as first written would have measured as success — MASK-1's shape in a lever

`closing-lines.yml` had no cache step, so the barren record could never survive
to a second run and the three-run threshold was unreachable. It would have
reported **"implemented, 0 credits saved"** — indistinguishable from *"the
provider started pricing everything"*.

**MASK-1 was a test whose outcome was masked by a downstream corrective. This is
the same shape in a production lever**: a component that cannot work, reporting
a value that a working component could legitimately produce. The generalisation
is now explicit:

> **A lever's null result must be distinguishable from its success. If "it did
> nothing" and "there was nothing to do" produce the same measurement, the lever
> is unverifiable and the replay-against-real-outcomes step is not optional.**

Replay is what caught it: 73 real request outcomes, 4 avoided, 0 wrongly
skipped. A unit test on the cache would have passed in every one of these cases,
because the cache was correct — **it was the deployment that was inert.**

### Postscript: cp1251, fourth instance

Hit again while writing the provenance scan — this time on `print()` to a
cp1251 console, not a subprocess capture, so `test_subprocess_encoding.py`
correctly did not catch it. Recorded to keep the count honest and to mark the
boundary of that guard: it pins *captures*, where failure yields `None`
silently. Console encoding failures raise loudly and are a different, lesser
problem. **The guard's scope is still right; the incident count is now four.**

---

# STAGE 16 — WHAT IS THE CHECKPOINT FOR?

**Order of work, declared as required: Part A first, then Part B in full, then
Part C.** The effect-size argument was written and its inputs measured before any
comparison to the observed interval was made.

**Contamination, stated honestly rather than denied.** I cannot claim to have
been blind to the observed mean: it appears in the Stage 16 prompt itself
(−0.509%, CI [−1.4%, +0.3%]) and has been in context since Stage 14. What is
checkable instead is this: **every input to Part B is a measured overround, a
config value, or a stated assumption from betting economics — and the thresholds
it produces (+2%, +4%) are six to eleven standard errors away from anything in
the observed data.** There is no path by which a mean of −0.587% is reverse
engineered into +2%. The argument stands or falls on its inputs, which are listed
so a reader can check rather than trust.

## 500 is the fourth instance, and the most consequential

README line 11: *"Real money stays off the table until **500 valid closing
lines** exist"*; line 60: *"100 → 200 → 500 (decision-grade)"*. **No derivation
is attached to any of the three.** After `212 credits/month`, the `~97%` egress
reduction and `256 credits/month, 96% coverage`, this is the fourth untagged
number — and the first that decides whether real money is ever staked.

*(README line 11 also says "~3.5 months away". Stage 15 measured March 2027,
about seven months. Stale, and flagged here rather than edited — no changes this
stage.)*

## PART A — what 500 is powered to detect

MEASURED 2026-08-25 from `pick_observations`, MODEL attribution, n = 46:

| quantity | value |
| --- | --- |
| mean | **−0.587%** |
| SD | 2.859% |
| SE | 0.422% |
| median | 0.000% |
| skewness / excess kurtosis | −0.335 / +0.170 |
| beat the close | 15 / 46 (32.6%) |

### A1. Symmetry, and whether the normal approximation holds

It does, and it was checked rather than assumed. Skew is mild (−0.335) and tails
are near-normal (+0.170). A 200,000-resample bootstrap gives **[−1.420%,
+0.218%]** against the normal **[−1.413%, +0.240%]** — agreement to 0.02pp. The
bootstrap interval is quoted throughout.

**The 21.7% point mass at exactly zero was investigated before being accepted**,
because a pile-up at "no change" is what a stale re-read would look like. It is
genuine price quantisation, not an artifact: every closing observation is
**≥190.7 minutes after its `taken_at`** (mean 285) and **14.6–115 minutes before
kickoff** (mean 51). The strictly-after rule and the 180-minute validity limit
both hold with margin on every row.

### A2. Precision as a function of n

**Design effect: 1.00, confirmed from the data** — 46 observations across 46
distinct fixtures. This is a consequence of `s5.3`'s one-pick-per-match cap, not
a property of CLV, and it stops holding if `max_picks_per_match` is ever raised.
*The README's `18.9% of fixtures carry two picks` is a PRE-`s5.3` HISTORICAL
figure and should be labelled as such.*

**Variance stability: the projection inherits real uncertainty.** SD = 2.859% ±
0.301% (11% relative, from 46 observations over 11 days of one season). Every
half-width below carries that ±11%.

| n | 95% half-width | range from SD uncertainty |
| --- | --- | --- |
| 46 | ±0.826% | 0.739 – 0.913% |
| 100 | ±0.560% | 0.501 – 0.619% |
| 150 | ±0.458% | 0.409 – 0.506% |
| 200 | ±0.396% | 0.354 – 0.438% |
| 300 | ±0.324% | 0.289 – 0.358% |
| **500** | **±0.251%** | 0.224 – 0.277% |
| 750 | ±0.205% | 0.183 – 0.226% |

### A3. Stated plainly

> **At n = 500, this experiment can distinguish a mean CLV of ±0.25% from zero**
> (95% interval half-width), with 80% power against a true effect of ±0.36%.

## PART B — what effect size would justify real money

*Written before Part C. Inputs labelled.*

| input | value | provenance |
| --- | --- | --- |
| settled record | 1,320 picks, 52.121% win, **−5.396% flat ROI**, avg odds 1.886 | MEASURED 2026-08-25 |
| overround, 1xBet (most represented book) | **4.86%** | MEASURED, 2,772 complete books |
| overround, Pinnacle | 3.76% | MEASURED, 1,517 books |
| overround, best line across books | **≥1.85%** | MEASURED, 3,241 matches — LOWER BOUND |
| overround, exchange (Matchbook / Betfair) | 0.88 – 1.09% | MEASURED |
| exchange commission | ~2% of net winnings | FROM LITERATURE |
| Kelly fraction / max stake | 0.25 / 4.0% of bankroll | MEASURED (config) |
| devigged close = true probability | — | **ASSUMED** (standard efficient-close model) |

*The best-line figure is a lower bound: it takes each book's latest price, which
may straddle timestamps, so it manufactures phantom arbitrage (582 of 3,241 sum
below 1.0). The true simultaneous best-line overround is higher, which makes B's
thresholds conservative in the direction that matters.*

### B1. Derivation

CLV here is `price_clv = taken / closing − 1`, measured against the **vigged**
close (`src/evaluation/clv.py:137`). Under the assumption above, the true
probability is `p = (1/O_close) / R` where `R` is the closing overround, so:

```
EV = p·O_taken − 1 = (O_taken/O_close)/R − 1 = (1 + CLV)/R − 1
```

**Break-even therefore requires CLV = R − 1: you must beat the closing line by
the vig just to reach zero.**

| where you bet | break-even CLV |
| --- | --- |
| retail (1xBet) | **+4.86%** |
| Pinnacle | +3.76% |
| best-line shopping | **+1.85%** |
| exchange (+commission) | ~+0.88% + commission |

**A sanity check that validates the model, and independently confirms the
2026-08-07 audit.** Betting *at* the close (CLV = 0) at retail prices predicts
`1/R − 1 = −4.64%`. The measured flat ROI over 1,320 settled picks is
**−5.396%**. The 0.76pp gap is the model's entire contribution, and it is well
inside the ~±2.6pp standard error of an ROI estimate on 1,320 picks. **The
system's realised loss is, to measurement precision, exactly the vig it pays.**
The 2026-08-07 audit reached "the model adds no information over the price"
three independent ways on settled outcomes; this is a fourth, from prices.

### B2. Range

Actionability, not just significance: at avg odds 1.886, quarter-Kelly stakes
`0.25·e/0.886 = 0.282·e`.

- **Minimum decision-relevant: CLV ≥ +2%.** Below the best-line overround of
  1.85%, expected value is at or below zero *under the most favourable
  assumption available* — disciplined line-shopping. An edge cannot exist below
  this.
- **Comfortable / actionable: CLV ≥ +4%.** Clears the retail overround, survives
  degradation of the best-line assumption, and yields an edge of ~2.1% → a
  quarter-Kelly stake of ~0.6% of bankroll. Below roughly a 1.8% edge (stake
  0.51%), operational cost and model-drift risk dominate the return.

## PART C — comparison

### C1. What n does B actually require?

80% power, α = 0.05 two-sided, at the measured SD of 2.859%:

| B's threshold | n required |
| --- | --- |
| +2.0% (minimum) | **17** |
| +4.0% (comfortable) | **5** |

**500 is over-specified by roughly 29× against the minimum decision-relevant
effect.** It is calibrated to resolve ±0.25% — **eight times finer than the
smallest difference that could change any decision.**

### C2. What the current sample already excludes

n = 46. Bootstrap 95% CI **[−1.420%, +0.218%]**. One-sided 95% upper bound
**+0.107%**.

| threshold | distance from observed mean |
| --- | --- |
| +2.0% | **6.1 SE** |
| +4.0% | **10.9 SE** |

**Both decision thresholds are already excluded, decisively.** At the observed
mean, **four observations** would suffice to rule out +2%; there are 46. The
question is no longer *when will we know* — it is **we already know**.

### C3. The three outcomes

1. **MODEL CLV clearly above +2%** → real money justified. **Excluded at 6.1 SE.**
2. **MODEL CLV clearly below +2%** → real money stays off permanently. **This is
   the observed result**, and it is the stage's answer.
3. **Indistinguishable** → more data needed. **Not the situation.** The interval
   is an order of magnitude tighter than the threshold it is being compared to.

Taking the prompt's instruction seriously: outcome 2 is **the experiment
succeeding**. It was built to test whether the model beats the closing line. It
does not, and the measurement is now precise enough to say so.

## PART D — recommendation (policy; no changes made)

> **RECOMMENDED CHECKPOINT — replace the count with a decision rule.**
>
> **Rule:** stop when the one-sided 95% upper bound on MODEL CLV lies below the
> best-line break-even overround (+1.85%).
> **Effect size:** +2.0% minimum decision-relevant (DERIVED, Part B).
> **Power:** ≥80% at n = 17; ≥99.9% at n = 46.
> **Status: ALREADY SATISFIED** — upper bound +0.107% vs +1.85% threshold.
> **Date:** 2026-08-25. **Provenance:** DERIVED from measured overrounds
> (2,772/1,517/3,241 books), measured SD (n = 46), and config staking rules.
>
> **If a count is preferred:** n = 50, not 500 — 80% power needs 17, and 50
> carries the +2% test at >99% power with margin for variance drift.

### What the current data supports, and what it does not

**Supported:** for this cohort, in these markets, the MODEL series does not beat
the closing line, and the shortfall against actionability is not marginal — it is
six standard errors. Real money is not justified, and no plausible continuation
of the current series changes that.

**NOT supported — and this is the honest limit of the stage.** The 46
observations span **11 days of one season**, one cohort (`s5.3`, which opened
inside an API-Football outage), and a narrow set of leagues concentrated in the
`h2h`+`totals` markets. **The statistical precision is ample; the external
validity is the weak point.** Whether the result holds across seasons, in winter
fixture lists, or in the four markets the refresh never prices is genuinely
unknown — and **more observations of the same 11-day regime do not answer it.**
That is a coverage-and-diversity question, not a sample-size question, and it is
the one thing 500 would have bought that 50 does not.

**Which reframes the whole frontier.** Stage 15 concluded that reaching 500 costs
seven months and that only money moves the date. Part C says the decision does
not need 500. **The binding question was never "how do we afford more
observations" — it was "more observations of what".**

### THE HABIT, third instance in the data layer

The overround query first returned empty: `odds.selection` carries **both**
`Home`/`Away` and `Home Win`/`Away Win`. Two vocabularies for one concept, in the
same column — after the display-vs-storage market taxonomy found in Stage 15 and
the two `180`s. Recorded, not fixed; no changes this stage.

## E. Downstream, deferred

**L6a remains unfixed and is now cheaper to justify.** It was priced at +34
credits/month against a binding cap. If the checkpoint falls from 500 to ~50, the
credit pressure that made it unaffordable falls with it, and it becomes a
correctness fix — the interval clock is reset by a `taken_at` write that can
never satisfy the strictly-after rule — rather than a purchase.

**L4 is SHIPPED AND KNOWN-INERT.** The 10:47 window is suppressed by exactly that
interval reset. Recorded here so that its zero is never read as a measurement:
this is a deployment whose null result is indistinguishable from its success,
which is the condition Stage 15 generalised. **It is inert by diagnosis, not by
observation.**

---

**STAGE 16 — CHECKPOINT DERIVED.**

500 is powered to detect ±0.25%. The smallest effect that could change any
decision is +2%. The checkpoint is over-specified by ~29×, and the decision it
gates was already resolved at n = 46 — six standard errors clear.

---

# STAGE 17 — IS THERE ANY SIGNAL LEFT?

**Pre-registration committed before analysis:
`6e3a59c04700bc1786d4272a71303465d4215a48`** (`docs/stage17-preregistration.md`).
It discloses the single query run before it was written — substrate extent only,
no predictor, no target, no relationship. **No code, config or schedule changes
were made in this stage; the working tree is clean.**

## Headline

**Three of the four pre-registered hypotheses cannot be tested on this data at
all. The one that can was real, replicated out of sample, and is less than half
the size required to be worth anything.**

## PART A — the substrate, and its limits

### A1. `opening_odds` is thinner and stranger than it looks

`odds` is unique on `(match_id, bookmaker, market_type, selection)` and
overwritten on refresh. `opening_odds` is first-seen-by-this-system, never
overwritten; `timestamp` is the LAST update. **There is no opening timestamp.**

MEASURED 2026-08-25, training window (kickoff 2026-02-28 → 06-30), 1X2:

| | |
| --- | --- |
| 1X2 rows | 99,943 |
| with `opening_odds` | 92,379 |
| **`opening` ≠ `current`** | **9,394 (10.2%)** |
| matches / with movement | 2,464 / 1,883 |
| selections unmapped by the two-vocabulary normaliser | **0** |

### A2. Selection: "movement" mostly records which matches were REFRESHED

The 400-day prune horizon (2025-07-21) has not yet reached any data in the study
window, so pruning survivorship is **not** yet a factor. The live selection
mechanism is a different and larger one:

| training matches | count | rows moved |
| --- | --- | --- |
| system took a pick | 391 | **16.64%** |
| system took none | 1,135 | **5.41%** |

**A 3× difference, and it is not the market.** Only pick-bearing matches get a
second observation from the closing refresh; the rest are seen once, so
`opening == current` by construction. **Movement in this database is largely a
record of the system's own pick selection.**

### A3. Two anomalies, investigated before being accepted (rule 5)

**First: max M = +1,129%.** Traced to one fixture — match 41242, Viking vs
Start, Eliteserien. Pinnacle held `Draw 7.79 → 7.79`, `Home 12.97 → 1.20`, and
**no Away row at all**. Viking at home are ~1.20; 12.97 is Start's away price
sitting in the Home row. **I initially generalised this into a claim that the
opening snapshot was broadly corrupt, and that was wrong** — checked against the
whole population, opening books are 98.9% complete three-leg with a 6.24% mean
overround and exactly one impossible book in 22,895. Viking/Start is an outlier.

**Second, and decisive: a median move of +37% and a shortened:drifted ratio of
43:1.** No market does that. It splits perfectly by provider:

| source | n | median M | shortened : drifted |
| --- | --- | --- | --- |
| API-Football books (10) | 2,709 | **+37.3%** | **2,647 : 62** |
| The Odds API books (26) | 1,418 | **−0.61%** | 661 : 757 |

The API-Football pairs carry the **already-documented two-way / draw-excluded
"Home/Away" trap** in `odds_value`: a shorter two-way price overwrites a genuine
1X2 opening, manufacturing a fake 37% "shortening". Ledger-recorded since Stage
13 and gated for the CLV series per `(match, book)` — **but never excluded from
the odds table, so it silently poisons any movement study that does not split by
provider.** The Odds API rows, by contrast, look like a real market: near
symmetric, mean +0.67%.

**The usable substrate is The Odds API rows only.**

### What the substrate can and cannot answer

It can answer **one cross-sectional question**: does a price that is an outlier
against its cross-book peers revert? It cannot answer anything requiring a price
*path*, because there are **exactly two observations per key and never three**,
and it cannot date the first of them.

## PART B — hypotheses

| | hypothesis | status |
| --- | --- | --- |
| **H1** | momentum | **UNTESTABLE** — needs t0→t1 to predict t1→t2; only two observations per key exist |
| **H2** | cross-book disagreement | **TESTED** |
| **H3** | injuries | **UNTESTABLE** — **34 injury rows in the entire database**, 10 teams, all dated 2026-08-17/18: zero in training, and what exists lies inside the sealed window |
| **H4** | elapsed time | **UNTESTABLE** — no opening timestamp; `matches.created_at` is unusable as a proxy because **53.5% of match rows were created AFTER their own kickoff** (mean +14 days), being backfill stamps |

**H3's 34 rows deserve their own line.** Stage 14 established that injuries reach
only the Claude review prompt and never the model. The reason this stage could
not test whether injury news moves lines is that **the system does not retain
injury history at all** — it holds a two-day snapshot.

### B2. The null

Clean Odds API population, training: 89.3% of rows unchanged, median 0.000%.
Distribution near-symmetric once the API-Football contamination is removed.

**M is an odds RATIO and is not comparable across price levels** — a longshot
moving 15.0 → 12.0 is +25% in M but ~1.7pp of implied probability. This is a
registered consequence of defining M in CLV units for comparability, and it is
why the raw pooled mean (+2.69%) is meaningless and the banded/provider-split
figures are the ones reported.

### B3. H2, training period, pre-registered quintiles

| quintile | above consensus | mean M | 95% CI |
| --- | --- | --- | --- |
| 1 | −6.04% | +0.724% | [+0.331, +1.118] |
| 4 | +1.70% | +0.465% | [+0.191, +0.739] |
| **5** | **+6.56%** | **+1.313%** | **[+0.695, +1.932]** |

Direction as pre-registered — an outlier-high price shortens. **The CI lower
bound is +0.695%, far below the +1.85% the decision rule requires.**

## PART D — the held-out period

Opened only after the above. Identical query, window swapped, no additions.

| quintile | above consensus | mean M | 95% CI | fixtures |
| --- | --- | --- | --- | --- |
| 1 | −5.90% | +0.166% | [−0.106, +0.438] | 400 |
| 4 | +1.78% | +0.429% | [+0.197, +0.660] | 402 |
| **5** | **+6.78%** | **+0.705%** | **[+0.412, +0.999]** | 402 |

**Differences from training, stated as required:**

1. **The direction replicates**, and more cleanly — held-out is close to
   monotone across quintiles, where training was U-shaped (Q1 +0.724% was the
   second-highest cell, which the mean-reversion story does not explain).
2. **The magnitude halves: +1.313% → +0.705%**, and the held-out interval
   **excludes the training point estimate.** The training figure was inflated.
3. **The conclusion does not change**, because both fail the same pre-registered
   threshold — and the held-out result fails it *decisively*: its **upper** bound
   (+0.999%) is below break-even (+1.85%). The effect is not merely unproven; it
   is excluded from being large enough.

**Verdict against the pre-registered rule: NO SIGNAL.**

### Not tested because it was not pre-registered

Listed for a future stage, not run:

- whether the *magnitude* of cross-book dispersion (rather than a selection's
  signed distance from consensus) predicts |M| — the U-shaped training result
  hints at it and it is a different hypothesis
- per-league and per-price-band variation in the H2 effect
- whether the effect differs between exchange and sportsbook closes
- H2 on `totals` rather than `1X2`

## PART C — economic size

| | value |
| --- | --- |
| held-out effect (Q5) | **+0.705%** [+0.412, +0.999] |
| best-line break-even | +1.85% |
| minimum decision-relevant | +2.00% |
| comfortable | +4.00% |

**Short of break-even by 1.15pp, and its entire confidence interval sits below
the threshold.** Capacity is not the constraint — Q5 spans 402 fixtures in 55
days (~220/month) — the size is. A +0.7% effect against a 1.85% cost is the
"smaller version of the same nothing" this stage was told to watch for.

**One observation worth carrying forward.** The system's own realised MODEL CLV
is **−0.587%** (Stage 16). Taking the outlier-high price instead yields
**+0.705%** — a ~1.3pp improvement that is pure execution, not prediction. **It
is still not enough to clear the vig**, so it changes nothing about whether to
bet real money; but it is the only concrete improvement this stage identified,
and it costs no model change.

## PART E — recommendation

**STAGE 17 — NO SIGNAL, ROUTE IDENTIFIED**, with the route stated honestly
rather than optimistically.

**There is no signal in this data.** The one hypothesis the substrate supported
produced a real, out-of-sample-replicating effect of +0.705% against a +1.85%
cost.

**The route is not a better model — it is retaining data already being
fetched and thrown away.** Three of four hypotheses failed for want of storage,
not for want of evidence:

| what is needed | what it enables | cost |
| --- | --- | --- |
| timestamped price snapshots instead of overwrite-in-place | H1 (momentum), H4 (time) — a price *path* rather than two endpoints | **zero API credits** — the prices are already fetched; the row is overwritten |
| retained injury history rather than a live snapshot | H3 | zero — already fetched for the review prompt |
| an `opening_odds` timestamp | dates the first observation | one column |

**Price momentum is the most-documented effect in this space and it has never
been tested here — not tested and absent, but untestable.** That is a genuine
reason to think a signal could exist that this stage could not reach.

**The honest counterweight, which belongs in the same paragraph.** The one
market-microstructure effect that *could* be measured was real, replicated, and
came to **less than half of break-even**. That is evidence — not proof — that
effects of this kind here are sub-vig. Anyone acting on the route above should
expect to find something small and real and insufficient, because that is
exactly what was found in the one place it was possible to look.

**Nothing here disturbs the four independent confirmations that the model adds
no information over the price.** This stage did not test the model; it tested the
market, and found the market's own inefficiencies too small to pay the vig.

---

**STAGE 17 — NO SIGNAL, ROUTE IDENTIFIED.**

Three of four pre-registered hypotheses were untestable on this substrate. The
fourth gave +0.705% held-out against a +1.85% cost, with its whole interval
below the threshold. The route to the untestable three is storage, not
modelling, and costs no API credits — but the one effect that could be measured
argues that what lies down it will also be too small.

---

# DAILY CI AUDIT — THE BACKLOG, CLEARED (audited 2026-08-25)

**343 runs, 2026-02-24 → 2026-08-25**, every run in the repository that had no
ledger verdict. Rows follow this section.

| verdict | runs |
| --- | --- |
| CLEAN | 276 |
| **DEGRADED** | **60** |
| **BROKEN** | **7** |

By workflow: daily-picks 247, closing-lines 84, paper-report 12.

## The number this audit exists to produce

> **63 of the 67 non-CLEAN runs were reported `success` by GitHub.**

Stage 13 Part A found 1 BROKEN and 9 DEGRADED green runs in a three-day window
and called it a window, not a census. It was right to. **Across six months the
same condition holds at 63 runs.** Every core step in `daily-picks.yml` carries
`continue-on-error: true`, so green means the runner survived, not that the
pipeline worked. DEL-2 narrowed this for pick-generation failures only; the
other 62 remain green by design and by decision.

## Tool validation before use

`scripts/ci_audit.py` was re-run over 2026-08-11 → 08-13 first and **reproduced
the manual pass exactly: 27 runs, 1 BROKEN, 9 DEGRADED, 17 CLEAN, zero
disagreements.** This mattered because Stage 15 changed `theodds_scraper.py`'s
`result=ok|no_rows` emission, which this script greps.

## What the counts mean — the half the script cannot do

### `injuries = 0` is not 24 incidents. It is one collapse and one recovery.

24 DEGRADED runs carry it, which reads as a chronic fault. Measured across 187
cached logs that mention injuries:

| month | runs | runs with injuries > 0 | max seen |
| --- | --- | --- | --- |
| 2026-03 | 22 | 9 | 128 |
| 2026-04 | 16 | 10 | 136 |
| 2026-05 | 23 | 18 | 198 |
| **2026-06** | **47** | **5** | **7** |
| 2026-07 | 53 | 23 | 70 |
| 2026-08 | 26 | 12 | 98 |

**June is a step change, not a run of bad luck** — 47 runs, 5 with any injuries
at all, and a maximum of 7 where the surrounding months reach 128–198. The
integration substantially stopped in June and partially recovered in July.
*(Counts extracted by pattern from cached logs; indicative, not authoritative.)*

**Recorded, not pursued**, per the standing rule.

### The two `injuries = 0` runs marked BROKEN are not a classification
inconsistency

`22796448665` and `29327973062` carry 3 and 2 traceback matches respectively and
escalate on those. The injuries finding merely displaced the traceback finding in
the truncated findings column. **Checked before being reported as a tool
defect.**

### Review coverage — checked against the database, not the logs

`saved_picks.review_action` (**not** `disposition` — see below), picks from
2026-08-14:

| date | picks | reviewed | **no decision** | CHANGE | actually changed |
| --- | --- | --- | --- | --- | --- |
| 08-14 | 26 | 25 | 1 | 10 | **9** |
| 08-15 | 57 | 39 | **18** | 14 | 14 |
| 08-16 | 46 | 32 | **14** | 15 | 15 |
| 08-17 | 13 | 12 | 1 | 11 | **10** |
| 08-18 → 08-24 | 59 | 59 | 0 | — | — |
| 08-25 | 1 | 0 | 1 | 0 | 0 |

**35 picks carry no review decision at all**, concentrated on 08-15 and 08-16
(18 of 57 and 14 of 46). The documented behaviour is that Claude reviews EVERY
pick. **And on 08-14 and 08-17, one CHANGE decision each was recorded but never
applied** — `review_action = CHANGE` while `model_market`/`model_selection` still
equal the final pick. That is the discarded-decision signature the script counts
from logs, confirmed independently in the data.

### A vocabulary trap, caught before it became a false finding

I first measured review coverage from `saved_picks.disposition` and found **1,336
of 1,338 picks NULL**, with the only non-null value being `consolidated`. That
reads as "the KEEP/CHANGE record does not exist". **It was the wrong column** —
`disposition` is the Stage 13 supersession marker; `review_action` holds the
decision. Caught by the standing rule that an implausible count is a vocabulary
bug until proven otherwise. **Third instance in the data layer** after the two
market taxonomies and `Home`/`Home Win`.

### Zeros checked against fixture availability, not assumed

| date | picks | fixtures available | reading |
| --- | --- | --- | --- |
| 2026-08-19 | 4 | 56 | API-Football suspended 10:10:28 UTC — expected |
| 2026-08-23 | 11 | 110 | **low; not explained by card size** |
| 2026-08-25 | 1 | 24 | **low; not explained by card size** |
| 2026-08-18 | 3 | 4 | genuinely thin card — CLEAN |
| 2026-08-20 | 1 | 5 | thin card — CLEAN |

**`pick_observations` = 2 × picks saved on all 12 days**, exactly. The dual
MODEL/FINAL attribution is intact end to end.

The 08-23 and 08-25 ratios are **recorded, not pursued**. 08-23 is the `s5.3`
cohort boundary, where a one-pick-per-match cap first applies; the cap explains
part of a drop from 110 fixtures but was not verified to explain all of it.

## Ledger rows

Notes carry counts, not adjectives. `—` in *steps failed* means GitHub reported
no failed step, which for this repository is the normal state of a broken run.

| run_id | workflow | started (UTC) | conclusion | steps failed | audited on | verdict | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 22356085285 | daily-picks | 2026-02-24 14:51 | success | — | 2026-08-25 | CLEAN | counts agree |
| 22390583197 | daily-picks | 2026-02-25 09:26 | success | — | 2026-08-25 | CLEAN | counts agree |
| 22435400466 | daily-picks | 2026-02-26 09:11 | success | — | 2026-08-25 | CLEAN | counts agree |
| 22441944420 | daily-picks | 2026-02-26 12:24 | success | — | 2026-08-25 | CLEAN | counts agree |
| 22479870379 | daily-picks | 2026-02-27 09:06 | success | — | 2026-08-25 | CLEAN | counts agree |
| 22488191886 | daily-picks | 2026-02-27 13:30 | success | — | 2026-08-25 | CLEAN | counts agree |
| 22517485924 | daily-picks | 2026-02-28 08:51 | failure | — | 2026-08-25 | CLEAN | counts agree |
| 22517577820 | daily-picks | 2026-02-28 08:57 | cancelled | — | 2026-08-25 | CLEAN | counts agree |
| 22523678012 | daily-picks | 2026-02-28 15:36 | success | — | 2026-08-25 | CLEAN | counts agree |
| 22538743687 | daily-picks | 2026-03-01 07:40 | cancelled | — | 2026-08-25 | CLEAN | counts agree |
| 22565905564 | daily-picks | 2026-03-02 07:31 | success | — | 2026-08-25 | **BROKEN** | 1 **traceback(s)** reached the log |
| 22612714122 | daily-picks | 2026-03-03 07:24 | success | — | 2026-08-25 | CLEAN | counts agree |
| 22634136633 | daily-picks | 2026-03-03 17:07 | cancelled | — | 2026-08-25 | CLEAN | counts agree |
| 22635284943 | daily-picks | 2026-03-03 17:37 | cancelled | — | 2026-08-25 | CLEAN | counts agree |
| 22635667867 | daily-picks | 2026-03-03 17:46 | cancelled | — | 2026-08-25 | CLEAN | counts agree |
| 22639036536 | daily-picks | 2026-03-03 19:24 | cancelled | — | 2026-08-25 | CLEAN | counts agree |
| 22641450764 | daily-picks | 2026-03-03 20:31 | cancelled | — | 2026-08-25 | CLEAN | counts agree |
| 22658999875 | daily-picks | 2026-03-04 07:11 | cancelled | — | 2026-08-25 | CLEAN | counts agree |
| 22662151730 | daily-picks | 2026-03-04 08:56 | cancelled | — | 2026-08-25 | CLEAN | counts agree |
| 22666672206 | daily-picks | 2026-03-04 11:06 | failure | — | 2026-08-25 | **BROKEN** | 2 **traceback(s)** reached the log |
| 22668603145 | daily-picks | 2026-03-04 12:05 | success | — | 2026-08-25 | CLEAN | counts agree |
| 22707041964 | daily-picks | 2026-03-05 07:24 | success | — | 2026-08-25 | CLEAN | counts agree |
| 22753178761 | daily-picks | 2026-03-06 07:12 | success | — | 2026-08-25 | **DEGRADED** | injuries **0** where the previous 7 runs produced some |
| 22756001792 | daily-picks | 2026-03-06 08:46 | cancelled | — | 2026-08-25 | CLEAN | counts agree |
| 22794362759 | daily-picks | 2026-03-07 07:03 | cancelled | — | 2026-08-25 | CLEAN | counts agree |
| 22796448665 | daily-picks | 2026-03-07 09:26 | failure | — | 2026-08-25 | **BROKEN** | injuries **0** where the previous 7 runs produced some |
| 22816159463 | daily-picks | 2026-03-08 07:06 | cancelled | — | 2026-08-25 | CLEAN | counts agree |
| 22820293045 | daily-picks | 2026-03-08 11:36 | cancelled | — | 2026-08-25 | CLEAN | counts agree |
| 22843020196 | daily-picks | 2026-03-09 07:33 | success | — | 2026-08-25 | CLEAN | counts agree |
| 22891339475 | daily-picks | 2026-03-10 07:13 | cancelled | — | 2026-08-25 | CLEAN | counts agree |
| 22893212883 | daily-picks | 2026-03-10 08:14 | success | — | 2026-08-25 | CLEAN | counts agree |
| 22941560722 | daily-picks | 2026-03-11 07:26 | success | — | 2026-08-25 | CLEAN | counts agree |
| 22991168777 | daily-picks | 2026-03-12 07:28 | success | — | 2026-08-25 | CLEAN | counts agree |
| 23040727447 | daily-picks | 2026-03-13 07:27 | success | — | 2026-08-25 | CLEAN | counts agree |
| 23083041755 | daily-picks | 2026-03-14 07:11 | success | — | 2026-08-25 | CLEAN | counts agree |
| 23105834679 | daily-picks | 2026-03-15 07:27 | success | — | 2026-08-25 | CLEAN | counts agree |
| 23133370856 | daily-picks | 2026-03-16 07:53 | success | — | 2026-08-25 | CLEAN | counts agree |
| 23183449389 | daily-picks | 2026-03-17 07:35 | success | — | 2026-08-25 | CLEAN | counts agree |
| 23234008813 | daily-picks | 2026-03-18 07:35 | success | — | 2026-08-25 | CLEAN | counts agree |
| 23284467373 | daily-picks | 2026-03-19 07:30 | success | — | 2026-08-25 | CLEAN | counts agree |
| 23333239155 | daily-picks | 2026-03-20 07:28 | success | — | 2026-08-25 | CLEAN | counts agree |
| 23374494765 | daily-picks | 2026-03-21 07:07 | failure | — | 2026-08-25 | CLEAN | counts agree |
| 23398025277 | daily-picks | 2026-03-22 07:12 | failure | — | 2026-08-25 | CLEAN | counts agree |
| 23426635615 | daily-picks | 2026-03-23 07:43 | failure | — | 2026-08-25 | CLEAN | counts agree |
| 23478251165 | daily-picks | 2026-03-24 07:35 | failure | — | 2026-08-25 | CLEAN | counts agree |
| 23530013971 | daily-picks | 2026-03-25 07:34 | failure | — | 2026-08-25 | CLEAN | counts agree |
| 23583046721 | daily-picks | 2026-03-26 07:41 | failure | — | 2026-08-25 | CLEAN | counts agree |
| 23636255597 | daily-picks | 2026-03-27 07:40 | failure | — | 2026-08-25 | CLEAN | counts agree |
| 23680318755 | daily-picks | 2026-03-28 07:28 | failure | — | 2026-08-25 | CLEAN | counts agree |
| 23704197572 | daily-picks | 2026-03-29 07:33 | failure | — | 2026-08-25 | CLEAN | counts agree |
| 23734505286 | daily-picks | 2026-03-30 08:08 | failure | — | 2026-08-25 | CLEAN | counts agree |
| 23786490086 | daily-picks | 2026-03-31 07:50 | failure | — | 2026-08-25 | CLEAN | counts agree |
| 23838336849 | daily-picks | 2026-04-01 08:00 | success | — | 2026-08-25 | CLEAN | counts agree |
| 23889918843 | daily-picks | 2026-04-02 07:46 | success | — | 2026-08-25 | CLEAN | counts agree |
| 23938461557 | daily-picks | 2026-04-03 07:38 | cancelled | — | 2026-08-25 | CLEAN | counts agree |
| 23938580915 | daily-picks | 2026-04-03 07:42 | success | — | 2026-08-25 | CLEAN | counts agree |
| 23941450983 | daily-picks | 2026-04-03 09:25 | success | — | 2026-08-25 | CLEAN | counts agree |
| 23974282950 | daily-picks | 2026-04-04 07:32 | success | — | 2026-08-25 | CLEAN | counts agree |
| 23996980830 | daily-picks | 2026-04-05 07:36 | success | — | 2026-08-25 | CLEAN | counts agree |
| 24024282796 | daily-picks | 2026-04-06 08:09 | failure | — | 2026-08-25 | CLEAN | counts agree |
| 24070722185 | daily-picks | 2026-04-07 07:55 | failure | — | 2026-08-25 | **BROKEN** | 1 **traceback(s)** reached the log |
| 24124518344 | daily-picks | 2026-04-08 07:58 | success | — | 2026-08-25 | CLEAN | counts agree |
| 24179358039 | daily-picks | 2026-04-09 08:00 | success | — | 2026-08-25 | CLEAN | counts agree |
| 24233149548 | daily-picks | 2026-04-10 08:05 | success | — | 2026-08-25 | CLEAN | counts agree |
| 24277721584 | daily-picks | 2026-04-11 07:32 | success | — | 2026-08-25 | CLEAN | counts agree |
| 24301797686 | daily-picks | 2026-04-12 07:48 | success | — | 2026-08-25 | CLEAN | counts agree |
| 24333465501 | daily-picks | 2026-04-13 08:27 | success | — | 2026-08-25 | CLEAN | counts agree |
| 24388243534 | daily-picks | 2026-04-14 08:10 | success | — | 2026-08-25 | CLEAN | counts agree |
| 24443729069 | daily-picks | 2026-04-15 08:12 | success | — | 2026-08-25 | CLEAN | counts agree |
| 24496880881 | daily-picks | 2026-04-16 07:04 | failure | — | 2026-08-25 | CLEAN | counts agree |
| 24499259777 | daily-picks | 2026-04-16 08:06 | failure | — | 2026-08-25 | CLEAN | counts agree |
| 24499477149 | daily-picks | 2026-04-16 08:11 | success | — | 2026-08-25 | CLEAN | counts agree |
| 24553301763 | daily-picks | 2026-04-17 07:26 | success | — | 2026-08-25 | CLEAN | counts agree |
| 24599391846 | daily-picks | 2026-04-18 07:01 | success | — | 2026-08-25 | CLEAN | counts agree |
| 24614508173 | daily-picks | 2026-04-18 21:41 | failure | — | 2026-08-25 | CLEAN | counts agree |
| 24623281052 | daily-picks | 2026-04-19 06:56 | failure | — | 2026-08-25 | CLEAN | counts agree |
| 24623910911 | daily-picks | 2026-04-19 07:34 | success | — | 2026-08-25 | CLEAN | counts agree |
| 24654148913 | daily-picks | 2026-04-20 07:32 | success | — | 2026-08-25 | CLEAN | counts agree |
| 24655009369 | daily-picks | 2026-04-20 07:54 | success | — | 2026-08-25 | CLEAN | counts agree |
| 24709639285 | daily-picks | 2026-04-21 07:24 | success | — | 2026-08-25 | CLEAN | counts agree |
| 24765713700 | daily-picks | 2026-04-22 07:22 | success | — | 2026-08-25 | CLEAN | counts agree |
| 24822553699 | daily-picks | 2026-04-23 07:24 | success | — | 2026-08-25 | CLEAN | counts agree |
| 24878564810 | daily-picks | 2026-04-24 07:50 | success | — | 2026-08-25 | CLEAN | counts agree |
| 24925306690 | daily-picks | 2026-04-25 07:04 | failure | — | 2026-08-25 | **BROKEN** | 6 **traceback(s)** reached the log |
| 24935579884 | daily-picks | 2026-04-25 16:39 | success | — | 2026-08-25 | CLEAN | counts agree |
| 24950740512 | daily-picks | 2026-04-26 07:03 | cancelled | — | 2026-08-25 | CLEAN | counts agree |
| 24951005409 | daily-picks | 2026-04-26 07:18 | success | — | 2026-08-25 | CLEAN | counts agree |
| 24955399010 | daily-picks | 2026-04-26 11:22 | cancelled | — | 2026-08-25 | CLEAN | counts agree |
| 24955692121 | daily-picks | 2026-04-26 11:38 | success | — | 2026-08-25 | CLEAN | counts agree |
| 24983785093 | daily-picks | 2026-04-27 08:08 | success | — | 2026-08-25 | CLEAN | counts agree |
| 25041553428 | daily-picks | 2026-04-28 08:10 | success | — | 2026-08-25 | CLEAN | counts agree |
| 25097679510 | daily-picks | 2026-04-29 08:03 | success | — | 2026-08-25 | CLEAN | counts agree |
| 25154580275 | daily-picks | 2026-04-30 08:07 | success | — | 2026-08-25 | CLEAN | counts agree |
| 25209335617 | daily-picks | 2026-05-01 09:16 | success | — | 2026-08-25 | CLEAN | counts agree |
| 25248259491 | daily-picks | 2026-05-02 08:50 | success | — | 2026-08-25 | CLEAN | counts agree |
| 25275009236 | daily-picks | 2026-05-03 09:05 | success | — | 2026-08-25 | CLEAN | counts agree |
| 25311888227 | daily-picks | 2026-05-04 09:38 | success | — | 2026-08-25 | CLEAN | counts agree |
| 25368488651 | daily-picks | 2026-05-05 09:28 | success | — | 2026-08-25 | CLEAN | counts agree |
| 25427906402 | daily-picks | 2026-05-06 09:42 | success | — | 2026-08-25 | CLEAN | counts agree |
| 25489125745 | daily-picks | 2026-05-07 10:00 | success | — | 2026-08-25 | CLEAN | counts agree |
| 25546708707 | daily-picks | 2026-05-08 08:56 | success | — | 2026-08-25 | CLEAN | counts agree |
| 25597139519 | daily-picks | 2026-05-09 09:00 | success | — | 2026-08-25 | **BROKEN** | 2 **traceback(s)** reached the log |
| 25624924915 | daily-picks | 2026-05-10 09:13 | success | — | 2026-08-25 | CLEAN | counts agree |
| 25665491812 | daily-picks | 2026-05-11 10:46 | success | — | 2026-08-25 | **DEGRADED** | 1 alert(s) **built but NOT delivered** |
| 25727593132 | daily-picks | 2026-05-12 10:05 | success | — | 2026-08-25 | CLEAN | counts agree |
| 25792324938 | daily-picks | 2026-05-13 10:05 | success | — | 2026-08-25 | CLEAN | counts agree |
| 25853766716 | daily-picks | 2026-05-14 09:55 | success | — | 2026-08-25 | CLEAN | counts agree |
| 25912019463 | daily-picks | 2026-05-15 10:02 | success | — | 2026-08-25 | CLEAN | counts agree |
| 25958160130 | daily-picks | 2026-05-16 09:12 | success | — | 2026-08-25 | CLEAN | counts agree |
| 25986929640 | daily-picks | 2026-05-17 09:19 | success | — | 2026-08-25 | CLEAN | counts agree |
| 26030780062 | daily-picks | 2026-05-18 11:30 | success | — | 2026-08-25 | CLEAN | counts agree |
| 26092333988 | daily-picks | 2026-05-19 10:47 | success | — | 2026-08-25 | CLEAN | counts agree |
| 26157007041 | daily-picks | 2026-05-20 10:34 | success | — | 2026-08-25 | CLEAN | counts agree |
| 26221514428 | daily-picks | 2026-05-21 10:52 | success | — | 2026-08-25 | CLEAN | counts agree |
| 26282758907 | daily-picks | 2026-05-22 10:33 | success | — | 2026-08-25 | CLEAN | counts agree |
| 26329139519 | daily-picks | 2026-05-23 09:21 | success | — | 2026-08-25 | CLEAN | counts agree |
| 26357587305 | daily-picks | 2026-05-24 09:28 | success | — | 2026-08-25 | CLEAN | counts agree |
| 26398420094 | daily-picks | 2026-05-25 11:35 | success | — | 2026-08-25 | CLEAN | counts agree |
| 26507319230 | daily-picks | 2026-05-27 11:04 | success | — | 2026-08-25 | **DEGRADED** | injuries **0** where the previous 7 runs produced some |
| 26570901520 | daily-picks | 2026-05-28 11:05 | success | — | 2026-08-25 | **DEGRADED** | Odds API **0 rows** where the previous 7 runs produced some |
| 26633321514 | daily-picks | 2026-05-29 10:56 | success | — | 2026-08-25 | CLEAN | counts agree |
| 26680425750 | daily-picks | 2026-05-30 09:30 | success | — | 2026-08-25 | CLEAN | counts agree |
| 26709558467 | daily-picks | 2026-05-31 10:00 | success | — | 2026-08-25 | CLEAN | counts agree |
| 26757106548 | daily-picks | 2026-06-01 13:13 | success | — | 2026-08-25 | **DEGRADED** | injuries **0** where the previous 7 runs produced some |
| 26772039386 | daily-picks | 2026-06-01 17:51 | failure | — | 2026-08-25 | CLEAN | counts agree |
| 27293639301 | daily-picks | 2026-06-10 17:25 | success | — | 2026-08-25 | CLEAN | counts agree |
| 27338242478 | daily-picks | 2026-06-11 09:44 | success | — | 2026-08-25 | **DEGRADED** | injuries **0** where the previous 7 runs produced some |
| 27339814324 | daily-picks | 2026-06-11 10:14 | success | — | 2026-08-25 | **DEGRADED** | injuries **0** where the previous 7 runs produced some |
| 27361989883 | daily-picks | 2026-06-11 16:31 | success | — | 2026-08-25 | **DEGRADED** | injuries **0** where the previous 7 runs produced some |
| 27416379675 | daily-picks | 2026-06-12 12:43 | success | — | 2026-08-25 | **DEGRADED** | injuries **0** where the previous 7 runs produced some |
| 27421481322 | daily-picks | 2026-06-12 14:17 | success | — | 2026-08-25 | CLEAN | counts agree |
| 27426338194 | daily-picks | 2026-06-12 15:42 | success | — | 2026-08-25 | CLEAN | counts agree |
| 27465524983 | daily-picks | 2026-06-13 11:30 | success | — | 2026-08-25 | **DEGRADED** | 1 briefing decision(s) **DISCARDED** — review ran, verdict not applied |
| 27469521509 | daily-picks | 2026-06-13 14:29 | success | — | 2026-08-25 | CLEAN | counts agree |
| 27497748287 | daily-picks | 2026-06-14 11:43 | success | — | 2026-08-25 | CLEAN | counts agree |
| 27501958069 | daily-picks | 2026-06-14 14:31 | success | — | 2026-08-25 | CLEAN | counts agree |
| 27555928145 | daily-picks | 2026-06-15 15:08 | success | — | 2026-08-25 | CLEAN | counts agree |
| 27557379483 | daily-picks | 2026-06-15 15:31 | success | — | 2026-08-25 | CLEAN | counts agree |
| 27564105033 | daily-picks | 2026-06-15 17:27 | success | — | 2026-08-25 | CLEAN | counts agree |
| 27616902878 | daily-picks | 2026-06-16 12:17 | success | — | 2026-08-25 | CLEAN | counts agree |
| 27625103306 | daily-picks | 2026-06-16 14:32 | success | — | 2026-08-25 | CLEAN | counts agree |
| 27635669856 | daily-picks | 2026-06-16 17:23 | success | — | 2026-08-25 | CLEAN | counts agree |
| 27690732492 | daily-picks | 2026-06-17 12:59 | success | — | 2026-08-25 | **DEGRADED** | 1 briefing decision(s) **DISCARDED** — review ran, verdict not applied |
| 27690859911 | daily-picks | 2026-06-17 13:01 | success | — | 2026-08-25 | CLEAN | counts agree |
| 27702775527 | daily-picks | 2026-06-17 16:07 | success | — | 2026-08-25 | CLEAN | counts agree |
| 27760010043 | daily-picks | 2026-06-18 12:38 | success | — | 2026-08-25 | CLEAN | counts agree |
| 27772060952 | daily-picks | 2026-06-18 15:54 | success | — | 2026-08-25 | CLEAN | counts agree |
| 27827223178 | daily-picks | 2026-06-19 13:00 | success | — | 2026-08-25 | CLEAN | counts agree |
| 27835285929 | daily-picks | 2026-06-19 15:44 | success | — | 2026-08-25 | CLEAN | counts agree |
| 27869868638 | daily-picks | 2026-06-20 11:31 | success | — | 2026-08-25 | **DEGRADED** | 2 briefing decision(s) **DISCARDED** — review ran, verdict not applied |
| 27874168621 | daily-picks | 2026-06-20 14:33 | success | — | 2026-08-25 | CLEAN | counts agree |
| 27903635203 | daily-picks | 2026-06-21 12:00 | success | — | 2026-08-25 | **DEGRADED** | 1 briefing decision(s) **DISCARDED** — review ran, verdict not applied |
| 27907566451 | daily-picks | 2026-06-21 14:34 | success | — | 2026-08-25 | CLEAN | counts agree |
| 27962743457 | daily-picks | 2026-06-22 15:07 | success | — | 2026-08-25 | CLEAN | counts agree |
| 27970588764 | daily-picks | 2026-06-22 17:13 | success | — | 2026-08-25 | CLEAN | counts agree |
| 28025876176 | daily-picks | 2026-06-23 12:23 | success | — | 2026-08-25 | **DEGRADED** | 2 briefing decision(s) **DISCARDED** — review ran, verdict not applied |
| 28036906306 | daily-picks | 2026-06-23 15:26 | success | — | 2026-08-25 | CLEAN | counts agree |
| 28097312462 | daily-picks | 2026-06-24 12:07 | success | — | 2026-08-25 | **DEGRADED** | 3 briefing decision(s) **DISCARDED** — review ran, verdict not applied |
| 28108684871 | daily-picks | 2026-06-24 15:10 | success | — | 2026-08-25 | **DEGRADED** | injuries **0** where the previous 7 runs produced some |
| 28168719073 | daily-picks | 2026-06-25 12:03 | success | — | 2026-08-25 | **DEGRADED** | 2 briefing decision(s) **DISCARDED** — review ran, verdict not applied |
| 28180659344 | daily-picks | 2026-06-25 15:18 | success | — | 2026-08-25 | **DEGRADED** | injuries **0** where the previous 7 runs produced some |
| 28236632011 | daily-picks | 2026-06-26 12:00 | success | — | 2026-08-25 | **DEGRADED** | 3 briefing decision(s) **DISCARDED** — review ran, verdict not applied |
| 28245636814 | daily-picks | 2026-06-26 14:48 | success | — | 2026-08-25 | **DEGRADED** | injuries **0** where the previous 7 runs produced some |
| 28287518114 | daily-picks | 2026-06-27 11:13 | success | — | 2026-08-25 | **DEGRADED** | injuries **0** where the previous 7 runs produced some |
| 28291554441 | daily-picks | 2026-06-27 14:07 | success | — | 2026-08-25 | **DEGRADED** | injuries **0** where the previous 7 runs produced some |
| 28320584590 | daily-picks | 2026-06-28 11:22 | success | — | 2026-08-25 | CLEAN | counts agree |
| 28325087751 | daily-picks | 2026-06-28 14:17 | success | — | 2026-08-25 | **DEGRADED** | injuries **0** where the previous 7 runs produced some |
| 28377137724 | daily-picks | 2026-06-29 13:52 | success | — | 2026-08-25 | CLEAN | counts agree |
| 28386237573 | daily-picks | 2026-06-29 16:13 | success | — | 2026-08-25 | **DEGRADED** | injuries **0** where the previous 7 runs produced some |
| 28442596088 | daily-picks | 2026-06-30 11:59 | success | — | 2026-08-25 | **DEGRADED** | injuries **0** where the previous 7 runs produced some |
| 28453089347 | daily-picks | 2026-06-30 14:44 | success | — | 2026-08-25 | **DEGRADED** | injuries **0** where the previous 7 runs produced some |
| 28517087943 | daily-picks | 2026-07-01 12:21 | success | — | 2026-08-25 | CLEAN | counts agree |
| 28527824544 | daily-picks | 2026-07-01 15:13 | success | — | 2026-08-25 | **DEGRADED** | injuries **0** where the previous 7 runs produced some |
| 28588019500 | daily-picks | 2026-07-02 11:55 | success | — | 2026-08-25 | **DEGRADED** | injuries **0** where the previous 7 runs produced some |
| 28597984718 | daily-picks | 2026-07-02 14:30 | success | — | 2026-08-25 | **DEGRADED** | injuries **0** where the previous 7 runs produced some |
| 28658837423 | daily-picks | 2026-07-03 11:53 | success | — | 2026-08-25 | CLEAN | counts agree |
| 28667064129 | daily-picks | 2026-07-03 14:33 | success | — | 2026-08-25 | **DEGRADED** | injuries **0** where the previous 7 runs produced some |
| 28704293790 | daily-picks | 2026-07-04 11:08 | success | — | 2026-08-25 | CLEAN | counts agree |
| 28708549879 | daily-picks | 2026-07-04 14:00 | success | — | 2026-08-25 | **DEGRADED** | injuries **0** where the previous 7 runs produced some |
| 28738980918 | daily-picks | 2026-07-05 11:18 | success | — | 2026-08-25 | CLEAN | counts agree |
| 28743543697 | daily-picks | 2026-07-05 14:11 | success | — | 2026-08-25 | **DEGRADED** | injuries **0** where the previous 7 runs produced some |
| 28793732702 | daily-picks | 2026-07-06 13:06 | success | — | 2026-08-25 | CLEAN | counts agree |
| 28804784525 | daily-picks | 2026-07-06 15:55 | success | — | 2026-08-25 | **DEGRADED** | injuries **0** where the previous 7 runs produced some |
| 28865408266 | daily-picks | 2026-07-07 12:16 | success | — | 2026-08-25 | CLEAN | counts agree |
| 28877446676 | daily-picks | 2026-07-07 15:17 | success | — | 2026-08-25 | CLEAN | counts agree |
| 28939008649 | daily-picks | 2026-07-08 11:26 | success | — | 2026-08-25 | **DEGRADED** | Odds API **0 rows** where the previous 7 runs produced some |
| 28951403546 | daily-picks | 2026-07-08 14:42 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29017818302 | daily-picks | 2026-07-09 12:20 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29030116785 | daily-picks | 2026-07-09 15:35 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29091930551 | daily-picks | 2026-07-10 12:14 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29102226093 | daily-picks | 2026-07-10 15:03 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29149814240 | daily-picks | 2026-07-11 10:43 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29155218848 | daily-picks | 2026-07-11 13:56 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29190030196 | daily-picks | 2026-07-12 10:58 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29195364533 | daily-picks | 2026-07-12 13:57 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29249532268 | daily-picks | 2026-07-13 12:21 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29261304347 | daily-picks | 2026-07-13 15:13 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29327973062 | daily-picks | 2026-07-14 11:10 | success | — | 2026-08-25 | **BROKEN** | injuries **0** where the previous 7 runs produced some |
| 29340035297 | daily-picks | 2026-07-14 14:15 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29410851248 | daily-picks | 2026-07-15 11:12 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29422577440 | daily-picks | 2026-07-15 14:13 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29494050000 | daily-picks | 2026-07-16 11:20 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29506513304 | daily-picks | 2026-07-16 14:25 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29575833777 | daily-picks | 2026-07-17 11:08 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29586509564 | daily-picks | 2026-07-17 14:04 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29641445761 | daily-picks | 2026-07-18 10:47 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29646944410 | daily-picks | 2026-07-18 13:52 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29684298183 | daily-picks | 2026-07-19 10:58 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29689821756 | daily-picks | 2026-07-19 13:55 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29740671362 | daily-picks | 2026-07-20 12:33 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29751216855 | daily-picks | 2026-07-20 14:34 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29826106817 | daily-picks | 2026-07-21 11:26 | success | — | 2026-08-25 | **DEGRADED** | Odds API **0 rows** where the previous 7 runs produced some |
| 29839016353 | daily-picks | 2026-07-21 14:25 | success | — | 2026-08-25 | CLEAN | counts agree |
| 29915776892 | daily-picks | 2026-07-22 11:27 | success | — | 2026-08-25 | CLEAN | counts agree |
| 30003266094 | daily-picks | 2026-07-23 11:28 | success | — | 2026-08-25 | **DEGRADED** | Odds API **0 rows** where the previous 7 runs produced some |
| 30089229751 | daily-picks | 2026-07-24 11:20 | success | — | 2026-08-25 | CLEAN | counts agree |
| 30155284880 | daily-picks | 2026-07-25 10:53 | success | — | 2026-08-25 | CLEAN | counts agree |
| 30199482479 | daily-picks | 2026-07-26 11:05 | success | — | 2026-08-25 | **DEGRADED** | API-Football **account suspended** |
| 30204273355 | daily-picks | 2026-07-26 13:33 | success | — | 2026-08-25 | CLEAN | counts agree |
| 30267093066 | daily-picks | 2026-07-27 12:43 | success | — | 2026-08-25 | CLEAN | counts agree |
| 30355496884 | daily-picks | 2026-07-28 11:36 | success | — | 2026-08-25 | **DEGRADED** | Odds API **0 rows** where the previous 7 runs produced some |
| 30448957612 | daily-picks | 2026-07-29 11:47 | success | — | 2026-08-25 | **DEGRADED** | Odds API **0 rows** where the previous 7 runs produced some |
| 30538844155 | daily-picks | 2026-07-30 11:30 | success | — | 2026-08-25 | **DEGRADED** | Odds API **0 rows** where the previous 7 runs produced some |
| 30628079467 | daily-picks | 2026-07-31 11:44 | success | — | 2026-08-25 | CLEAN | counts agree |
| 30696942420 | daily-picks | 2026-08-01 11:03 | success | — | 2026-08-25 | CLEAN | counts agree |
| 30744998217 | daily-picks | 2026-08-02 11:04 | success | — | 2026-08-25 | CLEAN | counts agree |
| 30814751466 | daily-picks | 2026-08-03 12:43 | success | — | 2026-08-25 | CLEAN | counts agree |
| 30906275362 | daily-picks | 2026-08-04 11:47 | success | — | 2026-08-25 | **DEGRADED** | Odds API **0 rows** where the previous 7 runs produced some |
| 31002235726 | daily-picks | 2026-08-05 11:38 | success | — | 2026-08-25 | **DEGRADED** | Odds API **0 rows** where the previous 7 runs produced some |
| 31112133734 | daily-picks | 2026-08-06 14:42 | success | — | 2026-08-25 | **DEGRADED** | Odds API **0 rows** where the previous 7 runs produced some |
| 31170325270 | daily-picks | 2026-08-07 10:30 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31252035375 | daily-picks | 2026-08-08 10:05 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31307570446 | daily-picks | 2026-08-09 10:07 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31380557463 | daily-picks | 2026-08-10 10:46 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31407338578 | closing-lines | 2026-08-10 16:08 | success | — | 2026-08-25 | **DEGRADED** | 2 credits claimed, **0 closing lines captured** |
| 31416982312 | closing-lines | 2026-08-10 18:01 | success | — | 2026-08-25 | **DEGRADED** | 2 credits claimed, **0 closing lines captured** |
| 31426838071 | closing-lines | 2026-08-10 19:59 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31436110648 | closing-lines | 2026-08-10 21:56 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31443804685 | closing-lines | 2026-08-10 23:49 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31792895823 | daily-picks | 2026-08-14 10:36 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31796284378 | paper-report | 2026-08-14 11:27 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31798148700 | closing-lines | 2026-08-14 11:55 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31808922756 | closing-lines | 2026-08-14 14:19 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31817584121 | closing-lines | 2026-08-14 16:04 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31827003802 | closing-lines | 2026-08-14 18:05 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31835392007 | closing-lines | 2026-08-14 19:54 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31843261983 | closing-lines | 2026-08-14 21:37 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31850797893 | closing-lines | 2026-08-14 23:33 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31878183679 | daily-picks | 2026-08-15 09:54 | success | — | 2026-08-25 | **DEGRADED** | 1 alert(s) **built but NOT delivered** |
| 31880839259 | paper-report | 2026-08-15 10:57 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31882302007 | closing-lines | 2026-08-15 11:32 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31887897469 | closing-lines | 2026-08-15 13:41 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31893054820 | closing-lines | 2026-08-15 15:33 | success | — | 2026-08-25 | **DEGRADED** | 4 credits claimed, **0 closing lines captured** |
| 31898638147 | closing-lines | 2026-08-15 17:33 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31904161875 | closing-lines | 2026-08-15 19:31 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31909759683 | closing-lines | 2026-08-15 21:33 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31915024968 | closing-lines | 2026-08-15 23:33 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31940324038 | daily-picks | 2026-08-16 09:55 | success | — | 2026-08-25 | **DEGRADED** | 1 alert(s) **built but NOT delivered** |
| 31943094556 | paper-report | 2026-08-16 10:58 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31944620086 | closing-lines | 2026-08-16 11:32 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31950625630 | closing-lines | 2026-08-16 13:43 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31956047427 | closing-lines | 2026-08-16 15:34 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31961920210 | closing-lines | 2026-08-16 17:32 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31967824022 | closing-lines | 2026-08-16 19:31 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31973813130 | closing-lines | 2026-08-16 21:32 | success | — | 2026-08-25 | CLEAN | counts agree |
| 31979444279 | closing-lines | 2026-08-16 23:32 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32018632341 | daily-picks | 2026-08-17 10:07 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32022968316 | paper-report | 2026-08-17 11:02 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32025815082 | closing-lines | 2026-08-17 11:37 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32036923026 | closing-lines | 2026-08-17 13:51 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32042840589 | closing-lines | 2026-08-17 15:36 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32051530862 | closing-lines | 2026-08-17 17:42 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32061612730 | closing-lines | 2026-08-17 19:40 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32072097546 | closing-lines | 2026-08-17 21:37 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32081051345 | closing-lines | 2026-08-17 23:35 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32124630406 | daily-picks | 2026-08-18 10:01 | success | — | 2026-08-25 | **DEGRADED** | Odds API **0 rows** where the previous 7 runs produced some |
| 32129707558 | paper-report | 2026-08-18 11:02 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32132644028 | closing-lines | 2026-08-18 11:37 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32145189728 | closing-lines | 2026-08-18 13:55 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32156275205 | closing-lines | 2026-08-18 15:44 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32166853804 | closing-lines | 2026-08-18 17:40 | success | — | 2026-08-25 | **DEGRADED** | Odds API **0 rows** where the previous 7 runs produced some |
| 32177739185 | closing-lines | 2026-08-18 19:38 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32188574141 | closing-lines | 2026-08-18 21:36 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32197745800 | closing-lines | 2026-08-18 23:34 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32240633728 | daily-picks | 2026-08-19 10:01 | success | — | 2026-08-25 | **DEGRADED** | injuries **0** where the previous 7 runs produced some |
| 32245546696 | paper-report | 2026-08-19 11:01 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32248451160 | closing-lines | 2026-08-19 11:36 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32260926223 | closing-lines | 2026-08-19 13:55 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32271817127 | closing-lines | 2026-08-19 15:44 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32282678520 | closing-lines | 2026-08-19 17:37 | success | — | 2026-08-25 | **DEGRADED** | 1 league request(s) returned **no_rows** (credits spent, nothing written) |
| 32293929208 | closing-lines | 2026-08-19 19:36 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32304880706 | closing-lines | 2026-08-19 21:38 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32313813438 | closing-lines | 2026-08-19 23:34 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32357064869 | daily-picks | 2026-08-20 10:04 | success | — | 2026-08-25 | **DEGRADED** | API-Football **account suspended** |
| 32361917103 | paper-report | 2026-08-20 11:03 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32364904225 | closing-lines | 2026-08-20 11:39 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32377235665 | closing-lines | 2026-08-20 13:57 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32388201580 | closing-lines | 2026-08-20 15:47 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32399202101 | closing-lines | 2026-08-20 17:43 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32410255699 | closing-lines | 2026-08-20 19:43 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32420683407 | closing-lines | 2026-08-20 21:41 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32429424699 | closing-lines | 2026-08-20 23:37 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32470865675 | daily-picks | 2026-08-21 10:03 | success | — | 2026-08-25 | **DEGRADED** | API-Football **account suspended** |
| 32475389199 | paper-report | 2026-08-21 11:02 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32478166285 | closing-lines | 2026-08-21 11:38 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32489492899 | closing-lines | 2026-08-21 13:56 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32499416119 | closing-lines | 2026-08-21 15:46 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32509601944 | closing-lines | 2026-08-21 17:43 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32519261167 | closing-lines | 2026-08-21 19:35 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32529285066 | closing-lines | 2026-08-21 21:36 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32537443805 | closing-lines | 2026-08-21 23:35 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32566236079 | daily-picks | 2026-08-22 09:55 | success | — | 2026-08-25 | **DEGRADED** | API-Football **account suspended** |
| 32569046033 | paper-report | 2026-08-22 10:58 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32570608261 | closing-lines | 2026-08-22 11:33 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32576577708 | closing-lines | 2026-08-22 13:43 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32582063138 | closing-lines | 2026-08-22 15:33 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32588112161 | closing-lines | 2026-08-22 17:33 | success | — | 2026-08-25 | **DEGRADED** | 4 league request(s) returned **no_rows** (credits spent, nothing written) |
| 32594025485 | closing-lines | 2026-08-22 19:32 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32599968642 | closing-lines | 2026-08-22 21:34 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32605543775 | closing-lines | 2026-08-22 23:33 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32632354001 | daily-picks | 2026-08-23 09:55 | success | — | 2026-08-25 | **DEGRADED** | API-Football **account suspended** |
| 32635235347 | paper-report | 2026-08-23 10:58 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32636878301 | closing-lines | 2026-08-23 11:33 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32643241921 | closing-lines | 2026-08-23 13:43 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32646469497 | daily-picks | 2026-08-23 14:46 | success | — | 2026-08-25 | **DEGRADED** | API-Football **account suspended** |
| 32648982545 | closing-lines | 2026-08-23 15:34 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32655184546 | closing-lines | 2026-08-23 17:32 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32661566816 | closing-lines | 2026-08-23 19:32 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32667908338 | closing-lines | 2026-08-23 21:33 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32673983048 | closing-lines | 2026-08-23 23:33 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32716289408 | daily-picks | 2026-08-24 10:20 | success | — | 2026-08-25 | **DEGRADED** | API-Football **account suspended** |
| 32720148367 | paper-report | 2026-08-24 11:06 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32723195851 | closing-lines | 2026-08-24 11:42 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32736037003 | closing-lines | 2026-08-24 14:00 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32747605838 | closing-lines | 2026-08-24 15:53 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32758407323 | closing-lines | 2026-08-24 17:44 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32769671034 | closing-lines | 2026-08-24 19:42 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32780903101 | closing-lines | 2026-08-24 21:42 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32789857642 | closing-lines | 2026-08-24 23:33 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32835286636 | daily-picks | 2026-08-25 10:04 | success | — | 2026-08-25 | **DEGRADED** | API-Football **account suspended** |
| 32840387478 | paper-report | 2026-08-25 11:04 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32840749965 | closing-lines | 2026-08-25 11:08 | success | — | 2026-08-25 | CLEAN | counts agree |
| 32843550194 | closing-lines | 2026-08-25 11:40 | success | — | 2026-08-25 | CLEAN | counts agree |

*Backlog cleared 2026-08-25. Every run in the repository now carries a verdict.*

---

# STAGE 18 — HALTED AT PART B BY A COHORT EVENT

**No code, config, schema or migration changes were made.** The stage stopped
where its own rule 5 and §B4 say it must: *"If any model or feature currently
reads those rows, it is [prediction-affecting], and that stops this stage."*

**A model feature does read them, and the contamination is worse than recorded.**

## PART A — sizing. Complete, and the answer is yes.

### A1. What the current table costs (MEASURED 2026-08-25)

| | |
| --- | --- |
| `odds` rows | **365,269** (the prompt's ~317,657 is stale) |
| `odds` total size | **74 MB** — 33 MB heap + **40 MB index** |
| per row, all-in | **211 bytes** |
| whole database | **108 MB** of a 500 MB free tier |
| rows eligible for the 400-day prune today | **0** |
| date the prune first bites | **2027-04-04** |

**Indexes cost more than the data** (40 MB vs 33 MB). Any snapshot design pays
that multiple on every retained row, which is why the per-row figure used below
is 211 bytes and not the ~90 bytes the heap alone would suggest.

The 400-day prune preserves rows for pick-bearing matches. It removes nothing
today and cannot until 2027-04-04, so **pruning is not a constraint on this
decision** — it is a ceiling that arrives after the research window.

### A2. What snapshots would cost — derived from write volume, not fixtures

Under snapshot semantics every upsert becomes an insert, so the growth driver is
**measured write volume**. August is the most complete month in the log cache
(146 of ~150 runs): **76,882 odds rows written in 25 days → ~92,000/month.**

| basis | rows/month | storage/month |
| --- | --- | --- |
| current (new keys only — 365,269 over 6 months) | ~61,000 | 12.9 MB |
| **snapshot (every write persists)** | **~92,000** | **19.4 MB** |
| **delta** | **+31,000** | **+6.5 MB** |

**Only a 1.5× multiplier.** Most writes already create new keys rather than
updating existing ones, which is why snapshotting is far cheaper here than the
"overwrite-in-place" framing suggests.

| horizon | projected database | % of 500 MB free tier |
| --- | --- | --- |
| today | 108 MB | 22% |
| +6 months | ~224 MB | **45%** |
| +12 months | ~341 MB | **68%** |

**Egress.** Stage 3 cut egress ~97% by column projection, and a taller `odds`
would undo part of that for any consumer that reads by `match_id` without a
latest-only filter. **The design below avoids this entirely rather than
mitigating it** — see C1.

### A3. Retention — keep everything

Full retention fits comfortably, so the honest answer is the simple one: **no
retention policy beyond the existing 400-day prune.** The research question needs
≥3 observations per key with meaningful spacing; the measured run cadence
supplies 1 pick-time write plus ~2–3 surviving closing refreshes per pick-bearing
key on match day. Designing a cleverer policy would add a mechanism to defend
without buying headroom that is needed.

**PART A VERDICT: affordable. `SUBSTRATE NOT AFFORDABLE` does not apply.**

## PART B — the contamination. This is where the stage stops.

### B2. Extent (MEASURED 2026-08-25)

The overround check works as a per-row detector, cleanly:

| source | complete 3-leg books | avg overround | **> 15%** |
| --- | --- | --- | --- |
| API-Football | 11,191 | **+18.90%** | **5,242 (47%)** |
| The Odds API | 32,468 | +6.63% | 166 (0.5%) |

A 47% vs 0.5% separation at a 15% threshold. **Mechanism confirmed
arithmetically:** a two-way Home/Away pair sums to ~1.03–1.05, and adding a
genuine 1X2 Draw leg (~0.25) yields ~1.30 — which is the ~32% overround observed,
not a coincidence of scale.

| | |
| --- | --- |
| contaminated (match, book) snapshots | **5,408** |
| **matches affected** | **2,608** |
| rows affected | **16,224** |
| books affected | 15 |
| period | **2026-02-01 → 2026-08-23** |
| saved picks on affected matches | **304** |
| pick observations on affected matches | 16 |

### B3/B4. Why the stage stops

| book | contaminated | avg overround |
| --- | --- | --- |
| **Bet365** | **2,565 / 2,775 = 92.4%** | **32.13%** |
| **Pinnacle** | 720 / 2,778 = **25.9%** | 11.68% |
| William Hill | 396 / 405 = 97.8% | 35.40% |
| Unibet | 392 / 396 = 99.0% | 35.67% |
| 10Bet, Betano | 255 / 255 = 100% | 34.5–36.1% |

`feature_engineer._get_bookmaker_features()` reads
`market_type IN ("1X2", "over_under", "btts", "team_goals")` across all
bookmakers, with a documented preference order of **Bet365 → Pinnacle → any**.

> **The model's first-choice bookmaker input is 92.4% contaminated, and has been
> since 2026-02-01.**

Consequently:

- **Refusing at the write (B3) is prediction-affecting.** It would stop Bet365
  1X2 rows being written for ~92% of matches, changing the bookmaker consensus,
  the de-vigged implied probabilities, and the 40% bookmaker blend. That is a
  cohort event.
- **Marking rows (B4) is prediction-neutral only while nothing filters on the
  mark.** Adding a nullable reason column changes no read. The moment any
  consumer honours it, it becomes the same cohort event.
- **Building Part C now would be actively harmful.** §B1 says the defect must be
  fixed *"before any history accumulates, or the accumulated history inherits
  it."* Snapshotting today would bake a 92%-contaminated Bet365 series into
  precisely the history the momentum research is meant to use. **Halting C is not
  literal compliance; it is the correct engineering call.**

### This corrects a previously recorded figure

The ledger records the blast radius as *"SEVEN books affected incl. Pinnacle
26%"*. **Pinnacle reproduces exactly at 25.9%** — but the true radius is **15
books**, and **Bet365 at 92.4% was never recorded at all.** The earlier
measurement understated the most important book in the pipeline.

## PARTS C AND D — not built

Designed but deliberately not implemented, recorded so the decision has something
concrete to act on:

- **C1 — a separate `odds_snapshots` table, not a taller `odds`.** Keeps the
  existing unique constraint and every current-price consumer untouched, so **no
  existing query reads a single extra row** and Stage 3's egress work is
  preserved by construction rather than by care. The same-snapshot rule becomes a
  timestamp comparison instead of an inference about row identity.
- **C2 — an explicit `first_seen_at`.** `created_at` cannot proxy for it: 53.5%
  of match rows were created after their own kickoff, mean +14 days.
- **C3 — injury history with a known-at timestamp.** Cohort-neutral: Stage 14
  established injuries reach only the Claude review prompt and never the model.
- **D — the proof-of-accumulation checks**, including the 20-of-20 replay
  standard for the two-way refusal, are not reached.

## The decision this stage hands back

Three options, none of which are mine to take:

1. **Fix at the write and accept a cohort break** — a new `CODE_REVISION`, a new
   `model_version`, and a documented boundary. Cleanest data, and every
   comparison across 2026-02-01 → now becomes two cohorts.
2. **Mark only, filter nothing, build the substrate** — cohort-neutral today, but
   the accumulated history inherits the defect and the momentum research runs on
   a 92%-contaminated primary book.
3. **Fix at the write, restricted to the snapshot table only** — `odds` keeps
   feeding the model exactly as today, `odds_snapshots` refuses contaminated
   writes. Cohort-neutral, research-clean, at the cost of two tables that
   disagree on purpose. **This is the option worth examining first**, and it was
   not examined here because examining it means designing the fix, which is
   Part C, which is halted.

**Note for whoever takes it:** four independent tests already say the model adds
nothing over the price. A 92% contaminated primary bookmaker input is a candidate
explanation for *why* that has never been tested, and option 1 is the only one
that would let it be.

---

**STAGE 18 — HALTED: COHORT EVENT AT PART B.**

Neither `SUBSTRATE BUILT` nor `SUBSTRATE NOT AFFORDABLE` applies. Part A shows
the free tier holds the substrate comfortably (68% at twelve months). Part B
shows the substrate cannot be built cleanly without changing what the model
reads. **The blocker is not storage. It is that the fix touches the predictive
core, and this stage is not allowed to.**

---

# STAGE 18 DIAGNOSTIC — THE HALT PREMISE WAS WRONG

Read-only. No code, config or schema changes.

## The correction, first

**Stage 18's halt rested on a stale docstring, not on the implementation.**

`_get_bookmaker_features()` opens with *"For each market the preferred bookmaker
order is: Bet365 → Pinnacle → any."* Thirty lines below, the code says:

> `# This used to de-vig ONE bookmaker, chosen Bet365 -> Pinnacle -> any.`
> `# Two problems, both measured on 2026-08-07: 1. Bet365's stored 1X2 was`
> `# corrupt for all 2,486 matches ... An overround plausibility gate runs`
> `# first, so a book whose market does not sum to a believable figure is`
> `# excluded outright rather than averaged in.`

**The docstring describes behaviour that was replaced in Stage 4 because of this
exact defect.** I read the docstring and reported "the model's first-choice
bookmaker input is 92.4% contaminated". The live code gates every book at
`[1.005, 1.25]` and then takes a **per-outcome median across surviving books**.

This is the project's own catalogued error class — *a definition read as an
occurrence* — committed by me, on the stage whose purpose was to find it.

**Two further claims in that commit are also wrong:**

- *"the true radius is FIFTEEN books"* — no. Fifteen counts books above a 1.15
  threshold, which sweeps in high-margin French Odds API books (`pmu_fr` 11.1%,
  `winamax_fr` 11.4%) that are not the two-way trap. **The real trap population
  is 9 bookmakers, 4,907 book-snapshots, 2,408 matches**, against the recorded
  seven. Only **3** Odds API books exceed 1.25 in the entire table.
- *"Bet365 at 92.4% was never recorded at all"* — no. `clean_dataset.py`'s module
  docstring records **"Bet365 92% of matches, William Hill 94%, Unibet 96%,
  Betfair 86%, 10Bet 98%, Betano 96%, 888Sport 97%, Pinnacle 26%"**. It was
  recorded in the evaluation harness; it was absent only from the ledger's
  blast-radius entry. **The Stage 4 measurement was substantially right and I
  reported it as substantially wrong.**

## The five paths (MEASURED 2026-08-25)

| # | path | reads contaminated rows? | protection |
| --- | --- | --- | --- |
| 1 | `_get_bookmaker_features()` | **no** | overround gate `[1.005, 1.25]` → per-outcome cross-book median |
| 2 | bookmaker blend, `w = 0.8` | **no** | reads the gated feature vector; inherits the gate |
| 3 | de-vig / EV path | **no** | `value_calculator._MARKET_PROB_KEYS` documents it consumes the gated consensus |
| 4 | CLV series | **no — 0 of 46** MODEL observations on trap matches | Stage 13 gate holds |
| 5 | `run_baseline` → `baseline.py` | **no** | `OVERROUND_3WAY = (1.005, 1.25)`, comment cites Bet365's 1.3524 |

### Residual exposure — the honest remainder

The gate rejects **4,910 of 5,408** contaminated books (**90.8%**). What survives:

| | |
| --- | --- |
| contaminated books passing the gate (1.15 < r ≤ 1.25) | **498** |
| as a share of the 38,250 passing books | **1.30%** |
| matches with ≥1 contaminated book in consensus | 379 of 3,191 (**11.9%**) |
| **matches where contaminated books are a MAJORITY of the consensus** | **25 (0.78%)** |
| single-book matches that are contaminated | **0** of 474 |

Consensus depth is a median of 5 books and a mean of 12. **A 1.3% minority
cannot move a median.** Only the 25 majority-contaminated matches could carry a
materially wrong consensus — 0.78% of the training population.

The user's point about normalisation is correct and the code already acts on it:
`_devig` returns **None** for an implausible book rather than normalising it,
with the comment *"an implausible book is dropped, not silently normalised into a
plausible-looking answer."*

## What this does to the decision

**The blend sweep stands. It does not need re-running.** The 2026-08-07 result
that the market beats the model monotonically to `w = 1.0` was computed through
`baseline.py`, which excludes these books by the same band. Both sides of that
comparison were clean.

**And that removes option 1's principal justification.** The case for accepting a
cohort break was that a 92%-corrupt primary market input might explain four
tests' worth of model inertness. **It cannot, because it was never in those
measurements.** The corruption has been gated out of every consumer since Stage
4. It explains neither why the model is bad nor why the market is good.

**A fourth option exists, and it is cohort-neutral by construction.** A write-side
refusal using **the same `[1.005, 1.25]` band the readers already apply** removes
exactly the rows every consumer already discards. **Prediction impact: zero, by
identity.** It fixes the storage defect, stops the momentum history inheriting
it, and needs no cohort break — and it is the one-definition consolidation §B3
asked for, since the band would then live once at the write instead of being
re-implemented in `feature_engineer`, `baseline` and `clean_dataset`.

Its limit, stated plainly: it leaves the 498 books in `(1.15, 1.25]` alone,
because tightening the band *is* prediction-affecting. Whether those 498 are
two-way contamination or genuinely wide markets was **not** established here and
should not be assumed.

## Status

**Stage 18's halt is withdrawn as to its premise.** Halting was still correct on
§B1's ordering — history must not accumulate on a defect — but the reason given
was wrong, and the decision it framed was framed around a danger that the
codebase had already neutralised in Stage 4.

Three of the five paths were protected by work this project had already done and
recorded. **The failure was mine in not reading it.**

---

# STAGE 18 — SUBSTRATE BUILT (Parts C, D, E)

Migration `009_price_and_injury_history` applied 2026-08-25. Additive only, with
a rollback. **`CODE_REVISION` stays `s5.3`; `model_version` unchanged.**

## PART C — what is now stored

### C1. `odds_snapshots` — a separate table, not a taller `odds`

The design decision that makes this cohort-neutral and egress-neutral **by
construction rather than by care**:

- `odds` keeps its unique key, its columns and its contents. **Every
  current-price consumer reads exactly what it read yesterday**, so Stage 3's
  ~97% column-projection egress work is untouched and no existing query scans a
  single extra row.
- `odds_snapshots` has **deliberately no unique constraint** on
  `(match_id, bookmaker, market_type, selection)`. That absence *is* the
  feature — a second row for the same key at a later `observed_at` is the entire
  point, and a unique index there would silently restore overwrite semantics.

Written from **both** odds writers via one shared helper
(`src/data/price_history.py`), because two scrapers needing identical behaviour
is precisely how `(1.005, 1.25)` came to exist three times.

An **update** is recorded, not just an insert. The update is exactly the
observation that overwrite semantics used to destroy.

**Fails open.** A snapshot insert that raises must not take down pick
generation: history is a research nicety, pricing picks is the job.

**The same-snapshot rule gets simpler, as predicted.** A closing observation
must come from a price observed strictly after `taken_at`; with a real
`observed_at` that is a timestamp comparison rather than an inference about row
identity.

### C2. `odds.first_seen_at`

Nullable, never backfilled. Existing rows stay NULL because **a guessed
first-sight time would look like evidence** — and `matches.created_at` cannot
proxy for it, since 53.5% of match rows were created after their own kickoff
(mean +14 days) as backfill stamps.

Write-once by construction: the only site that sets it sets it only when absent,
and the bulk `ON CONFLICT` clause deliberately omits it, so a refresh cannot
turn first-seen into last-seen.

### C3. `injury_observations`

`observed_at` is the whole point: an injury moves a line **when the news
arrives**, and a current-status snapshot cannot distinguish a two-week-old
absence from this morning's announcement. Recorded at both write sites,
including the UPDATE branch — the branch where a status change used to overwrite
what was previously known.

**Cohort-neutral, confirmed not assumed:** Stage 14 established injuries reach
only the Claude review prompt and never the model.

## PART D — proof, not trust

`tests/test_price_history_accumulates.py`, 9 checks. The failure this guards
against is L2's: **shipped, wired, and dead — which measures identically to
working.**

| check | result |
| --- | --- |
| three writes to one key produce three rows, ordered | **pass** |
| the natural key is NOT unique (accumulation possible at all) | **pass** |
| injury status changing three times keeps three rows | **pass** |
| `first_seen_at` does not move on a second write | **pass** |
| `first_seen_at` absent from the `ON CONFLICT` set | **pass** |
| **both** odds writers call the recorder | **pass** |
| **nothing reads the snapshot table** — cohort guard | **pass** |
| `odds` keeps its columns and its unique constraint | **pass** |
| two-way refusal: 20 contaminated refused, 20 legitimate admitted, refused set ≡ discarded set | **pass** (44 checks, previous commit) |

**819 + 9 = 828 tests pass. 26 invariants pass.**

**What could NOT be checked without waiting:** *"after one real day, at least one
key has three observations."* The tables are live and empty; the first rows land
on the next scraper run. The verification query is:

```sql
SELECT match_id, bookmaker, selection, count(*) AS observations
FROM odds_snapshots
GROUP BY 1,2,3 HAVING count(*) >= 3
ORDER BY observations DESC LIMIT 20;
```

**If that returns nothing after a full day, the substrate is inert and Stage 19
must not begin.**

## PART E — when each hypothesis becomes testable

**Derived from the measured observation rate, not estimated.**

Rate basis, MEASURED 2026-08-25: **274 of 875 August matches (24 days) already
receive ≥2 observations under overwrite — 11.4 matches/day.** Those become ≥2
snapshot rows; the pick-bearing subset gains a third from the closing refresh.
Conservatively taking half as reaching a genuine three-point trajectory gives
**~5.5 trajectory-fixtures/day**.

Target n = 50 independent fixtures — Stage 16's recommended checkpoint, chosen
because 17 suffices at 80% power for a +2% effect and 50 carries it with margin
for a movement variance that may differ from CLV's.

| hypothesis | needs | derived date |
| --- | --- | --- |
| **H1 momentum** | ≥3 observations per key, 50 fixtures at 5.5/day ≈ 9 days, +5 for fixture-calendar irregularity | **2026-09-08** |
| **H4 lead time** | `first_seen_at` plus one later observation — same writes, same rate | **2026-09-08** |
| **H3 injuries** | injury STATUS TRANSITIONS near kickoff, not just presence; needs ~30 days of daily snapshots | **2026-09-24** |

**H3 carries a live risk:** the June collapse (47 runs, 5 producing any
injuries, max 7 against 128–198 in March–May) is unexplained and deferred. If it
recurs, H3's date moves and nothing will announce that. **Check the injury
observation count before starting Stage 19, not after.**

## Open questions, recorded and not pursued

- **The 498 books in (1.15, 1.25].** They pass the band and may be trap residue
  or genuinely wide markets. **Tightening the band IS prediction-affecting**, so
  establishing which they are is a cohort decision, not a diagnostic.
- The June injury collapse; the 08-23/08-25 thin cards; the 35 picks with no
  review decision and the two unapplied CHANGE decisions — all from the
  2026-08-25 audit, all still deferred.

---

**STAGE 18 — SUBSTRATE BUILT.**

H1 and H4 testable **2026-09-08**, H3 **2026-09-24**, subject to the one-day
accumulation check above. Stage 19 is a research stage and must not begin before
that query returns rows.

---

# DAILY CI AUDIT — 2026-08-26, and two promoted findings

Read-only. No code, config or schema changes.

## Routine pass — 10 runs

| run_id | workflow | started (UTC) | conclusion | steps failed | audited on | verdict | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32856933640 | closing-lines | 2026-08-25 14:01 | success | — | 2026-08-26 | CLEAN | nothing to do. |
| 32869076321 | closing-lines | 2026-08-25 15:58 | success | — | 2026-08-26 | CLEAN | nothing to do. |
| 32879621488 | closing-lines | 2026-08-25 17:44 | success | — | 2026-08-26 | CLEAN | the ONLY run to write snapshots: 85 rows, 1 match. |
| 32891242489 | closing-lines | 2026-08-25 19:44 | success | — | 2026-08-26 | CLEAN | nothing to do. |
| 32902385047 | closing-lines | 2026-08-25 21:42 | success | — | 2026-08-26 | CLEAN | nothing to do. |
| 32911466157 | closing-lines | 2026-08-25 23:35 | success | — | 2026-08-26 | CLEAN | nothing to do. |
| 32957334452 | daily-picks | 2026-08-26 10:15 | success | — | 2026-08-26 | **DEGRADED** | **0 fixtures discovered, 0 picks.** AF suspended (1 request used); Flashscore 0 fixtures for all 3 leagues scraped; f-d.org 0 new fixtures. |
| 32961678226 | paper-report | 2026-08-26 11:06 | success | — | 2026-08-26 | CLEAN | 61 captured, CLV series unchanged. |
| 32962222174 | closing-lines | 2026-08-26 11:13 | success | — | 2026-08-26 | CLEAN | no pending picks. |
| 32964698782 | closing-lines | 2026-08-26 11:42 | success | — | 2026-08-26 | CLEAN | no pending picks. |

**AUDIT GAP, recorded:** `scripts/ci_audit.py` flagged run 32957334452 only for the
API-Football suspension. **It has no assertion for "zero fixtures analysed"**, so
the largest event of the day was invisible to the mechanical pass and was found
only because Niki said six fixtures existed. A day that discovers nothing is not
currently a finding.

## FINDING 1 — zero fixtures. It is BOTH failures, and they are unrelated.

### (a) Discovery failure — this is the cause of zero picks

**The six fixtures are ABSENT from `matches` as fixtures.** All three sources
were asked and all three returned nothing. From the log, not from intent:

| source | what it actually returned |
| --- | --- |
| API-Football | `account suspended`, then `update complete (1 requests used)` — the dead-integration signature: first call refused, a flag suppresses every later call *before* it logs |
| **Flashscore** | scraped `spain/laliga`, `portugal/primeira-liga`, `europe/champions-league` and returned **0 fixtures for all three** |
| football-data.org | `1 scores updated, 0 new fixtures added` |

→ `_check_empty_fixture_leagues: Flashscore: 0 fixtures / timeout for:
europe/champions-league`, and `0 fixtures for tier-1 league` for both
portugal/primeira-liga and spain/laliga
→ `get_daily_picks: No fixtures found for 2026-08-26`.

**The suspension explains only API-Football.** Flashscore is independent, was
asked for exactly the leagues Niki names, and returned zero. That is a second,
separate failure and it is the one that matters, because Flashscore is the
fallback the suspension was supposed to be survivable by.

UEFA discovery has a longer history: `europe/champions-league` was last
discovered **2026-08-19** — the day of the suspension — and
`europe/europa-conference-league` on **2026-08-14**. Those competitions have
**0 corrupt rows**, i.e. they only ever came from API-Football.

### (b) A separate corruption — results rows stamped `now()`

The 20 rows dated 2026-08-26 are **not fixtures**. All 20 carry a score, and all
20 were created 10:22–10:33 — *after* `No fixtures found` at 10:22:25. They are
completed matches from the post-picks results scrape.

`flashscore_scraper._parse_match_date(element, default, is_result)` returns
`default=datetime.now()` when the time text cannot be parsed. **Silently.** So an
unparseable kickoff becomes "now", which is exactly why every corrupt row has
`match_date ≈ created_at`:

| created | matches | `match_date` == `created_at` | % |
| --- | --- | --- | --- |
| 08-19 | 56 | 51 | 91.1% |
| 08-22 | 50 | 16 | 32.0% |
| 08-23 | 110 | 97 | 88.2% |
| 08-24 | 39 | 33 | 84.6% |
| 08-25 | 24 | 23 | 95.8% |
| **08-26** | **20** | **20** | **100.0%** |

### (c) THIS CORRECTS TWO EARLIER AUDIT ENTRIES OF MINE

The 2026-08-25 audit recorded *"08-23: 11 picks from 110 fixtures"* and
*"08-25: 1 from 24"* as **"low and not explained by card size"**. **Those
denominators were mostly phantom.** Removing the results-artifacts:

| day | real fixtures | picks | conversion |
| --- | --- | --- | --- |
| 08-23 | 13 | 11 | 85% |
| 08-24 | 6 | 6 | **100%** |
| 08-25 | 1 | 1 | **100%** |
| 08-26 | 0 | 0 | — |

**There was never a pick-selection collapse.** The pipeline converted nearly
every real fixture it was given. What collapsed is **fixture discovery**, and the
growing pile of `now()`-stamped results rows disguised it by inflating the
denominator — a failure that looked like a filter problem because the count it
was measured against was wrong.

### (d) Do the three days share one cause?

**Yes, one cause, and it is monotone.** Real fixtures discovered: 13 → 6 → 1 →
**0**. Picks: 11 → 6 → 1 → 0. The team-identity gate, `club_pick_min_coverage`,
the odds requirement and the briefing freeze are all **irrelevant here** — no
`SKIPPED on team-identity mismatch` line appears, and nothing reached those
filters because nothing reached the analyser at all. **§7 stays UNTESTABLE: a
gate that never ran is not a gate that passed.**

## FINDING 2 — substrate verification, and the date is withdrawn

Part D's deferred check, run 2026-08-26:

```
keys with >= 3 observations : 0
total odds_snapshots rows   : 85
distinct matches            : 1
observation window          : 2026-08-25 17:44:33.637 -> .642  (a single 5ms burst)
injury_observations         : 0
odds.first_seen_at populated: 0
```

**The machinery is NOT inert — 85 rows prove the wiring fires.** It has simply
had almost nothing to record: one match, in one run, ever. `first_seen_at` is 0
because those 85 writes were all UPDATES to existing keys, and it is set only on
insert — correct by design, but it means the column fills only as genuinely new
(match, book, market, selection) keys appear.

### Re-derived from what is actually accumulating

Stage 18 derived 2026-09-08 for H1 and H4 from **11.4 matches/day with ≥2
observations, measured in August**. That rate assumed fixtures were being
discovered. Measured since deployment:

| | August basis | actual since deploy |
| --- | --- | --- |
| matches accumulating observations | 11.4 / day | **1 total** |
| keys reaching 3 observations | — | **0** |
| injury observations | — | **0** |

> **STAGE 19 HAS NO TRIGGER DATE.** 2026-09-08 and 2026-09-24 are withdrawn. At
> the observed rate H1 and H4 are never testable, because a price path needs a
> fixture to have a price, and no fixtures are being discovered.

**This is the failure mode flagged in Stage 18 arriving early on a different
input.** The warning was written about H3's injury dependency — *"if it recurs
the date moves and nothing will announce it"* — and the same structure applied
to fixture discovery, which was not identified as a dependency at all. **The
substrate's accumulation rate is a function of the pipeline's health, and Stage
18 treated it as a constant.**

**Fixture discovery must be understood before Stage 19 has any date**, and the
accumulation query is the honest trigger — not a calendar date.

## Recorded, not pursued

- `_parse_match_date`'s silent `datetime.now()` default (read-only stage; this is
  a fix, and it needs its own decision — the rows it has already written are
  historical data, and this project has twice decided such rows are marked, not
  deleted)
- Why Flashscore returns 0 fixtures for tier-1 leagues it successfully reaches
- `ci_audit.py` has no "zero fixtures analysed" assertion
- The June injury collapse; the 35 picks with no review decision; the two
  unapplied CHANGE decisions; the 498 books in (1.15, 1.25]

---

# STAGE 19 — DISCOVERY DIAGNOSED, NOT RESTORED

Four reports. **Part A halts for a cohort decision. Part B's cause is found but
not reproduced. Part D is shipped and proven both directions.**

## PART A — the phantom rows

### A1. The defect, named as the family it belongs to

`_parse_match_date` returned `default` on parse failure, and both callers passed
`datetime.now()` (results) or `datetime.now() + 1 day` (fixtures). An unparseable
kickoff silently became "kicks off at this instant".

**This is the fourth instance of one family: a default that makes a failure look
like a success** — after the swallowed `IntegrityError`, the short-circuit that
suppressed its own logging, and the safety net that masked its mechanism. The
answer has been the same every time: fail closed and say so.

### A2. Quantified, MEASURED 2026-08-26

**The test, and its precision.** A real kickoff is scheduled on a whole minute; a
`now()` stamp carries seconds *and* microseconds. Two candidate tests:

| test | rows |
| --- | --- |
| `EXTRACT(SECOND FROM match_date) <> 0` | **510** |
| within 60s of `created_at` | 378 |
| both | 378 |
| sub-minute but NOT near creation | **132** |
| **near creation but on a whole minute** | **0** |

The last row is what establishes precision: **no row is near its creation time
while carrying a legitimate whole-minute kickoff**, so the 60-second heuristic is
a strict subset and the precision test is the correct one. The 132 extra rows
resolve by offset — **495 at 0h** (the results default) and **7 at 24h** (the
fixtures default, which fired only until 2026-03-09).

**Separated from Stage 17's finding, which they were never distinguished from.**
Stage 17 attributed "53.5% of match rows created after their own kickoff, mean
+14 days" to backfill stamps. The populations are **completely disjoint**:

| population | rows | mean days created-after-kickoff |
| --- | --- | --- |
| created after kickoff | 37,072 | **637.3** |
| phantoms | 510 | **max 0.014** (20 minutes) |

**Stage 17's 53.5% was genuine backfill. None of it is this defect.**

**Consumers:**

| consumer | exposure |
| --- | --- |
| saved picks | **0** |
| pick observations | **0** |
| odds rows | 142 |
| **the fitting set** (`is_fixture=false AND home_goals IS NOT NULL AND training_exclusion_reason IS NULL`) | **503 of 510** |
| already excluded | **0** |

**All 503 sit inside the 180-day half-life** — stamped "now", they carry
**maximum recency weight** in the time-decayed Poisson.

### A3. Forward fix SHIPPED; historical marking HALTED

**Shipped:** `_parse_match_date` now returns `None` on failure. Both callers
refuse the row, log the **raw unparsed text** (`_raw_time_text`), and increment
`_unparseable_dates` for the run summary. A refusal that does not quote what it
could not parse makes a selector change look like a quiet drop in counts.

**HALTED, per rule 8 and §A3.** Marking the 503 removes them from fitting. Their
scores and teams are real — only the *date* is wrong — so this is not cleanup:

- **exclude** → lose 503 real results
- **keep** → 503 real results carry a false date at maximum decay weight
- **repair** → recover true kickoffs from source; best, still prediction-affecting

**All three change what the model learns. This is a cohort event and it is
Niki's decision.**

## PART B — why Flashscore returns zero

### B2 first, because it reframes B1

| | |
| --- | --- |
| **last day Flashscore produced ANY fixture** | **2026-05-30** |
| consecutive days at zero | **88** |
| attempts in that window | 200+ |
| fixtures produced | **0** |
| non-zero scrapes before that | 406, totalling 2,592 fixtures |

**Fixture discovery has been dead since 2026-05-30 and nothing reported it**,
because API-Football silently covered for it until its suspension on 2026-08-19.
The 13 → 6 → 1 → 0 decline is not Flashscore degrading — it is **cached
API-Football knowledge ageing out of a source that had already stopped working
three months earlier.**

### An amplifier I initially mistook for the cause

```python
_fixture_leagues = [l for l in _ordered_leagues
                    if l in _important        # has today's fixtures or pending picks
                    and l not in _fs_skip_fixtures]
```

`_important = _today_leagues | _pending_leagues`, both derived from fixtures
**already known**. So a league is scraped for fixtures only if it is already
known to have fixtures — circular, and it can only shrink. Today: **30 leagues
configured, 26 scraped for results, 3 for fixtures.**

**But it is not the binding cause, and I reported it as one before checking.**
Scraping all 30 would still yield zero, because the parser produces nothing for
any league. Recorded as a real defect of its own.

### B1. Where it breaks, from the log

| league | duration | reading |
| --- | --- | --- |
| `spain/laliga` | **13.0s** | page fetched, parser found no rows |
| `portugal/primeira-liga` | **10.0s** | page fetched, parser found no rows |
| `europe/champions-league` | **1.2s** | **too fast for a page load — likely never fetched** |

CI runs **Chrome/Selenium under Xvfb**, not camoufox
(`_get_driver: Xvfb detected — running Chrome in headed mode`).

**The decisive discrimination:** results and fixtures use the **same selectors**
(`.event__match.event__match--static.event__match--twoLine`, falling back to
`.event__match`) on the **same domain in the same run** — and **results
succeeded**, creating 20 rows. So this is **not** bot-blocking, **not** the
driver, and **not** the session. It is specific to `/fixtures/`.

### B3. NOT REPRODUCED, and that is the finding

Camoufox is not installed locally and I did not fetch the live site from here, so
**the fixtures page was never inspected directly**. What is established bounds
the fix: the failure is on the fixtures page, with working selectors, in a run
where the results page worked. **What it would take:** one run against
`flashscore.com/football/spain/laliga/fixtures/` capturing the served HTML, to
separate *page had nothing* from *page had something unrecognised*. Until then,
iterating on selectors would be guessing.

## PART C — football-data.org

**Its map contains exactly the leagues that went missing.**

```
"CL": europe/champions-league        "PD": spain/laliga
```

Nine club competitions plus the World Cup, of **30 configured leagues — a 30%
discovery floor** if both other sources stay unavailable.

`sync_fixtures(days_ahead=0)` fetches **today only**, with no lookahead, and
admits only `SCHEDULED`/`TIMED` matches that also resolve through
`TEAM_NAME_MAP`.

**Why it reported `0 new fixtures added` is NOT established.** The integration is
demonstrably live — `1 scores updated` in the same run — so the key works and the
API answers. The remaining candidates are: the API returned nothing for today;
it returned matches filtered out by status; or names failed to resolve. **The log
conflates all three and I could not separate them**: `FOOTBALL_DATA_ORG_KEY` is a
CI secret and is absent locally, so the call is unreproduced.

**This is the highest-value open thread in the stage.** If football-data.org can
supply Champions League and LaLiga and simply is not, discovery is recoverable
without Flashscore and without API-Football.

## PART D — the audit could not see it. Now it can.

**The assertion was never implemented for discovery.** Stage 14 specified the
self-calibrating shape and discovery never received one.

Added to `ci_audit.py`, keyed on **the scraper's own warning**
(`returned 0 fixtures for X — expected ≥1 for active season`), which already
excludes `off_season_leagues`. **A second off-season list in the audit would be a
sixth data-layer copy of a definition; inheriting the scraper's judgement avoids
it**, and `test_discovery_assertion.py` pins that reasoning.

**Proven by replay, both directions — the positive control matters as much as
the assertion:**

| window | result |
| --- | --- |
| 2026-08-23 → 08-26 | **fires on all four days** (8, 7, 6, 5, 3 active leagues at zero) |
| 2026-05-28 → 05-31 | **silent — all CLEAN** (discovery still worked; 05-30 is the last good day) |

### The deeper point, recorded as a proposal

**The audit measures what the pipeline reports about itself.** It has no external
reference for "how many matches were actually played today", so a pipeline that
sees nothing and says nothing reads as a quiet day — which is exactly what
happened for 88 days. The new assertion closes this instance by watching the
scraper's own expectation, but not the class.

An external check — one weekly call to any free fixture list, compared against
what the pipeline discovered — would close the class. **Proposal only, not
implemented:** it is a new external dependency and belongs to a decision, not to
this stage.

## Recorded, not pursued

- `_important` circularity in fixture-league selection (real, not binding)
- The 1.2s Champions League scrape — a different failure from the 10–13s ones
- cp1251, **fifth instance**, in this stage's own diagnostic output
- `feature_engineer`'s hand-copied training predicate (known since Stage 13)

---

**STAGE 19 — DISCOVERY DIAGNOSED, NOT RESTORED.**

Neither declaration in the prompt applies. Discovery is **not restored**: the
Flashscore fixtures parser has produced nothing since 2026-05-30 and was not
reproduced. It is **not established as unrestorable** either — football-data.org
covers Champions League and LaLiga and has not been ruled out, and that is where
the next work belongs.

**833 tests pass. 26 invariants pass. `CODE_REVISION` unchanged at `s5.3`.**

---

# THE SILENT SUBSTITUTION — the class, recorded in its own terms

**Flashscore's fixture scraper died on 2026-05-30 and nothing noticed for 88
days, because API-Football silently covered for it.**

This is the same family as MASK-1's safety net, and it is the strongest instance
the project has found:

> **A fallback that substitutes silently makes the primary's failure invisible.
> Redundancy that is never tested per component is not redundancy — it is one
> working source and two unverified claims.**

The system had three fixture sources and believed itself resilient. It was
running on one. The other two had been dead or degraded for months and every
aggregate measurement — fixtures discovered, picks generated, runs green — said
the pipeline was healthy, because the survivor's output filled the hole exactly
where a monitor would have looked.

**It generalises past this incident.** Any component with a fallback needs a
per-component liveness check, or its failure is deferred until the fallback also
fails — at which point two failures surface together and the older one is
invisible inside the newer. That is precisely how this presented: the operator
saw "zero fixtures today" on 2026-08-26, seven days after API-Football's
suspension and **88 days after the actual failure**.

## Item 1 — the assertion I shipped carried the same blindness

**Tested against the day that matters, 2026-05-31:**

| source | fixtures |
| --- | --- |
| Flashscore | **0** |
| football-data.org | **0** |
| API-Football | **13** |

**Total discovery was healthy, so the aggregate assertion did NOT fire.** The
check written in response to an 88-day blindness would have reproduced it.

**Fixed: per source, never aggregate.** Each fixture source is now watched on its
own — `src_flashscore_fixtures`, `src_footballdataorg_fixtures`,
`src_apifootball_fixtures` — through the existing self-calibrating mechanism, so
a source that produced within the last 7 runs and now produces none alarms
regardless of what the others do.

**Gated to the day's FIRST run.** The first per-source version fired repeatedly on
2026-03-03, a day discovery was working, because a same-day re-run legitimately
finds no *new* fixtures. Only the day's first run exercises discovery from cold.

**Replayed both directions:**

| window | result |
| --- | --- |
| **2026-05-31** | **FIRES** — names Flashscore and football-data.org while API-Football is healthy |
| 2026-03-03 re-runs | **silent** (was firing 5× before the gate) |
| 2026-03-01 → 03-04 first runs | silent except one genuine one-day zero |

Nine tests pin it, including `test_one_dead_source_fires_even_while_others_produce`
and `test_a_same_day_rerun_does_not_fire`.

## Item 2b — PART B REPRODUCED. One upstream change caused both defects.

Camoufox installed (`pip install camoufox`, `python -m camoufox fetch`) and one
`/fixtures/` page loaded. **The blocker was a setup gap, as stated.**

`https://www.flashscore.com/football/spain/laliga/fixtures/` — **HTTP 200**,
title `LaLiga Fixtures - Football/Spain`, 1.77 MB of HTML, **113 fixture rows
present**, including `26.08. 22:00 | Real Madrid | Real Sociedad`.

**The page was fine. The selectors were not.**

| selector | found |
| --- | --- |
| `.event__match` (fallback) | **113** |
| `.event__match.event__match--static.event__match--twoLine` (**primary**) | **0** |
| `.event__time` (what `_parse_match_date` read) | **0** |
| `wcl-*` classes | **1,775** |

Flashscore's 2026 redesign renamed `event__match--static` →
`event__match--withRowLink` and moved the kickoff out of `.event__time` into
**build-hashed CSS-module classes**:

```
row: event__match event__match--withRowLink event__match--twoLine event__match--scheduled
     wcl-scores-simple-text-01_-OvnR wcl-scores_Na715 …  '26.08. 22:00'
```

**ONE CHANGE, BOTH DEFECTS:**

- the **primary** selector requires `--static` → **fixtures return 0** (Part B)
- the **fallback** still matches rows on the results page, but `.event__time` is
  gone → every kickoff unparseable → `datetime.now()` → **the phantoms** (Part A)

Two symptoms, 88 days apart in visibility, one cause.

**Fixed structurally, not cosmetically.** The hashed classes (`_-OvnR`, `_Na715`)
change on every Flashscore deploy, so selecting on them would be a countdown, not
a fix. The row class `event__match` is stable and the kickoff is read from the
row's **text** by shape (`\d{2}\.\d{2}\.(\d{4})?\s+\d{2}:\d{2}`). Verified against
the live page text: `'26.08. 22:00'` → `2026-08-26 22:00:00`, a whole-minute
kickoff rather than a `now()` stamp.

## Item 2a — PART C RESOLVED. The API has the fixture; the pipeline drops it.

`FOOTBALL_DATA_ORG_KEY` was in `.env` under a name I did not search for. **The
blocker was mine.**

Live call, 2026-08-26, masked key `e1e6…fea1`:

```
2 matches for 2026-08-26
  CLI FINISHED  Independiente del Valle vs CD Tolima     (not in COMPETITION_MAP)
  PD  ...       Real Madrid CF vs Real Sociedad de Fútbol  19:00Z
```

**The LaLiga fixture Niki named is in the API**, `PD` maps to `spain/laliga`, and
both names resolve through `TEAM_NAME_MAP` (`Real Madrid`, `Sociedad`). It should
have been created.

**Why it was not — and this is an upstream data defect.** The `status` field on
that match is malformed:

```
status : '2026-08-26 19:00:00Z'      <-- a timestamp where an enum belongs
utcDate: '2026-08-26T19:00:00Z'
```

A call minutes earlier returned `status: 'TIMED'`. **The field flaps.** The
pipeline's `status not in ("SCHEDULED","TIMED") → continue` then drops the
fixture, **with no log line at all**.

**And the log cannot distinguish this from any other zero.** `_get` returns
`None` on a non-200 or an exception, logging only at `logger.debug`;
`fetch_matches` turns `None` into `[]`; `sync_fixtures` reports `0 new fixtures
added` at INFO. Three different failures, one message. *A default that makes a
failure look like a success* — the fifth instance, now in the third source.

**Discriminated rather than assumed:** the CI log carries **46 DEBUG lines** and
**no football-data.org error among them**, so on 2026-08-26 `_get` returned 200
and the drop happened downstream, in the filter.

**Not fixed here** — the silent `continue` and the status-tolerance question are a
change to what gets discovered and belong with the operator, now that the cause
is known rather than suspected.

**Coverage, for the record:** football-data.org maps 9 club competitions of 30
configured leagues — a **30% discovery floor** — and `sync_fixtures(days_ahead=0)`
looks only at today, with no lookahead.

## Item 3 — the 510 phantoms EXCLUDED, and the cohort break taken once

Marked `training_exclusion_reason = 'phantom_kickoff_now_stamp'`. **Marked, never
deleted.** Stage 13's 29 `corrupt_team_identity` marks preserved; 0 phantoms left
unmarked; fitting set 39,290 → **38,787**.

**MEASURED EFFECT, recorded in the `s5.4` history entry as decay weight, because
the count understates it:**

| | |
| --- | --- |
| share by count | 503 / 39,290 = **1.280%** |
| **share by decay weight** (H = 540d) | 486.8 / 17,281.6 = **2.817%** |
| average weight, phantom | **0.9677** |
| average weight, real match | **0.4330** |

**Each phantom carried 2.23× the weight of an average real match** — because a
`now()` stamp sits at weight ≈1.0 by construction. Their harm was disproportionate
to their count, which is the opposite of the usual argument for tolerating 1.3%
of a fitting set.

### `CODE_REVISION` → `s5.4`, one break covering three changes

Folded together as instructed, so there is one boundary and not three:

1. the 510 exclusions
2. `_parse_match_date` failing closed (future rows refused, not invented)
3. the Flashscore selector/time repair, which changes **which fixtures are
   discovered**

`model_version`: `stage5_baseline_20260807.098437` → **`stage5_baseline_20260807.0976b8`**.
Both pins in `tests/experiment_pins.py` updated to match.

**Authorised on the ground that there is no longer an experiment to protect:**
Stage 16 established the MODEL CLV upper bound at **+0.107%** against a **+1.85%**
requirement.

---

**837 tests pass. 26 invariants pass. `CODE_REVISION` = `s5.4`.**

**Discovery is repaired but NOT yet demonstrated in production.** The declaration
`STAGE 19 — DISCOVERY RESTORED` requires a measured fixture count over a full day
against the real card, and the next scheduled run has not happened. **The honest
status is DISCOVERY REPAIRED, PENDING PROOF**, and the proof is tomorrow's run:
if `spain/laliga` and the other 29 leagues return non-zero fixture counts and the
per-source assertion stays silent, it is restored.

---

# STAGE 19 ITEM 2a — the silent drop removed, and s5.5 forced not chosen

## The cohort split was forced

`ac8bedb` was **already pushed** to the public remote when this landed
(`git ls-remote origin HEAD` = `ac8bedb…`), and rewriting a pushed commit is the
history rewrite this project has refused twice. So `s5.5`, not a second entry in
`s5.4`.

**It is materially costless, and that is measurable rather than asserted:** no
prediction was ever stamped `s5.4`. `saved_picks.model_version` holds
`…098437` (18 picks, s5.3), `…485823` (246), and NULL (1,074) — **zero rows
carry `…0976b8`**. The split separates an empty cohort from an empty one.

## The fix

`status` no longer answers "has this been played". `utcDate` does, because it is
verifiable rather than guessed. `status` stays authoritative only for what a
date cannot express — `POSTPONED`, `CANCELLED`, `SUSPENDED`, `AWARDED`.

- unrecognised status + **future** kickoff → **ADMIT**, log a warning naming the
  raw status
- unrecognised status + past kickoff → skip (ordinary, already played)
- unrecognised status + **unparseable** `utcDate` → **REFUSE**, log — when
  neither field can answer, nothing is assumed
- **never a bare `continue`**

Counters `_status_refusals` / `_status_recovered` so a run summary can report it:
a silent recovery is only marginally better than a silent drop.

## MEASURED EFFECT — and it is zero, which is the honest headline

Sampled 11 days (3 back, 7 ahead) on 2026-08-27, mapped competitions only:

```
102 matches   statuses: {FINISHED: 8, TIMED: 94}
admitted by the OLD filter : 94
admitted by the NEW filter : 94
recovered                  : 0
```

**The malformed status was TRANSIENT and had already resolved.** The repaired
filter admits exactly the same fixtures. **The discovery floor does not move.**

Recorded that way deliberately: this change is *insurance against a recurrence*,
not a gain, and a later reader must not credit it with fixtures it did not find.
What it removes is the **silence** — the next occurrence will be a logged
warning instead of a fixture that was never mentioned.

## A blocker asserted from a failed lookup is a claim about the lookup

Added to the guard-design notes beside the stale docstring.

I reported Part C blocked because `FOOTBALL_DATA_ORG_KEY` was "absent locally".
It was in `.env` at line 12. I had searched for `FOOTBALL_DATA_API_KEY` and
`FOOTBALLDATA_API_KEY` — neither of which is the name the code reads, which is
visible in `footballdataorg_scraper.py:290`.

**Both of this stage's "blockers" were my own setup gaps**: the key was there
under a name I did not search for, and camoufox was in `requirements.txt` at
line 13 and installs in one command. Two of the highest-value threads in the
stage were parked on claims about my own lookups.

> **The rule: "X is not available" is a finding about the search, not about the
> world, until the search itself has been checked.** It belongs with the stale
> docstring, which was the same error in the other direction — trusting a
> lookup that returned something wrong, rather than one that returned nothing.

---

# API-FOOTBALL REPLACEMENT ACCOUNT, and the four checks it made urgent

**A working API-Football account changes what has to be proven.** With three
sources alive, a non-zero fixture total proves nothing about any one of them —
it reproduces exactly the condition that hid Flashscore's death for 88 days.
Every check below is therefore **per source**, and the aggregate is treated as
the number that lied.

## OPS-2 — REPLACEMENT account, not a restoration

**OPS-1 does not close.** The original account's suspension on 2026-08-19
10:10:28 UTC remains true, is unappealed, and stays open in the record. What
exists now is a **different account**, and the distinction matters because the
cohort boundary and the provider's terms both attach to it.

| | |
| --- | --- |
| new key SHA-256 (SEC-5 — value never recorded) | `d10280aa8a78c292b73bd8be0b6924275719929a4d952f08b9cd49181f318ad7` |
| key length | 32 chars |
| **previous** key SHA-256 (suspended, now commented out in `.env`) | `fc9d3c031298b88c…` (56 chars) |
| **different credential** | **yes** — different fingerprint AND different length/format |
| account created | **2026-08-27** (derived: `/status` reports `subscription.end = 2027-08-27`, a one-year free term) |
| plan / state | `Free`, `active=true`, **0 / 100 requests used** |
| deployed to CI | GitHub secret `API_FOOTBALL_KEY` updated **2026-08-27T11:24:39Z** |
| local | `.env.local` (gitignored, `.gitignore:50`) — **not** `.env`, whose `API_FOOTBALL_KEY` line is commented out and holds the OLD key |

**ANTICIPATED, NOT MYSTERIOUS.** API-Football's published terms allow **one free
account per user**. This is a second free account on the same project after the
first was suspended. If it is swept, that is the rule being applied, **not a new
incident** — recorded now so the event reads as anticipated. The 56→32 character
format change also suggests a different registration route from the original,
which is worth knowing if the sweep ever needs explaining.

## Data-minimisation review — one real finding, removed

Enumerated every header, parameter and body this codebase sends to the three
providers.

| provider | transport | credential | other headers | parameters |
| --- | --- | --- | --- | --- |
| API-Football | aiohttp via `BaseScraper.fetch_json` | `x-apisports-key` header | **was: a spoofed `Chrome/120.0.0.0 Windows NT 10.0` User-Agent** | endpoint-specific only (`date`, `league`, `season`, `fixture`) |
| football-data.org | httpx, direct | `X-Auth-Token` header | httpx defaults only (`accept`, `accept-encoding`, `connection`, `user-agent: python-httpx/x.y`) | `date` only |
| The Odds API | aiohttp via `_get_session` | `apiKey` **query parameter** (provider's documented design, not our choice) | same spoofed UA | `regions=eu`, `markets=h2h,totals`, `oddsFormat=decimal`, `dateFormat=iso` |

**The expected finding was NOT confirmed, and that is why it was checked.**
API-Football did not receive "one header and nothing else": `_get_session()` set
a session-level **spoofed Chrome User-Agent** that aiohttp merged into every
request.

Its only two consumers are API-Football and The Odds API — **both JSON APIs that
authenticate by key**. Flashscore uses Selenium/camoufox and never this session,
so the browser string served **no anti-blocking purpose**. It was a fabricated
client identity transmitted where none is needed or asked for. **Removed**;
aiohttp's own default now stands, which names the library and nothing about the
machine, the account or the repository.

**Second finding, smaller and exactly the named case:** `weather_service.py` sent
`User-Agent: betting-agent/1.0` to Open-Meteo at two sites — **a repository name
in a User-Agent**. Removed.

**Nothing else leaves the system.** No email, no hostname, no machine
identifier, no account data, no debug field, in any header, parameter or body to
any of the three providers. Recorded plainly so the question does not need
asking again.

## Per-source counts are now structural

`scripts/ci_audit.py` prints `disc[fs=N fdo=N af=N]` on **every** row of a
workflow that attempts discovery — verdict or not — and it belongs in the ledger
note. Verified on a real row:

```
32835286636 daily-picks 2026-08-25 DEGRADED  disc[fs=0 fdo=1 af=0]  5 active-season league(s) returned 0 fixtures…
```

That one string says what three months of aggregates did not: **Flashscore and
API-Football produced nothing while football-data.org produced one.**

`-` and `0` are deliberately different: a source that did not report is not a
source that reported nothing, and a `closing-lines` run shows no `disc[…]` at
all rather than a misleading `fs=0`.

## The per-source assertion, re-proven with a source restored

It was proven by replay against 2026-05-31, when a source was dead. **All three
may now be alive, which is the condition under which it must not go back to
sleep** — so the condition was *injected* rather than waited for.

| case | result |
| --- | --- |
| all three producing normal counts | **silent** |
| Flashscore forced to 0, other two healthy | **fires, naming Flashscore** |
| football-data.org forced to 0, other two healthy | **fires, naming football-data.org** |
| API-Football forced to 0, other two healthy | **fires, naming API-Football** |
| in each case, the two healthy sources | **not named** |

The second row is the 2026-05-30 failure exactly, and a restored API-Football is
precisely when it stops being hypothetical. **13 tests.**

## Cross-source agreement — measured, NO threshold shipped

Zero is the last symptom, not the first. A source quietly returning half its
fixtures would be caught by nothing today.

**Two-source baseline** (Flashscore vs football-data.org, 7 leagues × 7 days,
2026-08-27 → 09-02):

```
league-days where either source saw a fixture : 26
exact agreement                               : 26 / 26
both non-zero but different                   : 0
```

**Three-source baseline** (adding API-Football, 2 days, 2 requests of 100):

| league | date | FS | FDO | AF |
| --- | --- | --- | --- | --- |
| spain/laliga | 08-27 | 2 | 2 | 2 |
| spain/laliga | 08-28 | 2 | 2 | 2 |
| PL, SA, BL1, FL1, PPL, DED | 08-28 | 1 | 1 | 1 |

**8 / 8 exact, three-way.**

**No threshold is being shipped, and that is deliberate.** One week of one season
is not a basis for one — the same discipline that made the settled-pick segment
thresholds a finding rather than a feature. What the baseline establishes is that
**exact agreement is the normal state on overlapping coverage**, which is a much
stronger starting point than expecting noise. Before a check ships it needs:

- **per-league, per-source-pair** treatment — coverage genuinely differs
  (`europe/champions-league` is absent from *all three* this week, correctly)
- enough days to see whether disagreement is ever legitimate (kickoff-date
  boundaries near midnight UTC are the obvious candidate)
- a decision on what a *single* day's disagreement means versus a sustained one

Recorded as the baseline to compare against, not as a rule.

## Champions League — RETRACTED: this is a three-source zero, not a confirmation

**The paragraph that stood here was wrong and is corrected rather than deleted.**
It read: *"`europe/champions-league` returns 0 fixtures from all three sources
… the prediction's clause that a UCL zero is CORRECT is therefore confirmed …
It is genuinely between rounds, not broken."*

**Two errors, and the second is the serious one.**

**1. It inverted the clause it claimed to confirm.** The prediction says a UCL
absence *"must NOT be scored against the selector repair, and **must NOT be
rounded up into success either**."* That withholds judgement. It was rounded up
into success — the precise move the clause was written to forbid.

**2. It used cross-source agreement as corroboration.** Agreement between
sources is evidence only if the sources can fail independently. **A shared blind
spot produces identical agreement**, and that is the failure mode this entire
stage documents: a fallback that substitutes silently makes the primary's
failure invisible. `disc[fs=N fdo=N af=N]` was built this same day *because*
aggregates conceal exactly this — and then agreement was used as proof.

### What is actually known

- **Niki watched four Champions League matches on 2026-08-26.** Those fixtures
  existed.
- **`COMPETITION_MAP` contains `"CL": "europe/champions-league"`**, so
  football-data.org maps the competition and should have returned them.
- The 2026-08-26 run scraped `europe/champions-league` and logged
  `Flashscore returned 0 fixtures for europe/champions-league`.
- Loaded directly under camoufox, the UCL fixtures page yields **0 rows of any
  kind**, against 110–120 rows for every domestic league — a structurally
  different result, not merely an empty date.

> **`disc[fs=0 fdo=0 af=0]` on a day with four real fixtures is a FINDING, not a
> quiet day.** Three sources returned nothing for a competition that was
> playing, and one of them demonstrably maps it.

### Status: OPEN, and deliberately not investigated yet

The 2026-08-27 LaLiga test comes first and must not be disturbed. This is
recorded now, before the verdict, because **"independently confirmed" and
"unexplained three-source zero" read very differently to whoever picks it up** —
and the wrong one was on the record.

Whether the 08-27 → 09-02 window is legitimately empty (between rounds) is a
separate question from why 08-26 returned zero, and neither has been
established. **Nothing about UCL is confirmed.**

---

# RUN 33075828280 — THE STAGE 19 VERDICT

`Daily Betting Picks`, **`workflow_dispatch`** 2026-08-27T13:14:48Z,
`conclusion: success`. Audited 2026-08-27. **The 09:37 scheduled run never
fired** — this is the manual trigger only.

*The prediction was re-read in full before any log line was opened.*

## What was registered, restated before the evidence

| # | registered claim |
| --- | --- |
| 1 | **Exactly 2 fixtures**, both `spain/laliga`: Celta v Osasuna, Barcelona v Ath Bilbao — strictly, *"exactly 2 within the sampled set and within football-data.org's nine competitions"* |
| 2 | Kickoffs near **18:30Z / 19:00Z**, `EXTRACT(SECOND FROM match_date) = 0` on both |
| 3 | UCL zero **scored NEITHER way** (per the correction of 2026-08-27) |
| 4 | Per-source assertion silent for producing sources, fires for a source that produced within 7 runs and now returns zero |
| 5 | `model_version = stage5_baseline_20260807.b16ec7` (`s5.5`) |

**Outcome definitions:** RESTORED requires **Flashscore's own count for
`spain/laliga` to be non-zero** — not the total, not "the fixtures arrived".
PARTIAL–fdo if fixtures arrive without Flashscore. REGRESSION on any sub-minute
kickoff. **Precondition: `spain/laliga` attempted unconditionally**, which was
supposed to make UNTESTED impossible.

## VERDICT: DISCOVERY RESTORED

All three registered conditions met, literally:

```
Scraping fixtures: spain/laliga          13:20:59
Scraped 2 fixtures from spain/laliga     13:21:15
```

| condition | evidence |
| --- | --- |
| **Flashscore's own count for `spain/laliga` non-zero** | **`Scraped 2 fixtures from spain/laliga`** — verbatim |
| both fixtures present | id 50140 Celta v Osasuna 18:30, id 50141 Barcelona v Ath Bilbao 19:00 |
| whole-minute kickoffs | `EXTRACT(SECOND)=0` → **True** on both |
| precondition held | `Scraping fixtures: spain/laliga` present — **attempted, confirmed from the log, not inferred** |

**NOT a regression:** **105 matches were created today and ZERO carry a
sub-minute kickoff.** The phantom defect did not return, tested against 105
opportunities rather than 2.

### Who created what — the distinction the definition was built to survive

Row `created_at` is **13:20:57.2 / 13:20:57.9**; football-data.org logged
`added 2 new fixtures to DB` at **13:20:58**; Flashscore scraped at
**13:20:59 → 13:21:15** and both rows carry `flashscore_id`.

**football-data.org CREATED both. Flashscore INDEPENDENTLY FOUND both and
matched them.**

This is why the definition keyed on *Flashscore's own count* rather than on row
provenance: had it keyed on creation, the answer would have been
PARTIAL–fdo-only and the working parser would have been scored as a failure.
**Flashscore parsed 2 fixtures from the page — the repair works in CI, under
Chrome, not merely under camoufox.**

## Per source, verbatim

**`disc[fs=2 fdo=2 af=39]`**

| league | Flashscore | notes |
| --- | --- | --- |
| **spain/laliga** | **2** | the two predicted fixtures |
| 21 other leagues attempted | 0 each | no fixtures inside the 1-day cutoff |
| **europe/champions-league** | **no result line at all** | `Fixtures page failed (TimeoutException)` → retry → `retry also failed` |

## Prediction accuracy — the headline was overtaken, the strict claim held

**36 fixtures exist for 2026-08-27, not 2**: the 2 LaLiga plus **34 Europa
League / Conference League**, all created by API-Football between 13:26:15 and
13:27:42.

- The **strict** claim — *"exactly 2 within the sampled set and within
  football-data.org's nine competitions"* — **HOLDS**. UEL and UECL are in
  neither `COMPETITION_MAP` nor the eight leagues sampled.
- The **headline** — "exactly 2 fixtures discovered" — **does not**. It was
  written while API-Football was suspended; a restored account found 34 more.

Recorded both ways rather than resolved in the flattering direction.

**Item 5 is stale, not failed:** picks carry
`stage5_baseline_20260807.60caed` (`s5.6`). The prediction registered `b16ec7`
(`s5.5`) and the s5.6 bump was made *after* it was written. Bookkeeping
consequence of the bump sequence, not a discrepancy in the run.

## UCL — evidence for the OPEN entry, NOT investigated

`europe/champions-league`: **`disc[fs=-]`** — attempted, timed out twice,
produced no count line. It consumed **13:21:25 → 13:22:55 = 90 seconds, 30% of
the entire fixtures budget**, and returned nothing.

**And UEFA fixtures plainly existed today: API-Football created 34 of them.**
That is direct evidence for the open three-source-zero question, since Flashscore
and football-data.org returned none of the 34. **Deliberately not investigated —
the entry stays OPEN, now with this run attached to it.**

## Registered post-run measurements

### Trailing-league set — non-empty, and it bit on day one

Fixtures block: **13:20:59 → 13:26:00 = 301s**, and
`Flashscore fixtures: time budget exhausted, skipping remaining leagues`
**fired**. Budget remaining: **zero**.

**23 of 30 configured leagues attempted. Never attempted:**

```
england/league-two   spain/laliga2      germany/2-bundesliga   italy/serie-b
france/ligue-2       europe/europa-league   europe/europa-conference-league
```

> **The trailing set contains `europe/europa-league` and
> `europe/europa-conference-league` — the two competitions carrying 34 of
> today's 36 fixtures.** The residual predicted in advance bit immediately, on
> exactly the competitions that mattered.

**Decision rule, applied as registered:** the set is **non-empty**. Stability
needs a second run and cannot be claimed from one. But the run already names the
dominant cost: **UCL alone consumed 90s of 300s producing nothing**, which points
at *per-league cost* rather than ordering — the registered rule's "varying →
attack the per-league cost" branch. **Rotation would move which leagues are
starved, not how much budget a 90-second timeout wastes.**

### The new API-Football account, as the run saw it

`API-Football update complete (51 requests used, 3 fixture(s) SKIPPED on
team-identity mismatch)`. **51 of 100 requests. Zero `errors.access`, zero
`errors.plan`, zero suspension messages.** The replacement account authenticated
and worked.

### §7 IS NO LONGER UNTESTABLE

Three `ERROR` lines, all the Stage 13 identity gate:
`TEAM IDENTITY MISMATCH — refusing to resolve` (API-Football ids 604, 3502, 531).
**The gate ran, fired, and failed closed on 3 fixtures.** A gate that never ran
is not a gate that passed — it has now run.

### The per-source assertion

**Silent**, correctly: all three sources produced (`fs=2 fdo=2 af=39`).

**But the run exposes a defect in the OTHER check I shipped.** The
`fixtures_zero_active` assertion fired **21 times**, and all 21 are false
positives: those leagues genuinely had no fixtures inside `max_days_ahead=1`.
The scraper's own `expected ≥1 for active season` warning does not account for
the 1-day cutoff, and my assertion inherited that. **On a normal day this check
will fire ~21 times, which is how a check gets ignored.** Recorded, not fixed.

### Fail-closed date parsing — working, and loudly

**139 `REFUSING fixture row … kickoff time did not parse` warnings**, all with
`raw='<no time element found>'`, concentrated in five leagues:

```
austria/bundesliga 36   bulgaria/efbet-league 35   denmark/superliga 30
switzerland/super-league 24   greece/super-league 14
```

**Those are precisely the leagues that manufactured phantoms on 2026-08-26.**
The rows that used to enter the database with a `now()` kickoff are now refused
and named. **Zero phantoms among 105 matches created confirms it end to end.**

Two caveats recorded rather than resolved: `_raw_time_text` reports
`<no time element found>` because it still looks only for the retired
`.event__time`, so the refusal message cannot show what it failed to parse; and
whether any refused row was an in-window fixture is **not established** — no
in-window fixture is known to have been lost, but that is absence of evidence.

### Substrate accumulation, and the Stage 18 trigger

| | 2026-08-26 | now |
| --- | --- | --- |
| `odds_snapshots` rows | 85 | **2,713** |
| distinct matches | 1 | **35** |
| **keys with ≥3 observations** | 0 | **0** |
| `injury_observations` | 0 | **4** |
| `odds.first_seen_at` populated | 0 | **2,628** |

**The substrate is alive** — 2,628 rows and 2,628 first-seen stamps in one run,
against a total of 85 before it.

**The trigger has NOT fired: zero keys carry three observations.** A run gives
each key its *first* sight; the second and third come from `closing-lines`
refreshes before kickoff.

**Trigger dates are NOT re-issued as calendar dates.** One run is not a rate, and
the last three attempts to put a date on this were withdrawn within a day. The
trigger remains the query:

```sql
SELECT match_id, bookmaker, selection, count(*) FROM odds_snapshots
GROUP BY 1,2,3 HAVING count(*) >= 3;
```

What *can* be said: 34 picks exist across 34 fixtures, so those leagues are now
`_pending` and will be refreshed by tonight's `closing-lines` runs. **The first
honest measurement of the rate is available after those run** — not before.

## Pipeline counts, checked against what each step should produce

| | |
| --- | --- |
| picks saved | **34**, across **34 fixtures** — 1:1, the `max_picks_per_match` cap holding |
| `pick_observations` | **68** = 2 × 34 exactly |
| review coverage | **34 of 34** — no unreviewed picks |
| discarded review decisions | **0** |
| `PICK_REJECTED` | 5, all `reason=same_fixture_limit` — the cap, working |
| Telegram | 3 sent, **0 failed** |
| tracebacks | **0** |
| settlement | 2 warnings: `stale date(s) but only N API requests left — skipping stale-result fetch to preserve odds` |

## Trigger independence — what holds for a scheduled run and what does not

**Holds regardless of trigger:**

- **The verdict.** `Scraped 2 fixtures from spain/laliga` is a statement about
  the parser and the page, not about the hour. The selector and date-shape
  repairs are trigger-independent.
- Zero phantoms across 105 matches.
- The identity gate firing 3 times.
- The new account authenticating with no access/plan errors.
- The 21 false-positive warnings — a property of the check, not the clock.

**Depends on the trigger, and is NOT generalised:**

- **The trailing set.** `_FIXTURES_BUDGET_S` truncation depends on per-league
  latency on the day. The *names* may differ on another run; that the set is
  non-empty is what this run establishes.
- **Which fixtures were in-window.** `max_days_ahead=1` was computed from 13:14,
  giving a window to 13:14 on 08-28 — which excludes the 08-28 evening kickoffs
  that a 09:37 run would also have excluded. So the 21 zero-fixture leagues are
  **not** an artefact of the manual hour, but a run at, say, 20:00 would include
  them and the count would differ.
- **Odds imminence and pending picks.** A 13:14 run makes a different set of
  fixtures imminent than 09:37, which affects the closing-lines interaction and
  the 51 API-Football requests — not compared against a scheduled baseline here.
- **The missed 09:37 schedule itself** is unexplained and is not this run's
  evidence to give.

### Ledger row

| run_id | workflow | started (UTC) | conclusion | steps failed | audited on | verdict | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 33075828280 | daily-picks | 2026-08-27 13:14 | success | — | 2026-08-27 | **DEGRADED** | `disc[fs=2 fdo=2 af=39]`. **STAGE 19 = DISCOVERY RESTORED**: `Scraped 2 fixtures from spain/laliga`, both whole-minute. 34 picks / 34 fixtures / 68 obs / 34 reviewed. 105 matches created, **0 sub-minute**. DEGRADED for: 21 false-positive `active-season` warnings (the check ignores `max_days_ahead=1`); UCL double timeout consuming 90s of 300s; budget exhausted with **7 leagues never attempted incl. europa-league + conference-league**, which carried 34 of today's 36 fixtures. Identity gate fired 3× (§7 no longer UNTESTABLE). AF 51/100 requests, no access/plan errors. 139 fail-closed date refusals in the 5 phantom-producing leagues. |

---

# STAGE 20 — THE LOOP'S THREE DEFECTS

`cohort_status.py` reported **BUMP** (s5.6 carries 34 picks), so Parts A and B —
both selection-affecting — land in **one** bump to `s5.7`. Part C changes only
logging and an audit pattern and is selection-neutral.

## PART A — the three identity-gate refusals

### A1. Each refusal, verified against the provider rather than recalled

All three fired the **lexical anchor** check (`:1523`), none the country check.
Classified by querying `GET /teams?id=` — the same standard the gate itself uses,
the payload in hand:

| AF id | provider says | stored row | check | verdict |
| --- | --- | --- | --- | --- |
| **604** | Maccabi Tel Aviv, **Israel**, Tel Aviv, founded 1906 | 124 `Telstar`, Netherlands, eredivisie | anchor | **CORRECT REFUSAL** |
| **531** | Athletic Club, **Spain**, **Bilbao**, 1898, code `BIL`, San Mamés | 58 `Ath Bilbao`, Spain, spain/laliga | anchor | **FALSE POSITIVE** |
| **3502** | FC Iberia 1999, **Georgia**, **Tbilisi**, founded **1999** | 1218 `Saburtalo`, a Tbilisi club founded 1999 | anchor | **FALSE POSITIVE** |

**The Athletic false positive cost a predicted fixture.** Barcelona vs Ath Bilbao
is one of the two fixtures Stage 19 registered; its row carries
`apifootball_id = NULL` because the gate refused it. **It exists only because
Flashscore found it independently** — the redundancy that Stage 19 proved was
working is the only reason the loss is invisible in the fixture count.

Independent corroboration for 531, from this project's own measurements taken
before the run: **football-data.org named the fixture `FC Barcelona vs Athletic
Club` while Flashscore named it `Barcelona | Ath Bilbao`.** Two sources, two
names, one fixture.

### A2. THE KNOWLEDGE FOR THE ATHLETIC CASE ALREADY EXISTED

The first attempt added `"athletic club"` to `team_names.NAME_ALIASES` — and the
HABIT guard written alongside it **failed immediately**, because:

```
TEAM_NAME_ALIASES["Athletic Club"] = "Ath Bilbao"     # already present
```

It never fired. `_get_or_create_team_id` matches by `apifootball_team_id` at
**step 0** and refuses there; the alias table is consulted at **step 2**, which
is only reached when step 0 finds nothing. **The alias was present and
unreachable from the gate that needed it.**

So the fix is not a new alias but a **reachability** change: canonicalise through
`TEAM_NAME_ALIASES` *before* the anchor test. **Only ONE alias was genuinely
missing** — `FC Iberia 1999 → Saburtalo`.

**The anchor rule itself is untouched.** No ratio, no threshold, no widened
country band. `team_names._tok_match` carries a `SequenceMatcher` ratio ≥ 0.75
and was deliberately **not** imported for that reason.

**THE HABIT, sixth instance, found and not extended.** Two alias tables already
exist — `TEAM_NAME_ALIASES` (177 entries, "API-Football name → historical name")
and `team_names.NAME_ALIASES` (22), already overlapping on `psg` and
`olympiakos piraeus`. The Stage 20 entry went into **one**, and
`test_the_alias_knowledge_lives_in_exactly_one_table` fails if it is duplicated.
`_ALIAS_LOWER` is *derived* from the table rather than maintained beside it, so
the index cannot drift from what it indexes.

### A3. The size of the problem — one day, not a rate

**Every identity-gate refusal since the gate shipped:**

```
2026-08-27 : 3       (all other days: none)
distinct pairs : 3   recurrence : none
```

That is the gate's **complete live history**. API-Football was suspended from
2026-08-19 until the replacement account went live on 08-27, so the gate had no
traffic to refuse. **Three refusals on one day is not a rate and no clustering or
recurrence can be claimed from it** — the honest statement is that the population
is one day deep. If the next runs produce recurring pairs, those aliases belong
in the table whether or not they appeared here.

### A4. Replayed to the standard the gate was held to

14 tests. Both false positives now resolve; **the Telstar/Maccabi impostor is
still refused**; `Pau FC` / `St. Pauli` still refused; `Rapid Vienna` /
`Rapid Bucuresti` still shares an anchor **by design** and is pinned as such, with
a separate test confirming the **country** check is what separates them. Missing
country information (`Europe`, `World`, `Other`, empty) still falls through
rather than refusing — which matters, because `Saburtalo` is stored as `Europe`.

## PART B — 95% of the budget bought nothing

### B1. Measured before touching the budget

Per-league durations from run 33075828280:

| | |
| --- | --- |
| total | **301.7s** of a 300s budget |
| **spent on leagues returning ZERO fixtures** | **285.7s — 95%** |
| leagues reached | 23 of 30 |
| mean per league | 13.1s |
| **mean excluding champions-league** | **9.6s** |
| **champions-league alone** | **90.2s — a 9.4× outlier** |
| fastest (fast-fail already possible) | finland/veikkausliiga, **1.2s** |

The 90.2s is two `WebDriverWait(driver, 45)` calls timing out back to back —
initial attempt and retry.

**Raising the budget treats the symptom.** Freeing 75s of the 90 reaches ~8 more
leagues at the 9.6s mean, which covers the 7 that were never attempted.

**Fixed:** `FIXTURES_WAIT_S = 20`, one definition, used by both call sites. Every
league that returned rows completed in **7.2–16.0s end to end**, so 20s leaves
margin over the slowest observed success. Expected recovery **50–70s per run**.
Whether that is sufficient is **not asserted** — it is the next run's
measurement, and the registered decision rule stands.

### B2. The page inspection — and it CORRECTS my own earlier finding

I recorded that the UCL fixtures page "returns zero rows of any kind, against
110–120 for every domestic league — an empty date still renders a table, so zero
rows means the page is not the shape the parser expects."

**That does not reproduce.** Loaded directly today:

| page | HTTP | `.event__match` rows |
| --- | --- | --- |
| `champions-league/fixtures/` | 200 | **144** |
| `europa-league/fixtures/` | 200 | 5 |
| `europa-conference-league/fixtures/` | 200, **redirects to `/conference-league/`** | 10 |

**The page is fine and the selectors are fine.** 144 rows found with the current
selector, 2,146 `wcl-*` classes, correct title. **No selector work is warranted,
and the discipline that found the `--static` rename is what prevented it here
too — the page was inspected first and the fix it would have justified turned out
to be unnecessary.**

Yesterday's zero was measured in an 8-league loop with a 3.5s settle; today's
144 came with 5s. So the earlier reading was a load-timing artefact of my own
harness, not a property of the page. **A single anomalous result was evidence
about the measurement, and the measurement was mine.**

**What remains unexplained:** a 45s `WebDriverWait` failing in CI where a ~5s
direct load succeeds. That is not slow rendering, and it is a Chrome/Selenium
question rather than a page question. Recorded, not pursued.

**Incidental finding:** `europe/europa-conference-league` **redirects** to
`/football/europe/conference-league/`. It works, and it costs a redirect
round-trip on every attempt. Recorded, not changed — the configured key is a
selection-affecting surface and the redirect is harmless.

### B3. Ordering

The trailing set after this change is **not asserted**. It is the next run's
measurement, and the registered rule requires it by name plus a stability check
across runs. What this run established is that truncation is real, deterministic
by config order, and that it starved `europe/europa-league` and
`europe/europa-conference-league` — **the two competitions carrying 34 of the
day's 36 fixtures.**

## PART C — a check that fires 21 times a day carries no information

All 21 firings on 2026-08-27 were false positives, and the old pattern fired
**260 times across the whole cached period**.

**Root cause: the scraper's own warning ignores the window it queried for.**
`_scrape_fixtures_page(url, max_days_ahead=1)` returns fixtures inside one day; a
league with none is *quiet*, not broken. The warning said "expected ≥1 for active
season" regardless, and the audit assertion inherited every false positive.

This is **zero is only an anomaly relative to what was asked for** — a rule this
project recorded and then broke in the check written after it. It is also DEL-1
in a third form: **an alert that never arrives and an alert that always fires
carry the same information, which is none.**

**Fixed at the source, not at the reader.** `_scrape_fixtures_page` now records
`_last_page_rows` — the rows the PAGE yielded, before the window filter — and the
warning distinguishes three cases:

| page rows | outcome |
| --- | --- |
| **0** — the page gave nothing | **WARNING** (a real anomaly) |
| **> 0**, none in range | INFO — a quiet league |
| **unknown** | **WARNING — fails closed** |

The unknown case is deliberate: a check that assumes the benign case when it does
not know is the failure this project keeps cataloguing. `_last_page_rows` is
reset before each league so a previous league's count cannot silence this one.

**Replay, with its limit stated.** The new pattern matches **0** cached logs —
but that is a property of the text change, **not** evidence the check is quiet.
The honest prediction from measurement: of the 21 leagues that warned, camoufox
showed 110–120 page rows for the domestic ones and 0 only for UCL, so on a
comparable day the count should fall from **21 to ~1**. **That is a prediction,
and the next run tests it.**

---

**STAGE 20 — LOOP REPAIRED**, with the four registered measurements outstanding.

`CODE_REVISION = s5.7`, `model_version = stage5_baseline_20260807.645bac`.
855 tests pass; 26 experiment invariants pass.

To be reported from the next run, and not asserted now:

1. **fixture count per league**
2. **the trailing set by name**, and whether it is stable
3. **the identity gate's refusal count**, each classified
4. **how many times `fixtures_zero_active` fires** — predicted ~1, was 21

---

# GUARD DESIGN — THE HABIT, INVERTED

**The Stage 20 Part A finding, promoted to its own heading because five
instances of the familiar shape did not predict it.**

`TEAM_NAME_ALIASES["Athletic Club"] = "Ath Bilbao"` existed. It was correct,
canonical, curated, and in the right module. It never fired, because
`_get_or_create_team_id` decides at **step 0** — match on
`apifootball_team_id`, verify, refuse — while the alias table is consulted at
**step 2**, reached only when step 0 finds nothing.

**The five recorded instances were two definitions drifting apart.** This is one
definition placed where the deciding caller never looks. **Same root — knowledge
in the wrong place — opposite symptom.** Nothing drifted; nothing was
duplicated; a grep for the alias would have found it and concluded the case was
handled.

> ## A lookup table is only as good as the earliest decision point that consults it.
>
> **Any decision taken before the canonical source is read is made in ignorance
> by construction** — not by oversight, not by drift, and not visibly. The table
> can be complete and correct and still be irrelevant to the caller that
> matters.

**Why the familiar guard would not have caught it.** The HABIT tests ask "does
this knowledge exist more than once?" Here it existed exactly once. The question
that catches this one is different: **"is it read before, or after, the first
decision that needs it?"** — a control-flow question, not a duplication question.

**How it surfaced, which is worth keeping.** Not by analysis. The first Stage 20
fix added the alias to the *other* table, and the HABIT guard written alongside
it failed on the duplicate — which is the only reason anyone looked at why the
existing entry had never fired. **A guard written for one failure mode found its
inverse by refusing a wrong fix.**

## Recorded and NOT opened: are there other early-exit paths?

`_get_or_create_team_id` decides at step 0 before reading the alias table. **The
question this raises is whether other paths do the same** — decide, return, or
refuse before consulting canonical knowledge that sits further down.

Candidate shape to look for: any `return` or `continue` that precedes a lookup
into `TEAM_NAME_ALIASES`, `NAME_ALIASES`, `MARKET_SPECS`, `COMPETITION_MAP`,
`LEAGUE_TO_THEODDS_SPORT` or `off_season_leagues`.

**This is the same audit shape as MASK-1 and the 23 sites, and it is not this
stage's work.** Recorded so it is a known question rather than a future
surprise.

---

# STAGE 20 — REGISTERED MEASUREMENTS, extended

Two additions to the four already registered, both falsifiable, both before the
run rather than after.

## 5. The timeout risk — a prediction, not a hope

**`FIXTURES_WAIT_S = 20` was derived from the wrong distribution.** The 7.2–16.0s
range is measured on leagues that **succeeded**. The failures it cuts took 45s
*because they were timeouts*, so they carry **no information about the tail of
the success distribution**. A league that would have succeeded at 25s is
invisible in that evidence.

So 20s can convert **slow successes into fast failures**.

> **PREDICTION: every league that produced a non-zero fixture count on
> 2026-08-27 must produce a non-zero count again.**
>
> Baseline, from run 33075828280: **`spain/laliga` = 2. It was the only
> non-zero league**, so that is the whole of the direct check — stated plainly
> rather than dressed up as broader coverage than it has.
>
> **If it drops to zero while fixtures exist for it, the timeout is too tight,
> and the correct response is to RAISE it — not to conclude the league went
> quiet.**

**The symptom is NOT indistinguishable, and Part C is why — accidentally.**
Verified in the code rather than assumed: `_last_page_rows` is assigned *after*
the `WebDriverWait`, so a `TimeoutException` raises before it is ever set, and
it was reset to `None` before the league began.

| outcome | log signature |
| --- | --- |
| **timeout** | `Fixtures page failed (TimeoutException)` **and** a WARNING (page rows unknown → fails closed) |
| **genuinely quiet** | INFO `no fixtures within the requested window (N row(s) on the page, none in range)` |

**A zero with a row count is a quiet league. A zero without one is a timeout.**
That discriminator was not designed — it fell out of Part C's fail-closed
unknown case, and it is what makes prediction 5 checkable at all.

**Also to record:** per-league wall time for every league. **Any league
completing in 16–20s sits in the band the evidence never covered**, and its zero
must be checked against an independent source before being accepted.

## 6. The redirect must land on the competition that was asked for

`europe/europa-conference-league` **redirects** to
`/football/europe/conference-league/`. It returns HTTP 200 and 10 rows, so it
resolves — but it was recorded as incidental, and that is too generous given
UEL and UECL carried **34 of 36 fixtures** and were never attempted.

**A redirect that resolves to the right competition is fine. One that resolves
to a different competition, or to a season index, would produce rows that PARSE
AND ARE WRONG — which is worse than zero, and would not announce itself.**

> **CHECK, on the next run in which these leagues are reached:** for
> `europe/europa-league` and `europe/europa-conference-league`, confirm the
> fixtures actually created belong to the competition requested — by team
> names against the public calendar, not by the count alone. A plausible count
> of the wrong competition's fixtures is exactly the failure this check exists
> for.

The same question applies to any other configured key whose URL redirects; that
has not been enumerated and is not claimed to be enumerated.

---

# OPS-3 — SCHEDULED FIRINGS MISSED (open, watching)

Recorded against the open entry from the Stage 19 verdict, where the missed
09:37 `daily-picks` schedule was noted as unexplained.

## 2026-08-27 — measured, not reported

Both manual triggers are confirmed: `daily-picks` ran `workflow_dispatch` at
13:14 and `paper-trading-report` at 17:50; neither has a `schedule` run today.

**But the count is larger than two, and the "not repository-wide" reading does
not survive the check.** Due-by-17:51 UTC against actual `schedule` runs:

| workflow | cron(s) due today | due | fired | missed |
| --- | --- | --- | --- | --- |
| Daily Betting Picks | 09:37 | 1 | **0** | **1** |
| Paper Trading Report | 10:47 | 1 | **0** | **1** |
| **Closing Line Capture** | 10:47, 11:17, 13:17, 15:17, 17:17 | **5** | **1** | **4** |

**6 of 7 due firings missed, across all three workflows.**

**The one closing-lines run does not rescue it.** It fired at **04:30**, which
matches **no closing-lines cron** — the schedule is `47 10` and `17 11,13,15,17,
19,21,23`. The nearest preceding cron is **23:17 on 2026-08-26**, making it a
firing **5h13m late**, inside the 0.5–5.7h GitHub scheduler delay already
documented for this repository.

> **On 2026-08-27, no cron has fired at its own hour for any workflow.** The
> last scheduled firing that landed near its slot was **2026-08-26 22:11** (the
> 21:17 cron, 54 minutes late).

**So the pattern is repository-wide, not confined to two workflows.** It reads
as one scheduler affecting every workflow in the repository rather than
something specific to `daily-picks` and `paper-trading-report` — which is a
different hypothesis, and a broader one.

## Why it is being watched rather than investigated

One miss is GitHub's queue. **Six on one day, across three workflows and seven
distinct clock hours, is a pattern** — but this repository has a *documented*
0.5–5.7h delay distribution, and every one of today's misses is still inside a
window where a very late firing could arrive. **A missed cron and a cron delayed
past the next audit are indistinguishable at the moment of looking**, which is
exactly why this is a watch and not yet a finding.

**Escalation criterion, fixed now so it is not adjusted later:**

> **If 2026-08-28 also produces no on-time scheduled firing for any workflow, it
> stops being load and becomes something to investigate.** Two consecutive days
> of repository-wide silence is outside anything this repository has recorded.

**What to check when that happens, recorded so the investigation starts from a
list rather than from scratch:** GitHub's own incident history for Actions; the
60-day inactivity auto-disable for scheduled workflows (the repository has been
pushed to repeatedly today, so this is unlikely but is cheap to exclude);
whether the crons still parse after the Stage 20 edits — `closing-lines.yml`
gained the `47 10` entry in Stage 15 and the file has been edited since; and
whether the default branch is what the schedules run from.

**Not investigated today. Recorded with the count corrected.**

---

# OPS-3 — CORRECTED. Nothing was missed; everything was ~10 hours late.

**Read-only verification, 2026-08-28. No cron, concurrency setting or workflow
was changed.**

## The correction, first

The OPS-3 entry above recorded **"6 of 7 due firings missed"** on 2026-08-27,
assessed at 17:51 UTC that day. **Every one of them subsequently fired.**

| workflow | cron | fired (schedule) | late by |
| --- | --- | --- | --- |
| Daily Betting Picks | 09:37 | **19:58** | **10h 21m** |
| Paper Trading Report | 10:47 | **20:45** | **9h 58m** |
| Closing Line Capture | 10:47 | **20:47** | **10h 00m** |

**Nothing was missed. The delay envelope is ~10h21m — nearly double the 0.5–5.7h
previously documented for this repository.**

**The entry warned about exactly this failure and then committed it anyway.** It
said: *"a missed cron and a cron delayed past the next audit are
indistinguishable at the moment of looking, which is exactly why this is a watch
and not yet a finding"* — and then tabulated a count of "missed". The word was
wrong at the moment it was written, and the table read as evidence.

## The three causes, separated

### 1. Scheduler delay — CONFIRMED, and it is the only cause with evidence

The envelope must be restated: **~10h21m observed**, against 5.7h previously
recorded. Every 08-27 firing arrived; none was lost to this cause.

### 2. Concurrency queueing behind a manual run — NOT SUPPORTED

**The three workflows are in DIFFERENT concurrency groups** — `daily-picks`,
`closing-lines`, `paper-trading-report`, each `cancel-in-progress: false`. **A
manual run of one workflow cannot queue a scheduled run of another.** Cross-
workflow queueing is impossible by construction, so the hypothesis can only
apply within a workflow.

Within a workflow it does not hold either:

- **Only 4 manual runs exist in the last 120** — 2026-08-27 at 13:14
  (daily-picks), 17:50 (paper-report), 17:55 (closing-lines), and 2026-08-23 at
  14:46 (daily-picks). **None on 08-24, 08-25 or 08-26.** The premise that
  manual triggering has been going on for two days is not supported by the run
  history.
- **Every 08-27 manual run came AFTER its own workflow's due slot** (13:14 vs a
  09:37 cron; 17:50 vs 10:47; 17:55 vs 10:47). A run that starts later cannot
  have blocked one that was due earlier.
- **`closing-lines` missed the most and had no manual run until 17:55**, after
  four of its five due slots had already passed.
- `daily-picks`' manual run finished around 13:40; the scheduled run fired at
  **19:58**, six hours later. It was not waiting on the group.

**Concurrency queueing explains none of the observed lateness.**

### 3. Pending-run displacement — PLAUSIBLE for `closing-lines`, and NOT self-inflicted

**Zero cancelled runs across the last 120** (all 120 conclude `success`). That is
weak evidence on its own, because a displaced pending run may leave no record at
all — which is precisely what makes this hypothesis hard to falsify.

But the counts point somewhere:

| workflow | due 2026-08-27 | scheduled firings delivered |
| --- | --- | --- |
| Daily Betting Picks | 1 | 1 |
| Paper Trading Report | 1 | 1 |
| **Closing Line Capture** | **8** (10:47 + 17-past seven odd hours) | **~3** (20:47, 21:00, and 05:21 on 08-28) |

**The two single-cron workflows lost nothing. The eight-cron workflow delivered
about three.** That is consistent with GitHub holding at most one pending run
per workflow: while one firing sits pending for ten hours, later crons come due
and coalesce rather than accumulate.

**If that is what happened, the cause is the DELAY, not manual triggering.** The
only manual `closing-lines` run was at 17:55, after most of the day's slots. **So
this is not self-inflicted and would not be fixed by triggering less** — a
correction to the framing that prompted this check.

## The escalation criterion was wrong, and is replaced

The criterion recorded yesterday: *"If 2026-08-28 also produces no on-time
scheduled firing for any workflow, it stops being load."*

**It is unfalsifiable too early.** Assessed now (2026-08-28 13:38 UTC), today has
produced one `closing-lines` run at 05:21 and nothing else — which *looks* like a
repeat. But 08-27 established that firings arrive up to **10h21m** late, so a
09:37 cron cannot be judged before ~20:00. **Checking four hours in and calling
it a miss is the same error the correction above is about.**

> **REPLACEMENT CRITERION.** A firing counts as MISSED only if it has not arrived
> **12 hours** after its cron — a margin above the largest delay yet observed
> (10h21m). Assessment for a given day therefore happens no earlier than
> **12h after the last cron due that day**, not during it.
>
> Escalate only when a firing is missed under that definition. **Lateness, however
> extreme, is cause 1 and is GitHub's queue.**

## What is genuinely open

- **The delay envelope has roughly doubled** (5.7h → 10h21m). That is the finding
  from these two days, and it is about GitHub, not this repository.
- **Whether `closing-lines` loses firings to coalescing** when a pending run sits
  for hours. Its 8-due/~3-delivered ratio on 08-27 is the only evidence, it is
  one day, and displacement leaves no record — so it is recorded as plausible and
  unproven.

**Neither is investigated here, and nothing was changed.**

## A separate, real defect found while checking the crons

`paper-trading-report.yml`'s header documents the schedule as:

```
#   11:17 UTC  closing-lines — first capture slot of the day
#   10:47 UTC  THIS JOB      — after settlement, before the first capture
```

and its cron comment claims *"~30 min before the first capture"*.

**Both are false.** `closing-lines.yml` now carries `47 10 * * *` as its **first**
cron, so the first capture slot is **10:47** and the two jobs are **simultaneous**,
not 30 minutes apart.

**I introduced this.** The `47 10` entry is the Stage 19 L4 early window, added
to `closing-lines.yml` without checking what depended on the 11:17 slot.

**It matters beyond documentation, exactly as suspected.** The header states the
design intent plainly — *"Running between the daily pipeline and the first
capture means the report always describes a settled, quiet state rather than one
mid-write."* Running simultaneously with the first capture means the report can
read **mid-capture**: some picks with closing observations written, others not,
producing coverage figures for a state that never settled.

**Not changed** — this is a read-only check, and both the cron and the comment
are decisions rather than typos. **Recorded as a live defect, not as
documentation drift**, and it is the second time a Stage 19/20 schedule edit has
had an unexamined downstream reader.

---

## OPS-3 addendum — the ordering was never enforced, and the gap never guaranteed it

**The stale header is the symptom. The design assumption underneath is the
defect, and it predates the collision.**

The `paper-trading-report` header states the requirement plainly: the report must
describe *"a settled, quiet state rather than one mid-write"*. What was supposed
to guarantee that was a **30-minute clock gap** — report at 10:47, first capture
at 11:17.

**Against a measured delay envelope of 10h 21m, a 30-minute gap guarantees
nothing.** The scheduler's variance exceeded the margin by a factor of ~20, and
it did so long before the L4 window collided with it.

**MEASURED, 2026-08-27, both firing ~10h late:**

| | |
| --- | --- |
| Paper Trading Report (schedule) | **20:45:50 → 20:46:16** |
| Closing Line Capture (schedule) | **20:47:08 → 20:47:43** |
| nominal cron gap | 1,800s |
| **actual separation** | **52s** |
| compression | **97%** |

**The ordering held, and it held by 52 seconds of luck.** Two firings whose
crons are half an hour apart landed within a minute of each other. Nothing
enforced the sequence; the margin simply happened not to be consumed.

**So there were never two states — nominal-and-safe, then collided-and-unsafe.
There was one: an ordering that was hoped for.** The L4 window turned a nominal
ordering into a simultaneous one, and neither was ever real.

> **"Restore the 30-minute gap" is the fix that looks obvious and restores
> nothing.** It would return the schedule to a margin already demonstrated to be
> 20× too small, while reading as though the problem were solved.

**The actual requirement, recorded so the next person does not move a cron and
believe it is fixed:** if the report must observe a settled state, it needs a
**dependency** — `workflow_run` on the capture completing, a state check that
refuses to report while a capture is in flight, or a claim on the same
concurrency group — **not a time gap**. That is a design change and it is not
this stage's work.

## PROPOSAL (not built — this stage is read-only): pin the schedule contract

**A schedule is an interface.** Changing a cron changes a contract other
components depend on — and unlike code there is **no compiler, no import graph,
and nothing to grep**. The dependents are expressed in prose comments and in
implicit ordering assumptions that no test asserts. That is why this has now
happened twice.

**The guard:** parse the crons from the three workflow files and assert the
ordering invariants the comments claim. Concretely, the claims currently made in
prose:

- `daily-picks` runs before `paper-trading-report` (settlement before reporting)
- `paper-trading-report` runs before the **first** `closing-lines` slot
- the first `closing-lines` slot is the earliest of its crons

**It would have failed the moment `47 10` was added to `closing-lines.yml`** —
the edit that created this defect — and it fails again on the next edit that
contradicts a documented sequence.

**Its limitation, stated rather than discovered later: it cannot enforce the
ordering at runtime.** The scheduler defeats that, as the 52-second measurement
above proves. **It pins the CLAIM, not the behaviour** — so a change to the
claim becomes deliberate rather than accidental, and the comment cannot drift
away from the crons.

That is the same shape as the exemption-count pin and the `filter_generation`
digest: **the guarantee is unenforceable, so pin the claim and make changing it
an explicit act.** A guard that made the runtime ordering true would be a
different and much larger thing, and it is the dependency described above.

**Recorded as a proposal. Not built here.**

---

# AUDIT 2026-08-30 — 24 runs, and lateness priced

Read-only. No cron, concurrency setting or workflow changed.

## PART A — the audit

24 runs, 2026-08-26 → 2026-08-30: **20 CLEAN, 4 DEGRADED, 0 BROKEN.** All
`conclusion: success`, which remains no evidence of anything.

| run | workflow | started | verdict | notes |
| --- | --- | --- | --- | --- |
| 33100876420 | closing-lines | 08-27 17:55 | **DEGRADED** | 2 league requests returned `no_rows` — credits spent, empty event list |
| 33111032974 | daily-picks | 08-27 19:58 | **DEGRADED** | `disc[fs=19 fdo=0 af=1]`; **NO FIXTURES FOUND** — nothing analysed |
| 33210509071 | daily-picks | 08-28 20:58 | **DEGRADED** | `disc[fs=91 fdo=0 af=10]`; **NO FIXTURES FOUND** — nothing analysed |
| 33258220733 | daily-picks | 08-29 14:40 | CLEAN | `disc[fs=71 fdo=0 af=11]` |
| 33260734153 | closing-lines | 08-29 15:36 | **DEGRADED** | 2 credits claimed, 0 closing lines captured |
| 19 others | closing-lines / paper-report | — | CLEAN | |

**`fdo=0` on all three daily-picks runs.** football-data.org added no fixtures on
any of the three days. Recorded, not pursued.

### Stage 20's four registered measurements

**1. Fixture count per league** — Flashscore now returns fixtures broadly, which
it had not done since 2026-05-30:

| run | non-zero leagues | total fixtures |
| --- | --- | --- |
| 08-27 19:58 | 15 | 19 |
| 08-28 20:58 | 20 | 91 |
| 08-29 14:40 | 21 | 71 |

**The Stage 20 prediction holds: `spain/laliga` produced non-zero on every run**
(2, 3, 3). The timeout reduction did not convert a slow success into a fast
failure — the specific risk registered against `FIXTURES_WAIT_S = 20`.

**2. The trailing set — NOT stable. It varies.**

| run | trailing | names |
| --- | --- | --- |
| 08-27 | 6 | league-one, league-two, laliga2, 2-bundesliga, serie-b, ligue-2 |
| 08-28 | 3 | 2-bundesliga, serie-b, ligue-2 |
| 08-29 | 4 | serie-b, ligue-2, **europa-league, europa-conference-league** |

Only `italy/serie-b` and `france/ligue-2` trail on all three days. **The budget
was exhausted on all three runs even after the timeout cut**, and leagues
attempted rose 23 → 24, 27, 26.

**Registered decision rule applied as written:** the set **varies**, so the rule
says *attack the per-league cost, not the ordering*. Rotation would not help — it
would merely rotate which leagues are starved.

**3. Identity gate — 4 refusals over three days, and one is MY REGRESSION**

| date | incoming | stored | verdict |
| --- | --- | --- | --- |
| 08-27 | `Maccabi Tel Aviv` | `Telstar` | **CORRECT** — the documented impostor |
| 08-28 | `Maccabi Tel Aviv` | `Telstar` | **CORRECT** — recurring |
| 08-29 | `Cracovia Krakow` | `Rakow` | **CORRECT** — two different Polish clubs (Kraków vs Częstochowa) |
| 08-29 | `Standard Liege` | `St. Liege` | **FALSE POSITIVE — caused by Stage 20** |

**The Cracovia case reveals a second Telstar-shaped corruption:** stored row 411
named `Rakow` carries **API-Football id 350, which is Cracovia's**. The refusal is
correct *and* the stored row is wrong. Recorded, not repaired.

### THE REGRESSION I INTRODUCED, and it is live

`Standard Liege` / `St. Liege` **shares an anchor** — `{lieg, liege}` is in both.
It was refused anyway, because Stage 20's canonicalisation rewrites it first:

```
TEAM_NAME_ALIASES["Standard Liege"] = "Standard"     # pre-existing entry
"Standard Liege" -> "Standard"   anchors {stan, standard}
"St. Liege"      -> (no alias)   anchors {lieg, liege}
intersection: EMPTY  ->  refused
```

**I applied a one-directional provider→canonical map to BOTH sides of a
symmetric comparison, and canonicalising can REMOVE the token that was the shared
anchor.** Stage 20 fixed two false positives and created a third.

**The fix is to union rather than replace** — compare
`anchors(raw) ∪ anchors(aliased)` on each side. That can only ever *add* anchors,
so it cannot refuse anything the pre-Stage-20 gate accepted, while still
resolving `Athletic Club`/`Ath Bilbao` and `FC Iberia 1999`/`Saburtalo`. Verified
by hand on all four pairs, including that `Maccabi Tel Aviv`/`Telstar` stays
refused.

**Not applied — this stage is read-only.** It is a live defect losing a
Jupiler Pro League fixture per occurrence and should be fixed first thing.

**4. `fixtures_zero_active` — predicted ~1, measured 2, 0, 0.**
Against **21** on 2026-08-27 pre-fix and **260** across the cached period. The
Part C fix worked and the prediction was close.

### The redirect check — NOT YET ANSWERABLE

Registered: confirm UEL/UECL fixtures belong to the competition requested, by
team names. Measured:

```
europe/europa-conference-league   25 fixtures   via_flashscore=0   via_apifootball=25
europe/europa-league             24 fixtures   via_flashscore=0   via_apifootball=24
```

**Flashscore has created ZERO fixtures for either competition**, so there is
nothing to check the redirect against. Both leagues were in the trailing set on
08-29 and were never attempted. **The check stays registered and unanswered.**

### The Stage 18 substrate trigger HAS FIRED

| | 08-27 | now |
| --- | --- | --- |
| `odds_snapshots` rows | 2,713 | **12,888** |
| **keys with ≥3 observations** | **0** | **91** |
| `injury_observations` | 4 | **286** |
| MODEL observations | 46 | 48 |

**91 keys now carry three observations.** The accumulation query — the trigger
Stage 18 registered in place of a calendar date — is satisfied. H1 and H4 are
testable for the first time. **Not this stage's work.**

## PART B — what lateness actually costs

| day | delay | fixtures in window @cron | @actual | **kicked off by start** | picks |
| --- | --- | --- | --- | --- | --- |
| 08-27 | **10h 21m** | 36 | 29 | **36 of 36** | **0** |
| 08-28 | **11h 21m** | 29 | 103 | **29 of 29** | **0** |
| 08-29 | 5h 03m | 103 | 76 | 57 of 103 | 35 |

**On 08-27 and 08-28 every fixture in the on-time window had already kicked off
by the time the run started. Both days produced zero picks.** The `NO FIXTURES
FOUND` finding in the audit is that, exactly.

### The compensating-effect hypothesis is FALSIFIED

The proposition was that a late run picking tomorrow's card at a longer lead is a
different pipeline, not a broken one. **It is not what happens.**

| | median lead |
| --- | --- |
| 08-29 late run (n=35) | **2.1h** (min 0.1h, max 4.1h) |
| on-time runs 08-14 → 08-25 | **4.4 – 8.8h** |

**A late run takes prices CLOSER to kickoff, not further from it.** The window is
`max_days_ahead = 1` and the pick generator works the imminent card, so starting
five hours late does not reach forward — it compresses the lead.

**That is strictly worse for the MODEL series**, which measures movement between
the taken price and the close: less time to the close means less room to move.
**Lateness is a pure loss on this evidence, in both directions — fewer picks, and
worse observations from the picks that survive.**

## The two damages, which OPS-3 conflated

**They are different in kind and only one is recoverable.**

**IRRECOVERABLE — missed closing-lines windows.** The price ceases to exist.

| pick_date | picks | captured | missing | late |
| --- | --- | --- | --- | --- |
| 2026-08-27 | 34 | **1** | 16 | 17 |
| 2026-08-29 | 35 | **0** | 17 | 18 |

**One capture from 69 picks across two days**, against 61 from ~116 in-market
picks in the 08-14 → 08-22 window. Those observations are gone permanently.

**RECOVERABLE IN PRINCIPLE — late daily-picks.** It shifts or destroys what is
picked, but a pick not made is an opportunity forgone, not evidence destroyed.
The fixtures still exist and the day can be re-run — as 2026-08-27 was, manually.

**OPS-3 treats lateness as one thing. It is two, and the closing-lines half is
the one that cannot be undone.**

## PART C — the options, evaluated

**Stated plainly first: `37 9 * * *` is 09:37 UTC, which is 12:37 Sofia. The cron
already satisfies "started before 13:00 Sofia".** The delay defeats it, and
**shifting the cron shifts the delay with it** — the firing times observed
(19:58, 20:58, 14:40) show no wall-clock attractor, so a delay distribution
applies from wherever the cron sits.

### Option 1 — move the cron earlier

For a 10:00 UTC worst-case start against the measured 10h21m envelope, the cron
lands at **≈23:39 UTC the previous day**.

- **Settlement breaks.** The previous day's evening fixtures (19:00–21:00 UTC)
  finish 2–4 hours before 23:39. Results are frequently not yet available from
  Flashscore or football-data.org at that lag, so settlement would run against
  incomplete results — a new defect in exchange for the old one.
- **Fixture window improves.** `max_days_ahead = 1` from 23:39 covers the whole
  of the next day's card, which the current 09:37 start does not.
- **Pick lead time improves substantially** — picks taken at 23:39 for next-day
  evening fixtures carry a **19–21h lead** against the current 4.4–8.8h. Given
  Part B, that is the one change that would *help* the MODEL series rather than
  merely protect it.
- **It is a cohort event.** Different lead times produce a different CLV
  distribution; this cannot be done without a `CODE_REVISION` bump.

### Option 2 — redundant crons with an idempotency guard

**The pipeline is only partly idempotent, and the expensive part is not.**
`--update` spends API-Football credits *before* the per-match cap, briefing
freeze or dedup key apply. Measured: **51 of the 100/day free tier on 2026-08-27
in a single run.** A second full run costs another ~51 and **exceeds the free
tier**; a third is impossible.

So this option requires an `--update`-level "already ran today" guard before it
is affordable at all. **That guard does not exist and is the real work here**,
not the extra cron lines.

### Option 3 — external trigger via `repository_dispatch`

Removes the GitHub-scheduler dependency and adds an external one: a host that
must stay up and a long-lived PAT that must be stored, rotated and never leaked.
**This project has already had one credential incident and one account
suspension**; adding a long-lived token with `repo` scope is a real cost and
should not be priced as free.

### Option 4 — change nothing

**Three days is not a trend, and fitting decisions to small samples is this
project's cardinal error** — the settled-pick segment thresholds, the fifteen
data-fitted thresholds, and the trigger dates withdrawn within a day of being
derived. Note also that **08-29's 5h03m delay is INSIDE the historical 0.5–5.7h
envelope**, so only **two** of three days exceeded it.

### RECOMMENDATION: Option 4 now, Option 1 when the envelope is established

**Recommended: change nothing yet**, with an explicit revisit trigger, because a
schedule redesigned around 10h21m becomes wrong in the other direction if the
envelope reverts — and one of the three observations is already within the old
bound.

**Assumption stated: that the 10–11h delays of 08-27/08-28 are an episode rather
than a new baseline.** That assumption is what the trigger tests.

> **REVISIT TRIGGER: if, over the next 7 days, 3 or more `daily-picks` firings
> arrive more than 6 hours after their cron, adopt Option 1** — move the cron to
> ≈23:39 UTC **and** split settlement into its own later job, since Option 1
> breaks settlement as it stands. That is a cohort event and needs a bump.

**What would change this recommendation immediately:** evidence that the
irrecoverable damage is larger than measured. **One capture from 69 picks is
already severe**, and if the next two days repeat it, the small-sample argument
stops outweighing a permanent loss of the experiment's only instrument.

---

# STAGE 21 — CLOCK MOVED

`cohort_status.py`: s5.7 carried 35 picks → **BUMP**. One bump, `s5.8`,
`stage5_baseline_20260807.dfe302`. 867 tests pass; 26 invariants pass.

## PART A — the regression removed, and the rule it belongs to

**Fixed by UNION, not replacement.** `names_share_an_anchor` now compares
`anchors(raw) ∪ anchors(aliased)` per side. Unioning can only ever ADD anchors,
so **no pair the pre-Stage-20 gate accepted can now be refused** — which is the
property that makes the fix safe rather than merely correct on the one case.

**Verified to the standard the gate was verified to:** 10 previously-passing
pairs all pass (`Standard Liege`/`St. Liege`, `Union St. Gilloise`/`St. Gilloise`,
`NEC Nijmegen`/`Nijmegen`, `Heart Of Midlothian`/`Hearts`, `Red Bull Salzburg`/
`Salzburg`, `CFR 1907 Cluj`/`CFR Cluj`, `Universitatea Craiova`/`Univ. Craiova`,
`Ferencvarosi TC`/`Ferencvaros`, plus the two Stage 20 fixes). All three
impostors still refused: `Maccabi Tel Aviv`/`Telstar`, `Pau FC`/`St. Pauli`,
`Cracovia Krakow`/`Rakow`. `Rapid Vienna`/`Rapid Bucuresti` still shares an
anchor by design — the country check separates it, as pinned.

### THE RULE — canonicalisation is not symmetric-safe

> **A one-directional map — many provider forms to ONE canonical form — applied
> to BOTH SIDES of a comparison can delete the very token the comparison depends
> on.**

**And the sharper form, which the audit below produced:**

| test shape | canonicalising both sides |
| --- | --- |
| **equality** (`set(ta) == set(tb)`) | **SAFE** — can only increase agreement |
| **overlap** (intersection non-empty, or a ratio) | **UNSAFE** — can remove the overlapping token |

That distinction is why `same_team_strict` was never affected and the anchor gate
was: one asks whether two names are *the same*, the other whether they *share
anything*.

### The symmetric-normalisation audit

Every map applied inside a comparison, checked:

| site | shape | verdict |
| --- | --- | --- |
| `apifootball.names_share_an_anchor` | overlap, alias applied to both sides | **was the bug; fixed** |
| `team_names._norm` → `same_team_strict` | **equality**, `NAME_ALIASES` both sides | safe by shape |
| `team_names._norm` → `team_names_similar` | **overlap ratio ≥0.7**, `NAME_ALIASES` both sides | **same hazard, currently benign** |
| `COMPETITION_MAP.get` | one-directional, applied once to incoming data | safe |
| `LEAGUE_TO_THEODDS_SPORT.get` | one-directional, logging only | safe |
| `TEAM_NAME_ALIASES.get` (step 2) | applied to the incoming name only | safe |
| `market_spec.extract_legs` | per-leg aliases, incoming dict only | safe |

**`team_names_similar` carries the identical hazard and does not currently
bite.** Scanned every alias whose canonical form deletes a token (20 of 22 do),
against every counterpart sharing a deleted token. In each case either both sides
alias to the same canonical (`olympiakos piraeus`/`olympiakos` → still similar) or
the pair is genuinely different (`korea republic`/`czech republic` → correctly
dissimilar). **It is benign by the accident of the table's contents, not by
construction.** Reported, not fixed — the instruction was to fix only the one.

### Fifth identity corruption, recorded not repaired

**Row 411 (`Rakow`) carries API-Football id 350, which is Cracovia's.** The gate
found it by refusing a real fixture — correctly, since Cracovia Kraków and Raków
Częstochowa are different clubs. **The refusal is right AND the stored row is
wrong.** The population of such rows is still unestablished and a repair pass is
its own decision.

## PART B — the clock

### B1. The weekend asymmetry, quantified

**Card size and shape by day** (MEASURED 2026-08-01 → 08-30):

| day | fixtures/day | median KO | % KO ≤ 14:00 |
| --- | --- | --- | --- |
| Mon | 46.2 | 13:13 | 80% |
| Tue | 15.8 | 12:52 | 56% |
| **Wed** | 29.8 | **10:28** | 84% |
| Thu | 29.8 | 17:00 | 7% |
| Fri | 18.2 | 18:30 | 3% |
| **Sat** | **58.6** | 14:00 | 51% |
| **Sun** | **55.2** | 14:00 | 50% |

**Fraction of the card already kicked off, by delay from a 09:37 cron:**

| day | +1h | +3h | +5h | +8h | +11h |
| --- | --- | --- | --- | --- | --- |
| Mon | 5% | 20% | **80%** | 90% | 100% |
| Tue | 38% | 43% | 56% | 76% | 100% |
| **Wed** | **60%** | **84%** | 84% | 89% | 100% |
| Thu | 3% | 7% | 7% | 59% | 100% |
| Fri | 3% | 3% | 3% | 32% | 100% |
| **Sat** | 0% | 20% | **54%** | 77% | 100% |
| Sun | 23% | 39% | 56% | 92% | 100% |

**Saturday's 54% at +5h independently reproduces the 55% measured on 2026-08-29**
— the same number from a different method.

**And a finding the weekend framing would have missed: Wednesday is worse than
Saturday.** Its median kickoff is **10:28 UTC — 51 minutes after the cron** — so
**60% of Wednesday's card is gone at just one hour of delay**, and 84% at three.
Thursday and Friday, by contrast, tolerate five hours at a cost of 7% and 3%.
**The binding days are Wed, Mon, Sun, Sat. Weekday-evening cards are not the
problem; early cards are, and one of them is midweek.**

### B2. How early can it run? NOT answerable from `first_seen_at`

The registered method was to read `first_seen_at`. **It cannot answer this, and
the reason matters.**

Measured: the maximum lead across **every** market is **5.4h**, and **zero**
fixtures carry a priced market 6h before kickoff. Bookmakers demonstrably price
days ahead, so that is not market behaviour.

> **`first_seen_at` records when THIS PIPELINE LOOKED, not when the market
> opened.** Every recent run was late (14:40, 19:58, 20:58), so first sight is
> late by construction.

**Using it would have been circular**: the runs were late → first-sight is late →
"odds are not available early" → do not move the cron earlier, on evidence
produced by the cron being late. **A single anomalous result was evidence about
the measurement, and the measurement was the one the stage prescribed.**

**Measured directly against the provider instead** (MEASURED 2026-08-30, threshold
**≥80% declared before looking**):

| lead | fixtures priced |
| --- | --- |
| **≥24h** | **9/10 = 90%** — **PASS** |
| ≥18h | 9/10 = 90% |
| ≥12h | 9/10 = 90% |
| ≥9h | 10/10 = 100% |

**Sample: n=10, one bookmaker (Bet365), one date** — the free plan restricts odds
to a 3-day window. Small, and stated as small. **Odds availability does not
bind** at any hour the other constraints allow.

### B3. The choice, with the arithmetic

**Three constraints, each binding from a different side:**

| constraint | measured | implication |
| --- | --- | --- |
| earliest kickoff | **10:04 UTC** (Sunday p05 10:06) | a ~20-min run must START by **09:45** |
| settlement | latest KO **19:30** → football ends **~21:30** | cron must not precede results publication |
| odds | 90% priced at ≥24h | does not bind |

**Therefore a cron at `09:45 − D` tolerates D of delay:**

| cron | delay tolerance | settlement margin | verdict |
| --- | --- | --- | --- |
| 04:45 | 5h00m | 7h15m | misses the 5.7h historical max by 42m |
| **03:00** | **6h45m** | **5h30m** | **covers the historical envelope with margin** |
| 01:45 | 8h00m | 4h15m | more tolerance, thinner settlement |
| 23:30 (prev) | 10h15m | **2h00m** | **REJECTED — settles before results publish** |

**CHOSEN: `0 3 * * *` — 03:00 UTC, 06:00 Sofia.**

**Projected pick lead at 03:00** (picks written ~03:20):

| | lead |
| --- | --- |
| p10 kickoff (10:23 UTC) | **7.1h** |
| median kickoff (14:00) | **10.7h** |
| p90 kickoff (18:45) | **15.4h** |
| *current on-time baseline* | *4.4 – 8.8h* |
| *measured late run 08-29* | *2.1h median* |

**Lead roughly doubles**, and Part B of the previous audit established that this
is the one effect that *helps* the MODEL series rather than merely protecting it:
a longer lead leaves more room for the price to move before the close, which is
exactly what CLV measures.

### B5. What this does not fix — stated plainly

> **Moving the cron buys margin. It does not remove the dependency.** GitHub's
> scheduler has produced 0.5h to 11h21m of delay and nothing here changes that.
> **A 03:00 cron delayed 11h lands at 14:00 UTC and still misses a Saturday
> afternoon card** — 54% of it, by the B1 table.

**The schedule is now robust to the TYPICAL delay and not to the OBSERVED
MAXIMUM.** OPS-3 stays open with its 12-hour escalation criterion.

## PART C — the severity framing, corrected

The previous entry called the capture collapse *"permanent loss of the
experiment's only instrument"* — 1 capture from 69 picks against 61 from ~116.

**That states the severity against a question Stage 16 already closed.** 500 was
over-specified **~29×**; seventeen observations suffice to exclude a
decision-relevant effect; and at n=46 the one-sided upper bound was **+0.107%
against a +1.85% threshold**. **Lost captures buy precision on an axis that is
already resolved** — the same finding Stage 15 reached about the seven months to
March 2027.

**The correct framing:** the CLV instrument retains **forward** value. If a model
with a plausible edge is ever built, this is how it would be tested. **That is a
reason to keep it working, not a reason to treat each lost observation as
urgent.**

**And it strengthens the previous recommendation rather than weakening it.** The
argument for not panicking was one part small-sample; it is now two parts —
**the loss is also cheap.**

---

**STAGE 21 — CLOCK MOVED.**

**Cron:** `37 9 * * *` → **`0 3 * * *`** (03:00 UTC / 06:00 Sofia).

**Three constraints satisfied simultaneously:** earliest kickoff 10:04 UTC gives
6h45m of delay tolerance; settlement has 5h30m of margin after football ends at
~21:30; odds are priced at ≥24h for 90% of fixtures, so availability does not
bind.

**Measured lead-time change:** 4.4–8.8h → **7.1–15.4h (median 10.7h)**.

**The day-of-week table that justified it is above**, and its unexpected half is
that **Wednesday, not Saturday, is the least delay-tolerant day** — 60% of its
card gone at one hour.

---

## Stage 21 follow-ups — schedule prediction registered, alias hazard pinned

Both cohort-neutral: a prediction document and a test. `CODE_REVISION` stays
`s5.8`; no config, no model, no workflow touched.

### 1. `docs/stage21-schedule-prediction.md` — committed before the first firing

`0 3 * * *` first fires **2026-08-31, a Monday**. Three outcomes defined before
looking: **WITHIN TOLERANCE** (start ≤ 08:42 UTC), **LATE BUT COVERING**
(08:42 → 09:45), **BEYOND TOLERANCE** (> 09:45, margin insufficient).

**And a caveat registered in advance rather than discovered after: tomorrow may
not test the margin at all.** Tomorrow's known card (football-data.org, its nine
competitions) starts at **16:30 UTC**, so *any* delay up to ~13h is harmless for
those eight fixtures — even the observed 11h21m maximum would cost nothing.

| run start | historical Monday (n=185) | tomorrow's known card (n=8) |
| --- | --- | --- |
| 03:00 / 08:42 / 09:45 | 0% | 0% |
| 14:00 (+11h, observed max) | **80%** | **0%** |

**A `WITHIN TOLERANCE` result tomorrow is therefore weak evidence** — it would
confirm the run fires and stamps `s5.8`, but not that 6h45m of margin suffices,
because the card would not have demanded it. **The margin is genuinely tested on
a day whose card starts early — a Wednesday (median KO 10:28) or a weekend
(p05 ~10:06).** That is the day to read the result on.

### 2. `tests/test_alias_symmetry_hazard.py` — the hazard pinned, not fixed

`team_names_similar` applies `NAME_ALIASES` to **both sides of an overlap ratio**
— the unsafe half of Stage 21's equality-versus-overlap rule, and the same defect
that reached production in the identity gate on 2026-08-29.

**Why pin now.** The alias table is *growing*: the identity gate has found five
corruptions and is still finding them one at a time, and **every false positive it
produces is repaired by adding an alias.** Each repair is a chance to activate
this.

**What is pinned is the INVENTORY, not the absence.** Seven hazard pairs exist
today and all seven are benign, for two distinct reasons — three because both
sides reach the same canonical (`united states`/`united states of america` → `usa`),
four because the entities genuinely differ and losing the overlap is *correct*
(`korea republic`/`czech republic`). A new alias creating an eighth fails at
commit time, forcing whoever adds it to classify it rather than discover it
months later as a silent false refusal.

The pair set is **derived** from the table, never maintained beside it, so the
scan cannot drift from what it scans — and a second test fails if the inventory
lists a pair that no longer exists, so it cannot accumulate stale entries that
mask a real one.

**POSITIVE CONTROL RUN, because a guard that never fires is worth nothing:**
injecting `"standard liege" → "standard"` alongside a `"st. liege"` counterpart —
exactly the shape that reached production — makes the pin fire and name the pair.
Removed again; the inventory returns to zero new hazards.

**The site is NOT fixed.** Fixing it symmetrically is a design change and belongs
with the guard work already deferred. 871 tests pass.

---

## What the Stage 20 regression cost — counted, not characterised

The `s5.8` entry said Stage 20 "created a third false positive". That was
unquantified. This is the count.

**Window** — from the commit that introduced it to the commit that removed it:

| | commit | UTC |
| --- | --- | --- |
| introduced | `1b29397` | 2026-08-27 **17:42** |
| removed | `ca3a7f2` | 2026-08-30 **14:38** |

**Four `daily-picks` runs fall inside it**, including the 08-30 14:30 run, which
started **eight minutes before the fix was pushed** and therefore belongs to the
*before* column.

### Every refusal in the window, classified under the union rule now in place

| date | incoming | stored | league | verdict |
| --- | --- | --- | --- | --- |
| 08-27 | Maccabi Tel Aviv | Telstar | netherlands/eredivisie | correct |
| 08-28 | Maccabi Tel Aviv | Telstar | netherlands/eredivisie | correct |
| 08-29 | Cracovia Krakow | Rakow | poland/ekstraklasa | correct |
| **08-29** | **Standard Liege** | **St. Liege** | **belgium/jupiler-pro-league** | **FALSE POSITIVE** |
| 08-30 | Cracovia Krakow | Rakow | poland/ekstraklasa | correct |
| **08-30** | **Standard Liege** | **St. Liege** | **belgium/jupiler-pro-league** | **FALSE POSITIVE** |

**6 refusals: 4 correct, 2 false positives.**

### Two refusals, ONE fixture

The 08-30 refusal is **not a second fixture.** Its log context reads
`API-Football: 1365 fixtures on 2026-08-29` → refusal →
`2026-08-29: 0 created, 101 updated`: the run was re-processing the **previous
day's** card and hit the same pair again.

**So the regression cost one fixture, refused on two consecutive days.**

### The fixture, and what "skipped" actually meant

`id=50291` — **Leuven vs St. Liege**, 2026-08-29 18:45, Jupiler Pro League.

**The row survived**: Flashscore created it (`flashscore_id` present). What was
lost was the **API-Football path**, and with it the odds. Against its own
same-day peers:

| fixture | `apifootball_id` | odds rows | picks |
| --- | --- | --- | --- |
| RAAL La Louviere v KV Mechelen | ✓ | 77 | 0 |
| Kortrijk v Charleroi | ✓ | 175 | 1 |
| **Leuven v St. Liege** | **✗** | **0** | **0** |
| Cercle Brugge v Lommel SK | ✓ | 79 | 1 |

**Zero odds against a peer median of 79, and no pick where two of three
comparable fixtures produced one.** Of 9 Belgian fixtures across 08-28 → 09-01,
**8 carry an API-Football id and 9 carry a Flashscore id** — the single gap is
this one.

> **THE COST: one Jupiler Pro League fixture rendered unpickable for three days.**
> Not a missing fixture — an unpriced one, which is quieter, because a fixture
> with no odds looks identical to a fixture nobody wanted to bet.

### The baseline this sets for the next run

**After `ca3a7f2` the false-positive count must be ZERO.** Any refusal that
survives is one of exactly two things:

* **a genuine corruption** — `Maccabi Tel Aviv`/`Telstar` (row 124 holds AF id
  604) or `Cracovia Krakow`/`Rakow` (row 411 holds AF id **350**, Cracovia's),
  both confirmed and both deliberately unrepaired; or
* **a sixth corruption to investigate** — a pair not on that list.

**A refusal that is neither is a regression**, and this table is what it would be
measured against.

*No post-fix `daily-picks` run had occurred at the time of writing
(2026-08-31 07:40 UTC); the first will be the run under the new 03:00 cron.*

---

# UNPRICED vs UNWANTED — the assertion, and what the correct refusals cost

**2026-08-31. One new check, one measurement, no repair.**

## PART 1 — the alarm

`src/data/coverage_checks.py`. **A fixture with zero odds is indistinguishable
from a fixture nobody wanted to bet** — identical in every output the pipeline
emits. That is how the Stage 20 regression survived three days.

**SELF-CALIBRATING, with no threshold and no maintained league list:** a fixture
is compared against its OWN same-league, same-day peers. Zero odds where peers
are priced is an anomaly; zero odds where nobody is priced is a quiet league, an
off-season, or an uncovered competition — silent by construction, not by
exclusion.

### The registered proof, replayed

| day | unpriced | **ALARM** | INFO |
| --- | --- | --- | --- |
| 2026-08-29 | 18 | **3** | 15 |
| 2026-08-30 | 10 | **0** | 10 |

```
ALARM  Avellino vs L.R. Vicenza      italy/serie-b            peers=5  med=91
ALARM  Leuven vs St. Liege           belgium/jupiler-pro      peers=3  med=79
ALARM  Radomiak Radom vs Cracovia    poland/ekstraklasa       peers=2  med=72
```

**It names Leuven vs St. Liege and is silent on all three of its peers** (77, 79,
175), which was the registered requirement.

### Two deviations from the literal specification, both stated

**1. The trigger is NOT "has a Flashscore id but no API-Football id".** That
signature describes the Leuven case exactly and **misses the Cracovia rows, which
carry neither id**. The id columns are *reported* because they name the likely
cause; they are not the trigger, because the trigger must catch the class rather
than the instance.

**2. It splits by cause, because unsplit it fires every day.** 18 alarms on a day
with 3 real problems is the `fixtures_zero_active` failure again — a check that
fires daily is a check that gets ignored. **No API-Football id → the team was
never resolved → ALARM. Has one → odds-budget coverage → INFO.**

### Two defects found in the check while building it

**THE SQL WAS POSTGRES-ONLY.** `match_date::date`, `percentile_cont` and `FILTER`
all raise on the SQLite fallback, where the `except` swallowed them and returned
an empty list. **The check would have reported "no unpriced fixtures" on exactly
the database where nobody would look.** The grouping now happens in Python. The
portable form is also the testable one — the pin runs on SQLite.

**A DEAD CHECK READ AS A CLEAN CHECK.** Run against a stale credential it printed
`2026-08-29: 0 unpriced` and returned success. Empty now means
*measured-and-clean*; **`None` means unmeasured, and the caller says so**:

```
UNPRICED FIXTURE CHECK DID NOT RUN — the query failed, so this run has NO
evidence either way about unpriced fixtures. Not a clean result.
```

Both shapes are the one this check exists to catch, reproduced inside it.

`ci_audit.py` greps both. **Not self-calibrating, deliberately:** a fixture that
could not be priced because a team would not resolve is wrong on its first
occurrence, not relative to history. `tests/test_unpriced_fixture_alarm.py` pins
the replay and all three properties. 877 tests pass. **Cohort-neutral** — the
check only reads and logs.

## PART 2 — what the two unrepaired corruptions cost, and three corrections

**Nothing repaired. The measurement corrects my own earlier statements first.**

| what I said earlier | measured 2026-08-31 |
| --- | --- |
| "the blocked clubs are Maccabi and Cracovia" | **Maccabi Tel Aviv has NO team row at all.** The gate refuses it at step 0 every time, so it has never been created. The only `%tel aviv%` row is Hapoel (4501), unrelated. |
| "12 of 12 August Cracovia fixtures unpriced" | **10 of those 12 are phantoms** — `match_date` equal to `created_at` to the microsecond, the already-documented class. **The real sample is n=2**, both unpriced. |
| "seven near-duplicate Cracovia vs Wieczysta rows — the corruption generates duplicates" | **Withdrawn. They are the phantom class**, not this corruption. I attributed one known defect's output to another. |

### Cracovia — MEASURED, and the two regimes disagree

| window | Cracovia priced | peers priced | gap |
| --- | --- | --- | --- |
| Aug 2026 (phantoms excluded) | **0 of 2 (0%)** | 25 of 33 (**76%**) | 76pp |
| 2025/2026 season | 8 of 37 (**21.6%**) | 117 of 324 (**36.1%**) | 14.5pp |

**The block is not total historically — Cracovia was priced 8 times in 2025/26.**
The recent regime is worse because the API-Football id path is now the primary
odds source. **Both numbers are reported because they disagree**, and the
resulting range is the honest answer rather than the more alarming half of it.

**Per season, at 34 Ekstraklasa fixtures per club** (306 fixtures / 18 clubs × 2,
MEASURED from 2024/2025) and **0.543 picks per priced Ekstraklasa fixture**
(19 picks / 35 priced, MEASURED since 2026-06-01):

| basis | fixtures lost/season | **picks lost/season** |
| --- | --- | --- |
| current-regime gap (76pp) | ~26 | **~14** |
| season-long gap (14.5pp) | ~5 | **~3** |

### Maccabi Tel Aviv — BOUNDED, not measured, and the reason is structural

**`other/israel` is not a configured league**, so this club reaches the system
only through `europe/*`. **Its fixtures cannot be counted because they were never
created** — that is the corruption. Observable instead: **2 refusals in the 3 days
API-Football has been healthy** (08-27, 08-28), which proves it is live in a
configured competition this season.

**ASSUMED** (stated as an assumption): a European participant plays 6 league-phase
fixtures plus up to 8 qualifiers → **6–14 fixtures/season**, conditional on
qualifying, → **~4–9 picks/season** at 0.669 picks per priced fixture (694 picks
/ 1038 priced, MEASURED since 2026-06-01).

### THE TOTAL, with its uncertainty intact

> **~7 to ~23 picks per season across both corruptions — 0.25% to 0.8% of
> annual pick volume** (694 picks in 3 months → ~2,800/year, MEASURED).

**The cost is real, recurring, and small.** It is also **not a bug** — every one
of those refusals is the gate working correctly against a wrong stored integer.

### What this changes about the repair decision

**It removes the urgency and keeps the case.** A repair pass is worth doing on
its own schedule, not as an emergency, and the population is still
unestablished — **two known corrupt rows out of an unmeasured number.**

**And the alarm above is what makes the population measurable.** It names
unresolved fixtures every day, so the population now accumulates as evidence
rather than requiring a one-off sweep whose completeness could not be checked.
**That is an argument for waiting a few days before repairing, not for repairing
now.**

*Put to Niki, unrepaired, 2026-08-31.*

---

# AUDIT 2026-08-31 — the first run under `0 3 * * *`

**Read-only. Nothing fixed.** Both registered files were read before the logs:
`docs/stage21-schedule-prediction.md` and the regression count in `f4f5c2d`.

## PART A — the schedule outcome

**Run `33375724727`, started `2026-08-31T09:02:37Z`. Delay against `0 3 * * *`:
6h 02m 37s.**

| outcome | boundary | today |
| --- | --- | --- |
| WITHIN TOLERANCE | start ≤ 08:42 UTC | — |
| **LATE BUT COVERING** | **08:42 < start ≤ 09:45** | **← 09:02:37** |
| BEYOND TOLERANCE | start > 09:45 UTC | — |

**Not the predicted outcome.** Prediction 1 said a historical envelope puts the
start between 03:00 and 08:42; **6h02m exceeds the 5.7h historical maximum by
20 minutes**, so the run landed in the band the cron change was bought for.

### THE LIMIT, restated verbatim from the registered file

> **A `WITHIN TOLERANCE` result on 2026-08-31 is not evidence that 6h45m of
> margin is sufficient, and must not be recorded as such.**

**Today produced LATE BUT COVERING, and the same limit applies with more force.**
Today's earliest kickoff was **16:00 UTC**, so every delay up to ~13h was
harmless; "covering" was guaranteed by the card, not earned by the margin.
**Today tests that the cron fires and that picks stamp `s5.8`. Nothing more.**

**PRIMARY CHECKPOINT REMAINS Wednesday 2026-09-02**, and per the registered
instruction its own card must be checked for lateness *before* the result is
read.

### The four card measurements

| measurement | value |
| --- | --- |
| fixtures in window at execution | **28** |
| already kicked off at 09:02:37 | **0** |
| **fraction of the card lost** | **0.0%** |
| lead time, n=20 | p10 **6.4h** · median **7.9h** · p90 **9.6h** |

Against the registered projection of **7.1 / 10.7 / 15.4h**: **today came in
below projection at every point.** The projection assumed picks written ~03:20;
they were written ~09:37. **Today's median of 7.9h sits inside the on-time
baseline of 4.4–8.8h** — the lead was normal, not improved. The doubling the
cron change was expected to buy has **not been observed**, because the run has
not yet started near 03:00.

### A DEFECT IN THE REGISTERED ARITHMETIC, found by measuring it

The 09:45 boundary came from *"a ~20-min run must start by 09:45 UTC"* against a
10:04 earliest kickoff. **Measured today: the run took 63m37s, and picks were
written at 09:37 — 34m35s after start**, not ~20.

> **The true boundary is 10:04 − 0:35 = ~09:29, not 09:45. The margin is
> 6h29m, not 6h45m** — about 16 minutes thinner than registered.

Today's 09:02 start is still inside it, so **no outcome changes**; the boundary
itself was optimistic and is corrected here rather than after it costs a card.

## PART B — the post-`ca3a7f2` baseline: **MET**

**Zero `TEAM IDENTITY MISMATCH` refusals in the entire run.** (315 grep hits were
test names; the refusal logger emitted nothing.)

| class | required | observed |
| --- | --- | --- |
| **false positive** | **must be ZERO** | **0 — baseline met** |
| correct (Maccabi/Telstar, Cracovia/Rakow) | expected when those clubs play | **0 — neither club played** |
| a sixth | investigate against `GET /teams?id=` | **0** |

**`Standard Liege` / `St. Liege` did not recur.** The union fix holds on its
first post-fix run.

**Stated plainly: the correct-refusal half of the baseline was met vacuously.**
Neither corrupt club had a fixture today, so this run is **not** evidence that
the two known corruptions still recur — only that no false positive occurred.

**Cross-check with `33317038655` (08-30 14:30, eight minutes pre-fix)** confirms
`f4f5c2d`'s *before* column verbatim: `Cracovia Krakow`/`Rakow` (correct) and
`Standard Liege`/`St. Liege` (false positive), `2 fixture(s) SKIPPED`.

## PART C — the unpriced alarm's first live firing

**IT RETURNED A MEASURED RESULT.** Not `None`. **2 ALARM, 1 INFO.** The
distinction built two days ago was exercised and reported honestly.

| # | fixture | league | KO | peers | median | fs id | af id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Celta Vigo B vs Castellon | spain/laliga2 | 19:30 | 2 | 75 | ✓ | **✗** |
| 2 | Sporting Clube de Braga vs Guimaraes | portugal/primeira-liga | 19:15 | 2 | 73 | ✓ | **✗** |

### BOTH ARE A DIFFERENT DEFECT CLASS FROM THE ONE THE ALARM WAS BUILT FOR

**Neither is identity corruption. Both are duplicate match rows:**

| Flashscore row | af id | odds | twin | af id | odds | picked |
| --- | --- | --- | --- | --- | --- | --- |
| 50969 Sporting Clube de Braga v Guimaraes | ✗ | 86 | **50990 Braga v Guimaraes** | ✓ | 78 | 50969 |
| 50976 Celta Vigo B v Castellon | ✗ | **0** | **50991 Celta de Vigo II v Castellón** | ✓ | 81 | 50991 |

Same kickoff, same fixture, two rows each — one from Flashscore, one from
API-Football, never unified because `Sporting Clube de Braga` ≢ `Braga` and
`Celta Vigo B` ≢ `Celta de Vigo II`.

**The alarm's message asserts a cause it did not establish:** *"the team was
never resolved, so the fixture was never priced."* The first half is true; **the
second is false — the fixture WAS priced, on its twin.**

### ALARM 2 WAS FALSE ELEVEN SECONDS AFTER IT FIRED

| event | time |
| --- | --- |
| alarm fires on 50969, "0 odds" | **09:22:51.01** |
| 86 odds rows written to 50969 | **09:23:01.94** |

**A TIMING DEFECT.** The check sits after `API-Football update complete`, but
odds keep arriving from later paths. **50969 was priced 10.9s later and picked**
(pick 1475, 1X2 Home Win). Only 50976 is still unpriced now.

> **First live firing: 2 alarms, 1 false within eleven seconds, 0 of the class
> it was built to catch.** Recorded as it stands; not fixed.

### Cross-check against Part B, and the repair-decision series

**0 refusals today → 0 fixtures blocked by a correct refusal → today contributes
NOTHING to the population the deferred repair decision rests on.**

| day | ALARM | from a correct refusal | other class |
| --- | --- | --- | --- |
| 2026-08-29 | 3 | 1 (Cracovia) + 1 (the regression) | 1 (Avellino, unclassified) |
| 2026-08-30 | 0 | 0 | 0 |
| **2026-08-31** | **2** | **0** | **2 (duplicate rows)** |

**The ALARM class is broader than the corruption class**, which is a finding
against my own registered design: I stated the split by cause would isolate
identity corruption, and on live data it surfaced a duplicate-row defect
instead. **Useful, but not what was claimed.**

## PART D — cohort and substrate

**20 picks today, every one `stage5_baseline_20260807.dfe302` (`s5.8`).** No pick
dated 2026-08-31 carries an earlier fingerprint. Prior cohorts intact and not
collapsed: `485823` (246), `098437` (18), `60caed` (34), `645bac` (66).

| substrate | 08-30 | **08-31** |
| --- | --- | --- |
| `odds_snapshots` rows | 12,888 | **24,524** |
| **keys with ≥3 observations** | 91 | **108** |
| `injury_observations` | 286 | **636** |

**The Stage 18 accumulation trigger remains satisfied and is growing.** H1/H4 are
the next stage's work.

## PART E — the routine pass

**26 runs listed as unaudited; 10 are genuinely new.** The other 16 were recorded
in the 2026-08-30 entry as *"19 others"* **without run ids**, so the script
cannot match them. **A ledger row that names no id is not machine-checkable** —
recorded as a format defect, not re-audited.

| run | workflow | started | verdict | discovery |
| --- | --- | --- | --- | --- |
| 33271591573 | closing-lines | 08-29 19:43 | CLEAN | |
| 33280580415 | closing-lines | 08-29 23:16 | CLEAN | |
| 33285273030 | closing-lines | 08-30 01:15 | CLEAN | |
| 33317038655 | daily-picks | 08-30 14:30 | CLEAN | `disc[fs=48 fdo=13 af=1]` |
| 33318395344 | paper-report | 08-30 15:00 | CLEAN | |
| 33318488899 | closing-lines | 08-30 15:02 | CLEAN | |
| 33319296937 | closing-lines | 08-30 15:19 | CLEAN | |
| 33331934377 | closing-lines | 08-30 19:49 | CLEAN | |
| 33341871995 | closing-lines | 08-30 23:27 | CLEAN | |
| **33375724727** | **daily-picks** | **08-31 09:02** | **DEGRADED** | **`disc[fs=26 fdo=7 af=2]`** |

**`conclusion: success` on all ten remains no evidence** — nine steps carry
`continue-on-error`.

**Today is DEGRADED on the new unpriced assertion**, which is the assertion
working; the two alarms are analysed in Part C.

### Three signals recorded, none acted on

* **Flashscore kickoff-parse refusals: 207 today, 192 on 08-30.** Steady, not a
  regression. The Stage 19 fail-closed parser is refusing a large share of the
  rows it sees (`<no time element found>`), and **14 leagues produced zero via
  Flashscore**. Discovery fell `fs=48 → fs=26` day on day.
* **TheOddsAPI: 21 credits remaining, "~2 days left".** A resource cliff with a
  date on it.
* **Today's `closing-lines` and `paper-trading-report` have not fired.** Due
  10:47 UTC, now 12:53 UTC — **2h 06m late**. Under the OPS-3 12-hour rule this
  is LATE, **not** MISSED, and is not assessable before 22:47 UTC.

*Read-only audit, 2026-08-31.*

---

# DUPLICATE MATCH ROWS — the s5.3 guarantee is violated in production

**2026-08-31. Population established. No row repaired.**

**The alarm found this by firing for the wrong reason.** It was built to isolate
identity corruption; both of its live alarms turned out to be duplicate rows, a
defect class nobody was looking for and which is worse than the one it was
built for. **Recorded that way round deliberately** — the check's designed
purpose did not find this; its false alarm did.

## THE MECHANISM

`finalize_picks` groups candidates by `rec.match_id`; `max_picks_per_match: 1`
caps each group; `_filter_correlated_picks` also keys on the match.

> **Two rows for one real fixture are two `match_id`s, therefore two groups,
> therefore two independent pick slots. The cap cannot see it, and the
> correlation filter cannot see it.**

## THE VIOLATION — CONFIRMED, not inferred

**2026-08-30 17:30 UTC, `spain/laliga`, Deportivo La Coruña vs Valencia:**

| row | source | odds | pick | market | EV | result |
| --- | --- | --- | --- | --- | --- | --- |
| **50920** `RC Deportivo La Coruña v Valencia` | API-Football `1570356` | 74 | **1445** | Double Chance X2 @1.515 | −0.0271 | **loss** |
| **50927** `Dep. A Coruna v Valencia` | Flashscore | 91 | **1456** | Under 2.5 @1.56 | −0.1924 | *unsettled* |

**One real fixture. Two picks. Both `s5.7` (`645bac`).** Valencia cannot play
two fixtures in one competition at 17:30, so these rows are one match by
construction, not by name similarity.

**Two consequences, not one:**

1. **`max_picks_per_match: 1` was defeated** — the policy Stage 13 broke a
   cohort to establish.
2. **The correlation filter never saw the pair.** Double Chance X2 and Under 2.5
   on one fixture are positively correlated; the filter that exists to drop the
   lower-EV member of such a pair was keyed on `match_id` and saw two matches.
   **The lower-EV pick (−0.1924) is exactly what it would have dropped.**

**Today only one twin was priced, so only one pick was made. That is luck, not a
mechanism** — and 2026-08-30 is the day the luck ran out.

## HOW THEY ARE IDENTIFIED, AND WITH WHAT PRECISION

**Test:** two rows, **same league**, **identical kickoff minute**, whose home
teams and away teams each resolve to the same club — by **`apifootball_team_id`
equality** where both rows carry one, falling back to `team_names_similar`.

| set | count |
| --- | --- |
| candidate pairs (same league, identical kickoff) | 60,976 |
| **STRICT — both sides resolve to the same club** | **750** |
| ...of which **both sides by provider-id equality** | **242** |
| WEAK — one side resolves + complementary sources | 2 |

> **Precision is 1.0 by construction on the 242.** A club cannot play two
> fixtures in one competition at the same minute, so if both clubs match by
> provider id at an identical kickoff, the two rows ARE one fixture. **The
> remaining 508 lean on `team_names_similar` for at least one side and inherit
> that comparator's false-positive rate, which is unmeasured.**

**RECALL IS THE WEAKER HALF, and it is stated because it matters:** the one
confirmed violation is in the **WEAK** set, not the STRICT one. **The
high-precision test missed the only case that has actually cost anything.**
Identical-kickoff was chosen as the precision lever; sources that disagree on
kickoff time are invisible to it. **750 is a floor, not an estimate.**

An earlier, looser test (same league, ±4h) returned **1,873 pairs** and was
almost entirely false — simultaneous but different fixtures. Discarded.

## WHERE THEY COME FROM

**Two sub-populations, and only one is dangerous.**

| | total | with odds or picks |
| --- | --- | --- |
| STRICT pairs | 750 | **176 active**, 574 inert |
| created **>24h apart** | 709 | 170 |
| created **in the same run** | 28 | **5** |

**The bulk is historical backfill duplicating a live row** — an old row with
zero odds beside a priced one (`Bayer Leverkusen` 614, o=0 ‖ `Leverkusen` 30995,
o=221). Inert: no odds, no picks, no exposure.

**The dangerous class is the 5 same-run races**, and both of today's alarms are
in it. **Source pair for the active set:**

| pair | n |
| --- | --- |
| `AF+FS` (both live sources, same run) | 4 |
| `AFFS + none` | 107 |
| `AF + none` | 40 |
| `none + none` | 21 |
| `FS + none` | 4 |

**`none` rows carry neither provider id** — the historical backfill. **The live
race is API-Football ↔ Flashscore.**

**51 further pairs had one row picked and the twin not.** Not violations, but
the pick attaches to an arbitrary twin, which decides which odds row backs it.

## AT CREATION — and THE HABIT is NOT the cause

**Both comparators ARE consulted at the fixture-matching step.** This was checked
rather than assumed, because the fifth data-layer instance of THE HABIT had
exactly this shape:

| site | comparator | reached? |
| --- | --- | --- |
| `flashscore_scraper` fuzzy merge (±4h) | `team_names_similar` | **yes** |
| `apifootball_scraper._find_match_by_date_league` (±26h) | `team_names_similar` | **yes**, called at line 963 after `_find_match_id` |

> **This is not knowledge in the wrong place. The canonical comparator is read
> by the deciding caller, and it returns the wrong answer.**

**Measured on today's two, with the exact strings the provider sent:**

```
API-Football sent:  "SC Braga" vs "Vitória SC"
stored (Flashscore): "Sporting Clube de Braga" vs "Guimaraes"
    similar("SC Braga", "Sporting Clube de Braga")  -> True
    similar("Vitória SC", "Guimaraes")              -> False     <-- AND fails

API-Football sent:  "Celta de Vigo II" vs "Castellón"
stored (Flashscore): "Celta Vigo B" vs "Castellon"
    similar("Celta de Vigo II", "Celta Vigo B")     -> False     <-- AND fails
    similar("Castellón", "Castellon")               -> True
```

**Exactly one half of each AND gate failed, and the two halves fail for
different reasons:**

* **`Vitória SC` / `Guimaraes` share ZERO tokens.** Vitória Sport Clube plays in
  Guimarães; the two names have no lexical overlap at all. **This is the
  residual class already named in `test_identity_gate_aliases.py`** — *"a
  legitimate pair with ZERO shared tokens, which no lexical test can reach by
  construction, and that is precisely why a curated table exists."* It was
  documented as the identity gate's residual; it is also the fixture matcher's.
* **`Celta Vigo B` / `Celta de Vigo II` is a reserve-team suffix collision** —
  `B` and `II` denote the same thing and neither normaliser knows it.

**Zero fuzzy links succeeded in the whole run** (`0` occurrences of
`fuzzy-linked` / `Fuzzy-merged`), against 2 duplicate creations.

## WHAT IS NOT CLAIMED

**Settlement is NOT broken.** The twin 50927 carries no result and its pick is
unsettled, which looked like a second failure mode. **Measured: 0 of 1,372 picks
on fixtures more than two days old are unsettled.** The 08-30 twin is ~19h old
and inside the normal window. **Recorded as a risk to watch, not a finding.**

**And a limit of the alarm itself: it sees a duplicate only when one twin is
unpriced.** On 2026-08-30 `Celta v Ath Bilbao` (50926, o=91 ‖ 50962, o=75) both
twins were priced, so nothing alarmed. **The alarm is not a duplicate detector
and must not be treated as one.**

## THE ALARM'S TWO CORRECTIONS

**1. Moved to the end of `daily_update`, after every odds path.** It sat after
`API-Football update complete`, which looked like "after the odds path" and was
not — odds also arrive from TheOddsAPI and the low-coverage backfill.

**Replayed from the new position:**

| day | before | **after** |
| --- | --- | --- |
| 2026-08-29 | 3 | 3 |
| 2026-08-30 | 0 | 0 |
| **2026-08-31** | **2** | **1** |

**The count drops exactly where predicted.** 50969 no longer alarms because its
odds arrived; 50976 survives because it genuinely still has none.
`test_the_check_runs_after_every_odds_path` pins the ordering structurally,
because no unit test could see this.

**2. The message no longer asserts a cause it did not establish.** It read *"the
team was never resolved, so the fixture was never priced"* — **false in both
live alarms, because the fixture WAS priced, on its twin.** It now says only
what is measured: *this row carries no odds while same-league peers do*, and
names both candidate causes without choosing.

**878 tests pass. Cohort-neutral** — logging, docs and tests only; no
prediction- or selection-affecting change, so `s5.8` is not bumped despite now
carrying 20 picks.

## WHAT IS NOT DONE

**No row repaired, no matcher changed.** The population is established; the
repair is its own stage and needs a decision on all three of: merging existing
duplicate rows, the alias/suffix knowledge that would prevent new ones, and
whether the pick cap should key on something other than `match_id`.

*Established 2026-08-31. Read-only on the data.*

---

# THE THIRD CONSEQUENCE, and the key that does not depend on the matcher

**2026-08-31. Read-only on the data. Nothing repaired, nothing built.**

## PART 1 — the cluster-aware statistics: **CLEAN, and now measured**

Every CLV interval in this project resamples **fixtures**, not picks. Two rows
for one fixture are counted as two independent clusters, which understates the
design effect, overstates the effective sample size, and narrows every interval
derived from it.

**The query, run against a deliberately WIDE duplicate definition (±26h rather
than the ±0 strict test) because here a false positive is the safe direction:**

| | |
| --- | --- |
| MODEL observations with a closing price | **48** (46 at Stage 16) |
| distinct `match_id` behind them | **48** |
| duplicate-pair rows in the whole database (±26h) | 2,325 pairs / **4,329 rows** |
| **MODEL observations sitting on any of those rows** | **0** |
| **observations on BOTH rows of one pair** | **0** |

> ### VERDICT: CLEAN. `deff = 1.00` was measured on a fixture count containing no duplicates. **Stage 16's +0.107% upper bound stands.** The finding is forward-looking.

**And the mechanism behind the result, so it is not mistaken for luck:** captures
span **2026-08-14 → 2026-08-27**. Duplicate pairs in that window: the August 2026
count is **3**, dated 08-28, 08-30 and 08-31 — **all after the last capture.**
The separation is real but it is *narrow and temporal*, not structural. **The
next capture window overlaps the live duplicate class directly**, so this result
does not transfer forward.

## PART 2 — THE FIVE, and what they actually show

| # | date | league | A | B | created apart |
| --- | --- | --- | --- | --- | --- |
| 1 | 2026-02-28 15:00 | england/championship | 30997 `QPR v Sheffield United` | 34316 `QPR v Sheffield Utd` | 385s |
| 2 | 2026-05-03 14:45 | netherlands/eredivisie | 5987 `Sp Rotterdam v Go Ahead Eagles` | 6003 `Sparta Rotterdam v Go Ahead Eagles` | 64s |
| 3 | 2026-05-17 17:00 | spain/laliga | 41295 `Ath Bilbao v Celta` | 41360 `Ath Bilbao v Celta Vigo` | 400s |
| 4 | 2026-08-30 19:30 | spain/laliga | 50926 `Celta v Ath Bilbao` | 50962 `Celta Vigo v Ath Bilbao` | 557s |
| 5 | 2026-08-31 19:15 | portugal/primeira-liga | 50969 `Sporting Clube de Braga v Guimaraes` | 50990 `Braga v Guimaraes` | 406s |

**Source pair: `FS` ‖ `AF` or `none` ‖ `AFFS` in every case. It is the live
API-Football ↔ Flashscore race, without exception.**

### THE RESULT THAT REFRAMES THE FIX

> **All five pass `team_names_similar` on BOTH sides — `merge=True` for every
> one — when the two STORED rows are compared to each other.**

The matcher did not fail on these names. **It fails because of WHAT IT COMPARES
AT INGEST: the provider's raw name against the stored name.** By pick time both
sides have been through `_get_or_create_team_id`, so both are canonical stored
names carrying provider ids.

```
INGEST  (fails):  "Vitória SC"  vs stored "Guimaraes"     -> False
PICK TIME (works): stored "Guimaraes" vs stored "Guimaraes" -> True
```

**A pick-time check is not merely a different place to put the same test. It is
a strictly better-conditioned test, and that is measured on all five, not
argued.**

## PART 3 — THE FIXTURE-IDENTITY KEY: designed and priced, NOT BUILT

### The rule

> **Two pick candidates are the same fixture if they share a league, a kickoff
> minute, and AT LEAST ONE club resolved to the same provider id.**

### Why the false-refusal cost is ZERO BY CONSTRUCTION

**A club cannot play two fixtures in one competition at the same minute.** So if
two rows share a league, a kickoff minute and one resolved club, they *are* one
fixture — regardless of what the other slot says. **This is not a threshold, a
ratio or a tuned band; it is an impossibility argument**, and it is the reason
this key can fail closed without a cost to price.

### Measured against the database

| | |
| --- | --- |
| pairs the rule matches | **835** |
| ...both clubs shared (unambiguous) | 242 — *exactly the 242 provider-id pairs from the strict test, an independent cross-check* |
| ...exactly one club shared | 593 |
| **would have refused a second pick, historically** | **2 pairs, out of 1,458 saved picks** |
| catches the confirmed 08-30 violation | **YES** — Valencia resolves to 532 on both rows |

**It is precisely targeted: 835 pairs matched, 2 picks refused.**

### FAIL-CLOSED BEHAVIOUR, stated plainly as asked

**Three branches, and only the first is provable:**

1. **≥1 shared resolved club id → REFUSE the second pick.** Provable, zero false
   refusals. Catches all five and both violations.
2. **No shared provider id, but stored names similar on both sides → REFUSE.**
   Heuristic. Inherits `team_names_similar`'s unmeasured false-positive rate.
   **Marked as heuristic, not proof.**
3. **No shared id, names dissimilar → ALLOW.** The residual.

> **Branch 3 is the `Vitória SC` / `Guimaraes` class — and at pick time it is
> much smaller than at ingest, because all five of the live cases pass branch 2
> on stored names even though they failed at ingest. It is not empty, and it is
> not claimed to be.**

**Blanket fail-closed is NOT recommended.** Refusing every same-league,
same-minute pair with no shared id would refuse genuinely different simultaneous
fixtures, which are the norm on a final matchday. **The rule must fail closed on
identity evidence, not on the absence of it** — the same distinction the
`_parse_match_date` fix drew.

### Price

* **Selection-affecting** — it changes which picks are persisted. **Requires
  `s5.9`.** `s5.8` currently carries 20 picks, so this cannot amend.
* Runtime: a group-by over one day's candidates, ~120 rows. Negligible.
* Historical cost: **2 refused picks**, both of which are the violations.

**NOT BUILT. Design and price only, as instructed.**

## PART 4 — A SECOND VIOLATION, found by the key while pricing it

**2026-08-14 19:15, `portugal/primeira-liga`, Sporting CP vs Guimarães:**

| row | af id | odds | picks |
| --- | --- | --- | --- |
| 49496 `Sporting Clube de Portugal v Guimaraes` | — | 79 | **1** (Over 2.5 @1.55, win) |
| 49520 `Sporting CP v Guimaraes` | 1575463 | 74 | **2** (Away Over 0.5 @1.90 win; Under 3.5 @1.66 loss) |

**Three picks on one real fixture.**

**The two violations are NOT the same violation, and the distinction is dated:**

| | date | cap then in force | verdict |
| --- | --- | --- | --- |
| Sporting CP v Guimarães | 2026-08-14 | **2** (`max_picks_per_match: 1` landed 2026-08-23 in `bef66ca`) | 3 picks — exceeds the cap of 2, same mechanism |
| Deportivo v Valencia | 2026-08-30 | **1** | **2 picks — the s5.3 violation proper** |

**Only the 08-30 case violates s5.3.** The 08-14 case predates it and is
recorded as the same defect operating against the older cap.

## PART 5 — the fifty, and the twin-price question

**50 pairs, not 51** — recount. In **all 50 the picked row carries more odds
rows than its twin**, which is near-tautological: a row with no odds cannot
produce a pick.

**But odds-row count is book coverage, not price quality, so the real question
was tested directly** — best available `1X2` Home price on each twin:

| | n |
| --- | --- |
| pairs where both twins carry odds | 19 |
| ...comparable on best home price | **6** |
| picked twin **better** priced | 3 |
| picked twin **worse** priced | **2** |
| identical | 1 |

The two worse cases: `50920` 2.68 vs twin 2.70; `6002` 3.80 vs twin 3.90.

> **No evidence of systematic bias in the taken price, and n=6 with deltas of
> ≤0.10 is far too small to exclude one.** Recorded as an open question with its
> sample size attached, not as a clearance.

**The closing capture resolves against the row the pick landed on, so taken and
closing prices stay internally consistent** — the exposure is to the *level* of
the taken price, not to a taken/closing mismatch.

## RECALL — the label, kept

> **750 is a FLOOR, not an estimate.** Identical-kickoff bought precision and
> lost every pair whose sources disagree on kickoff time. **The one confirmed
> s5.3 violation sits in the WEAK set, outside the strict 750.** The ±26h
> membership test used in Part 1 implicates 4,329 rows — the true population is
> somewhere between, and is not established.

*Established 2026-08-31. No repair, no build.*

---

# GUARD DESIGN — POSITION, the second rule

**Filed beside *"a lookup table is only as good as the earliest decision point
that consults it"*, because they are the same insight from opposite sides.**

`team_names_similar` returns **False** on `"Vitória SC"` vs stored
`"Guimaraes"`, and **True** on stored `"Guimaraes"` vs stored `"Guimaraes"`.
Same comparator, same clubs, opposite answers — because team resolution happens
in between.

> ## A comparison's reliability is a property of its POSITION in the pipeline, not of the comparator.
>
> **The same test applied before normalisation and after it is not the same
> test.** Benchmark a comparator in isolation and you measure something the
> running system never does.

**The pair, stated together:**

| rule | failure it names |
| --- | --- |
| *A lookup table is only as good as the earliest decision point that consults it.* | knowledge exists but the deciding caller never reads it |
| **A comparison is only as good as the resolution state of its inputs.** | the caller reads it, at a point where its inputs are not yet comparable |

**Between them they explain three defects already in this ledger, which is the
test of whether a rule is worth keeping:**

* **the alias that never fired** — `TEAM_NAME_ALIASES["Athletic Club"]` was
  correct and consulted at step 2, while `_get_or_create_team_id` decided at
  step 0. *(Rule 1.)*
* **the identity gate's false positives** — the gate compares a provider name
  against a stored one at ingest, the least-resolved point available. *(Rule 2.)*
* **the duplicate rows** — same comparison, same position, and it is why five
  pairs that are trivially recognisable once both rows exist were never merged.
  *(Rule 2.)*

**And it is why s5.9 is a pick-time check rather than a better matcher.** Not a
preference: **measured.** All five live duplicate pairs pass
`team_names_similar` on both sides when the two STORED rows are compared, and
all five failed at ingest. **The position is doing the work, not the predicate.**

---

# s5.9 — THE CAP KEYS ON FIXTURE IDENTITY

**`src/data/fixture_identity.py`. Built, tested, cohort bumped.**

| | |
| --- | --- |
| `CODE_REVISION` | `s5.8` → **`s5.9`** |
| `model_version` | `stage5_baseline_20260807.dfe302` → **`.694a60`** |
| cohort | **0 picks stamped** — clean break, nothing to pool |
| tests | **887 pass** (9 new) |

## The rule, and the unequal evidence behind its branches

> **Two rows are the same fixture if they share a league, a kickoff minute, and
> at least one club resolved to the same provider id.**

| branch | evidence | measured |
| --- | --- | --- |
| **1 — provider id** | **PROVABLE.** A club cannot play two fixtures in one competition at one minute. No tunable part; cannot drift. | 835 pairs |
| **2 — stored names** | **HEURISTIC, measured not assumed.** | **89 refusals, 0 provably wrong** |
| **3 — residual** | **DECLARED.** `Vitória SC`/`Guimaraes`, zero shared tokens. | not empty |

### Branch 2's false-positive rate, on the hardest available sample

**The only part shipping on anything but proof, so it was measured before it
was built.** Run against every same-league same-minute pair in the database,
and again at a wider ±4h:

| | ±4h | exact minute |
| --- | --- | --- |
| pairs | 78,400 | 60,976 |
| branch 1 fires (excluded — branch 2 only runs where 1 does not) | 858 | 835 |
| branch 2 eligible | 77,542 | 60,141 |
| **branch 2 would refuse** | **89** | **89** |
| **provably different fixtures → FALSE POSITIVE** | **0** | **0** |

**Three independent deciders were applied to those 89, not one:**

| decider | pairs it condemns |
| --- | --- |
| both slots resolve on both rows and differ | **0** |
| both rows carry `matches.apifootball_id` and they differ | **0** |
| both rows carry `flashscore_id` and they differ | **0** |

> **MEASURED FALSE-POSITIVE RATE: 0 of 89.** The bound is **[0.0%, 40.4%]**
> only if every pair where *neither* row carries a provider fixture id is also
> wrong — which inspection contradicts: `Man United v Man City` ‖
> `Manchester Utd v Manchester City`, `Salernitana v AC Milan` ‖
> `US Salernitana 1919 v Milan`.

**DECISION: branch 2 REFUSES, it does not merely warn.** Stated before
building, on that number. **Had it come back high, branch 1 alone still catches
both confirmed violations** — every one of the four known duplicate pairs
groups via `branch=provider_id` — and branch 2 would have been demoted to a
logged warning. It did not, so it refuses.

**This is the difference between a measured heuristic and the fifteen
data-fitted thresholds the 2026-08-07 audit condemned:** those were fitted to
the sample they were measured on. This is measured on the *hardest* sample
available and would have been abandoned had the number gone the other way.

## Refusing on evidence, never on its absence

**60,141 of 60,976 same-league same-minute pairs are genuinely different
simultaneous fixtures** — the norm on a final matchday. **A rule that failed
closed on unresolvable pairs would reject the whole card.** The same
distinction `_parse_match_date` draws: **unknown is not wrong.**

## Verified against production, not only against fixtures

| pair | result | branch |
| --- | --- | --- |
| 2026-08-30 Deportivo v Valencia (**the s5.3 violation**) | **GROUPED** | `provider_id` |
| 2026-08-14 Sporting CP v Guimarães (3 picks) | **GROUPED** | `provider_id` |
| 2026-08-31 Braga v Guimarães | **GROUPED** | `provider_id` |
| 2026-08-30 Celta v Ath Bilbao | **GROUPED** | `provider_id` |

**Controls, which matter as much as the catches:**

* today's **20 picked fixtures → 20 groups. Nothing collapsed.**
* a full matchday, **80 fixtures on 2026-08-30 → 78 groups** — exactly the two
  known duplicates, nothing else.

## Two implementation details worth the ledger

**The twin is usually not a candidate.** On 08-31 row 50990 carried 78 odds and
produced no pick. A resolver seeing only candidates would have grouped nothing.
It loads every row sharing a `(league, kickoff)` with any input — and the saved
picks' `match_id`s go in too, because **a pick saved earlier today on the twin
must consume this fixture's slot.** That is the cross-run half of the guarantee.

**A bug the tests caught that production would have hidden.** The `IN :ids`
bind needs `expanding=True`; without it the driver raises, the defensive
`except` logs a warning, and the resolver degrades to row identity. **The
feature would have been a no-op while reporting that it was fine** — the exact
shape this ledger keeps finding. The fallback behaved correctly; the fallback
was also what would have concealed it. Pinned by
`test_a_broken_session_degrades_to_row_identity_and_does_not_raise`.

**And the SQLite/Postgres kickoff type again:** SQLite returns `match_date` as
text, Postgres as a datetime, and the bucket key is `(league, kickoff)`. Mixed
types put one fixture's rows in two buckets and group nothing. Normalised at
the boundary, once — the same trap `coverage_checks` hit two days ago.

## The number that set the deadline

**Not severity — ordering.** Zero of the 48 MODEL observations sit on a
duplicate row, but that separation is **temporal**: captures ran 08-14 → 08-27,
the live duplicates are 08-28 onward, and **the credit budget resets tomorrow,
which is when captures resume.** Every observation taken on a duplicated
fixture from that point inflates the cluster count that `deff` is computed
from, and `deff` underwrites every confidence interval in this project.

> **Built before the next capture window rather than after it.**

## Still a floor

**750 duplicate pairs under the strict identical-kickoff test, and the one
confirmed violation sits OUTSIDE it.** Identical kickoff bought precision and
lost every pair whose sources disagree on time. A ±26h membership test
implicates 4,329 rows. **The true population is between, and is not
established.** `test_a_different_kickoff_minute_is_a_different_bucket` pins the
limitation rather than papering over it.

**No row repaired.** s5.9 stops the guarantee being violated; it does not merge
the duplicates already there.

*Built 2026-08-31.*

---

# GUARD DESIGN — A FALLBACK MUST RECORD THAT IT FIRED

**The extension of the fail-closed rule, and the family it completes.**

The `IN :ids` bug degraded to row identity, logged, and returned a valid
mapping. **The feature would have been a no-op in production while reporting
that it was fine** — and the concealment came from defensive code doing exactly
what it was written to do.

| instance | what it concealed |
| --- | --- |
| `datetime.now()` default | a kickoff that never parsed |
| swallowed `IntegrityError` | a write that never landed |
| short-circuit suppressing its own logging | the branch it took |
| `[]` meaning both measured-and-clean and never-ran | a check that did not execute |
| **`except` → degrade to row identity** | **a guarantee that was not being enforced** |

**The new element: the previous four were mistakes. This one was correct code.**

> ## Degradation is acceptable. SILENT degradation is the defect.
>
> **A fallback that does not announce itself makes the thing it protects
> unfalsifiable** — every observation looks the same whether the protected
> mechanism ran or not.

**Corollary, and it is the operational half:** *the log level of a fallback is
part of its contract, not a stylistic choice.* `DEBUG` on a path that disables
a guarantee is the same defect as no log at all, because nothing reads DEBUG in
CI.

## Three live instances found by applying the rule to today's own work

**Written hours ago, by me, while recording the rule that condemns them.**

**1. The outermost catch on the unpriced check.** `daily_update` wrapped the
call in `except Exception: logger.debug(...)`. `report_unpriced_fixtures`
announces its own failure — but that line cannot be emitted if the call never
reaches it. An import error or a failed session landed in the outer handler at
DEBUG, and **the run would have looked clean while the check had not run.**
Now WARNING, with the same `CHECK DID NOT RUN` wording so both paths converge.

**2. `ci_audit.extract` silently dropped the detector built to catch exactly
this.** The `else` branch does `int(ms[-1])` inside
`except (TypeError, ValueError): pass`. `unpriced_check_dead` has **no numeric
capture group**, so it matched, failed to convert, and never reached `facts`.

> **The assertion written to detect a dead check had been dead itself since it
> was added two days ago, and nothing failed.** A pattern that matches but
> cannot be counted now prints that it was not counted.

**3. The population, sized rather than swept.** `except` handlers whose only
record is `logger.debug`, across `src/`:

| file | n |
| --- | --- |
| `betting_agent.py` | 20 |
| `match_briefing.py` | 8 |
| `flashscore_scraper.py` | 5 |
| `historical_loader.py` | 5 |
| `injury_scraper.py` | 5 |
| 13 others | 26 |
| **total** | **69 across 18 files** |

**Not claimed to be 69 defects.** Many guard genuinely inconsequential things —
a cache warm, a debug annotation. **The rule sorts them:** a handler that
degrades a *guarantee, a measurement, or a gate* must log at WARNING; one that
degrades a convenience need not. **Nobody has sorted these 69, and that is a
stage, not a paragraph.**

`test_every_way_the_check_can_fail_reaches_the_audit` pins both failure paths
of the one check that has been sorted.

---

# PROPOSAL — DUAL-ENGINE TESTS FOR IDENTITY-DECIDING QUERIES

**Not built. Recorded with its instances so it can make its own case.**

## The structural problem

`conftest.py` strips `DATABASE_URL` so the suite runs on **SQLite**; production
runs on **Postgres**. That strip is correct and must stay — it exists because
DB-backed tests once wrote to the live database.

> **Its consequence is that any behaviour differing between the two engines is
> invisible to the tests BY CONSTRUCTION.** Not because the tests are weak —
> because they execute against a different engine than the one that matters.

**Same shape as the audit measuring what the pipeline says about itself: the
instrument cannot see the thing it is meant to check.**

## Three instances, all real, none hypothetical

| # | defect | what passed anyway |
| --- | --- | --- |
| 1 | **missing `PRAGMA foreign_keys=ON`** (`database.py:116`, Stage 13 1c) — SQLite defaults it OFF | **39 tests passing from a state the production schema forbids** |
| 2 | **`coverage_checks`' Postgres-only SQL** — `match_date::date`, `percentile_cont`, `FILTER` | on SQLite it raised, was swallowed, and returned **"no unpriced fixtures"** — clean-looking, never run |
| 3 | **`fixture_identity`'s kickoff type** — SQLite returns text, Postgres a datetime, and the bucket key is `(league, kickoff)` | mixed types put one fixture's rows in **two buckets**, grouping nothing — **two days after instance 2** |

**Instances 2 and 3 are both mine, eight days apart in the same subsystem.**

## The proposal, deliberately narrow

> **Any query whose result becomes a KEY, a BUCKET, or a COMPARISON runs
> against both engines in the test suite, and the results must be identical.**

**Not the whole suite** — that would be expensive and mostly pointless, since
most queries fetch values rather than decide identity. **Only the queries where
a type or dialect difference changes an identity decision**, which is short and
enumerable. **All three instances sit in it**, which is the evidence that the
scope is drawn in the right place.

**Cost, honestly:** a Postgres instance in CI (a service container, or a
throwaway schema on the existing one), and a fixture that runs a named query
twice and diffs. **The `DATABASE_URL` strip must survive it** — the second
engine has to be a disposable test database, never the production URL, or this
proposal reintroduces the defect `conftest.py` exists to prevent. **That
constraint is the risky part and it is why this is a proposal rather than a
patch.**

## Its own falsification

> **If a fourth instance arrives before this is built, the proposal has made
> its own case and should stop being a proposal.**

Three instances, three different failure modes — a disabled constraint, a
swallowed dialect error, and a type mismatch in a key. **Recorded 2026-08-31.**

---

# SEQUENCE, agreed

| when | what |
| --- | --- |
| **tomorrow** | credit budget resets, captures resume — **the first CLV observations under `s5.9`**, into a pipeline where the per-fixture guarantee holds |
| **Wed 2026-09-02** | the schedule margin test, against the **corrected ~09:29 boundary**, and only after checking Wednesday's own card is not itself unusually late |
| **then** | **H1 / H4** — 24,524 snapshots, 108 keys with ≥3 observations. Price momentum is the last open question that could change this project's answer, and it has been unanswerable since the beginning |

## Deferred, carried forward unchanged

* the duplicate-row **merge** (s5.9 stops new violations; it repairs nothing)
* the **identity repair** for rows 124 and 411, plus creating Maccabi Tel Aviv
* **invariants 2 and 3** (known defective; declarations must not cite the count)
* **MASK-1's** mechanism tests
* the **alias-site symmetry** fix in `team_names_similar`
* **OPS-3**
* **and new today:** sorting the 69 DEBUG-only handlers, and the dual-engine
  proposal above

*2026-08-31.*

---

# AUDIT 2026-09-01 — s5.9's first live run, and the budget reset

**Read-only. Nothing fixed.** Registered files read first: the corrected **~09:29**
boundary in `stage21-schedule-prediction.md`, and the `s5.9` history entry.

## PART 1 — s5.9: IT RAN, AND IT REFUSED NOTHING

**Run `33485294975`. 29 fixtures analysed, 29 picks saved, every one stamped
`stage5_baseline_20260807.694a60`.** The cohort stamp is what proves the s5.9
code was live.

| measurement | value |
| --- | --- |
| `match_id` rows carrying picks | **29** |
| fixture-identity groups formed | **29** |
| collapsed | **none** |
| whole card (29 rows) | **29 groups, 0 collapsed** |
| `PICK_REJECTED` of any reason | **0** |
| refusals on branch `provider_id` / `stored_names` | **0 / 0** |

**No group contained more than one row. The key caught no duplicate, because
there was none to catch.**

### Did it run, or did it silently not execute?

**IT RAN. Three pieces of evidence, and the third is the only positive one:**

1. picks carry `694a60`, so the s5.9 code was the code that executed;
2. **no `fixture-identity resolution failed` WARNING** — that is the degrade
   path, it logs at WARNING, and its absence rules out the `_load` failure;
3. replaying `resolve_fixture_groups` over the same 29 rows today forms 29
   groups — consistent with the run's silence.

> ### BUT THE MODULE EMITS NO UNCONDITIONAL "I RAN" SIGNAL, and that is a defect I shipped yesterday.
>
> `resolve_fixture_groups` logs **only when it unions a group**. A clean day and
> a dead call produce byte-identical logs. I could separate them only because
> the degrade path happens to log at WARNING and because the cohort stamp is an
> independent witness — **neither of which is the module announcing itself.**

**This is the fifth instance of the family, in code written one day after
recording the rule that condemns it:** *a fallback must record that it fired* —
and its sibling, **a check must record that it ran.** A one-line unconditional
summary (`N rows -> M groups`) would close it. **Not fixed; this is read-only.**

### The guarantee, checked independently of the key

**By `(competition, kickoff minute, resolved teams)`, never by `match_id`:**

| | |
| --- | --- |
| duplicate fixture pairs on the day (wide ±4h) | **0** |
| pairs carrying picks on **both** rows | **0** |

> **THE GUARANTEE HELD. But it held because the day had no duplicates, not
> because the key refused anything. A clean day is not a demonstration.**

### The control, and it is stronger than yesterday's

Four `(league, kickoff)` buckets carry more than one pick — **all genuinely
different simultaneous fixtures, none collapsed:**

| league | kickoff | picks | distinct rows |
| --- | --- | --- | --- |
| england/championship | 18:45 | 6 | 6 |
| england/championship | 19:00 | 2 | 2 |
| england/league-one | 18:45 | 7 | 7 |
| **england/league-two** | **18:45** | **12** | **12** |

**Twelve simultaneous fixtures in one league, kept apart correctly.** That is
the false-collapse control at a scale yesterday's 20-fixture check never
reached.

## PART 2 — the budget reset, and a gate that does not read its own date

**API-Football (`api_budget`, keyed per day per provider):** a fresh row opened
for 2026-09-01, `used=91 limit=100`. **Nothing carried forward.** Clean.

**TheOddsAPI (monthly, persisted to `data/models/theodds_credits.json`):**

```
08:0x  ⚠️ TheOddsAPI credits low (from last run): 15 remaining
09:0x  TheOddsAPI update complete: 2343 odds rows written, 31 games matched,
       0 unmatched (credits remaining: 490)
```

**The stale August figure DID carry across the period boundary** and produced a
false low-credit warning, corrected seconds later by the live call. September
opened at ~500. **The ledger and the provider now agree; they did not at the
moment the run started.**

### THE LATENT DEFECT THIS EXPOSES

`theodds_scraper.py:626`:

```python
_persisted = _load_persisted_credits()
if _persisted is not None and _persisted <= _CREDITS_GATE_THRESHOLD:   # 10
    ... hard skip
```

The persisted object is `{"remaining": N, "updated": "YYYY-MM-DD"}`.

> **The gate reads `remaining` and never reads `updated` — the field in the
> same object that would tell it the figure belongs to a closed period.**
>
> **August ended at 15. The gate is ≤10. It passed by FIVE CREDITS.** Had
> August ended at ten or fewer, the first day of a fresh ~500-credit month
> would have been hard-skipped on a number that was already void.

**THE HABIT INVERTED, mild form:** the knowledge is present, in the same object,
and the deciding branch does not look at it. Recorded, not fixed.

### Captures have resumed

| day | odds rows | fixtures | snapshots |
| --- | --- | --- | --- |
| 2026-08-31 | 2,997 | 22 | 3,017 |
| **2026-09-01** | **4,533** | **29** | **4,643** |

**But no CLV observation has closed yet: 2026-09-01 is 29 `pending`, 0
`captured`.** So **zero MODEL observations exist under s5.9** — the first ones
are still ahead.

### A CORRECTION TO YESTERDAY'S deff CHECK

**The MODEL series is still 48, spanning 2026-08-14 → 08-27, unchanged.**

> **Yesterday I reported "0 MODEL observations on duplicate-pair rows". Under
> the predicate `s5.9` ACTUALLY SHIPS, the answer is 1.**

Yesterday's script required **both** sides to match; the shipped rule needs
**at least one shared provider id OR both names similar**. The narrower
predicate missed `49496` — `Sporting Clube de Portugal v Guimaraes`, the 08-14
pair, which shares `Guimaraes` at provider id 224.

**The conclusion survives, and the margin is one capture:**

| row | picks | MODEL obs | **captured** |
| --- | --- | --- | --- |
| 49496 `Sporting Clube de Portugal v Guimaraes` | 1 | 1 | **1** |
| 49520 `Sporting CP v Guimaraes` | 2 | 2 | **0** |

**48 observations → 48 distinct fixture ids. The cluster count is NOT inflated
and `deff = 1.00` stands, now measured with the right instrument.**

> **It stands because two observations on the twin were never captured. Had
> either closed, the cluster count would have been inflated and Stage 16's
> +0.107% would have needed recomputing.** The separation is one closing line
> wide, not a margin.

**And the general point, which is the one that matters:** *a verification is
only as good as the predicate it uses, and mine was not the predicate the
system runs.* The right answer arrived anyway, which is exactly how a flawed
instrument survives.

## PART 3 — the schedule: a second point, not a verdict

**Started `2026-09-01T08:05:40Z`. Delay against `0 3 * * *`: 5h 05m 40s.**

| outcome | boundary | Tuesday |
| --- | --- | --- |
| **WITHIN TOLERANCE** | **start ≤ 08:42** | **← 08:05:40** |
| LATE BUT COVERING | 08:42 → ~09:29 (corrected) | — |
| BEYOND TOLERANCE | beyond ~09:29 | — |

**First firing since the change to land inside the historical 0.5–5.7h
envelope.**

### Tuesday's earliest kickoff, recorded BEFORE any conclusion

> **17:30 UTC.** The card ran 17:30 → 19:00.

**Tuesday is therefore uninformative about the margin, for exactly the reason
Monday was.** Any delay under ~9 hours would have scored identically. **Stated
before drawing the conclusion, not discovered after it.**

| measurement | value |
| --- | --- |
| fixtures in window at execution | **29** |
| already kicked off at 08:05:40 | **0** |
| **fraction of the card lost** | **0.0%** |
| lead time (n=29) | min 8.8h · **p10/median/p90 all 10.1h** · max 10.3h |

**Against the projected 7.1 / 10.7 / 15.4h: the median of 10.1h is the closest
any day has come, and it sits ABOVE the entire on-time baseline of 4.4–8.8h.**

**The first day the lead-time gain is visible — and it is confounded.** The card
is compressed into a 90-minute evening band, which lifts lead time regardless of
when the run starts. **This is not yet evidence that the cron change produced
it.**

### Wednesday cannot be pre-checked, and that is an ordering problem

**The registered instruction is to check Wednesday's card for lateness BEFORE
reading its result. As of 2026-09-02 07:16 UTC there are ZERO fixtures in the
database for 2026-09-02** — because the fixtures are created by the very run
being awaited.

> **The pre-check the checkpoint depends on cannot be performed from this
> system's own data until the run it is meant to qualify has already happened.**

An independent source (football-data.org's calendar) would be needed to do it
properly. **Recorded as a limitation of the registered method, not worked
around.**

**Today's run has not fired at 07:16 UTC — 4h16m past cron, inside the envelope,
and not MISSED under the OPS-3 12-hour rule (not assessable before 15:00 UTC).**

## PART 4 — the routine pass

**14 runs listed, 12 genuinely new** (two are already in the ledger and are
re-listed because the tool matches on id).

| run | workflow | started | verdict | discovery |
| --- | --- | --- | --- | --- |
| 33421026022 | paper-report | 08-31 17:42 | CLEAN | |
| **33421027825** | **closing-lines** | **08-31 17:42** | **DEGRADED** | 7 league requests returned `no_rows` — credits spent, empty event lists |
| 33422515296 | closing-lines | 08-31 17:59 | CLEAN | |
| 33449620550 | closing-lines | 08-31 23:11 | CLEAN | |
| 33460156282 | closing-lines | 09-01 01:48 | CLEAN | |
| **33485294975** | **daily-picks** | **09-01 08:05** | **DEGRADED** | **`disc[fs=17 fdo=8 af=12]`** — 1 active-season league returned 0 fixtures |
| 33524168975 | paper-report | 09-01 15:11 | CLEAN | |
| 33524281030 | closing-lines | 09-01 15:12 | CLEAN | |
| 33525843869 | closing-lines | 09-01 15:27 | CLEAN | |
| 33552321881 | closing-lines | 09-01 19:55 | CLEAN | |
| 33570292806 | closing-lines | 09-01 23:16 | CLEAN | |
| 33577785282 | closing-lines | 09-02 01:02 | CLEAN | |

**`conclusion: success` on all twelve remains no evidence.**

### YESTERDAY'S FIX EARNED ITS KEEP ON DAY ONE

The "pattern matched but produced no number" line added yesterday **fired
immediately**, on a *different* pattern:

```
ci_audit: pattern 'src_apifootball_fixtures' matched but produced no number
          — NOT COUNTED.
```

> **A second silently-dropped pattern, found within a day of making the drop
> visible.** `unpriced_check_dead` was not the only one. Recorded, not fixed.

### The three carried items

**Unpriced alarm — MEASURED, not `None`.** The run logged no `UNPRICED` line at
all, and the replay confirms why: **0 unpriced rows on 09-01 and on 09-02, so
0 ALARM and 0 INFO.** Silence here is the clean case, and it is corroborated by
replay rather than assumed.

**Identity gate — 0 refusals. False positives remain 0, and again VACUOUSLY:**
neither Maccabi Tel Aviv nor Cracovia played. This is the second consecutive
day where the correct-refusal half of the `f4f5c2d` baseline went untested.

**Substrate:**

| | 08-31 | **09-01** |
| --- | --- | --- |
| `odds_snapshots` | 24,524 | **29,167** |
| keys with ≥3 observations | 108 | **123** |
| `injury_observations` | 636 | **756** |

**Two further series, steady:** Flashscore kickoff-parse refusals **187**
(vs 207, 192) — no regression. Flashscore discovery continues to fall,
`fs=48 → 26 → 17`, while API-Football rises, `af=1 → 2 → 12`.

## PART 5 — the cohort transition, observed for the first time

```
CODE_REVISION : s5.9
model_version : stage5_baseline_20260807.694a60
picks stamped : 29   (of 1487 saved picks)
VERDICT: BUMP
```

**The amend-while-empty window closed the moment the first pick landed**, and
the transition is clean:

* **29 picks dated 2026-09-01 carry `694a60`. ZERO carry an earlier
  fingerprint.**
* every prior cohort intact and unmerged — `485823` (246), `098437` (18),
  `60caed` (34), `645bac` (66), `dfe302` (20).

**A further prediction- or selection-affecting change now requires `s5.10`.**

*Read-only audit, 2026-09-02, of the 2026-09-01 run.*

---

# THE PHANTOM AUDIT — every schedule figure re-derived

**2026-09-02. Read-only. The `0 3 * * *` decision is NOT reversed; its basis is
narrowed and restated.**

## 1a. The day-of-week table that chose 03:00

**August 2026. Left: as it was computed. Right: phantom-free.**

| day | n | /day | earliest | median | %≤14:00 | | **n** | **/day** | **earliest** | **median** | **%≤14:00** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mon | 213 | 42.6 | 10:34 | 13:13 | 69% | → | **65** | **13.0** | **15:00** | **17:30** | **0%** |
| Tue | 63 | 15.8 | 10:13 | 12:52 | 56% | → | **28** | **7.0** | **15:00** | **18:00** | **0%** |
| Wed | 119 | 29.8 | 10:15 | 10:28 | 84% | → | **19** | **4.8** | **16:00** | **19:00** | **0%** |
| Thu | 119 | 29.8 | 10:25 | 17:00 | 7% | → | 111 | 27.8 | 15:00 | 17:00 | **0%** |
| Fri | 73 | 18.2 | 10:25 | 18:30 | 3% | → | 71 | 17.8 | 15:00 | 18:30 | **0%** |
| **Sat** | 294 | 58.8 | **10:15** | 14:00 | 50% | → | **267** | **53.4** | **10:15** | 15:00 | **45%** |
| **Sun** | 325 | 65.0 | 10:04 | 15:00 | 43% | → | **226** | **45.2** | **10:15** | 15:00 | **34%** |

> ### EVERY WEEKDAY'S EARLY-KICKOFF CLAIM COLLAPSES TO ZERO. Only Saturday and Sunday survive.

**Mon/Tue/Wed lost 60–84% of their rows and their medians moved by four to nine
hours.** Thu and Fri kept their rows but their `%≤14:00` still fell to 0 —
because the few phantoms they had were the *only* pre-14:00 entries.

**Saturday is the most robust row in the table** (294→267, earliest unchanged at
10:15), which is why the contamination was invisible: the day the decision most
depended on was the day least affected.

## 1b. The earliest kickoff that set the boundary — **it was itself a phantom**

| | earliest | p05 | n |
| --- | --- | --- | --- |
| as computed | `10:04:29.761924` | `10:13:51.680390` | 1,206 |
| **PHANTOM-FREE** | **10:15:00** | **11:30:00** | **787** |

**`10:04:29.761924` carries microseconds.** The figure that anchored the whole
tolerance calculation was a phantom row, and p05 moves by **76 minutes**.

### The boundary, recomputed — and my own correction overshot

Picks are written **34m35s** after start (MEASURED 2026-08-31).

| basis | boundary | margin from 03:00 |
| --- | --- | --- |
| 10:04 − ~19m (original, assumed run length) | 09:45 | 6h45m |
| 10:04 − 34m35s (my correction, phantom earliest) | **09:29** | 6h29m |
| **10:15 − 34m35s (REAL earliest)** | **09:40:25** | **6h 40m 25s** |

> **The true boundary is 09:40. The original 09:45 was five minutes too late;
> my correction to 09:29 was eleven minutes too early.** Two errors — an
> assumed run length and a phantom kickoff — pushed in opposite directions, and
> the original was closer by accident. **Correcting one of two compensating
> errors moved the number further from the truth**, which is the case for
> re-deriving both rather than patching either.

## 1c. The 2026-08-29 figure — **UNCHANGED**

| | kicked off by 14:40 |
| --- | --- |
| as reported | 57 of 104 (55%) |
| **PHANTOM-FREE** | **57 of 104 (55%)** |

**2026-08-29 contains zero phantom rows, so the figure that priced lateness
survives intact.** (Reported as 103 at the time; 104 on re-query — a one-row
difference, not material.)

**The days that were mostly phantom are the ones already flagged:**

| date | rows | phantom | real |
| --- | --- | --- | --- |
| 2026-08-23 | 111 | **97 (87%)** | 14 |
| 2026-08-25 | 24 | **23 (96%)** | 1 |
| 2026-08-27 | 36 | 0 | 36 |
| 2026-08-28 | 29 | 0 | 29 |

**The phantom class is concentrated, not spread** — which is why one clean day
(08-29) carried a valid measurement while the day-of-week aggregate did not.

## 1d. Card lost by delay, phantom-free — **the justification narrows to weekends**

| day | +1h | +3h | +5h | +8h | +11h |
| --- | --- | --- | --- | --- | --- |
| Mon–Fri | **0%** | **0%** | **0%** | **0%** | **0%** |
| **Sat** | 0% | 0% | 0% | **4%** | **45%** |
| **Sun** | 0% | 0% | 0% | **2%** | **34%** |

> ### THE DECISION SURVIVES, AND ITS BASIS NARROWS TO SATURDAY AND SUNDAY.
>
> **On a weekday, an eleven-hour delay from 03:00 costs nothing.** The entire
> delay-tolerance case for moving the cron rests on two days a week — the two
> days that also carry the most fixtures (53.4 and 45.2 per day against 4.8 to
> 27.8).

**03:00 remains correct.** Real weekend cards start at **10:15**, picks land
34m35s after start, so the boundary is **09:40** and 03:00 buys **6h40m**. The
settlement constraint (football ends ~21:30) is unaffected by phantoms because
it was derived from latest kickoffs, which phantoms do not extend.

**What is withdrawn: the claim that the change protects the weekday card.** It
does not, because the weekday card was never at risk.

---

# 3. SATURDAY 2026-09-05 — thresholds re-derived BEFORE the day

**Registered here on 2026-09-02, three days ahead, from phantom-free data.**

| start | Sat (registered, contaminated) | **Sat PHANTOM-FREE** | **Sun PHANTOM-FREE** |
| --- | --- | --- | --- |
| 09:40 (the boundary) | — | **0.0%** | **0.0%** |
| 09:45 | 0% | **0.0%** | **0.0%** |
| 11:00 | 9.2% | **4.1%** | **1.8%** |
| 12:00 | **20.5%** | **12.4%** | **15.5%** |
| 13:00 | — | **16.1%** | **26.1%** |
| 14:00 | **50.5%** | **45.3%** | **34.1%** |
| 15:00 | — | 51.3% | 55.3% |

**n = 267 (Sat), 226 (Sun).**

**The registered gradient was overstated at every point** — 20.5%→**12.4%** at
12:00, 50.5%→**45.3%** at 14:00.

### The outcome bands, applied to the corrected curve

Bands unchanged as registered — **MARGIN HELD** 0%, **MARGIN CONSUMED** >0–20%,
**MARGIN INSUFFICIENT** >20%. What moves is *where the run start has to land*:

| outcome | start time (contaminated) | **start time (CORRECTED)** |
| --- | --- | --- |
| MARGIN HELD | ≤ ~09:45 | **≤ ~09:40** |
| MARGIN CONSUMED | ~09:45 → ~12:00 | **~09:40 → ~13:20** |
| MARGIN INSUFFICIENT | beyond ~12:00 | **beyond ~13:20** |

> **The 20% crossing moves roughly 80 minutes later, from ~12:00 to ~13:20.**
> A run starting at 12:30 would have been scored MARGIN INSUFFICIENT on the old
> table and is **MARGIN CONSUMED** on the corrected one.

**Saturday is now the PRIMARY checkpoint. Sunday 2026-09-06 is the secondary**,
and it is a genuine second test rather than a repeat: it crosses 20% earlier
(between 12:00 and 13:00) despite a later median, because its losses accumulate
more evenly.

**Registered 2026-09-02, three days before the day it governs.** *The previous
entry was committed four minutes after its run began; this one is not.*

---

# 2. THE THREE RULES ARE ONE RULE

**Filed together, because separately they read as three lessons and they are
not.**

| | rule | how the caller fails |
| --- | --- | --- |
| 1 | **A lookup table is only as good as the earliest decision point that consults it.** | decides **too early** — before the knowledge is read |
| 2 | **A comparison is only as good as the resolution state of its inputs.** | compares **too early** — before the inputs are comparable |
| 3 | **An exclusion is only as good as the queries that apply it.** | writes a **new query** that does not know the knowledge exists |

> ## THE KNOWLEDGE EXISTS AND THE CALLER DOES NOT USE IT.
>
> Not because it is missing, not because it is wrong, and not because anyone
> forgot — but because **correctness is a property of the call site, and the
> call site cannot see what it does not consult.** Curating the knowledge
> better does nothing for any of the three.

**Five defects, all accounted for by the same sentence:**

| defect | rule |
| --- | --- |
| `TEAM_NAME_ALIASES["Athletic Club"]` — correct, curated, never fired | **1** |
| the identity gate's false positives — provider text vs stored name at ingest | **2** |
| duplicate match rows — the same comparison at the same position | **2** |
| the `deff` predicate — verified with a narrower rule than the one shipped | **3** |
| **the phantom-contaminated schedule analysis** | **3** |

**And the diagnostic that distinguishes them, since the remedy differs:** ask
*where* the failure sits relative to the knowledge. **Before it** → rule 1, move
the read earlier. **Beside it, on unresolved inputs** → rule 2, move the
comparison later. **Outside it, in new code** → rule 3, and there is no
"earlier" or "later" to move to — **rule 3 is the one that recurs forever**,
because every future query inherits the contamination by default.

---

# 4. THE ODDS API GATE — a monthly near-miss, WITH A DEADLINE

**`theodds_scraper.py:626`:**

```python
_persisted = _load_persisted_credits()
if _persisted is not None and _persisted <= _CREDITS_GATE_THRESHOLD:   # 10
    ... hard skip
```

The persisted object is `{"remaining": N, "updated": "YYYY-MM-DD"}`. **The gate
reads `remaining` and never reads `updated`.**

**MEASURED 2026-09-01: August closed at 15 against a gate of 10. It passed by
five credits**, then the live call reported ~490.

> **A persisted figure whose `updated` date is not the current period is not a
> LOW reading. It is NO reading — and the response to no reading is to PROBE,
> not to skip.** Exactly the `[]` versus `None` distinction already drawn in
> `coverage_checks`: empty means measured-and-clean, absent means unmeasured.

**`GET /v4/sports` is free**, so the probe costs nothing.

> ## DEADLINE: end of September 2026 — before the 2026-10-01 boundary.
>
> **Recorded with a date, not on the undated list.** It recurs on the first of
> every month and its blast radius is a full month of odds coverage skipped on
> a number describing a month that had ended.

---

# 6. PROPOSAL — A GUARD THAT CAN SILENTLY DO NOTHING MUST SAY IT RAN

**`resolve_fixture_groups` emits no unconditional line. Written one day after
recording the rule that condemns exactly that.**

> **That is the evidence the rule cannot be applied by intention alone.** I
> recorded it, agreed with it, and violated it inside 24 hours in the code I
> wrote to satisfy it.

**The structural form:**

1. **every guard that can silently do nothing emits a count** — `N in → M out`,
   unconditionally, at INFO;
2. **a test asserts the line exists**, so the silence is a test failure rather
   than a reading-comprehension failure.

**Same stage as sorting the 69 DEBUG-only handlers, and it is one problem seen
from both ends:**

| | |
| --- | --- |
| handlers | say **too little when something fails** |
| guards | say **nothing when nothing does** |

**Both make a run unfalsifiable from its own log**, which is the property this
ledger exists to prevent. **Not built.**

*Recorded 2026-09-02.*

---

# GUARD DESIGN — COMPENSATING ERRORS, and why a plausible number invites no audit

**Filed beside the three rules, because it is about how they stay hidden.**

The 09:45 boundary had **two** wrong inputs: an assumed ~19-minute run length,
and a kickoff of `10:04:29.761924` that was a phantom row. I measured the run
length, corrected it to 34m35s, and moved the boundary to 09:29.

| | boundary | error |
| --- | --- | --- |
| original (assumed run, phantom kickoff) | 09:45 | **5 min too late** |
| my correction (measured run, phantom kickoff) | 09:29 | **11 min too early** |
| **both inputs correct** | **09:40** | — |

> ## When a derived number has several inputs, correcting one and not the others can move it further from the truth than correcting none.
>
> **A number's inputs must be audited together, not one at a time as each is
> discovered.** Partial correction is not partial progress; on compensating
> errors it is regression, and it arrives wearing the authority of a
> measurement.

**And the second half, which explains the survival:** the two errors pointed in
opposite directions, so **09:45 was closer to right than it deserved to be.**
A number that looks reasonable is not audited. **The same family as every defect
in this ledger — the failure that presents as a plausible value**, alongside the
missing kickoff that became `now()`, the swallowed write, the `[]` that meant
both clean and never-ran, and the fallback that degraded silently.

**Operational form:** when correcting one input to a derived figure, enumerate
**every** input and state which have been verified and which have not. A
correction that does not name its unverified inputs is not a correction; it is a
new number with an unstated provenance.

---

# PROPOSAL — THE PHANTOM EXCLUSION MUST BE THE DEFAULT, NOT A THING EACH QUERY REMEMBERS

**Not built. Read-only pass. Proposed with its instances and its sizing.**

## Why this one cannot be fixed by moving a call

Rules 1 and 2 are **positional**: the alias is read too early, the comparison
runs before resolution. Both are fixed by moving the call site.

> **Rule 3 has no position to move to.** The defect is in what the data layer
> hands out. **Every query written from now on starts contaminated unless its
> author remembers — and authors do not remember. I did not, one file after
> quantifying the class.**

## The sizing, measured 2026-09-02

| | |
| --- | --- |
| sites in `src/` reading `match_date` | **269** |
| ...that exclude the phantom class | **0** |
| phantom rows | **510** of 40,473 (**1.26%**) |

**The contamination is concentrated, which is what makes it dangerous rather
than negligible:**

| month | phantom | real | % phantom |
| --- | --- | --- | --- |
| 2026-06 | 1 | 103 | 1.0% |
| 2026-07 | 28 | 370 | 7.0% |
| **2026-08** | **419** | **787** | **34.7%** |
| 2026-09 (to date) | **0** | 51 | **0.0%** |

**A 1.26% table-wide rate became 34.7% in the exact window the schedule
analysis used.** An aggregate contamination figure is not a bound on any
particular query.

**The class appears to have STOPPED ACCRUING** — zero in September, consistent
with the Stage 19/20 fail-closed `_parse_match_date` change that refuses a row
rather than defaulting its kickoff. **Two days is not proof**, and it does not
reduce the need: **510 rows are permanently present and every future query
inherits them.**

## The five instances

| # | analysis | effect |
| --- | --- | --- |
| 1 | **Wednesday profile** (n, earliest, median) | 119→19 rows; earliest 10:15→16:00; **the cliff did not exist** |
| 2 | **day-of-week table** that chose `0 3 * * *` | every weekday's early-kickoff claim → **0%** |
| 3 | **earliest-kickoff anchor** for the boundary | `10:04:29.761924` was a phantom; real 10:15 |
| 4 | **Saturday gradient** | overstated at every point (20.5%→12.4%, 50.5%→45.3%) |
| 5 | **`fixtures_zero_active` / card-loss tables** | denominators including 87–96% phantom days (08-23, 08-25) |

**All five were written after the class was quantified.** Knowing about it
protected none of them.

## The proposal, in the form this project already uses twice

**`pick_filters.py` solved the rules-1-and-2-shaped problem by defining
`live_only()` and `valid_evidence()` once and importing them.** The row-set
equivalent:

1. **An accessor or view that excludes phantoms by construction** — the default
   way to reach match rows, so a new query is correct without its author
   knowing the class exists.
2. **Raw `matches` reserved for the queries that must see everything** —
   settlement, the phantom audit itself, migrations.
3. **Those queries declare themselves with a named marker**, in the same form
   the training-exclusion exemptions already use: a marker comment in `src/`
   naming exactly one category.
4. **A count pinned in `tests/experiment_pins.py`** with a test that greps
   `src/` and fails when the count changes — so a sixth exemption is a
   deliberate edit a reviewer sees, not an unobserved paste.

> **This is the THIRD guard of the same family: one definition, imported, with a
> test that fails when a second appears.** `TRAINING_EXCLUSION_EXEMPTIONS = 6`
> and `EVIDENCE_GATE_EXEMPTIONS = 4` are the existing two, and the mechanism is
> proven — it is the shape that caught the drift both previous times.

**The known cost, stated:** 269 call sites is a large migration, and a
mechanical sweep is exactly the kind of change that introduces a defect while
fixing one. **It should be staged — the accessor first, then the analytical
queries that feed decisions, and the remaining incidental reads last or never.**
The 269 are not equally important; the ones that produced these five instances
are.

**Not built. This is a read-only pass.**

---

# ONE ADDITION TO THE SCHEDULE RECORD — the choice is now MORE conservative than its justification requires

**So the next reader does not mistake it for tight.**

| | as believed | **as measured** |
| --- | --- | --- |
| boundary | 09:29 | **09:40** |
| margin from 03:00 | 6h29m | **6h 40m** |
| Saturday 20% crossing | ~12:00 | **~13:20** |

**Every correction moved in the same direction: there is MORE margin than the
registered figures claimed, not less.** The card is later than the phantoms
made it look, so the delay a 03:00 cron can absorb is larger.

> **`0 3 * * *` is more conservative than its own justification now requires.**
> That is a reason to leave it alone, not to move it later — the cost of the
> extra margin is zero, and the settlement constraint (football ends ~21:30) is
> the binding one on the other side.

*Recorded 2026-09-02.*

---

# AUDIT 2026-09-02 and 2026-09-03 — two MARGIN HELD results that prove nothing about the margin

**Read-only. Nothing fixed.** Registered files read first: the corrected **09:40**
boundary, the re-derived Saturday thresholds, and the phantom-free day-of-week
table.

## 1. THE ROUTINE PASS — 8 new runs

| run | workflow | started | verdict | discovery |
| --- | --- | --- | --- | --- |
| **33603479759** | **daily-picks** | **09-02 07:25** | **DEGRADED** | **`disc[fs=22 fdo=4 af=0]`** |
| 33643768967 | paper-report | 09-02 14:42 | CLEAN | |
| 33643887421 | closing-lines | 09-02 14:43 | CLEAN | |
| 33646243925 | closing-lines | 09-02 15:05 | CLEAN | |
| 33668570075 | closing-lines | 09-02 18:39 | CLEAN | |
| 33687021054 | closing-lines | 09-02 21:47 | CLEAN | |
| 33702269433 | closing-lines | 09-03 01:05 | CLEAN | |
| **33728324296** | **daily-picks** | **09-03 07:29** | **DEGRADED** | **`disc[fs=11 fdo=2 af=0]`** |

**Every step `success` on both daily-picks runs; nine carry `continue-on-error`,
so that remains no evidence.**

### THE `af=0` ALARM IS A FALSE POSITIVE, and the defect is in my own assertion

Both runs fired *"API-Football fixtures = 0 while other sources still produce"*.
**API-Football was healthy on both days:**

```
09-02   API-Football: 206 fixtures on 2026-09-01 -> 0 created, 29 updated
        API-Football: 325 fixtures on 2026-09-02 -> 0 created, 22 updated
        API-Football update complete (59 requests used)
09-03   API-Football: 325 fixtures on 2026-09-02 -> 0 created, 22 updated
        API-Football: 161 fixtures on 2026-09-03 -> 0 created, 11 updated
        API-Football update complete (63 requests used)
```

> **`disc[af=N]` counts rows CREATED, not fixtures discovered.** A source that
> fetches 325 fixtures and successfully matches every one onto an existing row
> scores **zero** — indistinguishable from a source that returned nothing.

**Same defect class as everything else in this ledger: a metric that reads
identically for "healthy" and "dead".** The per-source assertion was built to
separate one dead source from a quiet day, and it cannot separate a *fully
matched* source from a dead one. **Recorded, not fixed.**

**The discovery series, with that correction understood:**

| | 08-30 | 08-31 | 09-01 | 09-02 | 09-03 |
| --- | --- | --- | --- | --- | --- |
| `fs` | 48 | 26 | 17 | 22 | 11 |
| `fdo` | 13 | 7 | 8 | 4 | 2 |
| `af` | 1 | 2 | 12 | **0** | **0** |

**All three sources are declining together**, which is the international break,
not a defect. Flashscore kickoff-parse refusals steady at **214 / 218**
(vs 187, 207, 192).

**One new WARNING worth carrying forward** (09-02): *"Calibration drift check
(last 30d, n=75): avg predicted 63% vs actual 55% (gap −8%) — overconfidence
returning."* Recorded; not this stage's work.

## 2. CARD MEASUREMENTS — the interpretation stated BEFORE the numbers

> **Phantom-free, every weekday loses 0% of its card at every delay out to +11h.
> Wednesday and Thursday CANNOT test the margin. Two MARGIN HELD results this
> week prove the cron fires and nothing about whether 6h40m suffices.**

| | **2026-09-02 (Wed)** | **2026-09-03 (Thu)** |
| --- | --- | --- |
| actual start | 07:25:16 | 07:29:02 |
| **delay against `0 3 * * *`** | **4h 25m 16s** | **4h 29m 02s** |
| outcome under the 09:40 boundary | **WITHIN TOLERANCE** | **WITHIN TOLERANCE** |
| fixtures in window at execution | 22 | 11 |
| earliest kickoff | **16:30** | **16:00** |
| already kicked off at start | **0** | **0** |
| **fraction of card lost** | **0.0% — MARGIN HELD** | **0.0% — MARGIN HELD** |
| lead time | n=22 · 8.5 / **10.8** / 11.0h | n=11 · 7.9 / **10.4** / 10.9h |

**Against the projected 7.1 / 10.7 / 15.4h: the medians of 10.8h and 10.4h sit
essentially ON the projected median of 10.7h.** Third and fourth consecutive
days above the pre-change on-time baseline of 4.4–8.8h — **and still confounded
every time by a late card**, which lifts lead independently of start time.

### The delay series

| date | delay | inside 0.5–5.7h envelope? |
| --- | --- | --- |
| 2026-08-27 | 10h 21m | no |
| 2026-08-28 | 11h 21m | no |
| 2026-08-29 | 5h 03m | yes |
| 2026-09-01 | 5h 05m | yes |
| **2026-09-02** | **4h 25m** | **yes** |
| **2026-09-03** | **4h 29m** | **yes** |

**Four consecutive firings inside the historical envelope, and the last three
are tightening (5h05 → 4h25 → 4h29).** That is consistent with the 10–11h
episode of 08-27/08-28 being over. **Reported as a trend, NOT concluded** —
OPS-3's 12-hour criterion stays open, and two days of an episode never justified
a schedule redesign in the first place.

## 3. s5.9 ACROSS THREE LIVE DAYS

| day | picked rows → groups | collapsed | whole card → groups | collapsed |
| --- | --- | --- | --- | --- |
| 2026-09-01 | 29 → **29** | 0 | 29 → **29** | 0 |
| 2026-09-02 | 22 → **22** | 0 | 22 → **22** | 0 |
| 2026-09-03 | 11 → **11** | 0 | 11 → **11** | 0 |

**No group contained more than one row on any day. No second pick was refused
on either branch. The control holds: nothing genuinely distinct was collapsed.**

**The guarantee, checked independently by (competition, kickoff minute, resolved
teams):**

| day | duplicate pairs | pairs with picks on BOTH rows |
| --- | --- | --- |
| 2026-09-02 | **0** | **0 — held** |
| 2026-09-03 | **0** | **0 — held** |

### Did the key announce itself? NO — still inferring

> **Three live days, and `resolve_fixture_groups` has never emitted an
> unconditional line.** Its only log fires when it unions a group, and it has
> unioned nothing.

**I am again inferring a clean run from the absence of a degrade warning**, plus
the `694a60` cohort stamp and a replay. **That is the same inference chain as
2026-09-01, for the third time.** The fifth instance of the family; the fix is
**proposed, not built**, and every day it stays unbuilt is another day where
"ran and found nothing" and "did not execute" are the same log.

## 4. THE FIRST CLV OBSERVATIONS UNDER s5.9 — clean by mechanism, on n=1

| attribution | **total captured** | **under s5.9 (≥09-01)** |
| --- | --- | --- |
| **MODEL** | **49** (was 48) | **1** |
| **FINAL** | **66** | **3** |

**Status mix:** 09-01 is 29 `pending` — its captures have not closed. 09-02
produced **1 MODEL / 3 FINAL captured, 15 missing, 6 pending**. 09-03 is 11
`pending`.

### The deff question, now live rather than historical

| | |
| --- | --- |
| captured MODEL observations | **49** |
| distinct `match_id` | **49** |
| **distinct FIXTURE identities** | **49** |
| **cluster count inflated** | **NO** |

> **`deff = 1.00` holds, and for the first time one observation is protected by
> the mechanism rather than by the calendar.**

**But the honest size of that: n = 1.** One observation under s5.9 that happens
not to sit on a duplicate is *consistent with* the mechanism working and is not
a demonstration of it. **The retrospective half is the solid one** — the 08-14
twin's window has passed and its two observations can never be captured, so the
historical count is fixed at 49 over 49.

**And a result that clears a whole class:**

| | |
| --- | --- |
| picks on phantom rows | **0** |
| observations on phantom rows | **0** |
| odds rows on phantom rows | 142 of 399,198 (**0.036%**) |

**A phantom row can never carry a pick or an observation**, so every
pick-based and observation-based figure in this ledger is **phantom-clean by
construction**, not by luck.

## 5. THE PUBLISHED-NUMBERS EXPOSURE LIST

**Worked backwards from the claims, as instructed, not forwards from 269 call
sites.**

| stage / figure | source | verdict |
| --- | --- | --- |
| Stage 15 — CLV/capture-window figures, `deff = 1.00`, the 09:17 / 0-of-54 analysis, kickoff-hour 11–12 buckets | `pick_observations` ⋈ `saved_picks` | **CLEAN** — no pick or observation can sit on a phantom |
| Stage 15 — credit/coverage frontier, `odds_refresh_min_interval_minutes` | credit ledger + `odds` | **CLEAN** |
| Stage 18 — `odds_snapshots` / `injury_observations` substrate counts | those tables | **CLEAN** |
| **Stage 18 — "53.5% of match rows were created after their own kickoff, mean +14 days"** | **`matches` directly** | **EXPOSED** (below) |
| **Stage 18 — "274 of 875 August matches … 11.4 matches/day … ~5.5 trajectory-fixtures/day"** | **`matches` count over an August window** | **EXPOSED** (below) |
| Stage 20 — the loop's three defects, timeout, identity gate, redirect | logs and `teams` | **CLEAN** — no fixture-count or kickoff-time claim in the section |
| Stage 21 — day-of-week table, kickoff distributions, boundary, Saturday gradient | `matches` | **EXPOSED — already corrected 2026-09-02** |
| the 2026-08-29 "57 of 103" | `matches` | **CLEAN** — that day has zero phantoms |

### Stage 18's "53.5% created after their own kickoff" — number exposed, conclusion intact

| window | with phantoms | **phantom-free** |
| --- | --- | --- |
| all time | 93.3% (n=40,484) | **93.3%** (n=39,974) |
| **Aug 2026** | **37.5%** (n=1,206) | **4.2%** (n=787) |
| Jul–Aug 2026 | 30.8% | **4.1%** |

**Phantoms are created ~110 seconds after their own "kickoff", so all 495 of
them land in the "created after" bucket.** On a recent window that inflates the
figure **nine-fold**.

**The exact 53.5% is not reproducible without knowing its window** — stated
rather than guessed. **But the conclusion it supported is unaffected:**
`matches.created_at` cannot proxy for `odds.first_seen_at`, and the all-time
figure of **93.3% — phantom-insensitive — carries that on its own**, because
the historical backfill stamps dominate. **`first_seen_at` was the right call
for a reason that survives.**

### Stage 18's accumulation rate — EXPOSED, and this one changes the next stage

**Published:** *"~5.5 trajectory-fixtures/day"*, target **50 independent
fixtures**, deriving **H1's date of 2026-09-08**.

**MEASURED 2026-09-03, over a 9-day snapshot span:**

| observations per key | keys |
| --- | --- |
| 1 | **32,512** |
| 2 | 1,379 |
| **3** | **123** |
| **≥4** | **0** |

| | |
| --- | --- |
| **distinct FIXTURES with a ≥3-observation key** | **4** |
| implied rate | **~0.44 fixtures/day**, against 5.5 projected |
| snapshots landing on a key first seen earlier | **0% on 09-01, 09-02 and 09-03** |

> ### THE SUBSTRATE IS ACCUMULATING BREADTH, NOT DEPTH.
>
> **No key has ever exceeded three observations, and every new snapshot lands on
> a key never seen before.** The ceiling is structural: a key is
> `(match, book, market, selection)`, so its observations stop at kickoff and
> are bounded by the same-day capture cadence.

**H1's target is 50 fixtures. There are 4. At the measured rate that is ~114
days, not the 9 that produced 2026-09-08.**

**The trigger Stage 18 registered — "keys with ≥3 observations" — was the wrong
unit.** It reads 123 and sounds satisfied; the quantity H1 needs is *fixtures
with a three-point trajectory*, which is **4**. The count also **stopped moving**
between 09-01 and 09-03 (123 → 123) while snapshots grew 22%.

**Recorded as a finding about the next stage's premise. Not investigated
further here.**

## 6. SATURDAY 2026-09-05 READINESS

**Registration confirmed at the point of use:** the corrected thresholds are in
`docs/stage21-schedule-prediction.md` as well as this ledger — boundary
**09:40**, MARGIN CONSUMED to **~13:20**, 20% crossing ~80 minutes later than
the contaminated table said.

**Saturday's own card, taken from an external source today, two days ahead:**

| | |
| --- | --- |
| fixtures listed | **29** |
| **earliest kickoff** | **12:30 BST = 11:30 UTC** |
| Premier League | 7 (12:30 ×1, 15:00 ×5, 17:30 ×1 BST) |
| EFL Championship / League One / League Two | 11 / 11 / 10, from 12:30 BST |

**Corroborated across two independent lookups** (a search and a direct fetch of
the dated page), which agree on the 12:30 earliest.

### What this means for the checkpoint — registered now, before the day

> **Saturday's card is REAL and EARLY, so the margin IS testable — the first day
> this week of which that is true.**

**But its earliest is 11:30 UTC, 75 minutes later than the 10:15 the August
profile is built on.** So on *this specific card*:

| | August-derived | **this Saturday** |
| --- | --- | --- |
| first loss at a start of | ~10:15 | **~11:30** |
| MARGIN HELD requires start ≤ | 09:40 (boundary) | **~11:30 on this card** |

**The 09:40 boundary is therefore CONSERVATIVE for Saturday by about 1h50m.** A
run starting anywhere up to 11:30 will score MARGIN HELD on the card that
actually exists.

**Stated plainly so it is not over-read afterwards: a MARGIN HELD on Saturday
demonstrates the margin covers a start up to ~11:30 UTC on a 29-fixture early
card. It does not demonstrate the 09:40 boundary itself**, which would need a
start between 09:40 and 11:30 to be tested at all, and a start past 11:30 to be
falsified.

*Registered 2026-09-03, two days before the day it governs, with hours between
the measurement and the commit.*

## 7. SUBSTRATE — level and rate

| | 09-01 | **09-03** | change |
| --- | --- | --- | --- |
| `odds_snapshots` | 29,167 | **35,639** | **+6,472 (+22%)** |
| **keys with ≥3 observations** | 123 | **123** | **0** |
| `injury_observations` | 756 | **884** | +128 |

**Daily snapshot rate:** 4,643 (09-01) · 4,499 (09-02) · 1,973 (09-03, partial).
**Injury rate ~64/day.**

> **The rate that matters is zero.** `odds_snapshots` grew 22% and the ≥3-key
> count did not move at all. **Reporting the level alone (123, "the trigger is
> satisfied") would have concealed that** — which is the whole reason the rate
> was asked for.

*Read-only audit, 2026-09-03.*

---

# H1 IS BLOCKED BY A POLICY — the mechanism, the price, and the trade

**2026-09-03. Read-only. No config changed.**

## 1. THE MECHANISM — confirmed, and tighter than the hypothesis

**`betting.odds_refresh_window_minutes: 120` against
`betting.odds_refresh_min_interval_minutes: 180`.**

`TheOddsScraper.refresh_imminent` selects candidates via
`_imminent_league_fixtures(window_minutes)` — a fixture is eligible **only in
the last 120 minutes before kickoff** — and then drops any league refreshed
within `min_interval_minutes`.

> **The eligibility window (120) is SHORTER than the dedup interval (180), so a
> fixture can be refreshed AT MOST ONCE. Confirmed in code and in data.**

### What the data shows, and it is worse than "capped at three"

**Distinct PRE-KICKOFF snapshot times per fixture — the quantity that bounds
trajectory depth:**

| distinct pre-kickoff times | fixtures |
| --- | --- |
| 1 | **85** |
| 2 | **105** |
| 3 | **14** |
| **≥4** | **0** |

**And the 14 that reached three are not three points.** Their actual minutes
before kickoff:

```
match 51144  [664, 658, 4]      match 51152  [664, 658, 3]
match 51145  [664, 658, 4]      match 51156  [679, 673, 18]
match 51146  [679, 673, 19]     match 51158  [664, 658, 5]
match 51147  [529, 523, 106]    match 50140  [296, 293, 34]
```

> ### The first two points are SIX MINUTES apart, eleven hours before kickoff.
>
> They are the API-Football pass and the TheOddsAPI pass **inside the same daily
> run** — one observation written twice, not two points in a trajectory.

**So the real depth is TWO genuinely separated points: one at pick time, one
near kickoff (3–20 min). ZERO fixtures anywhere in the table have three
genuinely separated pre-kickoff observations.**

**Not 123 keys. Not 32 keys. Not 4 fixtures. Zero.**

**A further 6,686 keys carry ZERO pre-kickoff observations at all** — snapshots
taken entirely after kickoff, 19% of the substrate, useless for any pre-kickoff
question. Those are the 08-27/08-28 late runs.

### The shape is explained exactly

| observed | explained by |
| --- | --- |
| 32,512 keys at 1 observation | pick-time write only; no refresh reached them |
| 1,379 keys at 2 | pick-time + the single permitted refresh |
| 123 keys at 3 | the duplicated same-run write, or a post-kickoff row |
| **0 keys at ≥4** | **the policy forbids a second refresh** |
| **0% of new snapshots deepen an existing key** | **a key's observations end at kickoff** |

**H1 is not waiting for time to pass. The distribution is capped, and it will
read the same in 114 days.**

## 2. THE LEVER IS NOT L6b ALONE — a correction to the framing

**L6b was `min_interval` 180 → 120. That alone does NOT buy a third point.**

Within a **120-minute** eligibility window, two refreshes 120 minutes apart do
not both fit. **To get two refreshes inside the window requires
`window ≥ 2 × min_interval`.**

> **The binding constraint is the WINDOW, not the interval.** L6b was priced as
> a coverage lever, and as a coverage lever the interval is what mattered. As
> H1's precondition, the window is what matters and it was never on the board
> at all.

**The minimal change that yields three genuinely separated points:**

| | now | needed |
| --- | --- | --- |
| `odds_refresh_window_minutes` | 120 | **~300** |
| `odds_refresh_min_interval_minutes` | 180 | **~120** |

That permits pick-time (~11h out) + one at ~4h + one at ~1h. **Three points with
real separation, which is what a momentum test needs.**

## 3. THE PRICE, measured

**Credits are charged PER LEAGUE REQUEST, not per fixture** — `credits_for()`
multiplies requests by `CREDITS_PER_REQUEST`, and one request covers every
fixture in that league. **This is what makes the re-pricing different.**

**One extra refresh pass, measured against real cards:**

| date | leagues | fixtures | **cost of one extra pass** |
| --- | --- | --- | --- |
| 2026-08-29 | 23 | 104 | **46 credits** |
| 2026-08-30 | 23 | 80 | 46 |
| 2026-08-31 | 13 | 28 | 26 |
| 2026-09-01 | 5 | 29 | **10 credits** |
| 2026-09-02 | 8 | 22 | 16 |
| 2026-09-03 | 8 | 11 | 16 |

**A weekend pass costs ~46 credits and covers ~80–104 fixtures. A weekday pass
costs ~10–16 and covers 11–29.**

### What a three-point trajectory costs

**Target: 50 independent fixtures** (Stage 16's checkpoint, unchanged).

> **Two full weekend passes — 2 × 46 = ~92 credits — cover ~180 fixtures, well
> past the 50 target.** Spread over weekdays instead: ~6 passes at ~16 = **~96
> credits for ~100 fixtures.**
>
> ### ~90–100 credits buys the entire H1 sample. Not 116/month, and not recurring.

**That is the correction the re-pricing produces.** L6b was priced as a
**permanent +116 credits/month** for a *continuous* coverage gain. H1 needs a
**one-off sample of 50 fixtures**, which is a fixed purchase of ~90–100 credits
and then the window closes again.

### Against the tier

| | |
| --- | --- |
| free tier | ~500 credits/month |
| configured budget / safety margin | 400 / 50 → **350 spendable** |
| observed burn (break days) | 490 → 476 → 452 = **~19/day ≈ 570/month projected** |

**Current burn already runs at or above the free tier on a full month**, so ~95
credits is not free headroom — **it displaces roughly five days of closing-line
capture.**

## 4. THE TRADE, for Niki

**Both series draw on one budget. The question is which one gets ~95 credits.**

| | closing-line capture | **three-point trajectories (H1)** |
| --- | --- | --- |
| what it measures | CLV — does the model beat the close | **price momentum — the last open question** |
| status of the question | **CLOSED by Stage 16**: 500 was over-specified ~29×, 17 observations suffice, and at n=46 the one-sided upper bound was **+0.107% against a +1.85% threshold** | **OPEN — never measured, and unmeasurable until now** |
| current holding | **49 MODEL / 66 FINAL captured** | **ZERO fixtures with a real trajectory** |
| marginal value of more | precision on an axis already resolved | the only remaining measurement that could change this project's answer |
| cost | ~19/day ongoing | **~95 credits, one-off** |

> **Stage 16 closed the question the closing lines answer. H1's question is
> open. That asymmetry is the whole argument, and it is not this stage's
> decision to make.**

**Also worth Niki's attention: the ~95 credits need not come from capture.** The
measured burn includes pick-time pricing (L5: 213 cr/month, disqualified as a
cut because it *is* the product) and the barren-league exclusion already
recovered credits for free. **A one-off 95 is roughly five days of capture, or
one weekend's worth of the existing refresh budget reallocated.**

**Nothing changed. Reported for decision.**

---

# STAGE 18'S TRIGGER, RESTATED IN THE RIGHT UNIT

**`keys with ≥3 observations` was the wrong unit and it read as SATISFIED while
the real quantity stood at zero.**

| | the trigger as written | **as it should read** |
| --- | --- | --- |
| unit | keys with ≥3 observations | **fixtures with ≥3 SEPARATED PRE-KICKOFF observations** |
| value 2026-08-31 | 91 → "satisfied" | **0** |
| value 2026-09-03 | 123 → "still satisfied" | **0** |

**Three defects in one metric, and each alone would have been enough:**

1. **`keys`, not fixtures.** One fixture carries ~30 keys, so 123 keys is 4
   fixtures. H1's target is 50 *fixtures*.
2. **Post-kickoff rows counted.** 91 of the 123 keys reach three only by
   including an observation taken *after* kickoff.
3. **Same-run duplicates counted.** The remaining ones reach three via two
   writes six minutes apart in one run.

> **Per rule 3's spirit — the unit has to live where the query is.** A trigger
> stated in one unit and consumed in another is the same failure as an exclusion
> that lives in a dataset rather than a query: **the reader inherits the wrong
> quantity by default, and a satisfied-looking number is exactly what stops
> anyone checking.**

**The corrected trigger, for whoever opens H1:**

```sql
-- fixtures with >=3 pre-kickoff observations, separated by >=30 minutes
-- CURRENT VALUE: 0.  Target: 50.
```

**H1's derived date of 2026-09-08 is unreachable and is withdrawn.** It was
computed from ~5.5 trajectory-fixtures/day; the measured rate is **0/day**, and
it is 0 by policy rather than by sample size.

---

# `disc[af=N]` HAS THE 88-DAY BLIND SPOT IT WAS BUILT TO CLOSE

**The assertion exists because Flashscore died for 88 days behind a healthy
API-Football.** It now cannot see the mirror image.

```
2026-09-02   API-Football: 325 fixtures fetched -> 0 created, 22 updated   -> disc[af=0]
2026-09-03   API-Football: 161 fixtures fetched -> 0 created, 11 updated   -> disc[af=0]
```

> **A source that fetches 325 fixtures and matches every one onto an existing
> row scores ZERO — identical to a source that returned nothing.**

**The failure it was built for is a source going quiet behind a covering
source. If the COVERING source is the one that goes quiet, `af=N` reads 0 in
both the healthy case and the dead one** — and two days of false alarms is the
mild symptom; the severe one is the alarm being ignored when it is real.

**The fix, recorded not built: split created from matched.**

```
disc[fs=11 fdo=2 af=0]        ->    disc[fs=11 fdo=2 af=0c/11m]
```

**A source is alive if `created + matched > 0`; it is DEAD only when both are
zero.** Creation alone measures novelty, not health, and novelty legitimately
goes to zero whenever another source got there first — which is the normal
steady state, not a fault.

*Recorded 2026-09-03.*
