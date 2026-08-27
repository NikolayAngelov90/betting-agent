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
