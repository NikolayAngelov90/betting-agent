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
**0 of 27 were flagged by CI.** Every one reported `conclusion: success`.

---

## THE HABIT — read this before the individual findings

Every defect below is an instance of one habit:

> **When something needs to be used somewhere else, a second copy appears
> instead of the first being made reachable.** The copy is always the site that
> drifts.

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

# Stage 13.1 — s5.3 verification. **OPEN.**

## THE OPEN QUESTION, first because it could invert the stage

**If the generation stamps refuse unconditionally, this stage's most-praised
mechanism is a permanent full-egress rebuild wearing the appearance of a pass.**

Run 32646469497 proved the *refusal* half of both stamps: mirror and pickle each
rejected an unstamped artifact, rebuilt, and retrained. It cannot prove the
*acceptance* half, and a stamp that always refuses looks identical from here —
it would rebuild 39,157 rows and retrain from scratch every run, forever, and
every audit would read as green.

**This resolves in one scheduled run. 13.1 stays open until it does.**

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
