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
