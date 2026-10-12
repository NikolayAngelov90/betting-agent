# Rule Tallies — Claimed vs Corrected

**Audited 2026-10-12.** Read-only. This file exists because of one rule:

> **A count that cannot be enumerated is not cited as a count.** A rule with
> three verified instances and four unverifiable ones is a rule with three
> instances and a note.

`docs/ci-audit-ledger.md` carries **101** sentences containing the word
"instance", **35** of them an explicit ordinal ("the fifth instance of…").
Every one is cited as a measurement. They are maintained by accretion: each
addition is cheap, and until today none had been checked.

The audit below asks three things of each ordinal: **can the prior instances be
enumerated from the ledger at all**; **is each still true**; and **does the
sequence hold together**. Two rules fail the third test outright.

---

## SUMMARY

| rule | claimed | enumerable? | corrected |
| --- | --- | --- | --- |
| **PNC-1** (producer with no consumer) | **14** | **yes — a register exists** | **11 verified** (4 misclassified, 2 unrecorded) |
| **third-state family** | **13** | **no** | ordinals 7, 9, **9 again**; 8/10/11/12 never claimed |
| **THE HABIT** (one rule, many definitions) | **7** | **no** | two counters conflated, non-monotone |
| **"establish the clock"** | **5** | **no** | 1 site; the prior four never named |
| **a default that makes a failure look like a success** | **6** | partial | 4, 5, 6 claimed; 1–3 never named |
| **cp1251 / encoding** | **5** | partial | 4, 5 claimed; 1–3 never named |
| **a definition is not an occurrence** | **5** | partial | 2, 5 claimed; 1, 3, 4 never named |
| **UNI-1** (the unit is not the thing) | **6** | partial | 5, 6 claimed; 1–4 never named |
| **mark, never delete** | **4** | partial | 3, 4 claimed |
| **an exclusion is only as good as…** | **4** | **no** | only the 4th claimed |
| **the positive-control rule** | **3** | **no** | only the 3rd claimed |
| **a guard that enumerates a path and protects nothing** | **3** | **yes** | stated as a set of three, listed together — **holds** |
| **membership** (a workflow outside the audit's scope) | **4** | partial | "third and fourth" claimed together |
| **canonicalisation is not symmetric-safe** | *no ordinal* | **n/a** | carries a **measured population** (369 hazards) with its query window — **holds** |

**The two rules that survive the audit intact are the two that did not use an
accreted ordinal**: one stated its instances as a closed set of three, the
other quoted a measured population with the window it was measured over.

---

## PNC-1 — the only rule with a machine-readable register

The register is `KNOWN_INSTANCES` in `scripts/pnc_scan.py`; it is the reason
this rule could be audited at all. **Claimed 14 on 2026-10-11. Verified 11.**

### VERIFIED (11)

| id | reference | status |
| --- | --- | --- |
| `alarmed-append` | ledger §PNC-1, the naming entry | the original; documented with its five-surface table |
| `h1-stop-no-caller` | `0ed4cad` | measured: `allocate` TEST-ONLY, `COLLECTION_TERMINAL_STATES` ORPHAN |
| `odds-snapshots-unread` | `8ab8b21` → clean at `abb053e` | measured, before/after pair |
| `format-series-census` | `bafe876` | measured: TEST-ONLY @ `ci_audit.py:1200` |
| `same-team-strict` | `team_resolution.py:265` | real then, **since fixed** by the unified resolver |
| `remaining-vs-updated` | `theodds_scraper.py:88` | real then, **since fixed** — `updated` is read |
| `conclusion-requested-unread` | `ci_audit.py:429` | **STILL OPEN** — requested in the `--json` list, read nowhere |
| `requirements-not-read` | ledger line 17064 | real then, since fixed; today's three omissions are deliberate |
| `nothing-reads-snapshot-test` | retired 2026-10-10 | the inverted guard, documented |
| `injury-observations-unread` | `price_history.py:66` | **STILL OPEN** — 5,136 rows, measured 2026-10-12 |
| `first-seen-at-unread` | `odds.first_seen_at` | **STILL OPEN** — 123,043 rows (25.2%), measured 2026-10-12 |

### MISCLASSIFIED (4) — kept on file, out of the denominator

| id | why it is not PNC-1 |
| --- | --- |
| `lookup-table-not-consulted` | **a RULE filed as an instance.** The ledger says it explains three defects; counting the rule as one occurrence double-counts in both directions. |
| `five-terminal-states` | **the inverse shape.** The constant is read; one value is unreachable because no input selects it. A dead branch is a consumer with no producer. |
| `report-discovered-margins` | **the inverse shape.** It had a consumer at `ci_audit.py:1962`; the producer could never evaluate without `DATABASE_URL`. |
| `scrape-league-fixtures` | **a per-mode claim read as a tree-wide one.** Called at `betting_agent.py:588` and `flashscore_scraper.py:546`. |

### One entry re-described rather than removed

`step-failed-alert` claimed "the `step(s) FAILED` alert text was never printed".
That is an **unreachable branch**, a different family. The PNC-1 the ledger
actually recorded at that site is the sentence beside it: *"`conclusion` is
requested at line 207 and read nowhere."* The entry now carries that, and it is
**still open** at `scripts/ci_audit.py:429`.

### The effect on the detector's coverage

| | reported 2026-10-11 | corrected 2026-10-12 |
| --- | --- | --- |
| denominator | 13 | **11** |
| found | 3 | **4** (`injury_observations` was found by the scan itself) |
| coverage | 23% | **36%** |

**The coverage rose because the denominator was wrong, not because the detector
improved.** That is the clearest available illustration of why an accreted
tally is not a measurement.

---

## third-state family — claimed 13, and the ledger supports at most 9

Ordinals actually present in the ledger:

| ordinal | line | finding |
| --- | --- | --- |
| seventh | 16038 | the DEL-2 branch |
| **ninth** | 19722 | first found in the apparatus |
| **ninth** | 19953 | VAC-1, inside the verification step |
| thirteenth | 25150 | the unreadable sha (2026-10-11, mine) |

Two defects:

1. **"Ninth instance" is claimed twice, for two different findings.** The
   counter was incremented to nine and then to nine again.
2. **Nothing ever claimed the eighth, tenth, eleventh or twelfth.** The jump
   from nine to thirteen has no basis in the ledger — it came from a working
   session's memory and was then written into a **source comment** in
   `scripts/ci_audit.py`, where it reads as settled fact.

The family itself is real and well evidenced — roughly eleven distinguishable
occurrences are discussed across eighteen sites (`UNAUDITABLE`/`IN_PROGRESS`,
`None` vs `[]`, `skipped` twice, `UNEVALUATED`, `UNKNOWN`, VAC-1, the
unreadable sha, and others). **It is a rule with an enumerable list of
occurrences and no reliable ordinal.** Cite the list.

---

## THE HABIT — claimed 7, with two counters conflated

| line | claim |
| --- | --- |
| 64 | the definition |
| 2456 | "third instance **in the data layer**" |
| 4736 | "**sixth** instance, found and not extended" |
| 6351 | "the **fifth data-layer** instance" |
| 20984 / 21276 | "**seventh** instance of THE HABIT" |
| 21386 | "instance of THE HABIT" (no ordinal) |
| 1503, 2109, 4892, 7060, 10029 | discussed without an ordinal |

Two sequences — a global one and a "data-layer" sub-count — are written with the
same word. The result is **non-monotone in file order**: the sixth is claimed at
line 4736 and the fifth at line 6351. Nothing in the ledger lists the members of
either sequence, so neither ordinal is checkable. **Cite the sites, not a
number.**

---

## "establish the clock" — claimed 5, enumerable 1

A single site (line 24715) claims "the fifth instance of 'establish the clock'
in three days". The prior four are **named nowhere**. The underlying events are
real and individually recorded elsewhere (the H1 cron set three hours wrong, the
probe's `+180` refusal, the string-timestamp defects), but they were never
gathered under this name, so the ordinal cannot be reconstructed from the
ledger. **One enumerable site.**

---

## The partially enumerable rules

For these the ledger claims the last two or three ordinals and never the
earlier ones. Each is a rule with real, individually documented occurrences and
an ordinal that cannot be reconstructed:

* **a default that makes a failure look like a success** — 4th (3772), 5th
  (4112), 6th (25516). The 6th is the `or ""` series exclusion, verified.
* **cp1251 / encoding** — 4th (2222), 5th (3961), across 8 mentions.
* **a definition is not an occurrence** — 2nd (11497/11508), 5th (13261).
* **UNI-1** — 5th (16194), 6th. Five mentions.
* **mark, never delete** — 3rd (21263), 4th (23359). All three mentions are
  accounted for, so this one is close to enumerable.
* **membership** — "the third and fourth instance" claimed together (25079),
  naming both, which is the right form; the first two are not named.

## The rules that hold

* **a guard that enumerates a path and protects nothing** — stated as
  **"three instances"** and **listed together** in the same entry. A closed
  set, enumerable, correct.
* **canonicalisation is not symmetric-safe** — carries no ordinal at all. It
  quotes a **measured population**: *369 new symmetric-canonicalisation hazards
  introduced*, with the query that produced it. Still checkable today.

---

## The rule this file establishes

> **A running tally is a claim and needs the same verification as any other
> number.** Counts maintained by accretion drift upward because each addition
> is cheap and none is checked. Of fourteen tallies audited here, **two
> survive** — the two that recorded a closed set or a measured population
> instead of an ordinal.
>
> From 2026-10-12: an ordinal is cited only where the prior instances are
> enumerated — in this file, or in the entry that makes the claim. Otherwise
> cite the rule and its sites, which is what the evidence supports.
