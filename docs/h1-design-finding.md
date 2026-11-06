# H1's Design Finding — Registered 2026-11-06

**Not a fix. The next window depends on it.** Registered while the first real
trajectories are still on disk, because the figures below are only measurable
from them.

> **H1 priced every selection in a league, and the analysis needs the PICKED
> selection.** Of 1,120 qualifying trajectories, 939 were for a selection the
> model did not pick and only 52 matched. **The vocabulary was not the limit —
> the design was.**

---

## 1. WHAT THE RUN ACTUALLY PRODUCED

`measured` 2026-11-06, grain stated per figure, over the closed window
2026-11-01..11-09.

| | |
|---|---|
| qualifying trajectories (grain=SERIES) | **1,120** |
| &nbsp;&nbsp;matched a pick | **52** |
| &nbsp;&nbsp;picked fixture, DIFFERENT selection | **939** |
| &nbsp;&nbsp;fixture carried no pick | 93 |
| &nbsp;&nbsp;uncanonicalisable (quarter lines, team-goals Unders) | 36 |
| actionable prices (grain=FIXTURE) | **5** |
| credits spent | 26 of 168 |

52 + 939 + 93 + 36 = 1,120. The classification sums to its input.

**The 939 is the finding.** Those are real, correctly-captured, in-band price
paths for bets this pipeline would never have placed. They cost credits and
they cannot enter the estimator.

## 2. THE SECOND, LARGER CONSTRAINT — AND IT IS NOT ABOUT THIS WINDOW

H1 requests `markets=h2h,totals`. The card picks from **six** market families.
`measured` 2026-11-06, grain=PICK, over all 2,795 non-consolidated picks:

| family | picks | share | requested by H1 |
|---|---|---|---|
| `over_under` | 1,068 | 38.2% | **yes** (totals) |
| `1X2` | 527 | 18.9% | **yes** (h2h) |
| `team_goals` | 460 | 16.5% | no |
| `double_chance` | 289 | 10.3% | no |
| `btts` | 280 | 10.0% | no |
| `draw_no_bet` | 171 | 6.1% | no |

> **57.1% of picks are reachable by what H1 requests. 42.9% — 1,200 of
> 2,795 — are unreachable BY DESIGN**, and no amount of vocabulary
> reconciliation changes that. The 2026-11-05 extraction fixed the
> *comparison*; it could not fix the *coverage*.

## 3. WHAT A FUTURE WINDOW MUST DO

**Track the trajectory of the selections actually picked.** That means
collection follows the card's keys rather than sweeping a league.

### It does NOT require a later start

The common objection is that the picked set is unknown until the card is
written, so collection cannot begin until then. **Measured: that is already
true and already satisfied.** `daily-picks` completed at 07:28Z on 2026-11-06
and the H1 collection instants in the real window were 14:11 / 14:32 / 16:35 /
18:38 — between **4h43m and 11h10m after the card existed**. The collector has
never run before the picks.

So the change is to the REQUEST, not the schedule.

### What it does to the band

The band is set by what a trajectory needs: three instants, gaps ≥ 30 min,
interval ratio < 2.0, all pre-kickoff. The real window's qualifying fixtures
had kickoffs 19:45–20:00 and used a 4h27m span. Following picked keys does not
move that requirement, but it changes **which** fixtures are eligible: today
the band selects fixtures by kickoff within two pinned leagues; a
pick-following collector selects fixtures by *being picked*, and the card's
fixtures are spread across all 30 configured leagues.

**The band therefore stops being a league filter and becomes purely a
kickoff-distance filter.** That is a loosening, and it is the direction the
supply failure of this window argues for: band-eligible pinned fixtures were
10, then 0, 1, 0, 0 across the five days.

### What it does to the credit cost

TheOddsAPI bills per market per call, and a market returns all of its
selections — **a single selection cannot be requested**. So "follow the picked
selection" means "request the market family containing it, for the fixtures
that were picked".

* **today**: 2 markets × 2 pinned leagues × 1 region × rounds — a league sweep
  whose cost is independent of how many of its fixtures were picked;
* **pick-following**: the cost scales with *distinct (league, market family)*
  pairs on the card, which is bounded by 6 families and in practice far fewer
  per day.

**This is not costed here, and that is deliberate.** The sizing needs a
measured distribution of distinct (league, family) pairs per card, which is a
query on the existing record and belongs in the next window's
pre-registration — not in a finding written from one window's 26 credits.

## 4. THE SECOND FINDING — `picked_selections` OVERWROTE. **FIXED.**

`picked_selections` built `{match_id: (market, selection)}`, which overwrites.
The per-match cap is **2**.

`measured` 2026-11-06: **2,587 fixtures carry a pick against 2,767 picks, so
180 fixtures were silently losing one of their two.** Any fixture whose second
pick was the one with a captured trajectory contributed nothing and reported no
error.

**Status: FIXED, 2026-11-05**, in the same stage that extracted the
vocabulary — it returns a list per fixture and `one_price_per_fixture` matches
against all of them. It is recorded here because it was found *by* the
vocabulary fix and would otherwise look like part of it: two independent
defects, one of which the other was hiding.

## 5. WHAT THIS DOES NOT DO

Changes no collection code, moves no window, spends no credits, and does not
re-open the closed experiment. ρ=0.42, σ=5.937%, `TARGET_N` 39 and the floor
of 33 are untouched. `H1 WINDOW FAILED` stands and is not extended.
`CODE_REVISION` stays `s5.19`.

*The next window's pre-registration must state, before collection: which
market families it requests, why, and the measured (league, family) cost of
that choice.*
