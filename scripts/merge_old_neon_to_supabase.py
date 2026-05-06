"""Merge old Neon PostgreSQL data into Supabase.

The old DB has 37K matches and 504 historical picks not in Supabase.
Since both DBs use auto-increment IDs starting from 1, this script:
  1. Deduplicates teams by apifootball_team_id -> (name, league) -> name
  2. Deduplicates matches by apifootball_id -> (home_id, away_id, date)
  3. Deduplicates players by transfermarkt_id -> (name, team_id)
  4. Remaps all FK columns before inserting into Supabase
  5. Uses ON CONFLICT DO NOTHING on odds unique constraint

Usage:
    python scripts/merge_old_neon_to_supabase.py
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg2
import psycopg2.extras
from psycopg2.extras import execute_values

OLD_URL = (
    "postgresql://neondb_owner:npg_dMcT5qkYVow3"
    "@ep-raspy-bonus-agqqv1pl-pooler.c-2.eu-central-1.aws.neon.tech"
    "/neondb?sslmode=require"
)
SUPA_URL = (
    "postgresql://postgres.nhlurscyrlvpjzapmqcr:ofA5FEPTmjHzEtkQ"
    "@aws-1-eu-central-1.pooler.supabase.com:5432/postgres?sslmode=require"
)

CHUNK = 5_000


def fetchall_dict(cur, sql, params=None):
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Step 1: Teams
# ---------------------------------------------------------------------------

def merge_teams(old, supa):
    print("\n[teams] Building ID map...")

    old_cur = old.cursor()
    supa_cur = supa.cursor()

    # Load existing Supabase teams into lookup structures
    supa_teams = fetchall_dict(supa_cur,
        "SELECT id, name, league, apifootball_team_id FROM teams")

    by_afid   = {t["apifootball_team_id"]: t["id"] for t in supa_teams
                 if t["apifootball_team_id"]}
    by_name_league = {(t["name"], t["league"] or ""): t["id"] for t in supa_teams}
    by_name   = {}
    for t in supa_teams:
        by_name.setdefault(t["name"], t["id"])

    # Load old teams
    old_teams = fetchall_dict(old_cur, "SELECT * FROM teams ORDER BY id")

    team_map = {}   # old_id -> supa_id
    inserted = 0
    matched  = 0

    for t in old_teams:
        old_id = t["id"]

        # Try match by apifootball_team_id
        if t.get("apifootball_team_id") and t["apifootball_team_id"] in by_afid:
            team_map[old_id] = by_afid[t["apifootball_team_id"]]
            matched += 1
            continue

        # Try match by (name, league)
        key = (t["name"], t["league"] or "")
        if key in by_name_league:
            team_map[old_id] = by_name_league[key]
            matched += 1
            continue

        # Try match by name only
        if t["name"] in by_name:
            team_map[old_id] = by_name[t["name"]]
            matched += 1
            continue

        # Insert new team
        supa_cur.execute("""
            INSERT INTO teams (name, country, league, flashscore_id,
                               transfermarkt_id, apifootball_team_id, created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id
        """, (t["name"], t.get("country"), t.get("league"),
              t.get("flashscore_id"), t.get("transfermarkt_id"),
              t.get("apifootball_team_id"), t.get("created_at")))
        new_id = supa_cur.fetchone()[0]
        supa.commit()

        team_map[old_id] = new_id
        by_afid[t.get("apifootball_team_id")] = new_id
        by_name_league[key] = new_id
        by_name[t["name"]] = new_id
        inserted += 1

    old_cur.close()
    supa_cur.close()
    print(f"  teams: {matched} matched, {inserted} inserted -> {len(team_map)} total mapped")
    return team_map


# ---------------------------------------------------------------------------
# Step 2: Matches
# ---------------------------------------------------------------------------

def merge_matches(old, supa, team_map):
    print("\n[matches] Building ID map (this may take a minute)...")

    old_cur = old.cursor()
    supa_cur = supa.cursor()

    # Load Supabase match lookups
    print("  Loading Supabase match index...", end="", flush=True)
    by_afid  = {}
    by_teams_date = {}
    supa_cur.execute(
        "SELECT id, apifootball_id, home_team_id, away_team_id, "
        "DATE(match_date) AS d FROM matches"
    )
    for row in supa_cur.fetchall():
        sid, afid, ht, at, d = row
        if afid:
            by_afid[afid] = sid
        by_teams_date[(ht, at, str(d))] = sid
    print(f" {len(by_afid)} afid, {len(by_teams_date)} team+date entries")

    # Stream old matches in chunks
    old_cur.execute("SELECT COUNT(*) FROM matches")
    total = old_cur.fetchone()[0]
    old_cur.execute("DECLARE _mc CURSOR FOR SELECT * FROM matches ORDER BY id")

    match_map = {}   # old_id -> supa_id
    inserted = matched = 0

    while True:
        old_cur.execute(f"FETCH {CHUNK} FROM _mc")
        rows = old_cur.fetchall()
        if not rows:
            break
        cols = [d[0] for d in old_cur.description]
        records = [dict(zip(cols, r)) for r in rows]

        batch_insert = []
        batch_old_ids = []

        for m in records:
            old_id = m["id"]
            new_ht = team_map.get(m["home_team_id"])
            new_at = team_map.get(m["away_team_id"])
            if not new_ht or not new_at:
                continue  # unmapped team — skip

            # Match by apifootball_id
            if m.get("apifootball_id") and m["apifootball_id"] in by_afid:
                match_map[old_id] = by_afid[m["apifootball_id"]]
                matched += 1
                continue

            # Match by (home_team_id, away_team_id, date)
            d_str = str(m["match_date"])[:10]
            key = (new_ht, new_at, d_str)
            if key in by_teams_date:
                match_map[old_id] = by_teams_date[key]
                matched += 1
                continue

            # New match — queue for batch insert
            batch_insert.append(m)
            batch_old_ids.append(old_id)

        # Batch insert new matches
        if batch_insert:
            for m, old_id in zip(batch_insert, batch_old_ids):
                new_ht = team_map[m["home_team_id"]]
                new_at = team_map[m["away_team_id"]]
                supa_cur.execute("""
                    INSERT INTO matches (
                        home_team_id, away_team_id, match_date, league, season,
                        home_goals, away_goals, ht_home_goals, ht_away_goals,
                        home_shots, away_shots, home_shots_on_target, away_shots_on_target,
                        home_possession, away_possession, home_corners, away_corners,
                        home_fouls, away_fouls, home_yellow_cards, away_yellow_cards,
                        home_red_cards, away_red_cards, home_xg, away_xg,
                        home_dangerous_attacks, away_dangerous_attacks,
                        home_saves, away_saves, home_offsides, away_offsides,
                        home_free_kicks, away_free_kicks,
                        referee, venue, venue_capacity,
                        regulation_home_goals, regulation_away_goals,
                        penalty_home_score, penalty_away_score,
                        apifootball_id, flashscore_id, is_fixture, created_at
                    ) VALUES (
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                    ) RETURNING id
                """, (
                    new_ht, new_at, m["match_date"], m.get("league"), m.get("season"),
                    m.get("home_goals"), m.get("away_goals"),
                    m.get("ht_home_goals"), m.get("ht_away_goals"),
                    m.get("home_shots"), m.get("away_shots"),
                    m.get("home_shots_on_target"), m.get("away_shots_on_target"),
                    m.get("home_possession"), m.get("away_possession"),
                    m.get("home_corners"), m.get("away_corners"),
                    m.get("home_fouls"), m.get("away_fouls"),
                    m.get("home_yellow_cards"), m.get("away_yellow_cards"),
                    m.get("home_red_cards"), m.get("away_red_cards"),
                    m.get("home_xg"), m.get("away_xg"),
                    m.get("home_dangerous_attacks"), m.get("away_dangerous_attacks"),
                    m.get("home_saves"), m.get("away_saves"),
                    m.get("home_offsides"), m.get("away_offsides"),
                    m.get("home_free_kicks"), m.get("away_free_kicks"),
                    m.get("referee"), m.get("venue"), m.get("venue_capacity"),
                    m.get("regulation_home_goals"), m.get("regulation_away_goals"),
                    m.get("penalty_home_score"), m.get("penalty_away_score"),
                    m.get("apifootball_id"), m.get("flashscore_id"),
                    m.get("is_fixture", False), m.get("created_at"),
                ))
                new_id = supa_cur.fetchone()[0]
                match_map[old_id] = new_id
                inserted += 1

                # Update lookups for any subsequent duplicates in same chunk
                if m.get("apifootball_id"):
                    by_afid[m["apifootball_id"]] = new_id
                by_teams_date[(new_ht, new_at, str(m["match_date"])[:10])] = new_id

            supa.commit()

        total_done = matched + inserted
        print(f"\r  matches: {total_done:,}/{total:,} ({inserted:,} new)", end="", flush=True)

    old_cur.execute("CLOSE _mc")
    old.commit()
    old_cur.close()
    supa_cur.close()

    print(f"\r  matches: {matched:,} matched, {inserted:,} inserted -> {len(match_map):,} total mapped")

    # Reset sequence
    _reset_seq(supa, "matches")
    return match_map


# ---------------------------------------------------------------------------
# Step 3: Players
# ---------------------------------------------------------------------------

def merge_players(old, supa, team_map):
    print("\n[players] Merging...")

    old_cur = old.cursor()
    supa_cur = supa.cursor()

    supa_players = fetchall_dict(supa_cur, "SELECT id, name, team_id, transfermarkt_id FROM players")
    by_tmid  = {p["transfermarkt_id"]: p["id"] for p in supa_players if p["transfermarkt_id"]}
    by_name_team = {(p["name"], p["team_id"]): p["id"] for p in supa_players}

    old_players = fetchall_dict(old_cur, "SELECT * FROM players ORDER BY id")

    player_map = {}
    inserted = matched = 0

    for p in old_players:
        old_id = p["id"]
        new_team_id = team_map.get(p.get("team_id"))

        if p.get("transfermarkt_id") and p["transfermarkt_id"] in by_tmid:
            player_map[old_id] = by_tmid[p["transfermarkt_id"]]
            matched += 1
            continue

        key = (p["name"], new_team_id)
        if key in by_name_team:
            player_map[old_id] = by_name_team[key]
            matched += 1
            continue

        supa_cur.execute("""
            INSERT INTO players (name, team_id, position, market_value, is_key_player, transfermarkt_id)
            VALUES (%s,%s,%s,%s,%s,%s) RETURNING id
        """, (p["name"], new_team_id, p.get("position"), p.get("market_value"),
              p.get("is_key_player", False), p.get("transfermarkt_id")))
        new_id = supa_cur.fetchone()[0]
        supa.commit()

        player_map[old_id] = new_id
        if p.get("transfermarkt_id"):
            by_tmid[p["transfermarkt_id"]] = new_id
        by_name_team[key] = new_id
        inserted += 1

    old_cur.close()
    supa_cur.close()
    print(f"  players: {matched} matched, {inserted} inserted")
    _reset_seq(supa, "players")
    return player_map


# ---------------------------------------------------------------------------
# Step 4: Injuries (small table — simple upsert)
# ---------------------------------------------------------------------------

def merge_injuries(old, supa, team_map, player_map):
    print("\n[injuries] Merging...")

    old_cur = old.cursor()
    supa_cur = supa.cursor()

    old_injuries = fetchall_dict(old_cur, "SELECT * FROM injuries")
    inserted = skipped = 0

    for inj in old_injuries:
        new_team_id   = team_map.get(inj.get("team_id"))
        new_player_id = player_map.get(inj.get("player_id"))

        try:
            supa_cur.execute("""
                INSERT INTO injuries (player_id, team_id, injury_type,
                    start_date, expected_return, status, source, updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """, (new_player_id, new_team_id, inj.get("injury_type"),
                  inj.get("start_date"), inj.get("expected_return"),
                  inj.get("status"), inj.get("source"), inj.get("updated_at")))
            supa.commit()
            inserted += 1
        except Exception:
            supa.rollback()
            skipped += 1

    old_cur.close()
    supa_cur.close()
    print(f"  injuries: {inserted} inserted, {skipped} skipped (duplicates)")
    _reset_seq(supa, "injuries")


# ---------------------------------------------------------------------------
# Step 5: Odds (large table — batch insert with ON CONFLICT DO NOTHING)
# ---------------------------------------------------------------------------

def merge_odds(old, supa, match_map):
    print("\n[odds] Merging (large table — batch insert)...")

    old_cur = old.cursor()

    old_cur.execute("SELECT COUNT(*) FROM odds")
    total = old_cur.fetchone()[0]

    # Check how many are already in Supabase so we can skip ahead
    with supa.cursor() as sc:
        sc.execute("SELECT COUNT(*) FROM odds")
        already_in_supa = sc.fetchone()[0]
    print(f"  Supabase currently has {already_in_supa:,} odds rows; source has {total:,}")

    old_cur.execute("DECLARE _oc CURSOR FOR SELECT * FROM odds ORDER BY id")

    inserted = skipped = 0
    BATCH = 1000  # rows per execute_values call

    while True:
        old_cur.execute(f"FETCH {CHUNK} FROM _oc")
        rows = old_cur.fetchall()
        if not rows:
            break
        cols = [d[0] for d in old_cur.description]
        records = [dict(zip(cols, r)) for r in rows]

        # Build batch — remap match_id, drop orphans
        batch = []
        for o in records:
            new_match_id = match_map.get(o["match_id"])
            if not new_match_id:
                skipped += 1
                continue
            batch.append((
                new_match_id, o.get("bookmaker"), o.get("market_type"),
                o.get("selection"), o["odds_value"], o.get("opening_odds"),
                o.get("timestamp"),
            ))

        # Insert in sub-batches of BATCH rows using execute_values (single roundtrip per batch)
        for i in range(0, len(batch), BATCH):
            sub = batch[i:i + BATCH]
            with supa.cursor() as supa_cur:
                execute_values(
                    supa_cur,
                    """INSERT INTO odds
                           (match_id, bookmaker, market_type, selection,
                            odds_value, opening_odds, timestamp)
                       VALUES %s
                       ON CONFLICT (match_id, bookmaker, market_type, selection) DO NOTHING""",
                    sub,
                    page_size=BATCH,
                )
                inserted += supa_cur.rowcount
            supa.commit()

        print(f"\r  odds: {inserted:,} inserted, {skipped:,} skipped — {inserted+skipped+already_in_supa:,}/{total:,}",
              end="", flush=True)

    old_cur.execute("CLOSE _oc")
    old.commit()
    old_cur.close()
    print(f"\r  odds: {inserted:,} inserted, {skipped:,} skipped (total {total:,})" + " " * 10)
    _reset_seq(supa, "odds")


# ---------------------------------------------------------------------------
# Step 6: Saved picks — the most valuable data
# ---------------------------------------------------------------------------

def merge_saved_picks(old, supa, match_map):
    print("\n[saved_picks] Merging historical picks...")

    old_cur = old.cursor()
    supa_cur = supa.cursor()

    # Load existing Supabase picks to skip duplicates
    supa_picks = fetchall_dict(supa_cur,
        "SELECT match_id, market, selection FROM saved_picks")
    existing = {(p["match_id"], p["market"], p["selection"]) for p in supa_picks}

    old_picks = fetchall_dict(old_cur, "SELECT * FROM saved_picks ORDER BY id")

    inserted = skipped_dup = skipped_no_match = 0

    for p in old_picks:
        new_match_id = match_map.get(p["match_id"])
        if not new_match_id:
            skipped_no_match += 1
            continue

        key = (new_match_id, p.get("market"), p.get("selection"))
        if key in existing:
            skipped_dup += 1
            continue

        supa_cur.execute("""
            INSERT INTO saved_picks (
                match_id, pick_date, match_name, league, market, selection,
                odds, predicted_probability, expected_value, confidence,
                kelly_stake_percentage, risk_level, used_fallback_odds,
                result, actual_home_goals, actual_away_goals, settled_at, created_at
            ) VALUES (
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
            )
        """, (
            new_match_id, p.get("pick_date"), p.get("match_name"), p.get("league"),
            p.get("market"), p.get("selection"), p.get("odds"),
            p.get("predicted_probability"), p.get("expected_value"),
            p.get("confidence"), p.get("kelly_stake_percentage"),
            p.get("risk_level"), p.get("used_fallback_odds", False),
            p.get("result"), p.get("actual_home_goals"), p.get("actual_away_goals"),
            p.get("settled_at"), p.get("created_at"),
        ))
        existing.add(key)
        inserted += 1

    supa.commit()
    old_cur.close()
    supa_cur.close()

    print(f"  saved_picks: {inserted} inserted, "
          f"{skipped_dup} duplicate, {skipped_no_match} no match found")
    _reset_seq(supa, "saved_picks")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_seq(conn, table):
    with conn.cursor() as cur:
        try:
            cur.execute(
                f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
                f"COALESCE((SELECT MAX(id) FROM {table}), 1))"
            )
            conn.commit()
        except Exception:
            conn.rollback()


def final_counts(supa):
    print("\nFinal row counts in Supabase:")
    with supa.cursor() as cur:
        for table in ["teams", "matches", "players", "injuries", "odds", "saved_picks"]:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            print(f"  {table:<15} {cur.fetchone()[0]:>8,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Merging old Neon -> Supabase")
    print("=" * 60)

    old  = psycopg2.connect(OLD_URL)
    supa = psycopg2.connect(SUPA_URL)
    old.autocommit  = False
    supa.autocommit = False

    try:
        # These always rebuild ID maps (fast when data already exists)
        team_map   = merge_teams(old, supa)
        match_map  = merge_matches(old, supa, team_map)
        player_map = merge_players(old, supa, team_map)
        merge_injuries(old, supa, team_map, player_map)
        merge_odds(old, supa, match_map)
        merge_saved_picks(old, supa, match_map)
        final_counts(supa)
    except Exception as exc:
        print(f"\nFATAL: {exc}")
        try:
            old.rollback()
        except Exception:
            pass
        try:
            supa.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            old.close()
        except Exception:
            pass
        try:
            supa.close()
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("Merge complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
