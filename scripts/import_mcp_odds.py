"""Import odds from MCP wagyu-sports JSON files into Neon DB."""
import sys, json, glob, os
from datetime import datetime, date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.database import DatabaseManager
from src.data.models import Match, Team, Odds

# Sport key -> our league key mapping
SPORT_TO_LEAGUE = {
    "soccer_epl": "england/premier-league",
    "soccer_germany_bundesliga": "germany/bundesliga",
    "soccer_spain_la_liga": "spain/laliga",
    "soccer_italy_serie_a": "italy/serie-a",
    "soccer_france_ligue_one": "france/ligue-1",
    "soccer_efl_champ": "england/championship",
    "soccer_england_league1": "england/league-one",
    "soccer_england_league2": "england/league-two",
    "soccer_netherlands_eredivisie": "netherlands/eredivisie",
    "soccer_portugal_primeira_liga": "portugal/primeira-liga",
    "soccer_turkey_super_league": "turkey/super-lig",
    "soccer_belgium_first_div": "belgium/jupiler-pro-league",
    "soccer_spl": "scotland/premiership",
    "soccer_spain_segunda_division": "spain/laliga2",
    "soccer_germany_bundesliga2": "germany/2-bundesliga",
    "soccer_italy_serie_b": "italy/serie-b",
    "soccer_france_ligue_two": "france/ligue-2",
    "soccer_austria_bundesliga": "austria/bundesliga",
    "soccer_switzerland_superleague": "switzerland/super-league",
    "soccer_greece_super_league": "greece/super-league",
    "soccer_poland_ekstraklasa": "poland/ekstraklasa",
}

PREFERRED_BOOKIES = ["bet365", "pinnacle", "unibet", "betfair_ex_eu", "betclic", "sport888", "williamhill"]

NAME_MAP = {
    "AFC Bournemouth": "Bournemouth",
    "Wolverhampton Wanderers": "Wolves",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Tottenham Hotspur": "Tottenham",
    "Newcastle United": "Newcastle",
    "West Ham United": "West Ham",
    "Brighton and Hove Albion": "Brighton",
    "Nottingham Forest": "Nott'm Forest",
    "Leicester City": "Leicester",
    "Sheffield Wednesday": "Sheff Wed",
    "Ipswich Town": "Ipswich",
    "Oxford United": "Oxford",
    "West Bromwich Albion": "West Brom",
    "Queens Park Rangers": "QPR",
    "Norwich City": "Norwich",
    "Hull City": "Hull",
    "Coventry City": "Coventry",
    "Blackburn Rovers": "Blackburn",
    "Stoke City": "Stoke",
    "Derby County": "Derby",
    "Preston North End": "Preston",
    "Swansea City": "Swansea",
    "Plymouth Argyle": "Plymouth",
    "Birmingham City": "Birmingham",
    "Luton Town": "Luton",
    "Cardiff City": "Cardiff",
    "Peterborough United": "Peterborough",
    "Milton Keynes Dons": "MK Dons",
    "AFC Wimbledon": "Wimbledon",
    "Doncaster Rovers": "Doncaster",
    "Wigan Athletic": "Wigan",
    "Northampton Town": "Northampton",
    "Mansfield Town": "Mansfield",
    "Wycombe Wanderers": "Wycombe",
    "Stevenage FC": "Stevenage",
    "Exeter City": "Exeter",
    "Barnsley FC": "Barnsley",
    "Barrow AFC": "Barrow",
    "Newport County": "Newport",
    "Swindon Town": "Swindon",
    "Fleetwood Town": "Fleetwood",
    "Shrewsbury Town": "Shrewsbury",
    "Crewe Alexandra": "Crewe",
    "Harrogate Town": "Harrogate",
    "Cheltenham Town": "Cheltenham",
    "Grimsby Town": "Grimsby",
    "Tranmere Rovers": "Tranmere",
    "Salford City": "Salford",
    "Chesterfield FC": "Chesterfield",
    "Oldham Athletic": "Oldham",
    "Accrington Stanley": "Accrington",
    "Cambridge United": "Cambridge",
    "Gillingham FC": "Gillingham",
    "Walsall FC": "Walsall",
    "Notts County": "Notts Co",
    "Barnet FC": "Barnet",
    "Bromley FC": "Bromley",
    "Bristol Rovers": "Bristol Rovers",
    "Atletico Madrid": "Ath Madrid",
    "Athletic Bilbao": "Ath Bilbao",
    "Athletic Club": "Ath Bilbao",
    "Real Sociedad": "Sociedad",
    "Real Betis": "Betis",
    "Celta Vigo": "Celta",
    "Rayo Vallecano": "Vallecano",
    "Deportivo Alaves": "Alaves",
    "RCD Mallorca": "Mallorca",
    "FC Barcelona": "Barcelona",
    "Real Valladolid": "Valladolid",
    "Deportivo La Coruna": "Dep. La Coruna",
    "CD Leganes": "Leganes",
    "FC Internazionale": "Inter",
    "Internazionale": "Inter",
    "Inter Milan": "Inter",
    "AC Milan": "Milan",
    "AC Monza": "Monza",
    "SSC Napoli": "Napoli",
    "SS Lazio": "Lazio",
    "AS Roma": "Roma",
    "ACF Fiorentina": "Fiorentina",
    "Hellas Verona": "Verona",
    "Udinese Calcio": "Udinese",
    "Genoa CFC": "Genoa",
    "Cagliari Calcio": "Cagliari",
    "FC Bologna": "Bologna",
    "US Sassuolo": "Sassuolo",
    "Empoli FC": "Empoli",
    "Torino FC": "Torino",
    "Venezia FC": "Venezia",
    "US Lecce": "Lecce",
    "Frosinone Calcio": "Frosinone",
    "Paris Saint Germain": "Paris SG",
    "Paris Saint-Germain": "Paris SG",
    "Olympique Marseille": "Marseille",
    "Olympique de Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "AS Monaco": "Monaco",
    "Stade Rennais": "Rennes",
    "RC Strasbourg": "Strasbourg",
    "Stade Brestois": "Brest",
    "Montpellier HSC": "Montpellier",
    "Saint Etienne": "St Etienne",
    "Saint-Etienne": "St Etienne",
    "FC Lorient": "Lorient",
    "AJ Auxerre": "Auxerre",
    "Stade de Reims": "Reims",
    "Le Havre AC": "Le Havre",
    "Angers SCO": "Angers",
    "OGC Nice": "Nice",
    "Ajax Amsterdam": "Ajax",
    "Feyenoord Rotterdam": "Feyenoord",
    "Sparta Rotterdam": "Sp Rotterdam",
    "NEC Nijmegen": "Nijmegen",
    "SC Heerenveen": "Heerenveen",
    "FC Groningen": "Groningen",
    "FC Utrecht": "Utrecht",
    "FC Twente": "Twente",
    "Go Ahead Eagles": "Go Ahead Eagles",
    "NAC Breda": "NAC Breda",
    "Willem II Tilburg": "Willem II",
    "Fortuna Sittard": "For Sittard",
    "FC Volendam": "FC Volendam",
    "SL Benfica": "Benfica",
    "FC Porto": "Porto",
    "Sporting CP": "Sporting Lisbon",
    "SC Braga": "Sp Braga",
    "Vitoria SC": "Guimaraes",
    "Gil Vicente FC": "Gil Vicente",
    "Moreirense FC": "Moreirense",
    "FC Arouca": "Arouca",
    "Borussia Dortmund": "Dortmund",
    "Bayer Leverkusen": "Leverkusen",
    "FC Bayern Munich": "Bayern Munich",
    "Bayern Munich": "Bayern Munich",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "VfB Stuttgart": "Stuttgart",
    "SC Freiburg": "Freiburg",
    "Borussia Monchengladbach": "M'gladbach",
    "TSG Hoffenheim": "Hoffenheim",
    "1899 Hoffenheim": "Hoffenheim",
    "FC Augsburg": "Augsburg",
    "Werder Bremen": "Werder Bremen",
    "Union Berlin": "Union Berlin",
    "VfL Wolfsburg": "Wolfsburg",
    "FC St. Pauli": "St. Pauli",
    "Holstein Kiel": "Holstein Kiel",
    "FC Heidenheim": "Heidenheim",
    "VfL Bochum": "Bochum",
    "VfL Bochum 1848": "Bochum",
    "1. FC Koln": "FC Koln",
    "Hamburger SV": "Hamburger SV",
    "Hertha BSC": "Hertha Berlin",
    "Hertha Berlin": "Hertha Berlin",
    "SC Paderborn": "Paderborn",
    "Eintracht Braunschweig": "Braunschweig",
    "Fortuna Dusseldorf": "Dusseldorf",
    "FC Nurnberg": "Nurnberg",
    "Trabzonspor": "Trabzon",
    "Fenerbahce SK": "Fenerbahce",
    "Galatasaray SK": "Galatasaray",
    "Club Brugge KV": "Club Brugge KV",
    "Union Saint-Gilloise": "St. Gilloise",
    "Royale Union Saint-Gilloise": "St. Gilloise",
    "Royal Charleroi SC": "Charleroi",
    "KVC Westerlo": "Westerlo",
    "OH Leuven": "Leuven",
    "Celtic FC": "Celtic",
    "Aberdeen FC": "Aberdeen",
    "Heart of Midlothian": "Hearts",
    "Motherwell FC": "Motherwell",
    "Hibernian FC": "Hibernian",
    "Kilmarnock FC": "Kilmarnock",
    "Livingston FC": "Livingston",
    "Falkirk FC": "Falkirk",
    "SV Ried": "Ried",
    "WSG Tirol": "Tirol",
    "SCR Altach": "Altach",
    "FC Zurich": "Zurich",
    "FC Sion": "Sion",
    "FC St. Gallen": "St. Gallen",
    "FC Lugano": "Lugano",
    "FC Thun": "Thun",
    "Grasshopper Club Zurich": "Grasshoppers",
    "Grasshoppers": "Grasshoppers",
    "Olympiacos Piraeus": "Olympiacos Piraeus",
    "Olympiakos Piraeus": "Olympiacos Piraeus",
    "Aris Thessaloniki": "Aris",
    "Jagiellonia Bialystok": "Jagiellonia",
    "Cracovia Krakow": "Cracovia",
    "1. FC Heidenheim": "Heidenheim",
    "Leeds United": "Leeds",
    "PSV Eindhoven": "PSV",
    "Club Brugge": "Club Brugge KV",
    "HSV Hamburg": "Hamburger SV",
    "Real Valladolid": "Valladolid",
    "Real Zaragoza": "Zaragoza",
    "FC Famalicao": "Famalicao",
    "Vitoria de Guimaraes": "Guimaraes",
    "Aris Thessaloniki FC": "Aris",
    "Panserraikos FC": "Panserraikos",
    "OFI Crete FC": "OFI Crete",
    "Asteras Tripolis FC": "Asteras Tripolis",
    "Huddersfield Town": "Huddersfield",
    "Burton Albion": "Burton",
    "RC Lens": "Lens",
    "Otelul Galati": "Otelul",
    "Rapid Bucharest": "FC Rapid Bucuresti",
    "Rapid Bucuresti": "FC Rapid Bucuresti",
    "Dinamo Bucharest": "Dinamo Bucuresti",
    "Grasshopper Club": "Grasshoppers",
    "Cultural Leonesa": "Cultural Leonesa",
    "Racing de Santander": "Racing Santander",
}


def main():
    db = DatabaseManager()
    today = date(2026, 3, 14)

    odds_dir = "C:/Users/nikit/.claude/projects/c--Users-nikit-Desktop-betting-agent/96dbda74-73d3-4554-b418-a594b2ae4ce3/tool-results/"
    files = sorted(glob.glob(os.path.join(odds_dir, "mcp-wagyu-sports-get_odds-*.txt")))

    all_events = []
    for f in files:
        with open(f) as fp:
            raw = json.loads(fp.read())
            data = json.loads(raw["result"])
            events = data.get("data", data) if isinstance(data, dict) else data
            all_events.extend(events)

    print(f"Loaded {len(all_events)} events from {len(files)} files")

    today_events = [
        ev for ev in all_events
        if ev.get("commence_time", "").startswith("2026-03-14")
        or ev.get("commence_time", "").startswith("2026-03-15")
    ]
    print(f"Today/tomorrow events: {len(today_events)}")

    saved = 0
    matched = 0
    skipped = []

    with db.get_session() as session:
        fixtures = session.query(Match).filter(
            Match.match_date >= today,
            Match.home_goals.is_(None)
        ).all()

        fixture_lookup = {}
        for fix in fixtures:
            home = session.query(Team).filter(Team.id == fix.home_team_id).first()
            away = session.query(Team).filter(Team.id == fix.away_team_id).first()
            if home and away:
                fixture_lookup[(home.name.lower(), away.name.lower())] = fix

        for ev in today_events:
            sport = ev.get("sport_key", "")
            league = SPORT_TO_LEAGUE.get(sport)
            if not league:
                continue

            home_raw = ev.get("home_team", "")
            away_raw = ev.get("away_team", "")
            home_norm = NAME_MAP.get(home_raw, home_raw)
            away_norm = NAME_MAP.get(away_raw, away_raw)

            fix = fixture_lookup.get((home_norm.lower(), away_norm.lower()))
            if not fix:
                skipped.append(f"{home_raw} -> {home_norm} vs {away_raw} -> {away_norm} ({league})")
                continue

            matched += 1

            # Collect best odds per (market, selection)
            best = {}
            for bookie in ev.get("bookmakers", []):
                bkey = bookie.get("key", "")
                bname = bookie.get("title", bkey)

                for market in bookie.get("markets", []):
                    mkey = market.get("key", "")

                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name", "")
                        price = outcome.get("price", 0)
                        point = outcome.get("point")

                        if mkey == "h2h":
                            db_market = "1X2"
                            if name.lower() == "draw":
                                sel = "Draw"
                            elif name == home_raw or name == home_norm:
                                sel = "Home"
                            elif name == away_raw or name == away_norm:
                                sel = "Away"
                            else:
                                sel = None
                        elif mkey == "totals":
                            db_market = "over_under"
                            if point is not None and name in ("Over", "Under"):
                                sel = f"{name} {point}"
                            else:
                                sel = None
                        else:
                            continue

                        if not sel:
                            continue

                        combo = (db_market, sel)
                        is_prio = bkey in PREFERRED_BOOKIES
                        ex = best.get(combo)
                        if not ex or (is_prio and not ex[2]) or price > ex[0]:
                            best[combo] = (price, bname, is_prio)

            for (mkt, sel), (price, bname, _) in best.items():
                existing = session.query(Odds).filter(
                    Odds.match_id == fix.id,
                    Odds.bookmaker == bname,
                    Odds.market_type == mkt,
                    Odds.selection == sel
                ).first()
                if not existing:
                    session.add(Odds(
                        match_id=fix.id,
                        bookmaker=bname,
                        market_type=mkt,
                        selection=sel,
                        odds_value=price,
                        timestamp=datetime.utcnow()
                    ))
                    saved += 1

        session.commit()

    print(f"\nMatched {matched} events to DB fixtures")
    print(f"Saved {saved} new odds rows")
    if skipped:
        print(f"\nUnmatched ({len(skipped)}):")
        for s in skipped:
            print(f"  {s}")


if __name__ == "__main__":
    main()
