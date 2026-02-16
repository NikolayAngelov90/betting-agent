"""One-time script to settle Feb 15 2026 picks and send report to Telegram."""

import asyncio
from datetime import date

# Feb 15 picks from Telegram message + actual results
PICKS = [
    {"match": "Feyenoord vs Go Ahead Eagles", "selection": "Home Win", "odds": 1.41, "stake": 5.0, "score": "1-0"},
    {"match": "FC Lausanne-Sport vs Servette", "selection": "Over 2.5 Goals", "odds": 1.91, "stake": 5.0, "score": "3-3"},
    {"match": "FC Lausanne-Sport vs Servette", "selection": "BTTS Yes", "odds": 1.80, "stake": 5.0, "score": "3-3"},
    {"match": "FC Lausanne-Sport vs Servette", "selection": "Over 3.5 Goals", "odds": 2.65, "stake": 5.0, "score": "3-3"},
    {"match": "Kilmarnock vs Celtic", "selection": "Away Win", "odds": 1.38, "stake": 4.1, "score": "2-3"},
    {"match": "Parma vs Hellas Verona", "selection": "Over 1.5 Goals", "odds": 1.66, "stake": 5.0, "score": "2-1"},
    {"match": "Napoli vs AS Roma", "selection": "Over 1.5 Goals", "odds": 1.66, "stake": 5.0, "score": "2-2"},
    {"match": "Napoli vs AS Roma", "selection": "Home Win", "odds": 2.70, "stake": 5.0, "score": "2-2"},
    {"match": "Torino vs Bologna", "selection": "Under 2.5 Goals", "odds": 1.83, "stake": 5.0, "score": "1-2"},
    {"match": "Torino vs Bologna", "selection": "BTTS No", "odds": 1.90, "stake": 5.0, "score": "1-2"},
    {"match": "Motherwell vs Aberdeen", "selection": "Home Win", "odds": 1.64, "stake": 5.0, "score": "2-0"},
    {"match": "Cremonese vs Genoa", "selection": "Over 1.5 Goals", "odds": 1.59, "stake": 4.2, "score": "0-0"},
    {"match": "Mallorca vs Real Betis", "selection": "Under 2.5 Goals", "odds": 1.81, "stake": 5.0, "score": "1-2"},
    {"match": "Mallorca vs Real Betis", "selection": "BTTS No", "odds": 1.90, "stake": 3.9, "score": "1-2"},
    {"match": "Lorient vs Angers", "selection": "Home Win", "odds": 2.05, "stake": 5.0, "score": "2-0"},
    {"match": "Lyon vs Nice", "selection": "BTTS Yes", "odds": 1.80, "stake": 5.0, "score": "2-0"},
    {"match": "Lyon vs Nice", "selection": "Over 2.5 Goals", "odds": 1.82, "stake": 5.0, "score": "2-0"},
    {"match": "Lyon vs Nice", "selection": "Home Win", "odds": 1.86, "stake": 4.2, "score": "2-0"},
    {"match": "Royal Antwerp vs Westerlo", "selection": "BTTS Yes", "odds": 1.80, "stake": 5.0, "score": "0-2"},
    {"match": "Royal Antwerp vs Westerlo", "selection": "Over 2.5 Goals", "odds": 1.86, "stake": 5.0, "score": "0-2"},
    {"match": "Metz vs Auxerre", "selection": "BTTS Yes", "odds": 1.80, "stake": 4.2, "score": "1-3"},
    {"match": "Metz vs Auxerre", "selection": "Over 2.5 Goals", "odds": 2.28, "stake": 5.0, "score": "1-3"},
    {"match": "Rangers vs Hearts", "selection": "Over 2.5 Goals", "odds": 2.00, "stake": 5.0, "score": "4-2"},
    {"match": "Heerenveen vs FC Zwolle", "selection": "BTTS Yes", "odds": 1.80, "stake": 3.1, "score": "4-2"},
    {"match": "FC Basel vs FC Lugano", "selection": "Under 2.5 Goals", "odds": 2.34, "stake": 5.0, "score": "1-1"},
    {"match": "FC Basel vs FC Lugano", "selection": "BTTS No", "odds": 1.90, "stake": 2.9, "score": "1-1"},
    {"match": "FC Thun vs FC Sion", "selection": "Home Win", "odds": 1.99, "stake": 4.5, "score": "1-0"},
    {"match": "FC Thun vs FC Sion", "selection": "BTTS Yes", "odds": 1.80, "stake": 1.9, "score": "1-0"},
    {"match": "Nacional vs FC Porto", "selection": "Under 2.5 Goals", "odds": 2.07, "stake": 4.8, "score": "0-1"},
    {"match": "Le Havre vs Toulouse", "selection": "Away Win", "odds": 2.51, "stake": 5.0, "score": "2-1"},
    {"match": "Le Havre vs Toulouse", "selection": "Over 2.5 Goals", "odds": 2.32, "stake": 5.0, "score": "2-1"},
    {"match": "Udinese vs Sassuolo", "selection": "BTTS No", "odds": 1.90, "stake": 1.4, "score": "1-2"},
]


def settle_pick(pick):
    """Determine if a pick won or lost based on the actual score."""
    hg, ag = map(int, pick["score"].split("-"))
    total = hg + ag
    btts = hg > 0 and ag > 0
    sel = pick["selection"]

    if sel == "Home Win":
        return hg > ag
    elif sel == "Away Win":
        return hg < ag
    elif sel == "Draw":
        return hg == ag
    elif sel == "Over 1.5 Goals":
        return total > 1.5
    elif sel == "Over 2.5 Goals":
        return total > 2.5
    elif sel == "Over 3.5 Goals":
        return total > 3.5
    elif sel == "Under 2.5 Goals":
        return total < 2.5
    elif sel == "BTTS Yes":
        return btts
    elif sel == "BTTS No":
        return not btts
    return False


async def main():
    # Settle all picks
    settled = []
    for pick in PICKS:
        won = settle_pick(pick)
        pick["result"] = "win" if won else "loss"
        settled.append(pick)

    # Stats
    wins = sum(1 for p in settled if p["result"] == "win")
    losses = sum(1 for p in settled if p["result"] == "loss")
    total = wins + losses
    win_rate = wins / total if total > 0 else 0

    profit = sum(
        p["stake"] * (p["odds"] - 1) if p["result"] == "win" else -p["stake"]
        for p in settled
    )

    print(f"Settlement Report - 15 Feb 2026")
    print(f"Record: {wins}W - {losses}L ({win_rate:.0%})")
    print(f"P/L: {profit:+.1f}% of bankroll\n")

    for p in settled:
        emoji = "WIN" if p["result"] == "win" else "LOSS"
        print(f"  [{emoji}] {p['match']} ({p['score']}) - {p['selection']} @ {p['odds']}")

    # Build Telegram message
    header = f"<b>📊 Settlement Report - 15 Feb 2026</b>\n"
    header += f"<b>Record: {wins}W - {losses}L ({win_rate:.0%})</b>\n"
    profit_emoji = "📈" if profit >= 0 else "📉"
    header += f"{profit_emoji} P/L: {profit:+.1f}% of bankroll\n"

    lines = [header]
    for p in settled:
        result_emoji = "✅" if p["result"] == "win" else "❌"
        lines.append(
            f"\n{result_emoji} <b>{p['match']}</b> ({p['score']})\n"
            f"    {p['selection']} @ {p['odds']:.2f} | Stake: {p['stake']:.1f}%"
        )

    message = "\n".join(lines)

    # Send to Telegram
    from telegram import Bot

    bot_token = "8588061989:AAGQ91GA7QzsPmljwOuL6JV_gA9btpwCJU4"
    chat_id = "-5290104839"

    if bot_token and chat_id:
        bot = Bot(token=bot_token)
        # Split if too long
        if len(message) > 4000:
            mid = len(settled) // 2
            part1_lines = [header]
            for p in settled[:mid]:
                result_emoji = "✅" if p["result"] == "win" else "❌"
                part1_lines.append(
                    f"\n{result_emoji} <b>{p['match']}</b> ({p['score']})\n"
                    f"    {p['selection']} @ {p['odds']:.2f} | Stake: {p['stake']:.1f}%"
                )
            await bot.send_message(chat_id=chat_id, text="\n".join(part1_lines), parse_mode="HTML")

            part2_lines = [f"<b>📊 Settlement (continued)</b>"]
            for p in settled[mid:]:
                result_emoji = "✅" if p["result"] == "win" else "❌"
                part2_lines.append(
                    f"\n{result_emoji} <b>{p['match']}</b> ({p['score']})\n"
                    f"    {p['selection']} @ {p['odds']:.2f} | Stake: {p['stake']:.1f}%"
                )
            await bot.send_message(chat_id=chat_id, text="\n".join(part2_lines), parse_mode="HTML")
        else:
            await bot.send_message(chat_id=chat_id, text=message, parse_mode="HTML")

        print("\nSettlement report sent to Telegram!")
    else:
        print("\nTelegram not configured")


if __name__ == "__main__":
    asyncio.run(main())
