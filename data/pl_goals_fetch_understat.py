"""
Premier League shot data pipeline.

Sources:
  - Primary:  Understat (shot-level data with X/Y pitch coordinates)
  - Enriched: FBref/StatsBomb (body part, play pattern - added later)

Understat season keys map as follows:
  "2022" -> 2022/23
  "2023" -> 2023/24
  "2024" -> 2024/25
"""

import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from understatapi import UnderstatClient

LEAGUE = "EPL"
SEASONS = ["2022", "2023", "2024"]
MAX_WORKERS = 10  # concurrent threads for match shot fetching

OUTPUT_DIR = Path(__file__).parent / "raw"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "pl_shots_understat.csv"


def fetch_shots_for_match(match: dict, season: str) -> list[dict]:
    """Fetch shot data for a single match (runs in a thread)."""
    match_id = match["id"]
    home = match["h"]["title"]
    away = match["a"]["title"]
    date = match["datetime"]

    try:
        with UnderstatClient() as understat:
            shots = understat.match(match_id).get_shot_data()
    except Exception as exc:
        print(
            f"  WARNING: could not fetch shots for match {match_id} ({home} vs {away}): {exc}"
        )
        return []

    rows = []
    for side, shot_list in shots.items():
        for shot in shot_list:
            shot["match_id"] = match_id
            shot["season"] = f"{season}/{str(int(season) + 1)[-2:]}"
            shot["home_team"] = home
            shot["away_team"] = away
            shot["match_date"] = date[:10]
            shot["side"] = side  # "h" or "a"
            rows.append(shot)
    return rows


def fetch_shots_for_season(season: str) -> pd.DataFrame:
    """Fetch all shot data for one EPL season using a thread pool."""
    print(f"Fetching match list for {LEAGUE} {season}...")
    with UnderstatClient() as understat:
        matches = understat.league(LEAGUE).get_match_data(season=season)

    # Only fetch matches that have been played (have shot data)
    played = [m for m in matches if m.get("isResult")]
    print(
        f"  {len(played)} played matches found — fetching shots with {MAX_WORKERS} threads..."
    )

    all_shots = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_shots_for_match, match, season): match
            for match in played
        }
        for i, future in enumerate(as_completed(futures), 1):
            rows = future.result()
            all_shots.extend(rows)
            if i % 50 == 0 or i == len(played):
                print(f"  {i}/{len(played)} matches done...")

    return pd.DataFrame(all_shots)


def fetch_all_seasons() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        df = fetch_shots_for_season(season)
        frames.append(df)
        print(f"  -> {len(df)} shots fetched for season {season}\n")
    return pd.concat(frames, ignore_index=True)


def process(df: pd.DataFrame) -> pd.DataFrame:
    """Tidy up columns across all shots."""
    shots = df.rename(
        columns={
            "X": "x",  # pitch length coordinate (0=own goal, 1=opp goal)
            "Y": "y",  # pitch width coordinate (0=left, 1=right)
            "xG": "xg",
            "situation": "situation",  # e.g. OpenPlay, Corner, DirectFreekick, Penalty
            "shotType": "shot_type",  # e.g. RightFoot, LeftFoot, Head
            "result": "result",  # e.g. Goal, SavedShot, MissedShots, BlockedShot
        }
    )

    keep = [
        "season",
        "match_date",
        "match_id",
        "home_team",
        "away_team",
        "side",
        "player",
        "minute",
        "x",
        "y",
        "xg",
        "situation",
        "shot_type",
        "result",
        "player_id",
        "id",
    ]

    keep = [c for c in keep if c in shots.columns]
    shots = shots[keep].copy()

    for col in ["minute", "x", "y", "xg"]:
        if col in shots.columns:
            shots[col] = pd.to_numeric(shots[col], errors="coerce")

    shots = shots.sort_values(["match_date", "match_id", "minute"]).reset_index(
        drop=True
    )
    return shots


def main():
    print("=== Fetching Premier League shot data from Understat ===\n")

    df_raw = fetch_all_seasons()
    print(f"Total shots across all seasons: {len(df_raw)}")

    df_shots = process(df_raw)

    df_shots.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")

    print("\nShots per season:")
    print(df_shots.groupby("season").size().to_string())

    print("\nShot results breakdown:")
    print(df_shots["result"].value_counts().to_string())

    print("\nTop 10 players by shots across all seasons:")
    print(
        df_shots.groupby("player")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .to_string()
    )

    return df_shots


if __name__ == "__main__":
    main()
