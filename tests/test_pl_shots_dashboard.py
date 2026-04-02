"""
Tests for pl_shots_dashboard.py

Covers the core analytics logic used in the Team Analysis tab:
  - Team metric derivation (xG/shot, shots/match, attack/defence diffs)
  - Season table and position calculation
  - ordinal() formatting helper

Streamlit rendering is not tested here — these tests exercise the underlying
pandas transformations using minimal fixture data.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_shots(
    *,
    match_id="m1",
    season="2023/24",
    home_team="Team A",
    away_team="Team B",
    side="h",
    result="SavedShot",
    xg=0.10,
    player="Player X",
    n=1,
):
    """Return a list of minimal shot-row dicts."""
    return [
        {
            "id": f"{match_id}-{i}",
            "match_id": match_id,
            "season": season,
            "home_team": home_team,
            "away_team": away_team,
            "side": side,
            "result": result,
            "xg": xg,
            "player": player,
            "minute": 45,
            "x": 0.8,
            "y": 0.5,
        }
        for i in range(n)
    ]


def _build_df(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Team metrics derivation (mirrors tab 8 logic)
# ---------------------------------------------------------------------------

def _compute_team_stats(df):
    """Reproduce the tab-8 aggregation logic from pl_shots_dashboard.py."""
    df = df.copy()
    df["shooting_team"]  = np.where(df["side"] == "h", df["home_team"], df["away_team"])
    df["defending_team"] = np.where(df["side"] == "h", df["away_team"], df["home_team"])

    attack = (
        df.groupby("shooting_team")
        .agg(
            shots_for=("id", "count"),
            goals_for=("result", lambda x: (x == "Goal").sum()),
            xg_for=("xg", "sum"),
        )
        .rename_axis("team")
    )
    defense = (
        df.groupby("defending_team")
        .agg(
            shots_against=("id", "count"),
            goals_against=("result", lambda x: (x == "Goal").sum()),
            xg_against=("xg", "sum"),
        )
        .rename_axis("team")
    )
    home_m = df.groupby("home_team")["match_id"].nunique().rename_axis("team")
    away_m = df.groupby("away_team")["match_id"].nunique().rename_axis("team")
    matches = home_m.add(away_m, fill_value=0).rename("matches")

    ts = attack.join(defense).join(matches)
    ts["xg_per_shot_for"]     = ts["xg_for"]     / ts["shots_for"]
    ts["xg_per_shot_against"] = ts["xg_against"] / ts["shots_against"]
    ts["shots_per_match"]     = ts["shots_for"]  / ts["matches"]
    ts["goals_per_match"]     = ts["goals_for"]  / ts["matches"]
    ts["attack_diff"]         = ts["goals_for"]  - ts["xg_for"]
    ts["defense_diff"]        = ts["xg_against"] - ts["goals_against"]
    ts["combined"]            = ts["attack_diff"] + ts["defense_diff"]
    return ts


class TestTeamMetrics:
    def _two_team_df(self):
        """Two shots in one match: one home (goal), one away (saved)."""
        rows = (
            _make_shots(match_id="m1", home_team="Alpha", away_team="Beta",
                        side="h", result="Goal", xg=0.30, n=1)
            + _make_shots(match_id="m1", home_team="Alpha", away_team="Beta",
                          side="a", result="SavedShot", xg=0.10, n=1)
        )
        return _build_df(rows)

    def test_shooting_team_derived_correctly(self):
        df = self._two_team_df()
        df["shooting_team"] = np.where(df["side"] == "h", df["home_team"], df["away_team"])
        assert set(df["shooting_team"]) == {"Alpha", "Beta"}

    def test_goals_for_counts_only_goal_results(self):
        ts = _compute_team_stats(self._two_team_df())
        assert ts.loc["Alpha", "goals_for"] == 1
        assert ts.loc["Beta",  "goals_for"] == 0

    def test_shots_for_counts_all_shot_types(self):
        ts = _compute_team_stats(self._two_team_df())
        assert ts.loc["Alpha", "shots_for"] == 1
        assert ts.loc["Beta",  "shots_for"] == 1

    def test_xg_per_shot_for_calculated_correctly(self):
        ts = _compute_team_stats(self._two_team_df())
        assert ts.loc["Alpha", "xg_per_shot_for"] == pytest.approx(0.30)
        assert ts.loc["Beta",  "xg_per_shot_for"] == pytest.approx(0.10)

    def test_xg_per_shot_against_is_opponents_shots(self):
        ts = _compute_team_stats(self._two_team_df())
        # Alpha conceded Beta's shot (xg=0.10); Beta conceded Alpha's shot (xg=0.30)
        assert ts.loc["Alpha", "xg_per_shot_against"] == pytest.approx(0.10)
        assert ts.loc["Beta",  "xg_per_shot_against"] == pytest.approx(0.30)

    def test_matches_per_team_counts_correctly(self):
        ts = _compute_team_stats(self._two_team_df())
        assert ts.loc["Alpha", "matches"] == 1
        assert ts.loc["Beta",  "matches"] == 1

    def test_shots_per_match(self):
        rows = (
            _make_shots(match_id="m1", home_team="A", away_team="B", side="h", n=5)
            + _make_shots(match_id="m1", home_team="A", away_team="B", side="a", n=3)
        )
        ts = _compute_team_stats(_build_df(rows))
        assert ts.loc["A", "shots_per_match"] == pytest.approx(5.0)
        assert ts.loc["B", "shots_per_match"] == pytest.approx(3.0)

    def test_attack_diff_positive_when_goals_exceed_xg(self):
        # 1 goal from a shot with xg=0.05 → overperforming
        rows = _make_shots(match_id="m1", home_team="A", away_team="B",
                           side="h", result="Goal", xg=0.05, n=1)
        ts = _compute_team_stats(_build_df(rows))
        assert ts.loc["A", "attack_diff"] == pytest.approx(1 - 0.05)

    def test_attack_diff_negative_when_goals_below_xg(self):
        # 0 goals from 3 shots each with xg=0.30 → underperforming
        rows = _make_shots(match_id="m1", home_team="A", away_team="B",
                           side="h", result="SavedShot", xg=0.30, n=3)
        ts = _compute_team_stats(_build_df(rows))
        assert ts.loc["A", "attack_diff"] == pytest.approx(0 - 0.90)

    def test_defense_diff_positive_when_fewer_goals_conceded_than_xg(self):
        # Away team (B) creates 3 shots worth 0.30 xG each but scores 0.
        # Include a nominal home shot so both teams appear in the attack join.
        rows = (
            _make_shots(match_id="m1", home_team="A", away_team="B",
                        side="a", result="SavedShot", xg=0.30, n=3)
            + _make_shots(match_id="m1", home_team="A", away_team="B",
                          side="h", result="SavedShot", xg=0.05, n=1)
        )
        ts = _compute_team_stats(_build_df(rows))
        # A's defence diff = xg_against − goals_against = 0.90 − 0 = +0.90
        assert ts.loc["A", "defense_diff"] == pytest.approx(0.90)

    def test_combined_is_sum_of_attack_and_defense_diff(self):
        ts = _compute_team_stats(self._two_team_df())
        for team in ["Alpha", "Beta"]:
            assert ts.loc[team, "combined"] == pytest.approx(
                ts.loc[team, "attack_diff"] + ts.loc[team, "defense_diff"]
            )

    def test_multiple_matches_accumulate_correctly(self):
        rows = (
            _make_shots(match_id="m1", home_team="A", away_team="B", side="h", result="Goal", xg=0.20, n=2)
            + _make_shots(match_id="m1", home_team="A", away_team="B", side="a", result="SavedShot", xg=0.10, n=1)
            + _make_shots(match_id="m2", home_team="C", away_team="A", side="a", result="Goal", xg=0.15, n=1)
            + _make_shots(match_id="m2", home_team="C", away_team="A", side="h", result="SavedShot", xg=0.05, n=1)
        )
        ts = _compute_team_stats(_build_df(rows))
        assert ts.loc["A", "shots_for"]  == 3   # 2 in m1 as home, 1 in m2 as away
        assert ts.loc["A", "goals_for"]  == 3
        assert ts.loc["A", "matches"]    == 2


# ---------------------------------------------------------------------------
# Season table / position logic (mirrors tab 8 league-table derivation)
# ---------------------------------------------------------------------------

def _compute_season_table(df):
    """
    Reproduce the match-result / points logic from pl_shots_dashboard.py.
    Returns a DataFrame with columns: season, team, pts, gd, gf.
    (Omits the position-rank step, which is just pandas .rank() on pts/gd/gf
    and doesn't need a dedicated test.)
    """
    df = df.copy()
    all_matches = (
        df.groupby(["match_id", "season", "home_team", "away_team"])
        .size()
        .reset_index()[["match_id", "season", "home_team", "away_team"]]
    )
    goal_rows = df[df["result"].isin(["Goal", "OwnGoal"])].copy()
    goal_rows["home_goal"] = (
        ((goal_rows["result"] == "Goal")    & (goal_rows["side"] == "h")) |
        ((goal_rows["result"] == "OwnGoal") & (goal_rows["side"] == "a"))
    )
    goal_rows["away_goal"] = (
        ((goal_rows["result"] == "Goal")    & (goal_rows["side"] == "a")) |
        ((goal_rows["result"] == "OwnGoal") & (goal_rows["side"] == "h"))
    )
    match_scores = (
        goal_rows.groupby("match_id")
        .agg(home_goals=("home_goal", "sum"), away_goals=("away_goal", "sum"))
        .reset_index()
    )
    mr = all_matches.merge(match_scores, on="match_id", how="left").fillna(0)
    mr["home_goals"] = mr["home_goals"].astype(int)
    mr["away_goals"] = mr["away_goals"].astype(int)
    mr["home_pts"] = np.where(mr["home_goals"] > mr["away_goals"], 3,
                     np.where(mr["home_goals"] == mr["away_goals"], 1, 0))
    mr["away_pts"] = np.where(mr["away_goals"] > mr["home_goals"], 3,
                     np.where(mr["home_goals"] == mr["away_goals"], 1, 0))
    mr["home_gd"] = mr["home_goals"] - mr["away_goals"]
    mr["away_gd"] = -mr["home_gd"]

    home_t = mr.groupby(["season", "home_team"]).agg(
        pts=("home_pts", "sum"), gd=("home_gd", "sum"), gf=("home_goals", "sum")
    ).rename_axis(["season", "team"])
    away_t = mr.groupby(["season", "away_team"]).agg(
        pts=("away_pts", "sum"), gd=("away_gd", "sum"), gf=("away_goals", "sum")
    ).rename_axis(["season", "team"])
    return (home_t + away_t).reset_index()


class TestSeasonTable:
    """
    Tests for the match-result / points derivation used to build the league table.
    Each fixture includes two matches (one per "home" side) so both teams have
    home and away appearances, which is required for the home_t + away_t join.
    """

    def _two_match_rows(self, m1_home_result, m1_away_result,
                        m2_home_result="SavedShot", m2_away_result="SavedShot"):
        """m1: Alpha (home) vs Beta; m2: Beta (home) vs Alpha."""
        return _build_df(
            _make_shots(match_id="m1", season="2023/24",
                        home_team="Alpha", away_team="Beta",
                        side="h", result=m1_home_result, xg=0.4, n=1)
            + _make_shots(match_id="m1", season="2023/24",
                          home_team="Alpha", away_team="Beta",
                          side="a", result=m1_away_result, xg=0.1, n=1)
            + _make_shots(match_id="m2", season="2023/24",
                          home_team="Beta", away_team="Alpha",
                          side="h", result=m2_home_result, xg=0.1, n=1)
            + _make_shots(match_id="m2", season="2023/24",
                          home_team="Beta", away_team="Alpha",
                          side="a", result=m2_away_result, xg=0.1, n=1)
        )

    def test_win_gives_three_points_loss_gives_zero(self):
        # m1: Alpha scores, Beta doesn't → Alpha wins; m2: 0-0 draw
        st = _compute_season_table(
            self._two_match_rows("Goal", "SavedShot")
        )
        alpha_pts = st.loc[st["team"] == "Alpha", "pts"].values[0]
        beta_pts  = st.loc[st["team"] == "Beta",  "pts"].values[0]
        assert alpha_pts == 4  # 3pts (win) + 1pt (draw)
        assert beta_pts  == 1  # 0pts (loss) + 1pt (draw)

    def test_draw_gives_one_point_each(self):
        # m1: both teams score once → draw; m2: 0-0 draw
        st = _compute_season_table(
            self._two_match_rows("Goal", "Goal")
        )
        assert st.loc[st["team"] == "Alpha", "pts"].values[0] == 2
        assert st.loc[st["team"] == "Beta",  "pts"].values[0] == 2

    def test_goal_difference_calculated_correctly(self):
        # m1: Alpha 1-0 Beta; m2: 0-0
        st = _compute_season_table(
            self._two_match_rows("Goal", "SavedShot")
        )
        assert st.loc[st["team"] == "Alpha", "gd"].values[0] == 1
        assert st.loc[st["team"] == "Beta",  "gd"].values[0] == -1

    def test_own_goal_credits_opposition(self):
        # m1: Alpha own goal → Beta scores; m2: 0-0
        rows = _build_df(
            _make_shots(match_id="m1", season="2023/24",
                        home_team="Alpha", away_team="Beta",
                        side="h", result="OwnGoal", xg=0.1, n=1)
            + _make_shots(match_id="m1", season="2023/24",
                          home_team="Alpha", away_team="Beta",
                          side="a", result="SavedShot", xg=0.1, n=1)
            + _make_shots(match_id="m2", season="2023/24",
                          home_team="Beta", away_team="Alpha",
                          side="h", result="SavedShot", xg=0.1, n=1)
            + _make_shots(match_id="m2", season="2023/24",
                          home_team="Beta", away_team="Alpha",
                          side="a", result="SavedShot", xg=0.1, n=1)
        )
        st = _compute_season_table(rows)
        # m1: Beta wins via Alpha own goal (Beta 3pts); m2: 0-0 (1pt each)
        assert st.loc[st["team"] == "Beta",  "pts"].values[0] == 4
        assert st.loc[st["team"] == "Alpha", "pts"].values[0] == 1


# ---------------------------------------------------------------------------
# ordinal() helper
# ---------------------------------------------------------------------------

def ordinal(n: float) -> str:
    """Copied from pl_shots_dashboard.py for isolated unit testing."""
    n = round(n)
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(
        n % 10 if n % 100 not in (11, 12, 13) else 0, "th"
    )
    return f"{n}{suffix}"


class TestOrdinal:
    @pytest.mark.parametrize("n,expected", [
        (1,  "1st"),
        (2,  "2nd"),
        (3,  "3rd"),
        (4,  "4th"),
        (11, "11th"),
        (12, "12th"),
        (13, "13th"),
        (21, "21st"),
        (22, "22nd"),
        (23, "23rd"),
        (20, "20th"),
    ])
    def test_ordinal_suffixes(self, n, expected):
        assert ordinal(n) == expected

    def test_rounds_float_input(self):
        assert ordinal(1.4) == "1st"
        assert ordinal(1.6) == "2nd"


# ---------------------------------------------------------------------------
# Per-season team stats (mirrors tab 9 logic)
# ---------------------------------------------------------------------------

def _compute_per_season_stats(df):
    """Reproduce the tab-9 per-season aggregation from pl_shots_dashboard.py."""
    df = df.copy()
    df["shooting_team"] = np.where(df["side"] == "h", df["home_team"], df["away_team"])

    attack = df.groupby(["season", "shooting_team"]).agg(
        shots_for=("id", "count"),
        goals_for=("result", lambda x: (x == "Goal").sum()),
        xg_for=("xg", "sum"),
    )
    attack.index.names = ["season", "team"]

    home_m = df.groupby(["season", "home_team"])["match_id"].nunique()
    home_m.index.names = ["season", "team"]
    away_m = df.groupby(["season", "away_team"])["match_id"].nunique()
    away_m.index.names = ["season", "team"]
    matches = home_m.add(away_m, fill_value=0).rename("matches")

    ts = attack.join(matches)
    ts["xg_per_shot"]     = ts["xg_for"]    / ts["shots_for"]
    ts["shots_per_match"] = ts["shots_for"]  / ts["matches"]
    ts["xg_per_match"]    = ts["xg_for"]     / ts["matches"]
    ts["goals_per_match"] = ts["goals_for"]  / ts["matches"]
    ts["finishing_diff"]  = (ts["goals_for"] - ts["xg_for"]) / ts["matches"]
    return ts


class TestPerSeasonStats:
    def _two_season_df(self):
        """Team A vs Team B across two seasons — one goal each season."""
        return _build_df(
            _make_shots(match_id="m1", season="2022/23",
                        home_team="A", away_team="B", side="h", result="Goal", xg=0.20, n=1)
            + _make_shots(match_id="m1", season="2022/23",
                          home_team="A", away_team="B", side="a", result="SavedShot", xg=0.10, n=1)
            + _make_shots(match_id="m2", season="2023/24",
                          home_team="A", away_team="B", side="h", result="Goal", xg=0.30, n=2)
            + _make_shots(match_id="m2", season="2023/24",
                          home_team="A", away_team="B", side="a", result="SavedShot", xg=0.05, n=1)
        )

    def test_seasons_are_kept_separate(self):
        ts = _compute_per_season_stats(self._two_season_df())
        assert ("2022/23", "A") in ts.index
        assert ("2023/24", "A") in ts.index

    def test_xg_per_shot_correct_per_season(self):
        ts = _compute_per_season_stats(self._two_season_df())
        assert ts.loc[("2022/23", "A"), "xg_per_shot"] == pytest.approx(0.20)
        assert ts.loc[("2023/24", "A"), "xg_per_shot"] == pytest.approx(0.30)

    def test_xg_per_match_correct(self):
        ts = _compute_per_season_stats(self._two_season_df())
        # 2022/23: 1 shot × 0.20 xG in 1 match
        assert ts.loc[("2022/23", "A"), "xg_per_match"] == pytest.approx(0.20)
        # 2023/24: 2 shots × 0.30 xG in 1 match
        assert ts.loc[("2023/24", "A"), "xg_per_match"] == pytest.approx(0.60)

    def test_goals_per_match_correct(self):
        ts = _compute_per_season_stats(self._two_season_df())
        assert ts.loc[("2022/23", "A"), "goals_per_match"] == pytest.approx(1.0)
        assert ts.loc[("2023/24", "A"), "goals_per_match"] == pytest.approx(2.0)

    def test_finishing_diff_positive_when_goals_exceed_xg(self):
        # 1 goal from xg=0.05 → large overperformance
        df = _build_df(
            _make_shots(match_id="m1", season="2023/24",
                        home_team="A", away_team="B", side="h", result="Goal", xg=0.05, n=1)
            + _make_shots(match_id="m1", season="2023/24",
                          home_team="A", away_team="B", side="a", result="SavedShot", xg=0.10, n=1)
        )
        ts = _compute_per_season_stats(df)
        assert ts.loc[("2023/24", "A"), "finishing_diff"] == pytest.approx(1 - 0.05)

    def test_finishing_diff_negative_when_goals_below_xg(self):
        # 0 goals from 3 shots each worth 0.30 xG
        df = _build_df(
            _make_shots(match_id="m1", season="2023/24",
                        home_team="A", away_team="B", side="h", result="SavedShot", xg=0.30, n=3)
            + _make_shots(match_id="m1", season="2023/24",
                          home_team="A", away_team="B", side="a", result="SavedShot", xg=0.10, n=1)
        )
        ts = _compute_per_season_stats(df)
        assert ts.loc[("2023/24", "A"), "finishing_diff"] == pytest.approx(-0.90)

    def test_league_avg_finishing_diff_near_zero_for_balanced_league(self):
        # One team overperforms (+0.5) and another underperforms (-0.5) — average should be 0
        df = _build_df(
            _make_shots(match_id="m1", season="2023/24",
                        home_team="A", away_team="B", side="h", result="Goal", xg=0.50, n=1)
            + _make_shots(match_id="m1", season="2023/24",
                          home_team="A", away_team="B", side="a", result="SavedShot", xg=0.50, n=1)
        )
        ts = _compute_per_season_stats(df).reset_index()
        league_avg = ts.groupby("season")["finishing_diff"].mean()
        # A scored 1 goal from 0.50 xG (+0.50); B scored 0 from 0.50 xG (-0.50)
        assert league_avg["2023/24"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# YoY percentage change (mirrors annotate_yoy logic in tab 9)
# ---------------------------------------------------------------------------

def _yoy_pct(vals, i):
    """Pure calculation extracted from annotate_yoy in pl_shots_dashboard.py."""
    if i == 0:
        return None
    prev = vals[i - 1]
    if not prev:
        return float("nan")
    return (vals[i] - prev) / prev * 100


class TestYoYPct:
    def test_first_position_returns_none(self):
        assert _yoy_pct([1.0, 1.5], 0) is None

    def test_positive_change(self):
        assert _yoy_pct([1.0, 1.5], 1) == pytest.approx(50.0)

    def test_negative_change(self):
        assert _yoy_pct([2.0, 1.5], 1) == pytest.approx(-25.0)

    def test_no_change_gives_zero(self):
        assert _yoy_pct([1.0, 1.0], 1) == pytest.approx(0.0)

    def test_zero_prev_returns_nan(self):
        result = _yoy_pct([0.0, 1.0], 1)
        assert result != result  # nan != nan

    @pytest.mark.parametrize("vals,i,expected", [
        ([1.0, 2.0, 3.0], 2, 50.0),   # uses i-1, not i=0
        ([1.0, 2.0, 1.0], 2, -50.0),
    ])
    def test_uses_immediately_prior_value(self, vals, i, expected):
        assert _yoy_pct(vals, i) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Man Utd overlay league position (mirrors tab 8 overlay logic)
# Note: this derivation only handles result=="Goal"; own goals are not counted,
# unlike the full _compute_season_table used elsewhere in tab 8.
# ---------------------------------------------------------------------------

def _compute_overlay_position(df, team):
    """Reproduce the simplified league-position logic from the tab-8 overlay block."""
    goal_rows = df[df["result"] == "Goal"].copy()
    goal_rows["home_goal"] = (goal_rows["side"] == "h")
    goal_rows["away_goal"] = (goal_rows["side"] == "a")
    match_scores = (
        goal_rows.groupby("match_id")
        .agg(home_goals=("home_goal", "sum"), away_goals=("away_goal", "sum"))
        .reset_index()
    )
    fixtures = (
        df.groupby(["match_id", "home_team", "away_team"]).size()
        .reset_index()[["match_id", "home_team", "away_team"]]
    )
    mr = fixtures.merge(match_scores, on="match_id", how="left").fillna(0)
    mr["home_pts"] = np.where(mr["home_goals"] > mr["away_goals"], 3,
                     np.where(mr["home_goals"] == mr["away_goals"], 1, 0))
    mr["away_pts"] = np.where(mr["away_goals"] > mr["home_goals"], 3,
                     np.where(mr["home_goals"] == mr["away_goals"], 1, 0))
    mr["home_gd"] = mr["home_goals"] - mr["away_goals"]
    mr["away_gd"] = -mr["home_gd"]
    tbl = (
        mr.groupby("home_team").agg(pts=("home_pts", "sum"), gd=("home_gd", "sum"), gf=("home_goals", "sum")).rename_axis("team")
        + mr.groupby("away_team").agg(pts=("away_pts", "sum"), gd=("away_gd", "sum"), gf=("away_goals", "sum")).rename_axis("team")
    )
    tbl["position"] = (
        tbl[["pts", "gd", "gf"]]
        .apply(lambda col: col.rank(method="min", ascending=False))
        .mean(axis=1)
        .astype(int)
    )
    return int(tbl.loc[team, "position"])


class TestOverlayPosition:
    def _three_team_df(self):
        """
        Full round-robin: each pair plays home and away, so every team appears
        in both home and away roles (required by the + join in the overlay logic).

        Results:
          m1: A (h) 1-0 B   m2: B (h) 0-0 A
          m3: A (h) 1-0 C   m4: C (h) 0-0 A
          m5: B (h) 1-0 C   m6: C (h) 0-0 B
        Points: A=8, B=5, C=2 → A 1st, B 2nd, C 3rd.
        """
        rows = []
        for mid, ht, at, h_res, a_res in [
            ("m1", "A", "B", "Goal",      "SavedShot"),
            ("m2", "B", "A", "SavedShot", "SavedShot"),
            ("m3", "A", "C", "Goal",      "SavedShot"),
            ("m4", "C", "A", "SavedShot", "SavedShot"),
            ("m5", "B", "C", "Goal",      "SavedShot"),
            ("m6", "C", "B", "SavedShot", "SavedShot"),
        ]:
            rows += _make_shots(match_id=mid, home_team=ht, away_team=at, side="h", result=h_res, xg=0.3, n=1)
            rows += _make_shots(match_id=mid, home_team=ht, away_team=at, side="a", result=a_res, xg=0.1, n=1)
        return _build_df(rows)

    def test_leader_is_first(self):
        assert _compute_overlay_position(self._three_team_df(), "A") == 1

    def test_second_place_correct(self):
        assert _compute_overlay_position(self._three_team_df(), "B") == 2

    def test_bottom_team_correct(self):
        assert _compute_overlay_position(self._three_team_df(), "C") == 3

    def test_own_goals_not_counted(self):
        # OwnGoal rows are ignored (result != "Goal") so both matches end 0-0 →
        # equal points → both teams at position 1.
        # Each team must appear as home AND away, so we include the reverse fixture.
        df = _build_df(
            _make_shots(match_id="m1", home_team="A", away_team="B", side="h", result="OwnGoal",   xg=0.1, n=1)
            + _make_shots(match_id="m1", home_team="A", away_team="B", side="a", result="SavedShot", xg=0.1, n=1)
            + _make_shots(match_id="m2", home_team="B", away_team="A", side="h", result="SavedShot", xg=0.1, n=1)
            + _make_shots(match_id="m2", home_team="B", away_team="A", side="a", result="SavedShot", xg=0.1, n=1)
        )
        assert _compute_overlay_position(df, "A") == 1
        assert _compute_overlay_position(df, "B") == 1

    def test_teams_equal_on_points_share_position(self):
        # The algorithm ranks by mean(pts_rank, gd_rank, gf_rank).astype(int).
        # When two teams are equal on all three measures the truncation means
        # both receive position 1 — this test documents that known behaviour.
        df = _build_df(
            _make_shots(match_id="m1", home_team="A", away_team="B", side="h", result="Goal",     xg=0.3, n=1)
            + _make_shots(match_id="m1", home_team="A", away_team="B", side="a", result="SavedShot", xg=0.1, n=1)
            + _make_shots(match_id="m2", home_team="B", away_team="A", side="h", result="Goal",     xg=0.3, n=1)
            + _make_shots(match_id="m2", home_team="B", away_team="A", side="a", result="SavedShot", xg=0.1, n=1)
        )
        # Both win once, lose once → equal pts, GD, GF → both position 1
        assert _compute_overlay_position(df, "A") == 1
        assert _compute_overlay_position(df, "B") == 1
