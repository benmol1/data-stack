"""
Tests for pl_goals_fetch_understat.py

UnderstatClient calls are mocked throughout — no network access required.
"""

import pandas as pd
from unittest.mock import MagicMock, patch

from pl_goals_fetch_understat import (
    fetch_shots_for_match,
    fetch_shots_for_season,
    process,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MATCH = {
    "id": "99999",
    "h": {"title": "Arsenal"},
    "a": {"title": "Chelsea"},
    "datetime": "2023-08-12 12:30:00",
    "isResult": True,
}

RAW_SHOT_DATA = {
    "h": [
        {
            "player": "Bukayo Saka",
            "player_id": "1001",
            "id": "s1",
            "minute": "34",
            "X": "0.85",
            "Y": "0.52",
            "xG": "0.23",
            "result": "Goal",
            "situation": "OpenPlay",
            "shotType": "RightFoot",
        }
    ],
    "a": [
        {
            "player": "Cole Palmer",
            "player_id": "1002",
            "id": "s2",
            "minute": "61",
            "X": "0.78",
            "Y": "0.48",
            "xG": "0.11",
            "result": "SavedShot",
            "situation": "OpenPlay",
            "shotType": "LeftFoot",
        }
    ],
}


# ---------------------------------------------------------------------------
# fetch_shots_for_match
# ---------------------------------------------------------------------------


class TestFetchShotsForMatch:
    def _make_client(self):
        """Return a mock UnderstatClient whose .match().get_shot_data() returns RAW_SHOT_DATA."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.match.return_value.get_shot_data.return_value = RAW_SHOT_DATA
        return mock_client

    def test_returns_one_row_per_shot(self):
        with patch(
            "pl_goals_fetch_understat.UnderstatClient", return_value=self._make_client()
        ):
            rows = fetch_shots_for_match(MATCH, "2023")
        assert len(rows) == 2

    def test_match_metadata_attached(self):
        with patch(
            "pl_goals_fetch_understat.UnderstatClient", return_value=self._make_client()
        ):
            rows = fetch_shots_for_match(MATCH, "2023")
        for row in rows:
            assert row["match_id"] == "99999"
            assert row["home_team"] == "Arsenal"
            assert row["away_team"] == "Chelsea"
            assert row["match_date"] == "2023-08-12"
            assert row["season"] == "2023/24"

    def test_side_field_set_correctly(self):
        with patch(
            "pl_goals_fetch_understat.UnderstatClient", return_value=self._make_client()
        ):
            rows = fetch_shots_for_match(MATCH, "2023")
        sides = {r["player"]: r["side"] for r in rows}
        assert sides["Bukayo Saka"] == "h"
        assert sides["Cole Palmer"] == "a"

    def test_returns_empty_list_on_api_error(self):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.match.return_value.get_shot_data.side_effect = Exception(
            "network error"
        )

        with patch(
            "pl_goals_fetch_understat.UnderstatClient", return_value=mock_client
        ):
            rows = fetch_shots_for_match(MATCH, "2023")
        assert rows == []


# ---------------------------------------------------------------------------
# fetch_shots_for_season
# ---------------------------------------------------------------------------


class TestFetchShotsForSeason:
    def _make_client(self, matches):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.league.return_value.get_match_data.return_value = matches
        return mock_client

    def test_returns_dataframe(self):
        with patch(
            "pl_goals_fetch_understat.UnderstatClient",
            return_value=self._make_client([MATCH]),
        ):
            with patch(
                "pl_goals_fetch_understat.fetch_shots_for_match",
                return_value=[{"player": "Saka"}],
            ):
                df = fetch_shots_for_season("2023")
        assert isinstance(df, pd.DataFrame)

    def test_skips_unplayed_matches(self):
        unplayed = {**MATCH, "isResult": False}
        with patch(
            "pl_goals_fetch_understat.UnderstatClient",
            return_value=self._make_client([unplayed]),
        ):
            with patch("pl_goals_fetch_understat.fetch_shots_for_match") as mock_fetch:
                fetch_shots_for_season("2023")
        mock_fetch.assert_not_called()

    def test_returns_empty_dataframe_when_no_matches(self):
        with patch(
            "pl_goals_fetch_understat.UnderstatClient",
            return_value=self._make_client([]),
        ):
            df = fetch_shots_for_season("2023")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------


def _raw_df(**overrides):
    """Build a minimal raw shot DataFrame as returned by fetch_all_seasons."""
    base = {
        "season": "2023/24",
        "match_date": "2023-08-12",
        "match_id": "99999",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "side": "h",
        "player": "Bukayo Saka",
        "player_id": "1001",
        "id": "s1",
        "minute": "34",
        "X": "0.85",
        "Y": "0.52",
        "xG": "0.23",
        "result": "Goal",
        "situation": "OpenPlay",
        "shotType": "RightFoot",
    }
    base.update(overrides)
    return pd.DataFrame([base])


class TestProcess:
    def test_renames_coordinate_columns(self):
        df = process(_raw_df())
        assert "x" in df.columns
        assert "y" in df.columns
        assert "X" not in df.columns
        assert "Y" not in df.columns

    def test_renames_xg_column(self):
        df = process(_raw_df())
        assert "xg" in df.columns
        assert "xG" not in df.columns

    def test_renames_shot_type_column(self):
        df = process(_raw_df())
        assert "shot_type" in df.columns
        assert "shotType" not in df.columns

    def test_numeric_columns_are_numeric(self):
        df = process(_raw_df())
        for col in ["minute", "x", "y", "xg"]:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} should be numeric"

    def test_result_column_preserved(self):
        df = process(_raw_df(result="SavedShot"))
        assert df["result"].iloc[0] == "SavedShot"

    def test_all_result_types_preserved(self):
        rows = pd.concat(
            [
                _raw_df(result=r, id=str(i))
                for i, r in enumerate(
                    ["Goal", "SavedShot", "MissedShots", "BlockedShot"]
                )
            ]
        )
        df = process(rows)
        assert set(df["result"]) == {"Goal", "SavedShot", "MissedShots", "BlockedShot"}

    def test_drops_unknown_columns(self):
        raw = _raw_df()
        raw["unexpected_column"] = "junk"
        df = process(raw)
        assert "unexpected_column" not in df.columns

    def test_sorted_by_date_match_minute(self):
        rows = pd.concat(
            [
                _raw_df(match_date="2023-08-19", minute="10", id="s3"),
                _raw_df(match_date="2023-08-12", minute="34", id="s1"),
                _raw_df(match_date="2023-08-12", minute="61", id="s2"),
            ]
        )
        df = process(rows)
        assert df["match_date"].tolist() == ["2023-08-12", "2023-08-12", "2023-08-19"]
        assert df["minute"].tolist() == [34.0, 61.0, 10.0]

    def test_handles_missing_optional_columns_gracefully(self):
        raw = _raw_df().drop(columns=["player_id", "id"])
        df = process(raw)
        assert "player_id" not in df.columns
        assert len(df) == 1
