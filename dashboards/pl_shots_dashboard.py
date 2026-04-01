"""
Premier League Shots Dashboard
Mirrors the pl_shots_analytics.ipynb notebook as an interactive Streamlit app.

Run with:
    streamlit run dashboards/pl_shots_dashboard.py
"""

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PL Shots Analytics",
    page_icon="⚽",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
DATA_PATH = "data/raw/pl_shots_understat.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["match_date"] = pd.to_datetime(df["match_date"])
    df["x_m"] = df["x"] * 105
    df["y_m"] = df["y"] * 68
    return df


df_all = load_data()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

seasons = sorted(df_all["season"].unique())
selected_seasons = st.sidebar.multiselect("Seasons", seasons, default=seasons)

min_shots = st.sidebar.slider(
    "Min shots (xG performance charts)", min_value=5, max_value=100, value=20, step=5
)

df = df_all[df_all["season"].isin(selected_seasons)].copy()
goals = df[df["result"] == "Goal"].copy()

n_matches = df["match_id"].nunique()

st.sidebar.markdown("---")
st.sidebar.metric("Total shots", f"{len(df):,}")
st.sidebar.metric("Total goals", f"{len(goals):,}")
conv = len(goals) / len(df) if len(df) else 0
st.sidebar.metric("Conversion rate", f"{conv:.1%}")

# ---------------------------------------------------------------------------
# Pitch drawing helper
# ---------------------------------------------------------------------------
def draw_half_pitch(ax, lw=1.2):
    pitch_colour = "#2d5a27"
    line_colour = "white"

    ax.set_facecolor(pitch_colour)
    ax.add_patch(
        patches.Rectangle((52.5, 0), 52.5, 68, fill=False, edgecolor=line_colour, lw=lw, zorder=3)
    )
    ax.plot([52.5, 52.5], [0, 68], color=line_colour, lw=lw, zorder=3)
    ax.add_patch(
        patches.Rectangle((88.5, 13.84), 16.5, 40.32, fill=False, edgecolor=line_colour, lw=lw, zorder=3)
    )
    ax.add_patch(
        patches.Rectangle((99, 24.84), 6, 18.32, fill=False, edgecolor=line_colour, lw=lw, zorder=3)
    )
    ax.add_patch(
        patches.Rectangle((105, 30.34), 2, 7.32, fill=False, edgecolor="gold", lw=lw * 1.5, zorder=3)
    )
    ax.plot(93.5, 34, "o", color=line_colour, ms=2, zorder=3)
    arc = patches.Arc(
        (93.5, 34), 18.3, 18.3, angle=0, theta1=128, theta2=232, color=line_colour, lw=lw, zorder=3
    )
    ax.add_patch(arc)
    ax.set_xlim(50, 107)
    ax.set_ylim(-1, 69)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Season Overview",
    "Top Scorers",
    "xG Performance",
    "Situations & Types",
    "Goals by Minute",
    "Shot Map",
    "Heatmap",
    "Team Analysis",
])

# ── Tab 1: Season Overview ──────────────────────────────────────────────────
with tab1:
    st.subheader("Season overview")

    season_summary = (
        df.groupby("season")
        .agg(
            shots=("id", "count"),
            goals=("result", lambda x: (x == "Goal").sum()),
            xg_total=("xg", "sum"),
        )
        .assign(conversion=lambda d: d["goals"] / d["shots"])
        .assign(xg_per_shot=lambda d: d["xg_total"] / d["shots"])
        .assign(goals_per_xg=lambda d: d["goals"] / d["xg_total"])
        .sort_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    seasons_idx = season_summary.index
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c"][: len(seasons_idx)]

    axes[0].bar(seasons_idx, season_summary["shots"], color=colours)
    axes[0].set_title("Total shots")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    axes[1].bar(seasons_idx, season_summary["goals"], color=colours)
    axes[1].set_title("Total goals")

    axes[2].bar(seasons_idx, season_summary["conversion"], color=colours)
    axes[2].set_title("Conversion rate")
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    for ax in axes:
        ax.tick_params(axis="x", labelrotation=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Season overview", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        season_summary.style.format(
            {
                "shots": "{:,.0f}",
                "goals": "{:,.0f}",
                "xg_total": "{:,.1f}",
                "conversion": "{:.1%}",
                "xg_per_shot": "{:.3f}",
                "goals_per_xg": "{:.3f}",
            }
        ),
        use_container_width=True,
    )

# ── Tab 2: Top Scorers ──────────────────────────────────────────────────────
with tab2:
    st.subheader("Top 20 scorers")

    n_top = st.slider("Number of players to show", min_value=10, max_value=40, value=20, step=5)

    top_scorers = (
        goals.groupby("player")
        .agg(
            goals=("id", "count"),
            xg=("xg", "sum"),
        )
        .assign(
            shots=lambda d: d.index.map(
                df.groupby("player")["id"].count()
            )
        )
        .assign(xg_diff=lambda d: d["goals"] - d["xg"])
        .sort_values("goals", ascending=False)
        .head(n_top)
    )

    fig, ax = plt.subplots(figsize=(10, max(5, n_top * 0.35)))
    ax.barh(top_scorers.index, top_scorers["goals"], color="#1f77b4", label="Actual goals")
    ax.scatter(
        top_scorers["xg"],
        range(len(top_scorers)),
        color="#ff7f0e",
        zorder=5,
        s=50,
        label="xG",
    )
    ax.invert_yaxis()
    ax.set_xlabel("Goals")
    ax.set_title(f"Top {n_top} scorers", fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Tab 3: xG Over/Underperformers ─────────────────────────────────────────
with tab3:
    st.subheader(f"xG over/underperformers (min. {min_shots} shots)")

    player_stats = (
        df.groupby("player")
        .agg(
            shots=("id", "count"),
            goals=("result", lambda x: (x == "Goal").sum()),
            xg=("xg", "sum"),
        )
        .query("shots >= @min_shots")
        .assign(xg_diff=lambda d: d["goals"] - d["xg"])
    )

    n_each = st.slider("Players per side", min_value=5, max_value=20, value=10, step=1)

    top_over = player_stats.nlargest(n_each, "xg_diff")
    top_under = player_stats.nsmallest(n_each, "xg_diff")
    combined = pd.concat([top_over, top_under]).sort_values("xg_diff")

    fig, ax = plt.subplots(figsize=(10, max(5, len(combined) * 0.35)))
    colours = ["#d62728" if v > 0 else "#1f77b4" for v in combined["xg_diff"]]
    ax.barh(combined.index, combined["xg_diff"], color=colours)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Goals minus xG")
    ax.set_title(f"xG over/underperformers (min. {min_shots} shots)", fontweight="bold")

    x_max = combined["xg_diff"].max()
    x_min = combined["xg_diff"].min()
    if x_max > 0:
        ax.annotate("Overperforming →", xy=(x_max * 0.6, 1), fontsize=8, color="#d62728")
    if x_min < 0:
        ax.annotate(
            "← Underperforming",
            xy=(x_min * 0.6, 1),
            fontsize=8,
            color="#1f77b4",
            ha="right",
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Tab 4: Situations & Shot Types ──────────────────────────────────────────
with tab4:
    st.subheader("Situation × shot type")

    metric = st.radio(
        "Colour by", ["Goals", "Conversion rate"], horizontal=True, key="tab4_metric"
    )

    pivot_goals = (
        df.groupby(["situation", "shot_type"])
        .agg(shots=("id", "count"), goals=("result", lambda x: (x == "Goal").sum()))
        .reset_index()
        .pivot(index="situation", columns="shot_type", values=["shots", "goals"])
    )
    shots_pivot = pivot_goals["shots"].fillna(0).astype(int)
    goals_pivot = pivot_goals["goals"].fillna(0).astype(int)
    conv_pivot  = (goals_pivot / shots_pivot.replace(0, np.nan)).fillna(0)

    if metric == "Goals":
        heat_data = goals_pivot
        cmap = "Blues"
        norm = None
        fmt_val  = lambda v: f"{int(v):,}"
        cbar_label = "Goals"
    else:
        heat_data = conv_pivot
        cmap = mcolors.LinearSegmentedColormap.from_list("wht_grn", ["white", "#2d5a27"])
        bounds = [0, 0.01, 0.05, 0.10, 0.20, 0.5, 0.75, 1.0]
        norm = mcolors.BoundaryNorm(bounds, ncolors=256)
        fmt_val  = lambda v: f"{v:.0%}"
        cbar_label = "Conversion rate"

    n_rows, n_cols = heat_data.shape
    fig, ax = plt.subplots(figsize=(max(7, n_cols * 1.4), max(4, n_rows * 0.9)))

    im = ax.imshow(heat_data.values, cmap=cmap, aspect="auto", norm=norm)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(heat_data.columns, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(heat_data.index, fontsize=10)

    # Annotate each cell with the coloured-metric value and (shots) underneath
    for r in range(n_rows):
        for c in range(n_cols):
            n_s = shots_pivot.iloc[r, c]
            cell_val = heat_data.iloc[r, c]
            # Pick text colour based on cell brightness
            norm_val = cell_val / (heat_data.values.max() or 1)
            txt_colour = "white" if norm_val > 0.6 else "black"
            ax.text(
                c, r,
                fmt_val(cell_val),
                ha="center", va="center",
                fontsize=11, fontweight="bold", color=txt_colour,
            )
            ax.text(
                c, r + 0.28,
                f"({n_s:,} shots)",
                ha="center", va="center",
                fontsize=7, color=txt_colour, alpha=0.8,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(cbar_label)
    if metric == "Conversion rate":
        cbar.set_ticks([0, 0.01, 0.05, 0.10, 0.20, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(["0%", "1%", "5%", "10%", "20%", "50%", "75%", "100%"])

    ax.set_title("Situation × Shot type", fontweight="bold", pad=12)
    ax.set_xlabel("Shot type")
    ax.set_ylabel("Situation")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Tab 5: Goals by Minute ──────────────────────────────────────────────────
with tab5:
    st.subheader("Goals by minute")

    goals_plot = goals.copy()
    goals_plot["minute_capped"] = goals_plot["minute"].clip(upper=95)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(
        goals_plot["minute_capped"],
        bins=range(1, 97),
        weights=[1 / n_matches] * len(goals_plot) if n_matches else None,
        color="#2ca02c",
        edgecolor="white",
        linewidth=0.3,
    )
    ax.axvline(45, color="grey", linestyle="--", linewidth=0.8, label="Half time")
    ax.set_xlabel("Minute")
    ax.set_ylabel("Goals per match")
    ax.set_title("Goals per match by minute (all selected seasons)", fontweight="bold")
    ax.set_xticks([1, 15, 30, 45, 60, 75, 90, 95])
    ax.set_xticklabels(["1", "15", "30", "45", "60", "75", "90", "90+"])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Half-time state split ──────────────────────────────────────────────
    st.subheader("Second-half goals by half-time score state")

    first_half = df[df["result"].isin(["Goal", "OwnGoal"]) & (df["minute"] <= 45)].copy()

    def ht_contributions(row):
        if row["result"] == "Goal":
            return (1, 0) if row["side"] == "h" else (0, 1)
        else:
            return (0, 1) if row["side"] == "h" else (1, 0)

    if not first_half.empty:
        contrib = first_half.apply(ht_contributions, axis=1, result_type="expand")
        contrib.columns = ["home", "away"]
        first_half = first_half.join(contrib)

        ht_scores = (
            first_half.groupby("match_id")[["home", "away"]]
            .sum()
            .rename(columns={"home": "ht_home", "away": "ht_away"})
        )
        ht_scores["ht_gd"] = (ht_scores["ht_home"] - ht_scores["ht_away"]).abs()
        ht_scores["ht_state"] = ht_scores["ht_gd"].apply(
            lambda gd: "Close (GD 0–1)" if gd <= 1 else "Comfortable (GD 2+)"
        )

        second_half_goals = (
            goals[goals["minute"] > 45]
            .merge(ht_scores[["ht_state"]], on="match_id")
            .copy()
        )
        second_half_goals["minute_capped"] = second_half_goals["minute"].clip(upper=95)

        states = ["Close (GD 0–1)", "Comfortable (GD 2+)"]
        colours = ["#1f77b4", "#d62728"]
        bins = range(46, 98, 3)

        fig, ax = plt.subplots(figsize=(12, 5))
        for state, colour in zip(states, colours):
            subset = second_half_goals[second_half_goals["ht_state"] == state]
            n = ht_scores[ht_scores["ht_state"] == state].shape[0]
            if n > 0:
                ax.hist(
                    subset["minute_capped"],
                    bins=bins,
                    weights=[1 / n] * len(subset),
                    histtype="step",
                    linewidth=2,
                    color=colour,
                    label=f"{state} (n={n:,} matches)",
                )

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="grey", alpha=0.7)
        ax.set_axisbelow(True)
        ax.set_xlabel("Minute")
        ax.set_ylabel("Goals per match")
        ax.set_title(
            "Second-half goals per match (3-min bins) — close vs comfortable HT scorelines",
            fontweight="bold",
        )
        ax.set_xticks(list(range(46, 93, 6)) + [95])
        ax.set_xticklabels([str(m) for m in range(46, 93, 6)] + ["90+"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No data available for the selected seasons.")

# ── Tab 6: Shot Map ─────────────────────────────────────────────────────────
with tab6:
    st.subheader("Shot locations on the pitch")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].set_facecolor("#2d5a27")
    sc = axes[0].scatter(
        df["x_m"], df["y_m"],
        c=df["xg"], cmap="YlOrRd", s=4, alpha=0.3, vmin=0, vmax=0.5, zorder=1,
    )
    draw_half_pitch(axes[0], lw=2)
    axes[0].set_title("All shots (coloured by xG)", color="white", fontweight="bold", pad=8)

    axes[1].set_facecolor("#2d5a27")
    axes[1].scatter(
        goals["x_m"], goals["y_m"],
        c=goals["xg"], cmap="YlOrRd", s=10, alpha=0.5, vmin=0, vmax=0.5, zorder=1,
    )
    draw_half_pitch(axes[1], lw=2)
    axes[1].set_title("Goals only (coloured by xG)", color="white", fontweight="bold", pad=8)

    fig.patch.set_facecolor("#2d5a27")
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=mcolors.Normalize(vmin=0, vmax=0.5))
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("xG", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.suptitle(
        "Shot locations — EPL (selected seasons)", color="white", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Tab 7: Heatmap ──────────────────────────────────────────────────────────
with tab7:
    st.subheader("Shot density heatmap")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    titles = ["All shots", "Goals only"]
    datasets = [df, goals]

    for ax, title, data in zip(axes, titles, datasets):
        ax.set_facecolor("#2d5a27")
        if not data.empty:
            h, xedges, yedges = np.histogram2d(
                data["x_m"],
                data["y_m"],
                bins=[np.linspace(52.5, 105, 30), np.linspace(0, 68, 20)],
            )
            ax.pcolormesh(
                xedges,
                yedges,
                h.T,
                cmap="hot",
                alpha=0.65,
                norm=mcolors.PowerNorm(gamma=0.4),
                zorder=1,
            )
        draw_half_pitch(ax, lw=2)
        ax.set_title(title, color="white", fontweight="bold", pad=8)

    fig.patch.set_facecolor("#2d5a27")
    plt.suptitle(
        "Shot density heatmap — EPL (selected seasons)", color="white", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Tab 8: Team Analysis ────────────────────────────────────────────────────
with tab8:
    st.subheader("Team analysis")

    # ── Derive shooting/defending team per shot ──────────────────────────────
    df_t = df.copy()
    df_t["shooting_team"]  = np.where(df_t["side"] == "h", df_t["home_team"], df_t["away_team"])
    df_t["defending_team"] = np.where(df_t["side"] == "h", df_t["away_team"], df_t["home_team"])

    attack = (
        df_t.groupby("shooting_team")
        .agg(
            shots_for=("id", "count"),
            goals_for=("result", lambda x: (x == "Goal").sum()),
            xg_for=("xg", "sum"),
        )
        .rename_axis("team")
    )
    defense = (
        df_t.groupby("defending_team")
        .agg(
            shots_against=("id", "count"),
            goals_against=("result", lambda x: (x == "Goal").sum()),
            xg_against=("xg", "sum"),
        )
        .rename_axis("team")
    )
    home_m = df_t.groupby("home_team")["match_id"].nunique().rename_axis("team")
    away_m = df_t.groupby("away_team")["match_id"].nunique().rename_axis("team")
    matches_per_team = home_m.add(away_m, fill_value=0).rename("matches")

    ts = attack.join(defense).join(matches_per_team)

    # ── Ever-present filter ──────────────────────────────────────────────────
    teams_per_season = (
        df_t.groupby("shooting_team")["season"].nunique().rename("seasons_present")
    )
    ever_present = teams_per_season[teams_per_season == len(selected_seasons)].index
    only_ever_present = st.checkbox(
        f"Only show teams present in all {len(selected_seasons)} selected season(s)",
        value=True,
    )
    if only_ever_present:
        ts = ts[ts.index.isin(ever_present)]

    ts["xg_per_shot_for"]     = ts["xg_for"]       / ts["shots_for"]
    ts["xg_per_shot_against"] = ts["xg_against"]    / ts["shots_against"]
    ts["shots_per_match"]     = ts["shots_for"]     / ts["matches"]
    ts["goals_per_match"]     = ts["goals_for"]     / ts["matches"]
    ts["attack_diff"]         = ts["goals_for"]     - ts["xg_for"]       # +ve = overperforming attack
    ts["defense_diff"]        = ts["xg_against"]    - ts["goals_against"] # +ve = overperforming defence

    # ── Derive average league finishing position from match results ───────────
    all_matches = (
        df_t.groupby(["match_id", "season", "home_team", "away_team"])
        .size()
        .reset_index()[["match_id", "season", "home_team", "away_team"]]
    )
    goal_rows = df_t[df_t["result"].isin(["Goal", "OwnGoal"])].copy()
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

    home_table = mr.groupby(["season", "home_team"]).agg(
        pts=("home_pts", "sum"), gd=("home_gd", "sum"), gf=("home_goals", "sum")
    ).rename_axis(["season", "team"])
    away_table = mr.groupby(["season", "away_team"]).agg(
        pts=("away_pts", "sum"), gd=("away_gd", "sum"), gf=("away_goals", "sum")
    ).rename_axis(["season", "team"])
    season_table = (home_table + away_table).reset_index()
    season_table["position"] = (
        season_table.groupby("season")[["pts", "gd", "gf"]]
        .apply(lambda g: g.rank(method="min", ascending=False).mean(axis=1))
        .reset_index(level=0, drop=True)
        .astype(int)
    )
    avg_position = (
        season_table.groupby("team")["position"].mean().rename("avg_position")
    )
    ts = ts.join(avg_position)

    def ordinal(n: float) -> str:
        n = round(n)
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10 if n % 100 not in (11, 12, 13) else 0, "th")
        return f"{n}{suffix}"

    def team_label(team: str, row) -> str:
        gpg = row["goals_per_match"]
        pos = row.get("avg_position", float("nan"))
        pos_str = ordinal(pos) if not np.isnan(pos) else "?"
        return f"{team} ({gpg:.1f} gpg, {pos_str})"

    # ── Chart 1: Four-quadrant — attack quality vs defensive quality ─────────
    st.markdown("### Shot quality: attack vs defence")
    st.caption(
        "Each axis shows average xG per shot — a proxy for chance quality. "
        "X-axis: quality of chances **allowed** (lower = better defence). "
        "Y-axis: quality of chances **created** (higher = better attack). "
        "Quadrant lines are league averages. Bubble size = goals per match."
    )

    mean_x = ts["xg_per_shot_against"].mean()
    mean_y = ts["xg_per_shot_for"].mean()

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(
        ts["xg_per_shot_against"], ts["xg_per_shot_for"],
        s=ts["goals_per_match"] * 120, color="#1f77b4", alpha=0.7, zorder=3,
    )
    for team, row in ts.iterrows():
        ax.annotate(
            team_label(team, row), (row["xg_per_shot_against"], row["xg_per_shot_for"]),
            fontsize=7.5, ha="center", va="bottom",
            xytext=(0, 5), textcoords="offset points",
        )
    ax.axvline(mean_x, color="grey", linestyle="--", linewidth=0.8, zorder=1)
    ax.axhline(mean_y, color="grey", linestyle="--", linewidth=0.8, zorder=1)

    # Quadrant labels — placed after drawing so limits are set
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    pad_x = (x_hi - x_lo) * 0.01
    pad_y = (y_hi - y_lo) * 0.01
    ax.text(x_lo + pad_x, y_hi - pad_y, "Good attack\nGood defence",  fontsize=8, color="green",  va="top")
    ax.text(x_hi - pad_x, y_hi - pad_y, "Good attack\nPoor defence",  fontsize=8, color="orange", va="top",    ha="right")
    ax.text(x_lo + pad_x, y_lo + pad_y, "Poor attack\nGood defence",  fontsize=8, color="orange", va="bottom")
    ax.text(x_hi - pad_x, y_lo + pad_y, "Poor attack\nPoor defence",  fontsize=8, color="red",    va="bottom", ha="right")

    ax.set_xlabel("Quality of defensive chances allowed")
    ax.set_ylabel("Quality of offensive chances created")
    ax.set_title("Shot quality: attack vs defence", fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Chart 2: xG over/underperformance — attack & defence ────────────────
    st.markdown("### xG over/underperformance")
    st.caption(
        "**Attack diff** = goals scored − xG for. Positive: scored more than chances deserved. "
        "**Defence diff** = xG against − goals conceded. Positive: conceded fewer than expected. "
        "Teams sorted by combined over/underperformance."
    )

    ts_sorted = (
        ts.assign(combined=ts["attack_diff"] + ts["defense_diff"])
        .sort_values("combined", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(11, max(6, len(ts_sorted) * 0.32)))
    y = np.arange(len(ts_sorted))
    height = 0.35

    ax.barh(y + height / 2, ts_sorted["attack_diff"],  height=height,
            label="Attack (goals − xG for)",            color="#1f77b4", alpha=0.85)
    ax.barh(y - height / 2, ts_sorted["defense_diff"], height=height,
            label="Defence (xG against − goals conceded)", color="#2ca02c", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(ts_sorted.index, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Goals relative to xG")
    ax.set_title("xG over/underperformance by team", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Chart 3: Shot volume vs shot quality ────────────────────────────────
    st.markdown("### Shot volume vs shot quality")
    st.caption(
        "X-axis: shots taken per match (volume). "
        "Y-axis: average xG per shot (quality). "
        "Bubble size = goals per match. "
        "Top-right = lots of high-quality chances; bottom-left = the opposite."
    )

    mean_vol  = ts["shots_per_match"].mean()
    mean_qual = ts["xg_per_shot_for"].mean()

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(
        ts["shots_per_match"], ts["xg_per_shot_for"],
        s=ts["goals_per_match"] * 120, color="#ff7f0e", alpha=0.7, zorder=3,
    )
    for team, row in ts.iterrows():
        ax.annotate(
            team_label(team, row), (row["shots_per_match"], row["xg_per_shot_for"]),
            fontsize=7.5, ha="center", va="bottom",
            xytext=(0, 5), textcoords="offset points",
        )
    ax.axvline(mean_vol,  color="grey", linestyle="--", linewidth=0.8, zorder=1)
    ax.axhline(mean_qual, color="grey", linestyle="--", linewidth=0.8, zorder=1)

    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    pad_x = (x_hi - x_lo) * 0.01
    pad_y = (y_hi - y_lo) * 0.01
    ax.text(x_hi - pad_x, y_hi - pad_y, "High volume\nHigh quality",  fontsize=8, color="green",  va="top",    ha="right")
    ax.text(x_lo + pad_x, y_hi - pad_y, "Low volume\nHigh quality",   fontsize=8, color="orange", va="top")
    ax.text(x_hi - pad_x, y_lo + pad_y, "High volume\nLow quality",   fontsize=8, color="orange", va="bottom", ha="right")
    ax.text(x_lo + pad_x, y_lo + pad_y, "Low volume\nLow quality",    fontsize=8, color="red",    va="bottom")

    ax.set_xlabel("Shots per match")
    ax.set_ylabel("Average xG per shot")
    ax.set_title("Shot volume vs shot quality", fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
