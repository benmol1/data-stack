# data-stack

A personal data engineering and analytics playground with two main domains: **Premier League shot analytics** and **Pokémon data science**. The stack combines Python data pipelines, dbt transformations on DuckDB, a Streamlit dashboard, and scikit-learn models.

## What's in here

### Premier League shots pipeline (`data/`)

`data/pl_shots_fetch_understat.py` fetches shot-level data for EPL seasons 2020/21 – 2024/25 from the Understat API. It uses a thread pool to fetch all matches concurrently and writes the result to `data/raw/pl_shots_understat.csv`. Incremental by design — re-running only fetches seasons not already present.

Each shot record includes: player, pitch coordinates (x/y), xG, shot type, situation (open play, corner, etc.), result (Goal, SavedShot, etc.), and match metadata.

### Premier League shots dashboard (`dashboards/`)

`dashboards/pl_shots_dashboard.py` is a Streamlit app that visualises the shot data across nine tabs:

| Tab | Content |
|-----|---------|
| Season Overview | Shot volume, goals, conversion rates by season |
| Top Scorers | Goals and xG by player |
| xG Performance | Players over/underperforming their xG |
| Situations & Types | Shot breakdown by situation (open play, set piece, etc.) and foot/head |
| Goals by Minute | Goal timing distribution across the 90 minutes |
| Shot Map | Scatter plot of shots on a half-pitch by player or team |
| Heatmap | Shot density heatmap on a half-pitch |
| Team Analysis | Team-level metrics: xG per shot, shots/match, attack/defence xG diff, league table |
| Man Utd Deep Dive | Per-season trend charts for Man United (xG, finishing diff, league position overlay) |

Run it with:

```bash
streamlit run dashboards/pl_shots_dashboard.py
```

### Pokémon dbt models (`dbt/`)

A dbt project using DuckDB as the warehouse. Source data is `data/raw/PokemonData.csv`.

| Layer | Models |
|-------|--------|
| Staging | `stg_pokemon` — typed and renamed raw columns |
| Intermediate | `int_pokemon_stats` — derived fields: `base_stat_total`, `total_offense`, `total_defense`, `is_dual_type`, `is_mega` |
| Marts | `mart_pokemon_by_type`, `mart_pokemon_by_generation`, `mart_legendary_comparison` |

Run with:

```bash
cd dbt
dbt run --profiles-dir .
dbt test --profiles-dir .
```

> **Note:** shut down any running Jupyter kernel before running `dbt run` — DuckDB allows only one writer at a time.

### Pokémon legendary classifier (`data-science/`)

`data-science/pokemon_predict_legendary.py` trains a `GradientBoostingClassifier` to predict whether a Pokémon is legendary, using the `int_pokemon_stats` dbt view as its data source. Outputs a confusion matrix and feature importance chart to `data-science/outputs/legendary_model.png`.

Run from the repo root:

```bash
python data-science/pokemon_predict_legendary.py
```

### Notebooks (`notebooks/`)

- `notebooks/pl_shots_analytics.ipynb` — exploratory analysis of the PL shot data (the notebook version of the Streamlit dashboard)
- `notebooks/pokemon_analytics.ipynb` — Pokémon data exploration

### Tests (`tests/`)

pytest suite covering the dashboard's core analytics logic (no Streamlit rendering tested):

- `test_pl_shots_dashboard.py` — team metric derivation, season table / points calculation, own goal handling, ordinal formatting, per-season stats, and year-on-year % change
- `test_pl_shots_fetch_understat.py` — pipeline fetching logic

Run tests:

```bash
pytest tests/
```

## Getting started

### Python environment

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

To reproduce a pinned environment exactly:

```bash
pip install -r requirements.lock
```

### Jupyter kernel

Register the venv as a Jupyter kernel (one-time setup):

```bash
pip install ipykernel
python -m ipykernel install --user --name data-stack --display-name "data-stack"
```

Then select **data-stack** as the kernel when opening notebooks.
