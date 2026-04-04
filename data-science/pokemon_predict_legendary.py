"""
Predict whether a Pokémon is legendary using scikit-learn.
Data source: int_pokemon_stats (dbt intermediate model in data_stack.duckdb)

Run from the repo root:
    python data-science/predict_legendary.py
"""

import os
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
# The dbt view uses a relative CSV path, so we must open DuckDB from the dbt/ directory.
_orig_dir = os.getcwd()
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dbt"))
con = duckdb.connect("data_stack.duckdb", read_only=True)
df = con.execute("SELECT * FROM main_intermediate.int_pokemon_stats").df()
con.close()
os.chdir(_orig_dir)

print(f"Loaded {len(df)} rows from int_pokemon_stats")
print(f"Legendary: {df['is_legendary'].sum()}  |  Non-legendary: {(~df['is_legendary']).sum()}\n")

# ---------------------------------------------------------------------------
# 2. Feature selection
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "hp",
    "attack",
    "defense",
    "special_attack",
    "special_defense",
    "speed",
    "base_stat_total",
    "total_offense",
    "total_defense",
    "generation",
]
BOOL_FEATURES = ["is_dual_type", "is_mega"]
CAT_FEATURES = ["primary_type"]  # secondary_type is sparse; is_dual_type captures its presence

TARGET = "is_legendary"

# Fill null secondary_type so encoder doesn't choke (not used as feature here)
X = df[NUMERIC_FEATURES + BOOL_FEATURES + CAT_FEATURES].copy()
X[BOOL_FEATURES] = X[BOOL_FEATURES].astype(int)
y = df[TARGET].astype(int)

# ---------------------------------------------------------------------------
# 3. Preprocessing pipeline
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", NUMERIC_FEATURES + BOOL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ]
)

model = Pipeline(
    [
        ("prep", preprocessor),
        (
            "clf",
            GradientBoostingClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=3,
                random_state=42,
            ),
        ),
    ]
)

# ---------------------------------------------------------------------------
# 4. Train / test split (80/20, stratified to preserve class balance)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows\n")

# ---------------------------------------------------------------------------
# 5. Fit on training set, evaluate on held-out test set
# ---------------------------------------------------------------------------
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("=== Classification report (test set) ===")
print(classification_report(y_test, y_pred, target_names=["Non-legendary", "Legendary"]))

# ---------------------------------------------------------------------------
# 6. Feature importances
# ---------------------------------------------------------------------------
ohe_feature_names = model.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(CAT_FEATURES).tolist()
all_feature_names = NUMERIC_FEATURES + BOOL_FEATURES + ohe_feature_names

importances = model.named_steps["clf"].feature_importances_
importance_df = (
    pd.DataFrame({"feature": all_feature_names, "importance": importances})
    .sort_values("importance", ascending=False)
    .head(15)
    .reset_index(drop=True)
)

print("\n=== Top 15 feature importances ===")
print(importance_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 6b. Misclassified Pokémon
# ---------------------------------------------------------------------------
test_df = df.loc[X_test.index].copy()
test_df["predicted_legendary"] = y_pred.astype(bool)
misclassified = test_df[test_df["is_legendary"] != test_df["predicted_legendary"]]

DISPLAY_COLS = [
    "pokemon_name",
    "primary_type",
    "secondary_type",
    "hp",
    "attack",
    "defense",
    "special_attack",
    "special_defense",
    "speed",
    "base_stat_total",
    "is_legendary",
    "predicted_legendary",
]
print(f"\n=== Misclassified Pokémon ({len(misclassified)}) ===")
print(misclassified[DISPLAY_COLS].to_string(index=False))

# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Non-legendary", "Legendary"])
disp.plot(ax=axes[0], colorbar=False)
axes[0].set_title("Confusion matrix (test set)")

# Feature importances bar chart
axes[1].barh(
    importance_df["feature"][::-1],
    importance_df["importance"][::-1],
    color="steelblue",
)
axes[1].set_xlabel("Importance")
axes[1].set_title("Top 15 feature importances")
axes[1].tick_params(axis="y", labelsize=9)

plt.tight_layout()
plot_path = os.path.join(os.path.dirname(__file__), "outputs\legendary_model.png")
plt.savefig(plot_path, dpi=150)
print(f"\nPlot saved to {plot_path}")
plt.show()
