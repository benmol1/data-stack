# data-stack

Experiments and tooling for data pipeline work, analytics, and data science — powered by Claude.

## Structure

- `pipelines/` — data pipeline definitions and orchestration
- `analytics/` — exploratory analysis and reporting
- `science/` — data science experiments and models
- `notebooks/` — Jupyter notebooks

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

### dbt

```bash
cd dbt
dbt run --profiles-dir .
dbt test --profiles-dir .
```

> **Note:** shut down any running Jupyter kernel before running `dbt run` — DuckDB allows only one writer at a time.

### Data science

```bash
python data-science/predict_legendary.py
```
