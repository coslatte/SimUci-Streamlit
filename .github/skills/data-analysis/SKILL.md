---
name: data-analysis
description: Guidance for using Pandas, NumPy, Statistics, and Machine Learning within the SimUCI context.
---

# Data Analysis & Science Skills

## 1. Pandas Best Practices
- **Vectorization**: Avoid iterating over rows (`iterrows`). Use vectorized operations for simulation data processing.
    - *Bad*: `[row['a'] + row['b'] for _, row in df.iterrows()]`
    - *Good*: `df['a'] + df['b']`
- **Missing Values**: Explicitly handle `NaN`. Simulation data must be clean.
    - Use `fillna()` or `dropna()` with valid reasoning documented.
- **Loading Data**:
    - Use `pathlib` for paths.
    - Load constants from `utils/constants/limits.py` if filtering during load.

## 2. Statistics (SciPy & Stats models)
- **Comparisons**: Use `uci.stats` module.
    - `Friedman` test for multiple group comparisons.
    - `Wilcoxon` for paired comparisons.
- **Distributions**: Refer to `uci.distribuciones` for random variable generation (e.g., `norm`, `expon`).

## 3. Machine Learning (Scikit-Learn)
- **Pipelines**: Use `sklearn.pipeline.Pipeline` for preprocessing + modeling.
- **Persistence**: Save/Load models using `joblib` in `models/` directory.
- **Reproducibility**: Always set `random_state` (or seed) for stochastic models.

## 4. Simulation Data (SimPy Integration)
- **Output**: Simulation runs should generate consistent DataFrames.
- **Structure**:
    - Rows: Individual simulation events or patient runs.
    - Columns: Metrics (time in UCI, survival status, cost).
- **Aggregations**: Calculate aggregate stats (mean, std, percentiles) after collecting full simulation batch results.
