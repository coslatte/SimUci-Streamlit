# SimUCI - Statistical Analysis Documentation

## Overview

The statistical analysis module (`uci/stats.py`) provides comprehensive tools for validating simulation models against real patient data.

---

## Validation Metrics

### 1. Coverage Percentage

**Purpose**: Measures how often the true values fall within the simulation's confidence intervals.

**Formula**:
$$\text{Coverage}_v = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[t_{i,v} \in CI_{i,v}] \times 100\%$$

Where:
- $t_{i,v}$ = true value for patient $i$, variable $v$
- $CI_{i,v}$ = confidence interval from simulation for patient $i$, variable $v$
- $N$ = number of patients

**Interpretation**:
| Coverage | Interpretation |
|----------|---------------|
| > 90% | Excellent calibration |
| 80-90% | Good calibration |
| 70-80% | Acceptable, may need adjustment |
| < 70% | Poor calibration, review model |

**Code Example**:
```python
from uci.stats import SimulationMetrics

metrics = SimulationMetrics(true_data, simulation_data)
metrics.evaluate(confidence_level=0.95)

# Access coverage by variable
for var, coverage in metrics.coverage_percentage.items():
    print(f"{var}: {coverage:.1f}%")
```

---

### 2. Error Metrics (RMSE, MAE, MAPE)

#### Root Mean Squared Error (RMSE)

$$RMSE = \sqrt{\frac{1}{N} \sum_{i,v} (\bar{x}_{i,v} - t_{i,v})^2}$$

- **Units**: Same as input (hours)
- **Sensitivity**: Penalizes large errors heavily
- **Use case**: When large deviations are particularly problematic

#### Mean Absolute Error (MAE)

$$MAE = \frac{1}{N} \sum_{i,v} |\bar{x}_{i,v} - t_{i,v}|$$

- **Units**: Same as input (hours)
- **Sensitivity**: Equal weight to all errors
- **Use case**: General error magnitude assessment

#### Mean Absolute Percentage Error (MAPE)

$$MAPE = 100 \times \frac{1}{N} \sum_{i,v} \frac{|\bar{x}_{i,v} - t_{i,v}|}{t_{i,v}}$$

- **Units**: Percentage
- **Sensitivity**: Relative error measure
- **Note**: Excludes zero values to avoid division by zero

**Interpretation Guide**:

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| RMSE | < 24h | 24-48h | > 48h |
| MAE | < 12h | 12-24h | > 24h |
| MAPE | < 10% | 10-25% | > 25% |

**Joint Interpretation**:

| RMSE | MAE | MAPE | Interpretation |
|------|-----|------|---------------|
| Low | Low | Low | Model fits well |
| High | Low | Low | Few large outliers |
| Low | Low | High | Errors on small values |
| High | High | Low | Systematic bias on large values |

---

### 3. Kolmogorov-Smirnov (KS) Test

**Purpose**: Tests whether two distributions are statistically different.

**Statistic**:
$$D = \sup_x |F_{sim}(x) - F_{true}(x)|$$

Where:
- $F_{sim}(x)$ = empirical CDF of simulated data
- $F_{true}(x)$ = empirical CDF of true data

**Interpretation**:

| p-value | Interpretation |
|---------|---------------|
| ≥ 0.05 | Cannot reject null hypothesis (distributions similar) |
| < 0.05 | Reject null hypothesis (distributions differ) |
| < 0.01 | Strong evidence of difference |

**Per-Variable Analysis**:

```python
ks_result = metrics.kolmogorov_smirnov_result
# Structure: {
#     "per_variable": {
#         "Tiempo VAM": {"statistic": 0.15, "p_value": 0.23},
#         "Estadia UCI": {"statistic": 0.08, "p_value": 0.67},
#         ...
#     },
#     "overall": {"statistic": 0.11, "p_value": 0.45}
# }

for var, result in ks_result["per_variable"].items():
    status = "✓" if result["p_value"] >= 0.05 else "✗"
    print(f"{var}: D={result['statistic']:.3f}, p={result['p_value']:.3f} {status}")
```

---

### 4. Anderson-Darling (AD) Test

**Purpose**: Similar to KS but more sensitive to differences in distribution tails.

**Features**:
- More powerful than KS for detecting tail differences
- Better for detecting outliers
- Uses k-sample variant for comparing two distributions

**Code**:
```python
ad_result = metrics.anderson_darling_result
# Structure: {"statistic": 1.234, "significance_level": 0.045}

if ad_result["significance_level"] < 0.05:
    print("Significant difference detected in distribution tails")
```

---

## Statistical Tests for Experiment Comparison

### Wilcoxon Signed-Rank Test

**Purpose**: Compare two related samples (paired observations).

**Use Cases**:
- Compare two different simulation configurations
- Before/after treatment effect analysis
- Comparing two experiments on the same patients

**Assumptions**:
- Paired samples (same patients)
- Differences are symmetric
- At least 5-10 paired observations

**Usage**:
```python
from uci.stats import Wilcoxon

# Load two experiment results
experiment1_vam = df1["Tiempo VAM"]
experiment2_vam = df2["Tiempo VAM"]

# Run test
wilcoxon = Wilcoxon(x=experiment1_vam, y=experiment2_vam)
wilcoxon.test()

print(f"Statistic: {wilcoxon.statistic}")
print(f"p-value: {wilcoxon.p_value}")

if wilcoxon.p_value < 0.05:
    print("Significant difference between experiments")
```

---

### Friedman Test

**Purpose**: Compare three or more related samples.

**Use Cases**:
- Compare multiple simulation configurations
- Ranking different treatment options
- Evaluating multiple parameter settings

**Assumptions**:
- Three or more related samples
- Same subjects across all samples
- Ordinal or continuous data

**Usage**:
```python
from uci.stats import Friedman

# Load multiple experiment results
samples = [
    df1["Tiempo VAM"].values,
    df2["Tiempo VAM"].values,
    df3["Tiempo VAM"].values,
]

# Run test
friedman = Friedman(samples=samples)
friedman.test()

print(f"Statistic: {friedman.statistic}")
print(f"p-value: {friedman.p_value}")

if friedman.p_value < 0.05:
    print("Significant difference among experiments")
```

---

## SimulationMetrics Class

### Full API Reference

```python
@dataclass
class SimulationMetrics:
    true_data: np.ndarray           # Shape: (n_patients, n_variables)
    simulation_data: np.ndarray     # Shape: (n_patients, n_runs, n_variables)
    
    # Results (populated after evaluate())
    coverage_percentage: Dict[str, float]
    error_margin: Dict[str, float]  # {"rmse": x, "mae": y, "mape": z}
    kolmogorov_smirnov_result: Dict[str, Any]
    anderson_darling_result: Dict[str, float]
    
    def evaluate(
        self,
        confidence_level: float = 0.95,
        random_state: int | None = None,
        result_as_dict: bool = False
    ) -> None:
        """
        Run all validation metrics.
        
        Args:
            confidence_level: For coverage calculation (0.80-0.95 recommended)
            random_state: Seed for reproducibility
            result_as_dict: Return results as dictionaries
        """
```

### Complete Validation Example

```python
import numpy as np
from uci.stats import SimulationMetrics
from utils.helpers import get_true_data_for_validation, simulate_all_true_data

# 1. Load and prepare data
df_true = get_true_data_for_validation()
sim_array = simulate_all_true_data(df_true, n_runs=200, seed=42)

# 2. Create metrics object
metrics = SimulationMetrics(
    true_data=df_true.to_numpy(),
    simulation_data=sim_array
)

# 3. Run evaluation
metrics.evaluate(
    confidence_level=0.95,
    random_state=42,
    result_as_dict=True
)

# 4. Print comprehensive report
print("=" * 50)
print("SIMULATION VALIDATION REPORT")
print("=" * 50)

# Coverage
print("\n📊 Coverage Percentage (95% CI):")
for var, cov in metrics.coverage_percentage.items():
    status = "✓" if cov >= 80 else "⚠"
    print(f"  {var}: {cov:.1f}% {status}")

# Error metrics
print("\n📏 Error Metrics:")
err = metrics.error_margin
print(f"  RMSE: {err['rmse']:.1f} hours")
print(f"  MAE:  {err['mae']:.1f} hours")
print(f"  MAPE: {err['mape']:.1f}%")

# KS test
print("\n📈 Distribution Comparison (KS Test):")
ks = metrics.kolmogorov_smirnov_result
for var, result in ks["per_variable"].items():
    status = "Similar" if result["p_value"] >= 0.05 else "Different"
    print(f"  {var}: p={result['p_value']:.3f} ({status})")

# Overall assessment
print("\n📋 Overall Assessment:")
avg_coverage = np.mean(list(metrics.coverage_percentage.values()))
if avg_coverage >= 80 and err['mape'] < 25:
    print("  ✓ Model validation: PASSED")
else:
    print("  ⚠ Model validation: NEEDS REVIEW")
```

---

## Confidence Interval Calculation

### StatsUtils.confidenceinterval

```python
class StatsUtils:
    @staticmethod
    def confidenceinterval(
        mean: np.ndarray,
        std: np.ndarray,
        n: int,
        coef: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence interval using normal approximation.
        
        Formula:
            CI = mean ± z × (std / √n)
        
        Where z = ppf(1 - α/2) for the given confidence coefficient.
        """
```

**Usage**:
```python
from uci.stats import StatsUtils

# Calculate CI for simulation means
means = np.array([100, 50, 30])  # hours
stds = np.array([20, 10, 8])
n_runs = 200

lower, upper = StatsUtils.confidenceinterval(means, stds, n_runs, coef=0.95)

for i, (l, u) in enumerate(zip(lower, upper)):
    print(f"Variable {i}: [{l:.1f}, {u:.1f}]")
```

---

## Data Shape Requirements

### Input Shapes

| Data | Shape | Description |
|------|-------|-------------|
| `true_data` | `(n_patients, n_variables)` | Real observed values |
| `simulation_data` | `(n_patients, n_runs, n_variables)` | Simulation outputs |

### Automatic Reshaping

The `SimulationMetrics` class handles various input shapes:

```python
# 1D true_data (single variable or single patient)
true_data = np.array([100, 120, 90, 110])  # Automatically broadcast

# 2D true_data (standard format)
true_data = np.array([
    [100, 50, 30, 120, 200],  # Patient 0
    [120, 55, 35, 140, 250],  # Patient 1
    ...
])

# 3D simulation_data (required)
simulation_data.shape == (n_patients, n_runs, n_variables)
```

---

## Best Practices

### 1. Sample Size Recommendations

| Purpose | Min Runs | Recommended |
|---------|----------|-------------|
| Quick validation | 50 | 100 |
| Standard validation | 100 | 200 |
| Publication quality | 200 | 500+ |

### 2. Confidence Level Selection

| Situation | Recommended Level |
|-----------|------------------|
| Exploratory analysis | 0.90 |
| Standard validation | 0.95 |
| High-stakes decisions | 0.99 |

### 3. Interpreting Combined Metrics

```
Scenario Analysis:

1. Coverage High + RMSE Low + KS p>0.05
   → Excellent model fit

2. Coverage High + RMSE High + KS p>0.05
   → Wide confidence intervals, consider reducing variance

3. Coverage Low + RMSE Low + KS p<0.05
   → Different distribution shapes, same mean

4. Coverage Low + RMSE High + KS p<0.05
   → Model needs revision
```

### 4. Handling Zero Values

MAPE automatically excludes zero values:
```python
# Positions where true values are zero are excluded
# to avoid division by zero

# If all values are zero, MAPE returns NaN
if np.all(true_values == 0):
    mape = float("nan")
```

---

## Visualization Integration

The statistics module integrates with `utils/visuals.py`:

```python
from utils.visuals import (
    plot_coverage,
    plot_error_margin,
    plot_ks,
    make_all_plots
)

# Generate all validation plots
figs = make_all_plots(metrics, df_true, sim_array)

# figs = {
#     "coverage": Figure,
#     "error": Figure,
#     "ks": Figure,
#     ...
# }
```
