# Changelog

All notable changes to the **SimUCI** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **Documentation**: Created `APP_INFO_ES.md` and `APP_INFO_EN.md` for in-app information tabs.

### Changed

- **Validation UI Modularization**: Refactored `render_validation` in `utils/validation_ui/validation_ui.py` into distinct sub-renderers for better maintainability (Separation of Concerns). Removed `print` debugging, added explicit `__all__`, type hints, and fixed LaTeX syntax for formulas.
- **Utils Refactoring**: Reorganized `utils` directory into modular sub-packages (`helpers`, `validation_ui`, `visuals`) for better project structure and maintainability.
- **`uci/` Package Refactoring** (end-to-end): Complete overhaul of all modules in the `uci/` package for correctness, consistency, and maintainability:
  - **`distribuciones.py`**: Fixed typo `tiemp_postUCI_clustet1` → `tiemp_postUCI_cluster1`. All samplers now return plain `float` instead of mixed `ndarray`/`float`. Replaced `print()` debug with `logging`. Centroids CSV cached via `@lru_cache`. Extracted `_sample_exponential` / `_sample_weibull` helpers. Fixed potential `None` return in `tiemp_postUCI_cluster0`. Added full type annotations and docstrings.
  - **`simulacion.py`**: Removed now-unnecessary `_to_scalar()` helper (14 lines). Removed unused `numpy` import. Simplified `cluster` type from `np.intp` to `int`. Added module/class/method docstrings.
  - **`experiment.py`**: Added type annotations to all attributes and parameters. Removed dead comments. Added module/class/function docstrings. Simplified `multiple_replication` dtype casting logic.
  - **`procesar_datos.py`**: Eliminated 9 copy-paste generator functions via shared `_iter_column` helper using `yield from`. Fixed typos in docstrings ("Genrador", "Genreador", variable `daignostico`). Replaced `print()` debug in `get_time_simulation` with `logging`. Fixed `get_diagnostico_list` missing `index_col=0`. Changed `sorted(list)[-1]` to `max()`. Added `Path` type support.
  - **`stats.py`**: Replaced ~15 `print()` calls with `logging` (warnings, results, errors). Replaced `traceback.print_exc()` with `logger.exception()`. Moved 3 in-method `import` statements to module top-level. Changed double-underscore name-mangling (`__method`) to single underscore (`_method`). Extracted `_coerce_true_data`, `_align_shape`, `_variable_label` helpers. Removed dead comments. Added full docstrings. Fixed Pylance type warnings by using named result attributes (`.statistic`, `.pvalue`).

### Fixed

- **Path Resolution**: Updated `utils/constants/paths.py` to use absolute paths anchored to `_PROJECT_ROOT`, fixing `FileNotFoundError` when loading documentation files in Streamlit.
- **UI Layout Bug**: Corrected `st.columns` unpacking logic in `app.py` to resolve `StreamlitInvalidColumnSpecError`.

### Planned

- Integration of additional machine learning models for mortality prediction
- Export functionality for validation reports (PDF/HTML)
- Multi-language support for the user interface
- Advanced clustering visualization tools

---

## [0.2.0-beta] - 2025-09-29

### Added

- **Validation UI Enhancements**: New expandable sections in `utils/validation_ui.py` explaining validation metrics (RMSE, MAE, MAPE) with detailed definitions, units, and interpretation guidelines
- **Distribution Comparison Expander**: New `st.expander` component explaining how to interpret distribution overlays, shifted peaks, and long tails
- **Statistical Test Documentation**: In-UI explanations for Kolmogorov-Smirnov (KS) and Anderson-Darling (AD) tests
- **Coverage Metrics Visualization**: Bar charts showing confidence interval coverage per variable with download capability
- **Session State Caching**: Validation results are now cached with timestamps to avoid redundant computations

### Changed

- **`app.py` Syntax Restoration**: Fixed corrupted code blocks after wrapping operations:
  - Restored `st.toggle(...)` call with correct arguments (`label`, `help`, `key`)
  - Re-indented and reorganized `try/except` block for prediction flow
  - Removed errant f-string literal from `st.dataframe()` call in Comparisons tab
- **Error Metrics Display**: RMSE and MAE now displayed in hours; MAPE shows percentage with proper handling of zero values
- **Improved Time Formatting**: `format_time_columns()` function now properly excludes specified rows and handles edge cases

### Fixed

- **Syntax Errors in `app.py`**: Resolved compilation errors caused by misplaced parentheses and keyword argument ordering
- **Prediction Block Indentation**: Fixed indentation issues in the simulation-to-prediction flow
- **Session State Key Consistency**: Ensured all `st.session_state` keys remain consistent across tabs

### Security

- No security-related changes in this release

---

## [0.1.0] - 2025-08-28

### Added

- **Initial Project Structure**: Complete application scaffolding with modular architecture
- **Simulation Engine**: Core simulation functionality using SimPy for discrete-event simulation
  - Single patient simulation (`simulate_true_data`)
  - Batch simulation for all patients (`simulate_all_true_data`)
  - Configurable number of replications (`multiple_replication`)
- **Clustering Module** (`uci/distribuciones.py`):
  - K-means clustering for patient classification
  - Cluster-specific probability distributions (Weibull, Exponential, Uniform)
  - Centroid-based patient assignment
- **Statistical Analysis Module** (`uci/stats.py`):
  - `SimulationMetrics` class for comprehensive model validation
  - Coverage percentage calculation with confidence intervals
  - Error margin metrics (RMSE, MAE, MAPE)
  - Kolmogorov-Smirnov test for distribution comparison
  - Anderson-Darling test implementation
  - Wilcoxon signed-rank test for paired comparisons
  - Friedman test for multiple sample comparisons
- **Prediction Model Integration**:
  - Machine learning model for mortality prediction
  - Probability and class prediction display
  - Delta tracking for prediction changes
- **Streamlit User Interface** (`app.py`):
  - Patient configuration panel with all clinical parameters
  - Simulation tab with customizable run parameters
  - Real Data Validation tab with patient selection
  - Comparisons tab for Wilcoxon and Friedman tests
  - Theme toggle (light/dark mode)
  - Global seed configuration for reproducibility
- **Data Processing Utilities** (`utils/helpers.py`):
  - `build_df_for_stats()` for statistical summary DataFrames
  - `format_time_columns()` for human-readable time display
  - `generate_id()` for unique patient identifiers
  - CSV data extraction and transformation functions
- **Visualization Module** (`utils/visuals.py`):
  - Coverage bar charts
  - Error margin plots
  - KS test result visualization
  - Distribution comparison charts (Plotly and Matplotlib)
- **Constants Organization** (`utils/constants/`):
  - Modular constants structure with separate files for:
    - Limits (age, APACHE, stay durations)
    - Messages and labels
    - Category mappings (diagnoses, ventilation types)
    - File paths
    - Theme colors

### Changed

- **`build_df_for_stats`**: Now supports calibration metrics (count/percentage of iterations within confidence interval) and external reference comparison
- **`format_df_stats`**: Accepts flexible label structures (`dict`, `list`, or `None`) with sensible defaults

### Fixed

- Initial release - no fixes from previous versions

### Dependencies

- Python >= 3.9 (recommended 3.13+)
- Streamlit >= 1.53.1
- SimPy >= 4.1.1
- SciPy >= 1.17.0
- scikit-learn >= 1.8.0
- Pandas >= 2.3.3
- NumPy >= 2.4.1
- Matplotlib >= 3.10.8
- Plotly >= 6.5.2
- Seaborn >= 0.13.2

---

## Version History Summary

| Version | Date | Description |
|---------|------|-------------|
| Unreleased | 2026-02-06 | `uci/` package end-to-end refactoring |
| 0.2.0-beta | 2025-09-29 | Validation UI improvements, syntax fixes |
| 0.1.0 | 2025-08-28 | Initial release with core functionality |

---

## Contributing

When contributing to this project, please:

1. Update this CHANGELOG.md with your changes under the `[Unreleased]` section
2. Follow the categories: Added, Changed, Deprecated, Removed, Fixed, Security
3. Include references to related issues or pull requests when applicable
4. Write clear, concise descriptions of changes

---

## Links

- **Repository**: [SimUCI on GitHub](https://github.com/coslatte/SimUci)
- **Deployment**: [SimUCI on Streamlit Community Cloud](https://simuci-v0.streamlit.app/)
