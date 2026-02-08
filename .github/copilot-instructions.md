# SimUCI AI Coding Instructions

You are working on **SimUCI**, a Python simulation application for ICU patient data, built with Streamlit and SimPy.

## Architecture & Boundaries
- **Frontend (UI)**: `app.py` is the monolithic entry point for the Streamlit interface. It handles layout, user input, and result visualization.
- **Core Logic (`uci/`)**:
  - `simulacion.py`: Contains the `Simulation` class using `simpy` for discrete-event modeling.
  - `experiment.py`: Manages simulation runs/experiments.
  - `distribuciones.py`: Statistical distributions source.
  - `stats.py`: Statistical analysis helpers (Friedman, Wilcoxon).
- **Configuration**: Avoid magic numbers/strings in code. Use `utils/constants/` (e.g., `limits.py`, `messages.py`) for all UI labels, limits, and defaults.

## Developer Workflow
- **Run App**: `streamlit run app.py` (Main entry point).
- **Tests**: `python -m pytest` (Run test suite).
- **Environment**: Python 3.9+ with dependencies in `requirements.txt`.

## Code style & Conventions
- **Language Mixed Context**:
  - **Code**: Use English for valid variable names, functions, and internal comments (e.g., `def calculate_mean()`).
  - **UI/Domain**: Use Spanish for all user-facing text, labels, and docstrings describing medical functionality (e.g., `st.title("Simulación de Pacientes")`).
- **Type Hinting**: Use modern Python type hints (`typing.TYPE_CHECKING`, annotations) especially in `uci/` modules.
- **Imports**: Group imports: Standard Lib -> 3rd Party (Pandas, Streamlit) -> Local (`uci`, `utils`).
- **Data Handling**: Prefer `pandas.DataFrame` for data manipulation. Data samples reside in `data/`.

## Critical Patterns
- **Simulation**: Logic follows `simpy.Environment` patterns. Events are modeled in `uci.simulacion.Simulation`.
- **Constants**: When adding new UI elements, define text in `utils/constants/messages.py` and numeric bounds in `utils/constants/limits.py` first, then import them in `app.py`.
- **Paths**: Use `os.path` or `pathlib` relative to project root. See `utils/helpers/helpers.py`.

## Data Science & Analysis Skills
- **Pandas**:
  - Prefer **vectorization** over loops for performance when processing patient data.
  - Use `pathlib` for robust file path handling when reading CSVs (e.g., `pd.read_csv(Path("data") / "file.csv")`).
  - Explicitly handle or explicitly ignore missing values (`NaN`); never assume data is clean.
  - Use clear variable names for DataFrames (e.g., `df_patients` vs `df`).
- **Statistics & ML**:
  - Use `scikit-learn` for ML pipelines and `scipy.stats` for statistical tests.
  - Ensure random seeds are fixed (where appropriate) for reproducibility in simulations.

## Streamlit Best Practices
- **Performance**: Use `@st.cache_data` for data loading and `@st.cache_resource` for loading heavy models.
- **Structure**: Keep `app.py` clean. Move complex calculation or simulation logic to `uci/` or `utils/`.
- **State**: Use `st.session_state` to manage simulation state across reruns/interactions.
- **Feedback**: Provide user feedback (spinners, progress bars) during long-running simulations.

## Documentation & Quality
- **Docstrings**: All new functions and classes must have docstrings (Google style preferred) explaining arguments and return values.
- **Comments**: Explain "Why", not just "What", especially for complex statistical formulas or simulation constraints.
- **Refactoring**: When modifying `app.py`, identify opportunities to extract logic into reusable functions in `utils/`.
