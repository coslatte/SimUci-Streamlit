---
name: streamlit-development
description: Best practices for building, structuring, and optimizing Streamlit applications in SimUCI.
---

# Streamlit Development Skills

## 1. Architecture Pattern
- **Logic Separation**: Keep `app.py` for View/Controller logic only.
  - Move complex calculations to `uci/`.
  - Move helper UI functions (plots, formatting) to `utils/visuals/`.
- **Monolith**: `app.py` is the single entry point. Do not create multi-page apps unless architecture changes.

## 2. Performance & Caching
- **Data**: Use `@st.cache_data` for loading CSVs and heavy dataframes.
  ```python
  @st.cache_data
  def load_data(path: Path) -> pd.DataFrame:
      return pd.read_csv(path)
  ```
- **Resources**: Use `@st.cache_resource` for loading ML models or DB connections.

## 3. Session State Management
- **Persistence**: Use `st.session_state` to store simulation results between re-runs.
- **Initialization**: Always check if key exists before access.
  ```python
  if 'simulation_results' not in st.session_state:
      st.session_state.simulation_results = None
  ```

## 4. User Experience (UI/UX)
- **Language**: All visible text (titles, buttons, warnings) must be in **Spanish**.
- **Variables**: Load text from `utils/constants/messages.py`.
- **Feedback**:
  - Use `st.spinner()` for running simulations.
  - Use `st.progress()` for batch experiments.
  - Use `st.error/warning/info` for status updates.

## 5. Visualizations
- **Library**: `plotly` (interactive) or `matplotlib/seaborn` (static).
- **Theme**: Respect user theme preference (Light/Dark) where possible, or stick to project defaults defined in `utils/constants/theme.py`.
