---
name: documentation
description: Standards for documenting code, creating docstrings, and updating project documentation.
---

# Documentation Standards

## 1. Code Docstrings
- **Style**: Google Style Python Docstrings.
- **Requirement**: All public functions, classes, and modules must have a docstring.
- **Language**: Spanish for domain-specific descriptions (User facing), English for technical details (Developer facing).
    - *Example (Mixed)*:
      ```python
      def calcular_supervivencia(pacientes: pd.DataFrame) -> float:
          """
          Calcula la tasa de supervivencia de los pacientes ventilados.

          Args:
              pacientes (pd.DataFrame): DataFrame with patient data containing 'status' column.

          Returns:
              float: The survival rate between 0.0 and 1.0.
          """
      ```

## 2. Inline Comments
- **Focus**: Explain "Why", not "What".
- **Context**: Critical for statistical formulas or SimPy simulation logic constraints.
- **Language**: English preferred for internal logic.

## 3. Project Documentation
- **Location**: `docs/documentation/`
- **Updates**: When adding significant features, update the relevant markdown file:
    - `SIMULATION.md` for changes in `uci/simulacion.py`.
    - `STATISTICS.md` for new metrics in `uci/stats.py`.
    - `UI_GUIDE.md` for new Streamlit views.
- **CHANGELOG**: Add an entry to `docs/documentation/CHANGELOG.md` (or `CHANGELOG_ES.md`) for versioned releases.

## 4. Readability
- Use meaningful variable names (English) that describe the data content (e.g., `df_patients_filtered` instead of `df`).
- Type hints are part of self-documentation. Use them extensively.
