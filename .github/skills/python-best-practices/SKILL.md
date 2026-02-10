---
name: python-best-practices
description: Core Python development standards, including type hinting, path handling, and constants usage.
---

# Python Best Practices

## 1. Type Hinting & Typing
- **Mandatory**: Use type hints for all function arguments and return values.
- **Modern Syntax**: Use Python 3.9+ syntax (e.g., `list[str]` instead of `List[str]`).
- **Imports**: Use `typing.TYPE_CHECKING` to avoid circular imports.
  ```python
  if TYPE_CHECKING:
      from simuci.experiment import Experiment
  ```

## 2. Constants & Configuration
- **No Magic Numbers**: Never hardcode limits, thresholds, or UI strings.
- **Location**:
    - Numeric limits -> `utils/constants/limits.py`
    - UI Strings/Messages -> `utils/constants/messages.py`
    - File Paths -> `utils/constants/paths.py`

## 3. Path Handling
- **Library**: Always use `pathlib.Path`.
- **Root**: relative to project root.
- **Cross-platform**: Do not use string concatenation for paths (e.g., use `Path("data") / "file.csv"`).

## 4. Testing
- **Framework**: `pytest`.
- **Run**: `python -m pytest` from root.
- **Coverage**: Ensure simulation logic in `simuci` is covered by unit tests.

## 5. Error Handling
- Use specific exceptions (e.g., `FileNotFoundError`, `ValueError`) rather than bare `Exception`.
- For Streamlit UI errors, use `st.error()` to inform the user instead of crashing the app.
