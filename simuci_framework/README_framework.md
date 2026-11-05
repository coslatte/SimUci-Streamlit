# SimUci — Framework de Evaluación de Incertidumbre y Robustez

**Autor:** Jorge Jesús Santos García  
**Tutora:** DrC. Deborah Raquel Galpert Cañizares  
**Centro:** UCLV — Departamento de Ciencias de la Computación  
**Versión:** 1.0.0 | **Python:** 3.10+

---

## Descripción

Marco de trabajo modular que extiende el modelo SimUci con dos componentes:

1. **Módulo de cuantificación de incertidumbre** — Bootstrapping paramétrico, Conformal prediction, Monte Carlo  
2. **Módulo de evaluación de robustez** — Stress-tests (ruido, faltantes, shift), índice R, semáforo clínico

Ambos módulos se integran en el dashboard Streamlit de SimUci sin modificar su núcleo.

---

## Estructura del proyecto

```
simuci_framework/
├── __init__.py                   # API de conveniencia de nivel superior
├── framework/
│   ├── __init__.py               # Exporta las clases principales
│   ├── uncertainty_module.py     # DistributionFitter, ConformalPredictor,
│   │                             # ParametricBootstrapper, MonteCarloQuantifier,
│   │                             # UncertaintyQuantifier (fachada Strategy)
│   ├── robustness_module.py      # RobustnessEvaluator, stress-tests,
│   │                             # índice R, semáforo clínico
│   ├── simuci_adapter.py         # Puente con run_experiment()/predict() reales
│   │                             # de utils.helpers (proyecto SimUci)
│   └── dashboard_integration.py  # Componentes Streamlit del dashboard
├── tests/
│   └── test_framework.py         # 80+ pruebas unitarias (pytest)
├── requirements_framework.txt
└── README_framework.md
```

Esta carpeta se copia completa a la raíz del proyecto SimUci-Streamlit
(junto a `app.py`, `utils/`, `models/`), sin modificar ningún archivo
existente (integración no invasiva, requisito RNF-03).

---

## Instalación

```bash
# 1. Desde la raíz del proyecto SimUci
pip install -r simuci_framework/requirements_framework.txt

# 2. Verificar instalación
python -c "from framework import UncertaintyQuantifier, RobustnessEvaluator; print('OK')"

# 3. Ejecutar la aplicación (integrada en SimUci)
streamlit run app.py
```

---

## Dependencias

```
numpy>=1.24.0
scipy>=1.11.0
pandas>=2.0.0
scikit-learn>=1.3.0
fitter>=1.7.0
crepes>=0.6.0
simpy>=4.1.0
streamlit>=1.28.0
plotly>=5.18.0
joblib>=1.3.0
kmodes>=0.12.0
```

---

## Uso programático

### Cuantificación de incertidumbre (Conformal Prediction)

```python
from simuci_framework import UncertaintyQuantifier, SimUciAdapter

# Configurar adaptador
adapter = SimUciAdapter()
patient = dict(
    n_runs=200, age=57, apache=18, vam_time=5,
    uti_stay=8, preuti_stay=2, percent=10,
    d1=1, d2=0, d3=0, d4=0,
    resp_insuf=1, artif_vent=0
)
adapter.configure(patient)

# Conformal prediction (método recomendado)
X_cal, y_cal, model = adapter.get_conformal_inputs()
uq = UncertaintyQuantifier(method='conformal', confidence=0.95, seed=42)
uq.fit(X_cal, y_cal, model)
interval = uq.predict_interval(adapter.patient_feature_vector())

print(f"Predicción: {interval.prediction:.1f} días")
print(f"IC 95%: [{interval.lower:.1f}, {interval.upper:.1f}] días")
print(f"Amplitud: {interval.width:.1f} días")
```

### Evaluación de robustez

```python
from simuci_framework import RobustnessEvaluator

cal_df, num_cols = adapter.get_calibration_data()
evaluator = RobustnessEvaluator(
    simulate_fn=adapter.simulate,
    calibration_data=cal_df,
    numeric_cols=num_cols,
    seed=42
)
report = evaluator.evaluate(patient, run_sensitivity=True)

print(f"Índice R: {report.R:.2f}")
print(f"Semáforo: {report.semaphore_label}")
print(f"Variable más influyente: {report.sensitivity_results[0].variable}")
```

### Bootstrapping paramétrico

```python
uq = UncertaintyQuantifier(method='bootstrap', confidence=0.95, B=1000, seed=42)
uq.fit(cal_df, num_cols)
interval = uq.predict_interval(adapter.simulate, patient)
print(f"IC bootstrap 95%: [{interval.lower:.1f}, {interval.upper:.1f}]")
```

### Monte Carlo

```python
uq = UncertaintyQuantifier(method='montecarlo', confidence=0.95, B=500, seed=42)
interval = uq.predict_interval(adapter.simulate, patient)
print(f"IC Monte Carlo 95%: [{interval.lower:.1f}, {interval.upper:.1f}]")
```

---

## Integración con el dashboard Streamlit

Desde `app.py` de SimUci (raíz del proyecto), añadir al final de la sección de resultados:

```python
from simuci_framework.framework.dashboard_integration import FrameworkDashboard

# Instanciar y renderizar el framework
fw = FrameworkDashboard()
fw.render()
```

Nota: `simuci_framework/` se copia completa a la raíz del proyecto SimUci
(junto a `app.py`, `utils/`, `models/`). Todos los imports internos del
framework son relativos, por lo que funciona correctamente tanto si se
importa como subpaquete (`simuci_framework.framework.xxx`, el caso real
de integración) como en pruebas unitarias aisladas (ver
`tests/test_framework.py`, que inserta `simuci_framework/` directamente
en `sys.path` para pruebas independientes del proyecto SimUci).

El dashboard agrega automáticamente al panel lateral los controles de
configuración del framework (nivel de confianza, método, réplicas B) y
presenta los resultados en cuatro secciones: métricas hero, incertidumbre,
robustez y dashboard ejecutivo.

---

## Ejecución de pruebas

```bash
# Suite completa
pytest tests/test_framework.py -v --tb=short

# Con cobertura de código
pytest tests/test_framework.py -v --cov=framework --cov-report=term-missing

# Pruebas específicas
pytest tests/test_framework.py::TestConformalPredictor -v
pytest tests/test_framework.py::TestRobustnessEvaluator -v
pytest tests/test_framework.py::TestEndToEnd -v
```

Resultado esperado: **80+ tests passed**, cobertura ~94%.

---

## Fórmulas clave

| Expresión | Descripción |
|-----------|-------------|
| `R = 0.4·r_ruido + 0.4·r_falt + 0.2·r_shift` | Índice de robustez ponderado |
| `r_i = 1 - min(1, Δŷ / umbral)` | Componente de robustez por stress-test |
| `q = Percentil((1-α)(1+1/n_cal)·100, scores)` | Umbral conformal |
| `IC = [ŷ - q, ŷ + q]` | Intervalo de predicción conformal |
| `umbral_clínico = 0.25` | Variación relativa máxima aceptable (25%) |

---

## Semáforo clínico

| Color | Condición | Interpretación |
|-------|-----------|----------------|
| 🟢 Verde | R ≥ 0.70 | Alta confianza — usar como referencia primaria |
| 🟡 Ámbar | 0.40 ≤ R < 0.70 | Confianza moderada — integrar con juicio clínico |
| 🔴 Rojo | R < 0.40 | Baja confianza — priorizar criterio médico experto |

---

## Referencias

- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.  
- Angelopoulos, A. N., & Bates, S. (2022). A gentle introduction to conformal prediction. arXiv:2107.07511.  
- Efron, B., & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. CRC Press.  
- Saltelli, A., et al. (2000). *Sensitivity Analysis*. Wiley.  
- Iman, R. L., & Conover, W. J. (1982). A distribution-free approach to inducing rank correlation. *Communications in Statistics*, 11(3), 311–334.
