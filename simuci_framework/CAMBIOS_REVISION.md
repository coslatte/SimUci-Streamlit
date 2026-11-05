# Cambios del código en la revisión de la tesis
- adapter: nueva partición entrenamiento/calibración/prueba (`test_split`, `get_test_data`), semillas deterministas para la salida simulada.
- robustness_module: el stress-test de faltantes solo imputa variables clínicas (antes imputaba n_runs, percent y _crn_seed); pesos de R documentados como propuestos.
- uncertainty_module: ParametricBootstrapper documentado como bootstrap por perturbación de entradas; se guarda `last_predictions_`; Monte Carlo no perturba percent ni _crn_seed.
- dashboard_integration: histograma y comparación de métodos con predicciones reales (antes muestras sintéticas y anchuras fijas); recomendaciones sin cifras no respaldadas.
- tests: 101 pruebas (evidencia en tests/evidencia_pytest.txt).
- experimentos/: ejecutar_experimentos.py (una ejecución reproducible, semilla 42) y generar_figuras.py; resultados/ con JSON, CSV y figuras.
