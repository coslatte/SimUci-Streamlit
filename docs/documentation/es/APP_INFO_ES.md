# Información de la Aplicación SimUci

## Acerca de SimUci

**SimUci** es una aplicación de simulación discreta para modelar el flujo de pacientes en unidades de cuidados intensivos (UCI). Utiliza técnicas de simulación basadas en eventos discretos para analizar y optimizar procesos hospitalarios críticos.

## Características Principales

### Simulación de Pacientes

- Modelado del flujo completo de pacientes en UCI
- Desde el ingreso hasta el alta o fallecimiento
- Consideración de múltiples variables clínicas y operativas

### Validación Estadística

- Comparación de resultados simulados vs. datos reales
- Métricas de error (RMSE, MAE, MAPE)
- Pruebas de Kolmogorov-Smirnov
- Análisis de cobertura de intervalos de confianza

### Interfaz Interactiva

- Configuración de parámetros de simulación
- Visualización de resultados en tiempo real
- Generación de reportes y gráficos

## Tecnologías Utilizadas

- **Backend**: Python con SimPy (simulación discreta)
- **Frontend**: Streamlit
- **Análisis Estadístico**: SciPy, NumPy, scikit-learn
- **Visualización**: Matplotlib, Plotly

## Arquitectura del Sistema

La aplicación está organizada en módulos especializados:

- `uci/`: Núcleo de simulación (distribuciones, experimentos, estadísticas)
- `utils/`: Utilidades y componentes de interfaz
- `data/`: Conjuntos de datos y modelos entrenados
- `docs/`: Documentación completa del sistema

## Uso Clínico

Esta herramienta está diseñada para ayudar a administradores hospitalarios y clínicos a:

- Optimizar la asignación de recursos en UCI
- Evaluar escenarios de carga de pacientes
- Mejorar protocolos de atención
- Planificar expansiones de capacidad

## Limitaciones

- Los resultados son simulaciones y no predicciones definitivas
- Requiere validación con datos específicos de cada institución
- No reemplaza el juicio clínico profesional
