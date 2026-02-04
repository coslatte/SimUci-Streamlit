# Registro de Cambios

Todos los cambios notables en el proyecto **SimUCI** serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/),
y este proyecto se adhiere al [Versionado Semántico](https://semver.org/lang/es/).

---

## [Sin publicar]

### Cambiado
- **Refactorización de Utils**: Se reorganizó el directorio `utils` en sub-paquetes modulares (`helpers`, `validation_ui`, `visuals`) para mejorar la estructura y mantenibilidad del proyecto.

### Planificado
- Integración de modelos adicionales de machine learning para predicción de mortalidad
- Funcionalidad de exportación para informes de validación (PDF/HTML)
- Soporte multilenguaje para la interfaz de usuario
- Herramientas avanzadas de visualización de clustering

---

## [0.2.0-beta] - 2025-09-29

### Añadido
- **Mejoras en la UI de Validación**: Nuevas secciones desplegables en `utils/validation_ui.py` que explican las métricas de validación (RMSE, MAE, MAPE) con definiciones detalladas, unidades y guías de interpretación
- **Desplegable de Comparación de Distribuciones**: Nuevo componente `st.expander` que explica cómo interpretar superposiciones de distribuciones, picos desplazados y colas largas
- **Documentación de Pruebas Estadísticas**: Explicaciones dentro de la interfaz para las pruebas de Kolmogorov-Smirnov (KS) y Anderson-Darling (AD)
- **Visualización de Métricas de Cobertura**: Gráficos de barras mostrando la cobertura del intervalo de confianza por variable con capacidad de descarga
- **Caché del Estado de Sesión**: Los resultados de validación ahora se almacenan en caché con marcas de tiempo para evitar cálculos redundantes

### Cambiado
- **Restauración de Sintaxis en `app.py`**: Corrección de bloques de código corruptos después de operaciones de envolvimiento:
  - Restaurada la llamada `st.toggle(...)` con argumentos correctos (`label`, `help`, `key`)
  - Re-indentado y reorganizado el bloque `try/except` para el flujo de predicción
  - Eliminado literal f-string errante de la llamada `st.dataframe()` en la pestaña de Comparaciones
- **Visualización de Métricas de Error**: RMSE y MAE ahora se muestran en horas; MAPE muestra porcentaje con manejo adecuado de valores cero
- **Formato de Tiempo Mejorado**: La función `format_time_columns()` ahora excluye correctamente filas especificadas y maneja casos extremos

### Corregido
- **Errores de Sintaxis en `app.py`**: Resueltos errores de compilación causados por paréntesis mal colocados y orden de argumentos de palabra clave
- **Indentación del Bloque de Predicción**: Corregidos problemas de indentación en el flujo de simulación a predicción
- **Consistencia de Claves de Estado de Sesión**: Asegurada la consistencia de todas las claves de `st.session_state` entre pestañas

### Seguridad
- Sin cambios relacionados con seguridad en esta versión

---

## [0.1.0] - 2025-08-28

### Añadido
- **Estructura Inicial del Proyecto**: Andamiaje completo de la aplicación con arquitectura modular
- **Motor de Simulación**: Funcionalidad central de simulación usando SimPy para simulación de eventos discretos
  - Simulación de paciente individual (`simulate_true_data`)
  - Simulación por lotes para todos los pacientes (`simulate_all_true_data`)
  - Número configurable de replicaciones (`multiple_replication`)
- **Módulo de Clustering** (`uci/distribuciones.py`):
  - Clustering K-means para clasificación de pacientes
  - Distribuciones de probabilidad específicas por cluster (Weibull, Exponencial, Uniforme)
  - Asignación de pacientes basada en centroides
- **Módulo de Análisis Estadístico** (`uci/stats.py`):
  - Clase `SimulationMetrics` para validación integral del modelo
  - Cálculo de porcentaje de cobertura con intervalos de confianza
  - Métricas de margen de error (RMSE, MAE, MAPE)
  - Prueba de Kolmogorov-Smirnov para comparación de distribuciones
  - Implementación de prueba de Anderson-Darling
  - Prueba de rangos con signo de Wilcoxon para comparaciones pareadas
  - Prueba de Friedman para comparaciones de múltiples muestras
- **Integración del Modelo de Predicción**:
  - Modelo de machine learning para predicción de mortalidad
  - Visualización de probabilidad y clase predicha
  - Seguimiento de delta para cambios en predicciones
- **Interfaz de Usuario Streamlit** (`app.py`):
  - Panel de configuración de paciente con todos los parámetros clínicos
  - Pestaña de simulación con parámetros de ejecución personalizables
  - Pestaña de Validación con Datos Reales con selección de paciente
  - Pestaña de Comparaciones para pruebas de Wilcoxon y Friedman
  - Alternador de tema (modo claro/oscuro)
  - Configuración de semilla global para reproducibilidad
- **Utilidades de Procesamiento de Datos** (`utils/helpers.py`):
  - `build_df_for_stats()` para DataFrames de resumen estadístico
  - `format_time_columns()` para visualización de tiempo legible
  - `generate_id()` para identificadores únicos de paciente
  - Funciones de extracción y transformación de datos CSV
- **Módulo de Visualización** (`utils/visuals.py`):
  - Gráficos de barras de cobertura
  - Gráficos de margen de error
  - Visualización de resultados de prueba KS
  - Gráficos de comparación de distribuciones (Plotly y Matplotlib)
- **Organización de Constantes** (`utils/constants/`):
  - Estructura modular de constantes con archivos separados para:
    - Límites (edad, APACHE, duraciones de estancia)
    - Mensajes y etiquetas
    - Mapeos de categorías (diagnósticos, tipos de ventilación)
    - Rutas de archivos
    - Colores del tema

### Cambiado
- **`build_df_for_stats`**: Ahora soporta métricas de calibración (conteo/porcentaje de iteraciones dentro del intervalo de confianza) y comparación con referencia externa
- **`format_df_stats`**: Acepta estructuras de etiquetas flexibles (`dict`, `list`, o `None`) con valores por defecto sensatos

### Corregido
- Versión inicial - sin correcciones de versiones anteriores

### Dependencias
- Python >= 3.9 (recomendado 3.13+)
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

## Resumen del Historial de Versiones

| Versión | Fecha | Descripción |
|---------|-------|-------------|
| 0.2.0-beta | 2025-09-29 | Mejoras en UI de validación, correcciones de sintaxis |
| 0.1.0 | 2025-08-28 | Versión inicial con funcionalidad central |

---

## Contribuir

Al contribuir a este proyecto, por favor:

1. Actualiza este CHANGELOG_ES.md con tus cambios bajo la sección `[Sin publicar]`
2. Sigue las categorías: Añadido, Cambiado, Obsoleto, Eliminado, Corregido, Seguridad
3. Incluye referencias a issues o pull requests relacionados cuando sea aplicable
4. Escribe descripciones claras y concisas de los cambios

---

## Enlaces

- **Repositorio**: [SimUCI en GitHub](https://github.com/coslatte/SimUci)
- **Despliegue**: [SimUCI en Streamlit Community Cloud](https://simuci-v0.streamlit.app/)
