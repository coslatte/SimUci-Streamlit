# SimUCI

**Simulación de la evolución pos-egreso de pacientes ventilados en Cuidados Intensivos**

Esta herramienta es un sistema de apoyo a la decisión clínica (CDSS) que combina simulación de eventos discretos y modelos de aprendizaje automático predictivo. Su objetivo es modelar la trayectoria clínica de los pacientes para estimar tiempos de estancia, uso de recursos y probabilidades de supervivencia.

## Arquitectura del Sistema

La aplicación integra dos motores principales de análisis:

1.  **Motor de Simulación Estocástica**: Utiliza distribuciones de probabilidad derivadas de datos históricos para generar escenarios posibles para nuevos pacientes.
2.  **Motor de Predicción (Machine Learning)**: Estima la probabilidad de fallecimiento utilizando modelos entrenados sobre variables fisiológicas y demográficas.

---

## Módulos Funcionales

### 1. Simulación de Pacientes Individuales
El módulo de simulación permite configurar un "paciente virtual" con parámetros clínicos específicos:
*   **Datos Demográficos**: Edad.
*   **Variables Fisiológicas**: Puntaje APACHE II, Insuficiencia Respiratoria.
*   **Diagnósticos**: Códigos diagnósticos de ingreso y egreso.
*   **Manejo Clínico**: Tiempo de Ventilación Artificial (VA), Estancia esperada en UCI.

El sistema asigna al paciente a un "clúster" estadístico (basado en K-Means) y muestrea eventos de las distribuciones correspondientes para generar múltiples corridas (escenarios). Esto permite calcular intervalos de confianza para las variables de tiempo.

### 2. Validación con Datos Históricos
Este módulo compara las salidas de la simulación contra un conjunto de datos real ("Ground Truth") para evaluar la calibración del modelo.
Se calculan métricas estadísticas rigurosas:
*   **Cobertura**: Porcentaje de datos reales que caen dentro del intervalo de confianza simulado (>90% indica excelente calibración).
*   **Error (RMSE, MAE)**: Cuantificación de la desviación en horas entre los valores simulados y reales.
*   **Prueba Kolmogorov-Smirnov (KS)**: Evalúa si las distribuciones simuladas y reales son estadísticamente indistinguibles.

### 3. Predicción de Riesgo
Utiliza un modelo clasificador (almacenado como `prediction_model.joblib`) para determinar la clase de riesgo del paciente (Fallece / No Fallece) y la probabilidad asociada. Las variables predictoras clave incluyen combinaciones diagnósticas, edad y tiempo de ventilación mecánica.

### 4. Análisis Comparativo Estadístico
Herramientas para comparar resultados entre diferentes escenarios de simulación o grupos de pacientes:
*   **Prueba de Wilcoxon**: Para comparar dos muestras relacionadas (ej. mismo paciente con diferentes parámetros de simulación).
*   **Prueba de Friedman**: Para detectar diferencias en múltiples tratamientos o configuraciones de simulación simultáneamente.

---

## Interpretación de Métricas

*   **RMSE (Raíz del Error Cuadrático Medio)**: Penaliza los errores grandes. Un valor bajo indica alta precisión.
*   **MAE (Error Absoluto Medio)**: Promedio de los errores absolutos. Interpretación directa en horas.
*   **Valor P (P-Value)**: En las pruebas comparativas, un valor p < 0.05 generalmente indica diferencias estadísticamente significativas entre los grupos comparados.

---
*Desarrollado con fines de investigación académica en ingeniería de sistemas de salud.*
