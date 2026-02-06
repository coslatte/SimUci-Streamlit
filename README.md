# SimUCI

Aplicación para la simulación de la evolución pos-egreso de pacientes ventilados en Cuidados Intensivos

## Introducción

SimUCI utiliza simulación de eventos discretos y modelos de aprendizaje automático para apoyar la toma de decisiones médicas. Permite predecir estadísticas sobre la estancia en UCI y la supervivencia de pacientes.

## Características Principales

* **Simulación de Pacientes**: Modelado de trayectorias en UCI.
* **Predicción y Validación**: Estimaciones de supervivencia y validación estadística.
* **Análisis Comparativo**: Comparación de datos simulados con reales.

## Inicio Rápido

### Requisitos

* Python 3.9 o superior

### Instalación

1. Instalar dependencias:

    ```powershell
    python -m pip install -r requirements.txt
    ```

### Ejecutar la Aplicación

```powershell
streamlit run app.py
```

### Ejecutar Pruebas

```powershell
python -m pytest
```

## Despliegue

La aplicación está disponible en línea:
[SimUCI en Streamlit Community Cloud](https://simuci.streamlit.app/)

> **Nota**: Desde Cuba el acceso requiere VPN.

## Documentación Detallada

Consulta la carpeta `docs/documentation` para más información:

* [Índice de Documentación](docs/documentation/INDEX.md)
