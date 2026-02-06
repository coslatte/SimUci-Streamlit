# SimUCI - Simulation Module

## Overview

The simulation module (`uci/`) is the core engine of the SimUCI application. It implements discrete-event simulation using SimPy and provides statistical validation tools.

## Architecture

```text
uci/
├── __init__.py            # Package marker
├── distribuciones.py      # Probability distributions & clustering
├── experiment.py          # Experiment data holder & runner functions
├── simulacion.py          # SimPy discrete-event process
├── procesar_datos.py      # CSV data loading & column generators
└── stats.py               # Statistical tests & validation metrics
```

### Module Dependency Graph

```text
experiment.py
  ├── distribuciones.py    (clustering)
  └── simulacion.py        (SimPy process)
        └── distribuciones.py  (distribution sampling)

stats.py                   (standalone — used by app.py & helpers)
procesar_datos.py           (standalone — used by helpers)
```

## Simulation Pipeline

1. **Patient Configuration**: Define diagnosis, age, APACHE score, etc. via `Experiment`.
2. **Cluster Assignment**: `distribuciones.clustering()` assigns the patient to a cluster using nearest-centroid classification (Euclidean distance). Centroids are loaded once and cached.
3. **Distribution Sampling**: `simulacion.Simulation.uci()` samples event times from cluster-specific distributions. All samplers return plain `float` values.
4. **Result Collection**: `experiment.multiple_replication()` runs N independent replications and compiles results into a `DataFrame`.

## Components

### `distribuciones.py` — Distributions & Clustering

Provides probability-distribution samplers for each cluster and a nearest-centroid classifier.

**Key design decisions:**

- All samplers return **plain Python `float`** — no numpy arrays, no downstream unpacking needed.
- Shared helpers `_sample_exponential(mean)` and `_sample_weibull(shape, scale)` eliminate code duplication.
- Centroid matrix loaded once via `@lru_cache` (`_load_centroids()`).
- Debug output uses `logging.getLogger(__name__)` instead of `print()`.

**Distributions per cluster:**

| Variable | Cluster 0 | Cluster 1 |
|----------|-----------|----------|
| VAM time | Exponential(mean=113.508) | Exponential(mean=200) |
| Post-ICU time | Mixture of 3 Uniforms | Weibull(3.63, 1214.29) |
| UTI stay | Weibull(1.38, 262.21) | Weibull(1.58, 472.87) |

### `experiment.py` — Experiment & Runners

- **`Experiment`** class: Holds patient parameters and a mutable `result` dict.
- **`single_run(experiment)`**: One simulation replication (cluster → SimPy process → result dict).
- **`multiple_replication(experiment, n_reps, as_int)`**: N replications → `DataFrame`.

### `simulacion.py` — SimPy Process

- **`Simulation.uci(env)`**: Generator that yields SimPy timeouts for pre-VAM → VAM → post-VAM → post-ICU stages.
- VAM is capped to not exceed UCI stay (retry loop with 1000-attempt limit).

### `procesar_datos.py` — Data Loading

- **`cargar_fichero(path, column)`**: Core loader — reads CSV, parses dates, returns one column sorted by admission date.
- Column generators (`get_fecha_ingreso`, `get_tiempo_vam`, etc.) use `yield from _iter_column()` to avoid code duplication.
- **`get_time_simulation(path)`**: Computes simulation horizon in hours.

### `stats.py` — Statistical Validation

See [STATISTICS.md](STATISTICS.md) for the full API reference.

## Logging

All modules use `logging.getLogger(__name__)`. To enable debug output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Clusters

Patients are grouped using nearest-centroid classification to assign appropriate statistical distributions for simulation variables.
