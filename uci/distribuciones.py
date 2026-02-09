"""Distribution sampling and clustering for ICU simulation.

Provides probability-distribution samplers for each cluster and a
nearest-centroid classifier used to assign patients to clusters.

Each sampler returns a **plain Python float** so that downstream code
never needs to unpack numpy arrays.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.stats as stats

from utils.constants import DFCENTROIDES_CSV_PATH, EXPERIMENT_VARIABLES_FROM_CSV

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_N_FEATURES: int = len(EXPERIMENT_VARIABLES_FROM_CSV)
"""Number of features expected by the clustering function."""


def _sample_exponential(mean: float) -> float:
    """Draw a single value from an exponential distribution with given *mean*."""
    value: float = float(stats.expon.rvs(scale=mean))

    logger.debug("exponential sample: mean=%.3f draw=%.3f", mean, value)

    return value


def _sample_weibull(shape: float, scale: float) -> float:
    """Draw a single value from a Weibull-minimum distribution."""
    value: float = float(stats.weibull_min.rvs(shape, scale=scale))

    logger.debug("weibull sample: shape=%.3f scale=%.3f draw=%.3f", shape, scale, value)

    return value


# ---------------------------------------------------------------------------
# Cluster 0 distributions
# ---------------------------------------------------------------------------


def tiemp_VAM_cluster0() -> float:
    """VAM time for cluster 0 — Exponential"""

    return _sample_exponential(mean=113.508)


def tiemp_postUCI_cluster0() -> float:
    """Post-ICU time for cluster 0 — mixture of three uniform ranges.

    A discrete selector chooses one of three branches with probabilities
    (0.6, 0.3, 0.1), each mapping to a different Uniform interval.
    """

    branch_values = np.array([1, 2, 3])
    branch_probs = np.array([0.6, 0.3, 0.1])

    branch_dist = stats.rv_discrete(name="branch_selector", values=(branch_values, branch_probs))
    branch: int = int(branch_dist.rvs())

    # (loc, scale) pairs per branch — scale = upper - loc for scipy.uniform
    branch_params: dict[int, tuple[float, float]] = {
        1: (0.0, 168.0),
        2: (192.0, 384.0),
        3: (408.0, 648.0),
    }

    loc, scale = branch_params[branch]
    value: float = float(stats.uniform.rvs(loc=loc, scale=scale))
    logger.debug("tiemp_postUCI_cluster0: branch=%d loc=%.1f scale=%.1f draw=%.3f", branch, loc, scale, value)

    return value


def estad_UTI_cluster0() -> float:
    """UTI stay for cluster 0 — Weibull"""

    return _sample_weibull(shape=1.37958, scale=262.212)


# ---------------------------------------------------------------------------
# Cluster 1 distributions
# ---------------------------------------------------------------------------


def tiemp_VAM_cluster1() -> float:
    """VAM time for cluster 1 — Exponential"""

    return _sample_exponential(mean=200.0)


def tiemp_postUCI_cluster1() -> float:
    """Post-ICU time for cluster 1 — Weibull"""

    return _sample_weibull(shape=3.63023, scale=1214.29)


def estad_UTI_cluster1() -> float:
    """UTI stay for cluster 1 — Weibull"""

    return _sample_weibull(shape=1.57768, scale=472.866)


# ---------------------------------------------------------------------------
# Clustering (nearest centroid)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_centroids() -> np.ndarray:
    """Load and cache the centroid matrix from the CSV file.

    Only the first ``_N_FEATURES`` numeric columns are retained so the
    matrix aligns with the feature vector built by :func:`clustering`.
    """

    df = pd.read_csv(DFCENTROIDES_CSV_PATH)
    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()

    if len(numeric_cols) >= _N_FEATURES:
        centroids = df[numeric_cols[:_N_FEATURES]].to_numpy(dtype=float)
    else:
        centroids = df.iloc[:, :_N_FEATURES].to_numpy(dtype=float)

    logger.debug("Loaded centroids with shape %s", centroids.shape)

    return centroids


def clustering(
    edad: int,
    diag_ing1: int,
    diag_ing2: int,
    diag_ing3: int,
    diag_ing4: int,
    apache: int,
    insuf_resp: int,
    va: int,
    estadia_uti: int,
    tiempo_vam: int,
    est_pre_uci: int,
) -> int:
    """Assign a patient to a cluster via nearest-centroid classification.

    Builds a feature vector from the patient parameters — including a
    derived *va_group* feature — and returns the index of the closest
    centroid using Euclidean distance.
    """

    va_group: int = 2 if va in (2, 3) else 1

    features = np.array(
        [
            edad,
            diag_ing1,
            diag_ing2,
            diag_ing3,
            diag_ing4,
            apache,
            insuf_resp,
            va,
            va_group,
            estadia_uti,
            tiempo_vam,
            est_pre_uci,
        ],
        dtype=float,
    ).reshape(1, -1)

    centroids = _load_centroids()

    # Defensive alignment: trim to the smaller dimension if mismatch
    n_cols = min(centroids.shape[1], features.shape[1])
    distances = np.linalg.norm(centroids[:, :n_cols] - features[:, :n_cols], axis=1)
    cluster: int = int(np.argmin(distances))

    logger.debug(
        "clustering: distances=%s  chosen=%d",
        distances.tolist(),
        cluster,
    )

    return cluster
