"""Discrete-event simulation process for a single ICU patient."""

from __future__ import annotations

from typing import TYPE_CHECKING

import simpy

from uci import distribuciones

if TYPE_CHECKING:
    from uci.experiment import Experiment


class Simulation:
    """SimPy process that models an ICU patient's journey through VAM stages."""

    def __init__(self, experiment: "Experiment", cluster: int) -> None:
        self.experiment = experiment
        self.cluster = cluster

    def uci(self, env: simpy.Environment):
        """Run the patient through pre-VAM → VAM → post-VAM → post-ICU stages."""

        is_cluster_zero: bool = self.cluster == 0

        post_uci = int(
            distribuciones.tiemp_postUCI_cluster0() if is_cluster_zero else distribuciones.tiemp_postUCI_cluster1()
        )
        uci = int(
            distribuciones.estad_UTI_cluster0() if is_cluster_zero else distribuciones.estad_UTI_cluster1()
        )

        # Ensure VAM ≤ UCI stay; retry up to 1000 draws then clamp
        for _ in range(1000):
            vam = int(
                distribuciones.tiemp_VAM_cluster0() if is_cluster_zero else distribuciones.tiemp_VAM_cluster1()
            )
            if vam <= uci:
                break
        else:
            vam = uci

        pre_vam = int((uci - vam) * self.experiment.porciento / 100)
        post_vam = uci - pre_vam - vam

        self.experiment.result["Tiempo Pre VAM"] = pre_vam
        self.experiment.result["Tiempo VAM"] = vam
        self.experiment.result["Tiempo Post VAM"] = post_vam
        self.experiment.result["Estadia UCI"] = uci
        self.experiment.result["Estadia Post UCI"] = post_uci

        yield env.timeout(pre_vam)
        yield env.timeout(vam)
        yield env.timeout(post_vam)
        yield env.timeout(post_uci)
