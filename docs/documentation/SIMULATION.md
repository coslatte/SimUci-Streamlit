# SimUCI - Simulation Module

## Overview

The simulation module (`uci/`) is the core engine of the SimUCI application. It implements discrete-event simulation using SimPy.

## Simulation Pipeline

1.  **Patient Configuration**: Define diagnosis, age, APACHE score.
2.  **Cluster Assignment**: Match patient to a statistical cluster.
3.  **Distribution Sampling**: Sample event times (ICU stay, Ventilation time) based on cluster distributions.
4.  **Result Collection**: Compile results into a DataFrame.

## Components

### Experiment Class (`experiment.py`)
Manages patient parameters.

### Clusters
Patients are grouped using K-Means clustering to assign appropriate statistical distributions for simulation variables.
