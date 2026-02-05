# SimUCI

**Simulation of Post-Discharge Evolution of Ventilated Patients in Intensive Care**

This tool is a Clinical Decision Support System (CDSS) that combines discrete-event simulation and predictive machine learning models. Its goal is to model the clinical trajectory of patients to estimate length of stay, resource usage, and survival probabilities.

## System Architecture

The application integrates two main analysis engines:

1.  **Stochastic Simulation Engine**: Uses probability distributions derived from historical data to generate possible scenarios for new patients.
2.  **Prediction Engine (Machine Learning)**: Estimates the probability of death using models trained on physiological and demographic variables.

---

## Functional Modules

### 1. Individual Patient Simulation
The simulation module allows configuring a "virtual patient" with specific clinical parameters:
*   **Demographics**: Age.
*   **Physiological Variables**: APACHE II Score, Respiratory Insufficiency.
*   **Diagnoses**: Admission and discharge diagnosis codes.
*   **Clinical Management**: Artificial Ventilation (VA) time, Expected ICU stay.

The system assigns the patient to a statistical "cluster" (based on K-Means) and samples events from the corresponding distributions to generate multiple runs (scenarios). This allows calculating confidence intervals for time variables.

### 2. Validation with Historical Data
This module compares simulation outputs against a real dataset ("Ground Truth") to evaluate model calibration.
Rigorous statistical metrics are calculated:
*   **Coverage**: Percentage of real data falling within the simulated confidence interval (>90% indicates excellent calibration).
*   **Error (RMSE, MAE)**: Quantification of deviation in hours between simulated and real values.
*   **Kolmogorov-Smirnov (KS) Test**: Evaluates if simulated and real distributions are statistically indistinguishable.

### 3. Risk Prediction
Uses a classifier model (stored as `prediction_model.joblib`) to determine the patient's risk class (Deceased / Survived) and associated probability. Key predictor variables include diagnostic combinations, age, and mechanical ventilation time.

### 4. Statistical Comparative Analysis
Tools to compare results different simulation scenarios or patient groups:
*   **Wilcoxon Test**: To compare two related samples (e.g., same patient with different simulation parameters).
*   **Friedman Test**: To detect differences across multiple treatments or simulation configurations simultaneously.

---

## Metric Interpretation

*   **RMSE (Root Mean Squared Error)**: Penalizes large errors. A low value indicates high precision.
*   **MAE (Mean Absolute Error)**: Average of absolute errors. Direct interpretation in hours.
*   **P-Value**: In comparative tests, a p-value < 0.05 generally indicates statistically significant differences between compared groups.

---
*Developed for academic research purposes in health systems engineering.*
