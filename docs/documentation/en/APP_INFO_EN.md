# SimUci Application Information

## About SimUci

**SimUci** is a discrete-event simulation application for modeling patient flow in intensive care units (ICUs). It uses discrete-event simulation techniques to analyze and optimize critical hospital processes.

## Main Features

### Patient Simulation

- Complete ICU patient flow modeling
- From admission to discharge or death
- Consideration of multiple clinical and operational variables

### Statistical Validation

- Comparison of simulated results vs. real data
- Error metrics (RMSE, MAE, MAPE)
- Kolmogorov-Smirnov tests
- Confidence interval coverage analysis

### Interactive Interface

- Simulation parameter configuration
- Real-time results visualization
- Report and chart generation

## Technologies Used

- **Backend**: Python with SimPy (discrete-event simulation)
- **Frontend**: Streamlit
- **Statistical Analysis**: SciPy, NumPy, scikit-learn
- **Visualization**: Matplotlib, Plotly

## System Architecture

The application is organized in specialized modules:

- `uci/`: Simulation core (distributions, experiments, statistics)
- `utils/`: Utilities and interface components
- `data/`: Datasets and trained models
- `docs/`: Complete system documentation

## Clinical Use

This tool is designed to help hospital administrators and clinicians to:

- Optimize ICU resource allocation
- Evaluate patient load scenarios
- Improve care protocols
- Plan capacity expansions

## Limitations

- Results are simulations and not definitive predictions
- Requires validation with institution-specific data
- Does not replace professional clinical judgment

## Contact

For technical support or inquiries about application usage, consult the complete documentation in the project repository.
