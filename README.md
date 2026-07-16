# OODA-MAT 3C

OODA-MAT 3C is a local Streamlit application for OODA-loop-based, AI-assisted optimization of alkali-free and cement-free construction materials.

The application supports experimental CSV upload, Gaussian-process surrogate modeling, constrained multi-objective candidate generation, hydration-related XRD/TG/DTG interpretation, sensitivity visualization, and experiment-packet export.

## Requirements

- Python 3.10 or later
- Packages listed in `requirements.txt`

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
streamlit run OODA_MAT_Fin.py
```

The application opens in a local web browser after Streamlit starts.

## Main capabilities

- OODA workflow for observing data, orienting model results, deciding candidate mixtures, and recording actions
- Gaussian-process surrogate models for engineering-performance and chemistry-related targets
- XRD phase-fraction and TG/DTG response modeling
- Hydration simulation and conceptual 3D product visualization
- Multi-objective mixture screening with uncertainty information
- Sensitivity analysis relative to a documented reference mixture
- CSV-based data import and experiment-packet export

## Engineering notice

Predictions are decision-support results, not proof of hydration mechanisms or field suitability. Proposed mixtures must be verified experimentally and evaluated under applicable test methods, safety requirements, and engineering standards.

## Repository contents

- `OODA_MAT_Fin.py`: standalone Streamlit application
- `requirements.txt`: Python dependencies

