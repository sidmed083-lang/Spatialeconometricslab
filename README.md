# SpatialEconometricsLab

Interactive spatial econometrics and GIS analysis platform built on GeoPandas, PySAL, Folium and Streamlit.

> **Status: under development.** Stage 2 (spatial data layer) is complete. Weights, autocorrelation,
> regression, maps and the Streamlit UI follow in later stages. The full README is written in Stage 8.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
