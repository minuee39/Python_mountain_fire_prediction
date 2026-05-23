# Gangwon Forest Fire Risk Models

This workspace trains two baseline models from the supplied CSV caches:

- `forest_fire_occurrence`: station-day probability that at least one mapped forest fire occurs.
- `forest_fire_spread`: conditional probability that a fire event reaches at least 1 ha.

The training script uses only `pandas` and `numpy` so it can run with the bundled Codex Python runtime.

## Run

```powershell
& 'C:\Users\minwoo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\train_models.py
```

Optional inputs:

```powershell
& 'C:\Users\minwoo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\train_models.py `
  --fire-csv 'C:\Users\minwoo\Desktop\data_code\cache\gangwon_forest_fire_raw_cache.csv' `
  --weather-csv 'C:\Users\minwoo\Desktop\data_code\cache\gangwon_weather_station_all_cache.csv' `
  --split-date 2024-01-01
```

## Outputs

- `models/occurrence_model.json`
- `models/spread_model.json`
- `outputs/metrics.json`
- `outputs/daily_risk_predictions.csv`
- `outputs/latest_risk_predictions.csv`
- `outputs/model_report.md`

## Notebook Walkthrough

Open `notebooks/forest_fire_model_walkthrough.ipynb` to inspect each step visually:

- raw CSV checks
- missing values and distributions
- station mapping
- feature engineering
- train/test split
- occurrence model evaluation
- spread model evaluation
- latest station risk table

## Score New Weather Rows

```powershell
& 'C:\Users\minwoo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\predict_risk.py --latest-only
```

For a new weather CSV with the same schema:

```powershell
& 'C:\Users\minwoo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\predict_risk.py `
  --weather-csv 'C:\path\to\weather.csv' `
  --output-csv outputs\new_weather_risk.csv
```

## Modeling Notes

The fire data is city/county based, while weather data has 8 station names. The script maps each fire location to a representative weather station before joining by date and station. The current model uses observed same-day weather; for forecasting use, replace those rows with forecast weather values in the same schema.
