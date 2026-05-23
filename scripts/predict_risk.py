from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from train_models import (
    BinaryModel,
    DEFAULT_WEATHER_PATH,
    add_weather_features,
    predict_with_model,
    risk_band,
)


def load_model(path: Path) -> BinaryModel:
    data = json.loads(path.read_text(encoding="utf-8"))
    return BinaryModel(
        feature_names=data["feature_names"],
        fill_values=data["fill_values"],
        means=data["means"],
        scales=data["scales"],
        coef=data["coef"],
        intercept=data["intercept"],
        threshold=data["threshold"],
        metadata=data["metadata"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score weather rows with trained forest fire risk models.")
    parser.add_argument("--weather-csv", type=Path, default=DEFAULT_WEATHER_PATH)
    parser.add_argument("--occurrence-model", type=Path, default=Path("models/occurrence_model.json"))
    parser.add_argument("--spread-model", type=Path, default=Path("models/spread_model.json"))
    parser.add_argument("--metrics-json", type=Path, default=Path("outputs/metrics.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/scored_weather_risk.csv"))
    parser.add_argument("--latest-only", action="store_true")
    args = parser.parse_args()

    occurrence_model = load_model(args.occurrence_model)
    spread_model = load_model(args.spread_model)
    metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    medium_threshold = metrics["risk_band_thresholds"]["medium_min"]
    high_threshold = metrics["risk_band_thresholds"]["high_min"]

    weather = add_weather_features(pd.read_csv(args.weather_csv))
    if args.latest_only:
        weather = weather[weather["date"] == weather["date"].max()].copy()

    weather["fire_probability"] = predict_with_model(weather, occurrence_model)
    weather["spread_probability_if_fire"] = predict_with_model(weather, spread_model)
    weather["spread_risk_score"] = weather["fire_probability"] * weather["spread_probability_if_fire"]
    weather["risk_band"] = [
        risk_band(score, high_threshold, medium_threshold) for score in weather["spread_risk_score"]
    ]

    columns = [
        "date",
        "station_name",
        "fire_probability",
        "spread_probability_if_fire",
        "spread_risk_score",
        "risk_band",
        "temperature",
        "humidity",
        "min_humidity",
        "max_wind_speed",
        "rainfall",
        "dry_days",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    weather[columns].sort_values(["date", "spread_risk_score"], ascending=[True, False]).to_csv(
        args.output_csv,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Wrote {len(weather):,} scored rows to {args.output_csv}")


if __name__ == "__main__":
    main()
