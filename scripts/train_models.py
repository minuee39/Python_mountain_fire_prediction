from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_FIRE_PATH = Path(r"C:\Users\minwoo\Desktop\data_code\cache\gangwon_forest_fire_raw_cache.csv")
DEFAULT_WEATHER_PATH = Path(r"C:\Users\minwoo\Desktop\data_code\cache\gangwon_weather_station_all_cache.csv")
DEFAULT_SPLIT_DATE = "2024-01-01"
LARGE_FIRE_THRESHOLD_HA = 1.0

LOCATION_TO_STATION = {
    "강릉": "강릉",
    "고성": "속초",
    "동해": "강릉",
    "삼척": "태백",
    "속초": "속초",
    "양구": "인제",
    "양양": "속초",
    "영월": "정선군",
    "원주": "원주",
    "인제": "인제",
    "정선": "정선군",
    "철원": "춘천",
    "춘천": "춘천",
    "태백": "태백",
    "평창": "정선군",
    "홍천": "홍천",
    "화천": "춘천",
    "횡성": "원주",
}

WEATHER_NUMERIC_COLUMNS = [
    "temperature",
    "max_temperature",
    "humidity",
    "min_humidity",
    "wind_speed",
    "max_wind_speed",
    "instant_wind_speed",
    "rainfall",
    "sunshine_duration",
    "solar_radiation",
    "ground_temperature",
    "rainfall_3d_sum",
    "rainfall_7d_sum",
    "dry_days",
    "min_humidity_3d_avg",
    "max_wind_3d_max",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "weekday_sin",
    "weekday_cos",
]


@dataclass
class BinaryModel:
    feature_names: list[str]
    fill_values: dict[str, float]
    means: dict[str, float]
    scales: dict[str, float]
    coef: list[float]
    intercept: float
    threshold: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "fill_values": self.fill_values,
            "means": self.means,
            "scales": self.scales,
            "coef": self.coef,
            "intercept": self.intercept,
            "threshold": self.threshold,
            "metadata": self.metadata,
        }


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def build_feature_frame(df: pd.DataFrame, station_levels: list[str] | None = None) -> pd.DataFrame:
    features = df[WEATHER_NUMERIC_COLUMNS].copy()
    if station_levels is None:
        station_levels = sorted(df["station_name"].dropna().astype(str).unique().tolist())
    for station in station_levels:
        features[f"station={station}"] = (df["station_name"].astype(str) == station).astype(float)
    return features


def fit_preprocessor(df: pd.DataFrame) -> dict[str, Any]:
    station_levels = sorted(df["station_name"].dropna().astype(str).unique().tolist())
    raw = build_feature_frame(df, station_levels)
    fill_values = raw.median(numeric_only=True).fillna(0.0)
    filled = raw.fillna(fill_values)
    means = filled.mean()
    scales = filled.std(ddof=0).replace(0.0, 1.0)
    return {
        "station_levels": station_levels,
        "feature_names": raw.columns.tolist(),
        "fill_values": fill_values.astype(float).to_dict(),
        "means": means.astype(float).to_dict(),
        "scales": scales.astype(float).to_dict(),
    }


def transform_features(df: pd.DataFrame, prep: dict[str, Any]) -> np.ndarray:
    raw = build_feature_frame(df, prep["station_levels"])
    raw = raw.reindex(columns=prep["feature_names"])
    fill_values = pd.Series(prep["fill_values"])
    means = pd.Series(prep["means"])
    scales = pd.Series(prep["scales"]).replace(0.0, 1.0)
    filled = raw.fillna(fill_values)
    x = (filled - means) / scales
    return x.to_numpy(dtype=float)


def fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 4500,
    learning_rate: float = 0.08,
    l2: float = 0.02,
) -> tuple[np.ndarray, float]:
    y = y.astype(float)
    n_features = x.shape[1]
    coef = np.zeros(n_features, dtype=float)
    intercept = 0.0

    positives = max(float(y.sum()), 1.0)
    negatives = max(float(len(y) - y.sum()), 1.0)
    weights = np.where(y == 1.0, len(y) / (2.0 * positives), len(y) / (2.0 * negatives))
    weight_sum = weights.sum()

    for _ in range(epochs):
        pred = sigmoid(x @ coef + intercept)
        error = (pred - y) * weights
        grad_coef = (x.T @ error) / weight_sum + l2 * coef
        grad_intercept = error.sum() / weight_sum
        coef -= learning_rate * grad_coef
        intercept -= learning_rate * grad_intercept

    return coef, float(intercept)


def roc_auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    y = y_true.astype(int)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    sum_pos_ranks = ranks[y == 1].sum()
    return float((sum_pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg))


def average_precision_score(y_true: np.ndarray, score: np.ndarray) -> float:
    y = y_true.astype(int)
    pos = int(y.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-score)
    sorted_y = y[order]
    tp = np.cumsum(sorted_y)
    precision = tp / (np.arange(len(y)) + 1)
    return float((precision * sorted_y).sum() / pos)


def best_f1_threshold(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.quantile(score, np.linspace(0.01, 0.99, 99)))
    best_threshold = 0.5
    best_f1 = -1.0
    y = y_true.astype(int)
    for threshold in candidates:
        pred = (score >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold, float(best_f1)


def classification_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    y = y_true.astype(int)
    pred = (score >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    top_n = max(1, int(math.ceil(len(score) * 0.1)))
    top_idx = np.argsort(-score)[:top_n]
    baseline_rate = y.mean() if len(y) else float("nan")
    top_rate = y[top_idx].mean() if len(top_idx) else float("nan")
    return {
        "rows": int(len(y)),
        "positive_rate": float(baseline_rate),
        "roc_auc": roc_auc_score(y, score),
        "average_precision": average_precision_score(y, score),
        "brier": float(np.mean((score - y) ** 2)),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "top_10pct_positive_rate": float(top_rate),
        "top_10pct_lift": float(top_rate / baseline_rate) if baseline_rate else float("nan"),
    }


def add_weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    weather = weather.copy()
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
    weather = weather.dropna(subset=["date", "station_name"]).sort_values(["station_name", "date"])
    for col in WEATHER_NUMERIC_COLUMNS[:11]:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")

    grouped = weather.groupby("station_name", group_keys=False)
    weather["rainfall_3d_sum"] = grouped["rainfall"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    weather["rainfall_7d_sum"] = grouped["rainfall"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    weather["min_humidity_3d_avg"] = grouped["min_humidity"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    weather["max_wind_3d_max"] = grouped["max_wind_speed"].transform(lambda s: s.rolling(3, min_periods=1).max())

    dry_days_parts = []
    for _, part in weather.groupby("station_name", sort=False):
        count = 0
        counts = []
        for rainfall in part["rainfall"].fillna(0.0):
            if rainfall < 1.0:
                count += 1
            else:
                count = 0
            counts.append(count)
        dry_days_parts.append(pd.Series(counts, index=part.index))
    weather["dry_days"] = pd.concat(dry_days_parts).sort_index()

    month = weather["date"].dt.month
    day_of_year = weather["date"].dt.dayofyear
    weekday = weather["date"].dt.weekday
    weather["month_sin"] = np.sin(2 * np.pi * month / 12)
    weather["month_cos"] = np.cos(2 * np.pi * month / 12)
    weather["doy_sin"] = np.sin(2 * np.pi * day_of_year / 366)
    weather["doy_cos"] = np.cos(2 * np.pi * day_of_year / 366)
    weather["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    weather["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    return weather


def load_fire_events(path: Path) -> pd.DataFrame:
    fire = pd.read_csv(path)
    fire["date"] = pd.to_datetime(
        {
            "year": pd.to_numeric(fire["startyear"], errors="coerce"),
            "month": pd.to_numeric(fire["startmonth"], errors="coerce"),
            "day": pd.to_numeric(fire["startday"], errors="coerce"),
        },
        errors="coerce",
    )
    fire["damagearea"] = pd.to_numeric(fire["damagearea"], errors="coerce")
    fire["station_name"] = fire["locgungu"].map(LOCATION_TO_STATION)
    fire = fire.dropna(subset=["date", "station_name", "damagearea"]).copy()
    fire["large_fire"] = (fire["damagearea"] >= LARGE_FIRE_THRESHOLD_HA).astype(int)
    return fire


def build_occurrence_dataset(weather: pd.DataFrame, fire: pd.DataFrame) -> pd.DataFrame:
    grouped_fire = (
        fire.groupby(["date", "station_name"])
        .agg(
            fire_count=("damagearea", "size"),
            max_damagearea=("damagearea", "max"),
            total_damagearea=("damagearea", "sum"),
        )
        .reset_index()
    )
    dataset = weather.merge(grouped_fire, on=["date", "station_name"], how="left")
    dataset["fire_count"] = dataset["fire_count"].fillna(0).astype(int)
    dataset["max_damagearea"] = dataset["max_damagearea"].fillna(0.0)
    dataset["total_damagearea"] = dataset["total_damagearea"].fillna(0.0)
    dataset["fire_occurrence"] = (dataset["fire_count"] > 0).astype(int)
    return dataset


def build_spread_dataset(weather: pd.DataFrame, fire: pd.DataFrame) -> pd.DataFrame:
    return fire.merge(weather, on=["date", "station_name"], how="inner", suffixes=("", "_weather"))


def train_binary_model(df: pd.DataFrame, target: str, split_date: pd.Timestamp, metadata: dict[str, Any]) -> tuple[BinaryModel, dict[str, Any], np.ndarray]:
    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()
    prep = fit_preprocessor(train_df)
    x_train = transform_features(train_df, prep)
    x_test = transform_features(test_df, prep)
    y_train = train_df[target].to_numpy(dtype=int)
    y_test = test_df[target].to_numpy(dtype=int)

    coef, intercept = fit_logistic_regression(x_train, y_train)
    positives = max(float(y_train.sum()), 1.0)
    negatives = max(float(len(y_train) - y_train.sum()), 1.0)
    intercept += math.log(positives / negatives)
    train_score = sigmoid(x_train @ coef + intercept)
    test_score = sigmoid(x_test @ coef + intercept)
    threshold, train_f1 = best_f1_threshold(y_train, train_score)

    metrics = {
        "train": classification_metrics(y_train, train_score, threshold),
        "test": classification_metrics(y_test, test_score, threshold),
        "train_best_f1": train_f1,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_positives": int(y_train.sum()),
        "test_positives": int(y_test.sum()),
    }
    metadata = dict(metadata)
    metadata.update(
        {
            "target": target,
            "split_date": split_date.strftime("%Y-%m-%d"),
            "station_levels": prep["station_levels"],
        }
    )
    model = BinaryModel(
        feature_names=prep["feature_names"],
        fill_values=prep["fill_values"],
        means=prep["means"],
        scales=prep["scales"],
        coef=coef.astype(float).tolist(),
        intercept=intercept,
        threshold=threshold,
        metadata=metadata,
    )
    return model, metrics, test_score


def predict_with_model(df: pd.DataFrame, model: BinaryModel) -> np.ndarray:
    prep = {
        "station_levels": model.metadata["station_levels"],
        "feature_names": model.feature_names,
        "fill_values": model.fill_values,
        "means": model.means,
        "scales": model.scales,
    }
    x = transform_features(df, prep)
    return sigmoid(x @ np.array(model.coef) + model.intercept)


def feature_signal_table(model: BinaryModel, top_n: int = 12) -> list[dict[str, Any]]:
    rows = []
    for name, coef in zip(model.feature_names, model.coef):
        rows.append(
            {
                "feature": name,
                "coefficient": float(coef),
                "direction": "risk_up" if coef > 0 else "risk_down",
                "abs_coefficient": abs(float(coef)),
            }
        )
    rows.sort(key=lambda row: row["abs_coefficient"], reverse=True)
    return rows[:top_n]


def risk_band(score: float, high_threshold: float, medium_threshold: float) -> str:
    if score >= high_threshold:
        return "high"
    if score >= medium_threshold:
        return "medium"
    return "low"


def write_report(
    output_path: Path,
    *,
    fire: pd.DataFrame,
    weather: pd.DataFrame,
    occurrence_metrics: dict[str, Any],
    spread_metrics: dict[str, Any],
    occurrence_model: BinaryModel,
    spread_model: BinaryModel,
    latest: pd.DataFrame,
    split_date: str,
) -> None:
    mapping_lines = "\n".join(f"- {k} -> {v}" for k, v in sorted(LOCATION_TO_STATION.items()))
    occurrence_signals = "\n".join(
        f"- {row['feature']}: {row['coefficient']:.3f} ({row['direction']})"
        for row in feature_signal_table(occurrence_model)
    )
    spread_signals = "\n".join(
        f"- {row['feature']}: {row['coefficient']:.3f} ({row['direction']})"
        for row in feature_signal_table(spread_model)
    )
    latest_lines = "\n".join(
        f"- {row.station_name}: occurrence={row.fire_probability:.3f}, spread_if_fire={row.spread_probability_if_fire:.3f}, score={row.spread_risk_score:.4f}, band={row.risk_band}"
        for row in latest.sort_values("spread_risk_score", ascending=False).itertuples()
    )

    text = f"""# Gangwon Forest Fire Risk Models

## Data
- Fire rows used: {len(fire):,}
- Weather rows used: {len(weather):,}
- Fire date range: {fire['date'].min().date()} to {fire['date'].max().date()}
- Weather date range: {weather['date'].min().date()} to {weather['date'].max().date()}
- Temporal split: train before {split_date}, test on/after {split_date}
- Large spread target: damagearea >= {LARGE_FIRE_THRESHOLD_HA:g} ha

## Station mapping
The fire data is city/county based, while weather data has 8 stations. Each fire was mapped to a representative station:

{mapping_lines}

## Test metrics
Occurrence model:
- ROC AUC: {occurrence_metrics['test']['roc_auc']:.3f}
- Average precision: {occurrence_metrics['test']['average_precision']:.3f}
- Brier score: {occurrence_metrics['test']['brier']:.4f}
- F1 at train-selected threshold: {occurrence_metrics['test']['f1']:.3f}
- Top 10% lift: {occurrence_metrics['test']['top_10pct_lift']:.2f}
- Test positives / rows: {occurrence_metrics['test_positives']:,} / {occurrence_metrics['test_rows']:,}

Spread model:
- ROC AUC: {spread_metrics['test']['roc_auc']:.3f}
- Average precision: {spread_metrics['test']['average_precision']:.3f}
- Brier score: {spread_metrics['test']['brier']:.4f}
- F1 at train-selected threshold: {spread_metrics['test']['f1']:.3f}
- Top 10% lift: {spread_metrics['test']['top_10pct_lift']:.2f}
- Test positives / rows: {spread_metrics['test_positives']:,} / {spread_metrics['test_rows']:,}

## Main model signals
Occurrence:
{occurrence_signals}

Spread if a fire occurs:
{spread_signals}

## Latest station risk
Risk score = occurrence probability * spread probability if a fire occurs.

{latest_lines}

## Notes and limitations
- This is a baseline predictive model built only from the two supplied CSV files.
- The occurrence model uses station-day rows; a positive means at least one mapped fire happened on that station/date.
- The spread model uses only fire-event rows and estimates the conditional probability of damagearea >= {LARGE_FIRE_THRESHOLD_HA:g} ha.
- Same-day weather is used. For operational forecasting, feed forecast weather values instead of observed daily weather.
- Location precision is limited by the 8-station weather data and representative station mapping.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Gangwon forest fire occurrence and spread risk models.")
    parser.add_argument("--fire-csv", type=Path, default=DEFAULT_FIRE_PATH)
    parser.add_argument("--weather-csv", type=Path, default=DEFAULT_WEATHER_PATH)
    parser.add_argument("--split-date", default=DEFAULT_SPLIT_DATE)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    fire = load_fire_events(args.fire_csv)
    weather = add_weather_features(pd.read_csv(args.weather_csv))
    split_date = pd.Timestamp(args.split_date)

    occurrence_df = build_occurrence_dataset(weather, fire)
    spread_df = build_spread_dataset(weather, fire)

    occurrence_model, occurrence_metrics, _ = train_binary_model(
        occurrence_df,
        "fire_occurrence",
        split_date,
        {
            "model_name": "forest_fire_occurrence",
            "description": "Predicts whether at least one mapped forest fire occurs on a station-day.",
        },
    )
    spread_model, spread_metrics, _ = train_binary_model(
        spread_df,
        "large_fire",
        split_date,
        {
            "model_name": "forest_fire_spread",
            "description": f"Predicts whether a fire event reaches at least {LARGE_FIRE_THRESHOLD_HA:g} ha.",
            "large_fire_threshold_ha": LARGE_FIRE_THRESHOLD_HA,
        },
    )

    occurrence_df["fire_probability"] = predict_with_model(occurrence_df, occurrence_model)
    occurrence_df["spread_probability_if_fire"] = predict_with_model(occurrence_df, spread_model)
    occurrence_df["spread_risk_score"] = occurrence_df["fire_probability"] * occurrence_df["spread_probability_if_fire"]

    train_scores = occurrence_df.loc[occurrence_df["date"] < split_date, "spread_risk_score"]
    medium_threshold = float(train_scores.quantile(0.80))
    high_threshold = float(train_scores.quantile(0.95))
    occurrence_df["risk_band"] = [
        risk_band(score, high_threshold, medium_threshold) for score in occurrence_df["spread_risk_score"]
    ]

    latest_date = occurrence_df["date"].max()
    latest = occurrence_df[occurrence_df["date"] == latest_date].copy()
    prediction_columns = [
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

    (args.models_dir / "occurrence_model.json").write_text(
        json.dumps(occurrence_model.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.models_dir / "spread_model.json").write_text(
        json.dumps(spread_model.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics = {
        "occurrence": occurrence_metrics,
        "spread": spread_metrics,
        "risk_band_thresholds": {
            "medium_min": medium_threshold,
            "high_min": high_threshold,
        },
        "location_to_station": LOCATION_TO_STATION,
    }
    (args.outputs_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    occurrence_df[prediction_columns].to_csv(args.outputs_dir / "daily_risk_predictions.csv", index=False, encoding="utf-8-sig")
    latest[prediction_columns].sort_values("spread_risk_score", ascending=False).to_csv(
        args.outputs_dir / "latest_risk_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_report(
        args.outputs_dir / "model_report.md",
        fire=fire,
        weather=weather,
        occurrence_metrics=occurrence_metrics,
        spread_metrics=spread_metrics,
        occurrence_model=occurrence_model,
        spread_model=spread_model,
        latest=latest,
        split_date=args.split_date,
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
