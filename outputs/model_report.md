# Gangwon Forest Fire Risk Models

## Data
- Fire rows used: 1,027
- Weather rows used: 44,961
- Fire date range: 2011-01-22 to 2026-05-19
- Weather date range: 2011-01-01 to 2026-05-22
- Temporal split: train before 2024-01-01, test on/after 2024-01-01
- Large spread target: damagearea >= 1 ha

## Station mapping
The fire data is city/county based, while weather data has 8 stations. Each fire was mapped to a representative station:

- 강릉 -> 강릉
- 고성 -> 속초
- 동해 -> 강릉
- 삼척 -> 태백
- 속초 -> 속초
- 양구 -> 인제
- 양양 -> 속초
- 영월 -> 정선군
- 원주 -> 원주
- 인제 -> 인제
- 정선 -> 정선군
- 철원 -> 춘천
- 춘천 -> 춘천
- 태백 -> 태백
- 평창 -> 정선군
- 홍천 -> 홍천
- 화천 -> 춘천
- 횡성 -> 원주

## Test metrics
Occurrence model:
- ROC AUC: 0.850
- Average precision: 0.088
- Brier score: 0.0164
- F1 at train-selected threshold: 0.147
- Top 10% lift: 5.20
- Test positives / rows: 121 / 6,977

Spread model:
- ROC AUC: 0.588
- Average precision: 0.180
- Brier score: 0.1475
- F1 at train-selected threshold: 0.281
- Top 10% lift: 0.50
- Test positives / rows: 19 / 124

## Main model signals
Occurrence:
- rainfall_7d_sum: -0.606 (risk_down)
- min_humidity_3d_avg: -0.489 (risk_down)
- max_temperature: 0.336 (risk_up)
- rainfall_3d_sum: -0.317 (risk_down)
- humidity: -0.290 (risk_down)
- min_humidity: -0.277 (risk_down)
- doy_sin: 0.218 (risk_up)
- month_sin: 0.209 (risk_up)
- station=춘천: 0.203 (risk_up)
- dry_days: 0.187 (risk_up)
- station=강릉: -0.143 (risk_down)
- sunshine_duration: 0.142 (risk_up)

Spread if a fire occurs:
- instant_wind_speed: 0.485 (risk_up)
- max_temperature: 0.369 (risk_up)
- doy_cos: 0.348 (risk_up)
- month_cos: 0.339 (risk_up)
- station=강릉: -0.319 (risk_down)
- humidity: -0.293 (risk_down)
- station=태백: 0.233 (risk_up)
- month_sin: 0.221 (risk_up)
- dry_days: 0.196 (risk_up)
- ground_temperature: 0.194 (risk_up)
- weekday_sin: 0.194 (risk_up)
- wind_speed: 0.193 (risk_up)

## Latest station risk
Risk score = occurrence probability * spread probability if a fire occurs.

- 춘천: occurrence=0.008, spread_if_fire=0.019, score=0.0002, band=low
- 홍천: occurrence=0.006, spread_if_fire=0.015, score=0.0001, band=low
- 인제: occurrence=0.004, spread_if_fire=0.023, score=0.0001, band=low
- 원주: occurrence=0.009, spread_if_fire=0.009, score=0.0001, band=low
- 정선군: occurrence=0.002, spread_if_fire=0.012, score=0.0000, band=low
- 태백: occurrence=0.000, spread_if_fire=0.005, score=0.0000, band=low
- 속초: occurrence=0.000, spread_if_fire=0.002, score=0.0000, band=low
- 강릉: occurrence=0.000, spread_if_fire=0.001, score=0.0000, band=low

## Notes and limitations
- This is a baseline predictive model built only from the two supplied CSV files.
- The occurrence model uses station-day rows; a positive means at least one mapped fire happened on that station/date.
- The spread model uses only fire-event rows and estimates the conditional probability of damagearea >= 1 ha.
- Same-day weather is used. For operational forecasting, feed forecast weather values instead of observed daily weather.
- Location precision is limited by the 8-station weather data and representative station mapping.
