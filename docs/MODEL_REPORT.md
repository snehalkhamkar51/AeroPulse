# AeroPulse Model Report

## Active model

- Model type: RandomForestRegressor
- Subset: FD001
- Features: 16
- Feature list: sensor_2, sensor_3, sensor_4, sensor_6, sensor_7, sensor_8, sensor_9, sensor_11, sensor_12, sensor_13, sensor_14, sensor_15, sensor_17, sensor_20, sensor_21, cycle
- RUL cap: 125 cycles
- Validation split: engine-wise, 20% validation
- Validation MAE: 9.9272 cycles
- Validation RMSE: 15.1065 cycles
- Official test MAE: 13.4338 cycles
- Official test RMSE: 18.1026 cycles
- NASA score (test): 637.57

## Comparison

| Model | Feature set | Val MAE | Val RMSE | Test MAE | Test RMSE |
|---|---|---:|---:|---:|---:|
| Random Forest (15 sensors + cycle) | 16 current-cycle features | 9.9272 | 15.1065 | 13.4338 | 18.1026 |
| Random Forest (17-feature) | 17 current-cycle features | 12.8908 | 18.2979 | 13.1159 | 18.0091 |
| Random Forest compact | 7 current-cycle features | 13.3469 | 18.8745 | 14.0952 | 19.0030 |
| Random Forest history-aware | 36 rolling/current features | 18.4610 | 26.3363 | 16.9352 | 24.5456 |
| Extra Trees history-aware | 36 rolling/current features | 18.7532 | 26.7251 | 16.8818 | 24.5759 |
| HistGradientBoosting history-aware | 36 rolling/current features | 19.6955 | 27.9655 | 17.6079 | 26.7630 |

## Decision

The active model was switched to Random Forest on 15 non-constant sensors plus cycle because it improved validation RMSE and test MAE/RMSE over the compact 7-feature model. Rolling-window history features were tried and did not help; they increased both validation and test error, so they are not used for prediction.

## Range

`/predict` returns `rul_low` and `rul_high` from the 10th and 90th percentiles of individual tree predictions for the active Random Forest.