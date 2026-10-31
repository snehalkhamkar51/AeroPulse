import pandas as pd

from aeropulse.history_features import build_history_features


def test_history_feature_builder_rolling_values():
    df = pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 1, 1],
            "time_cycles": [1, 2, 3, 4, 5],
            "sensor_11": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )
    built = build_history_features(df, sensors=["sensor_11"], window=5, min_periods=5)
    assert len(built) == 1
    row = built.iloc[0]
    assert row["sensor_11"] == 14.0
    assert row["sensor_11_roll_mean"] == 12.0
    assert row["sensor_11_slope"] == 1.0
    assert row["sensor_11_delta_baseline"] == 2.0
