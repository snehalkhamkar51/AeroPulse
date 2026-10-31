"""Shared feature definitions for AeroPulse."""

from __future__ import annotations

# Non-constant FD001 sensors retained for the active current-cycle model.
NONCONSTANT_SENSORS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]

# The active model uses the current cycle number in addition to the sensors above.
CURRENT_FEATURES = NONCONSTANT_SENSORS + ["cycle"]
