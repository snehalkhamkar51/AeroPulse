import pytest

from aeropulse.predict import classify_health


@pytest.mark.parametrize("rul,expected", [(9, "Critical"), (10, "Warning"), (30, "Warning"), (31, "Normal")])
def test_classification_boundaries(rul, expected):
    assert classify_health(rul) == expected


def test_classify_health_extra():
    assert classify_health(0) == "Critical"
    assert classify_health(100) == "Normal"
    assert classify_health(9.9) == "Critical"
    assert classify_health(30.1) == "Normal"
