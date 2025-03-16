import pytest
from .timestamp import timestamp

@pytest.mark.parametrize("time_params, expected_output", [
    # Standard case. No overflows or omissions.
    (
        {"hours": 1, "minutes": 5, "seconds": 27},
        "01:05:27"
    ),
    # Truncate milliseconds and microseconds when present.
    (
        {"hours": 1, "minutes": 5, "seconds": 27, "milliseconds": 100, "microseconds": 10},
        "01:05:27"
    ),
    # Check that 00:00:SS is displayed before the 1 minute mark.
    (
        {"seconds": 27},
        "00:00:27"
    ),
    # Verify that the library handles seconds overflow correctly.
    (
        {"seconds": 61},
        "00:01:01"
    ),
    # Verify that weeks and days are converted to hours.
    (
        {"weeks": 1, "days": 2, "hours": 3, "minutes": 4, "seconds": 5},
        "219:04:05"
    ),
])
def test_timestamp(time_params, expected_output):
    ts = timestamp(**time_params)
    assert str(ts) == expected_output
