from __future__ import annotations

from datetime import datetime, timezone

import pytest

from t4_devkit.extract import TimeOrigin, TimeRange, TimeSpec

FIRST = 1704067200000000  # 2024-01-01T00:00:00Z
LAST = 1704067202000000  # 2024-01-01T00:00:02Z


class TestTimeSpec:
    """Test cases for `TimeSpec`."""

    @pytest.mark.parametrize(
        ("expression", "expect"),
        [
            (1704067200000000, TimeSpec(1704067200000000.0, TimeOrigin.ABSOLUTE)),
            ("1704067200000000", TimeSpec(1704067200000000.0, TimeOrigin.ABSOLUTE)),
            (1704067200.5, TimeSpec(1704067200500000.0, TimeOrigin.ABSOLUTE)),
            ("2024-01-01T00:00:00Z", TimeSpec(1704067200000000.0, TimeOrigin.ABSOLUTE)),
            ("2024-01-01T00:00:00", TimeSpec(1704067200000000.0, TimeOrigin.ABSOLUTE)),
            ("2024-01-01 00:00:00+00:00", TimeSpec(1704067200000000.0, TimeOrigin.ABSOLUTE)),
            ("+1.5", TimeSpec(1.5, TimeOrigin.FIRST)),
            (" -1.5 ", TimeSpec(-1.5, TimeOrigin.LAST)),
            (
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                TimeSpec(1704067200000000.0, TimeOrigin.ABSOLUTE),
            ),
        ],
    )
    def test_parse(self, expression, expect: TimeSpec) -> None:
        assert TimeSpec.parse(expression) == expect

    def test_parse_timespec(self) -> None:
        """Test a `TimeSpec` is acceptable as an expression."""
        expect = TimeSpec(1.5, TimeOrigin.FIRST)
        assert TimeSpec.parse(expect) == expect

    @pytest.mark.parametrize("expression", [10, "10", "", "foo", "+foo", None, True])
    def test_parse_with_invalid_expression(self, expression) -> None:
        with pytest.raises(ValueError):
            TimeSpec.parse(expression)

    @pytest.mark.parametrize(
        ("expression", "expect"),
        [
            ("1704067201000000", 1704067201000000),
            ("+0.5", FIRST + 500000),
            ("-0.5", LAST - 500000),
            ("+0", FIRST),
        ],
    )
    def test_resolve(self, expression: str, expect: int) -> None:
        assert TimeSpec.parse(expression).resolve(FIRST, LAST) == expect


class TestTimeRange:
    """Test cases for `TimeRange`."""

    def test_resolve_without_any_specification(self) -> None:
        time_range = TimeRange.resolve(first=FIRST, last=LAST)

        assert time_range == TimeRange(FIRST, LAST)
        assert time_range.duration == pytest.approx(2.0)

    def test_resolve_with_relative_expressions(self) -> None:
        time_range = TimeRange.resolve("+0.5", "-0.5", first=FIRST, last=LAST)

        assert time_range == TimeRange(FIRST + 500000, LAST - 500000)

    def test_resolve_with_duration(self) -> None:
        time_range = TimeRange.resolve("+0.5", duration=1.0, first=FIRST, last=LAST)

        assert time_range == TimeRange(FIRST + 500000, FIRST + 1500000)

    def test_resolve_prefers_end_over_duration(self) -> None:
        time_range = TimeRange.resolve(None, "-0.5", duration=100.0, first=FIRST, last=LAST)

        assert time_range == TimeRange(FIRST, LAST - 500000)

    def test_resolve_with_negative_duration(self) -> None:
        with pytest.raises(ValueError):
            TimeRange.resolve(duration=-1.0, first=FIRST, last=LAST)

    def test_resolve_with_reversed_range(self) -> None:
        with pytest.raises(ValueError):
            TimeRange.resolve("-0.5", "+0.5", first=FIRST, last=LAST)

    def test_contains(self) -> None:
        time_range = TimeRange(FIRST, LAST)

        assert time_range.contains(FIRST)
        assert time_range.contains(LAST)
        assert time_range.contains(FIRST + 1)
        assert not time_range.contains(FIRST - 1)
        assert not time_range.contains(LAST + 1)
