from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum, unique
from typing import TYPE_CHECKING, Union

from attrs import define, field

from t4_devkit.common.timestamp import microseconds2seconds, seconds2microseconds

if TYPE_CHECKING:
    from typing_extensions import Self

    from t4_devkit.typing import ScalarLike

__all__ = ["TimeOrigin", "TimeSpec", "TimeSpecLike", "TimeRange"]

# Numeric values greater than this threshold are interpreted as unix time in [us],
# otherwise in [s]. Note that 1e12[us] and 1e12[s] correspond to 1970-01-12 and 33658-09-27.
_MICROSECONDS_THRESHOLD: float = 1e12

# Numeric values smaller than this threshold are too ambiguous to be an absolute unix time.
# Note that 1e8[s] corresponds to 1973-03-03.
_MINIMUM_ABSOLUTE_SECONDS: float = 1e8


@unique
class TimeOrigin(str, Enum):
    """An enum to represent the origin which a `TimeSpec` is relative to.

    Attributes:
        ABSOLUTE: The value is an absolute unix time in [us].
        FIRST: The value is an offset in [s] from the first timestamp of the target data.
        LAST: The value is an offset in [s] from the last timestamp of the target data.
    """

    ABSOLUTE = "absolute"
    FIRST = "first"
    LAST = "last"


@define
class TimeSpec:
    """A dataclass to represent a timestamp specified by a user.

    A `TimeSpec` is not necessarily an absolute timestamp: it can also be an offset from the
    first or the last timestamp of the target data. Call `resolve(...)` to get an absolute
    unix time in [us].

    Attributes:
        value (float): Unix time in [us] if `origin` is `ABSOLUTE`, otherwise an offset in [s].
        origin (TimeOrigin): Origin which `value` is relative to.
    """

    value: float = field(converter=float)
    origin: TimeOrigin = field(default=TimeOrigin.ABSOLUTE, converter=TimeOrigin)

    @classmethod
    def parse(cls, expression: TimeSpecLike) -> Self:
        """Parse a user friendly expression into a `TimeSpec`.

        The following expressions are supported:

        * Unix time in [us], such as `1704067200000000`.
        * Unix time in [s], such as `1704067200.5`.
        * ISO 8601 datetime, such as `2024-01-01T00:00:00`. A naive datetime is interpreted as UTC.
        * Offset in [s] from the first timestamp, such as `+1.5`.
        * Offset in [s] from the last timestamp, such as `-1.5`.

        Args:
            expression (TimeSpecLike): Expression to be parsed.

        Raises:
            ValueError: Expecting the expression is one of the supported formats.

        Returns:
            Parsed self instance.

        Examples:
            >>> TimeSpec.parse("1704067200000000")
            TimeSpec(value=1704067200000000.0, origin=<TimeOrigin.ABSOLUTE: 'absolute'>)
            >>> TimeSpec.parse("+1.5")
            TimeSpec(value=1.5, origin=<TimeOrigin.FIRST: 'first'>)
        """
        if isinstance(expression, TimeSpec):
            return cls(expression.value, expression.origin)

        if isinstance(expression, datetime):
            return cls(_datetime2microseconds(expression))

        # NOTE: `bool` is a subclass of `int`, but is not a valid timestamp.
        if isinstance(expression, (int, float)) and not isinstance(expression, bool):
            return cls(_number2microseconds(expression))

        if not isinstance(expression, str):
            raise ValueError(f"Unsupported timestamp expression: {expression}")

        stripped = expression.strip()
        if not stripped:
            raise ValueError("Timestamp expression must not be empty.")

        if stripped[0] in ("+", "-"):
            try:
                offset = float(stripped)
            except ValueError:
                raise ValueError(
                    f"Relative timestamp must be a number in [s], but got: {expression}"
                ) from None
            origin = TimeOrigin.FIRST if stripped[0] == "+" else TimeOrigin.LAST
            return cls(offset, origin)

        try:
            return cls(_number2microseconds(float(stripped)))
        except ValueError:
            pass

        try:
            return cls(_datetime2microseconds(_parse_datetime(stripped)))
        except ValueError:
            raise ValueError(
                f"Unsupported timestamp expression: {expression}. "
                "Specify a unix time in [s] or [us], an ISO 8601 datetime such as "
                "`2024-01-01T00:00:00`, or a relative offset in [s] such as `+1.5`/`-1.5`."
            ) from None

    def resolve(self, first: int, last: int) -> int:
        """Resolve myself into an absolute unix time in [us].

        Args:
            first (int): The first timestamp of the target data in [us].
            last (int): The last timestamp of the target data in [us].

        Returns:
            Absolute unix time in [us].
        """
        if self.origin == TimeOrigin.ABSOLUTE:
            return int(round(self.value))
        anchor = first if self.origin == TimeOrigin.FIRST else last
        return int(round(anchor + seconds2microseconds(self.value)))


TimeSpecLike = Union[TimeSpec, datetime, str, int, float]
"""Type alias for inputs which can be parsed by `TimeSpec.parse(...)`."""


@define
class TimeRange:
    """A dataclass to represent an absolute time range in [us].

    Attributes:
        start (int): Inclusive start of the range in [us].
        end (int): Inclusive end of the range in [us].
    """

    start: int = field(converter=int)
    end: int = field(converter=int)

    def __attrs_post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"`start` must not be later than `end`: {self.start} > {self.end}")

    @classmethod
    def resolve(
        cls,
        start: TimeSpecLike | None = None,
        end: TimeSpecLike | None = None,
        *,
        duration: float | None = None,
        first: int,
        last: int,
    ) -> Self:
        """Resolve a time range from user specified expressions.

        Args:
            start (TimeSpecLike | None, optional): Start of the range.
                If None, `first` is used.
            end (TimeSpecLike | None, optional): End of the range.
                If None, `duration` or `last` is used.
            duration (float | None, optional): Time length in [s] from `start`.
                This is ignored if `end` is specified.
            first (int): The first timestamp of the target data in [us].
            last (int): The last timestamp of the target data in [us].

        Returns:
            Resolved self instance.
        """
        start_us = TimeSpec.parse(start).resolve(first, last) if start is not None else first

        if end is not None:
            end_us = TimeSpec.parse(end).resolve(first, last)
        elif duration is not None:
            if duration < 0:
                raise ValueError(f"`duration` must be positive, but got: {duration}")
            end_us = int(round(start_us + seconds2microseconds(duration)))
        else:
            end_us = last

        return cls(start_us, end_us)

    @property
    def duration(self) -> float:
        """Return the time length of the range in [s]."""
        return microseconds2seconds(self.end - self.start)

    def contains(self, timestamp: ScalarLike) -> bool:
        """Indicate whether the input timestamp is included in the range.

        Args:
            timestamp (ScalarLike): Unix time in [us].

        Returns:
            Return True if the timestamp is included.
        """
        return self.start <= timestamp <= self.end


def _number2microseconds(value: ScalarLike) -> float:
    """Interpret a numeric value as an absolute unix time in [us].

    Args:
        value (ScalarLike): Unix time in [s] or [us].

    Raises:
        ValueError: Expecting the value is large enough to be an absolute unix time.

    Returns:
        Unix time in [us].
    """
    if abs(value) >= _MICROSECONDS_THRESHOLD:
        return float(value)

    if abs(value) >= _MINIMUM_ABSOLUTE_SECONDS:
        return seconds2microseconds(float(value))

    raise ValueError(
        f"Ambiguous timestamp: {value}. Specify a unix time in [s] or [us], "
        "or a relative offset in [s] such as `+1.5`/`-1.5`."
    )


def _parse_datetime(expression: str) -> datetime:
    """Parse an ISO 8601 datetime expression.

    Args:
        expression (str): ISO 8601 datetime expression.

    Returns:
        Parsed datetime.
    """
    # NOTE: `datetime.fromisoformat` does not accept the `Z` suffix on Python 3.10.
    normalized = expression[:-1] + "+00:00" if expression.endswith("Z") else expression
    return datetime.fromisoformat(normalized)


def _datetime2microseconds(value: datetime) -> float:
    """Convert a datetime into a unix time in [us].

    A naive datetime is interpreted as UTC.

    Args:
        value (datetime): Datetime to be converted.

    Returns:
        Unix time in [us].
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return seconds2microseconds(value.timestamp())
