from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from attrs import define, field

from t4_devkit.common.timestamp import microseconds2seconds
from t4_devkit.schema import SensorModality

from .timespec import TimeRange

if TYPE_CHECKING:
    from t4_devkit.schema import SampleData, Sensor
    from t4_devkit.tier4 import T4Devkit

    from .timespec import TimeSpecLike

__all__ = ["ChannelSummary", "resolve_channels", "select_sample_data", "summarize_channels"]


@define
class ChannelSummary:
    """A dataclass to summarize sensor data recorded in a particular channel.

    Attributes:
        channel (str): Sensor channel name.
        modality (SensorModality): Sensor modality.
        sensor_token (str): Token of the corresponding `sensor` record.
        num_frames (int): The number of `sample_data` records.
        first_timestamp (int | None): The first unix time in [us]. None if no frame is recorded.
        last_timestamp (int | None): The last unix time in [us]. None if no frame is recorded.
    """

    channel: str
    modality: SensorModality
    sensor_token: str
    num_frames: int
    first_timestamp: int | None = field(default=None)
    last_timestamp: int | None = field(default=None)

    @property
    def duration(self) -> float:
        """Return the time length of the recorded frames in [s]."""
        if self.first_timestamp is None or self.last_timestamp is None:
            return 0.0
        return microseconds2seconds(self.last_timestamp - self.first_timestamp)

    @property
    def fps(self) -> float:
        """Return the average frame rate in [Hz]."""
        return (self.num_frames - 1) / self.duration if 0.0 < self.duration else 0.0


def resolve_channels(
    t4: T4Devkit,
    identifiers: str | Sequence[str] | None = None,
    *,
    modality: SensorModality | None = None,
) -> list[str]:
    """Resolve sensor identifiers into the corresponding channel names.

    Both a channel name, which is case insensitive, and a token of the `sensor` record are
    acceptable as an identifier.

    Args:
        t4 (T4Devkit): T4Devkit instance.
        identifiers (str | Sequence[str] | None, optional): Sensor channel name(s) or
            `sensor` token(s). If None, every channel of the specified modality is resolved.
        modality (SensorModality | None, optional): Specify if you want to restrict
            the sensor modality.

    Raises:
        ValueError: Expecting all identifiers refer to an existing sensor.

    Returns:
        List of the resolved channel names.
    """
    candidates: list[Sensor] = [
        sensor for sensor in t4.sensor if modality is None or sensor.modality == modality
    ]

    if not candidates:
        raise ValueError(
            f"No {modality.value if modality is not None else 'sensor'} channel is "
            f"registered in {t4.dataset_id}."
        )

    if isinstance(identifiers, str):
        identifiers = [identifiers]

    if not identifiers:
        return [sensor.channel for sensor in candidates]

    channel_map = {sensor.channel.lower(): sensor.channel for sensor in candidates}
    token_map = {sensor.token: sensor.channel for sensor in candidates}

    resolved: list[str] = []
    for identifier in identifiers:
        channel = token_map.get(identifier, channel_map.get(identifier.strip().lower()))
        if channel is None:
            raise ValueError(
                f"Unexpected sensor identifier: {identifier}. "
                f"Available channels: {[sensor.channel for sensor in candidates]}"
            )
        if channel not in resolved:
            resolved.append(channel)

    return resolved


def select_sample_data(
    t4: T4Devkit,
    channel: str,
    *,
    start: TimeSpecLike | None = None,
    end: TimeSpecLike | None = None,
    duration: float | None = None,
    max_frames: int | None = None,
) -> list[SampleData]:
    """Select valid `sample_data` records of the specified channel within the specified time range.

    Args:
        t4 (T4Devkit): T4Devkit instance.
        channel (str): Sensor channel name.
        start (TimeSpecLike | None, optional): Start of the time range.
            If None, the first frame of the channel is used.
        end (TimeSpecLike | None, optional): End of the time range.
            If None, `duration` or the last frame of the channel is used.
        duration (float | None, optional): Time length in [s] from `start`.
            This is ignored if `end` is specified.
        max_frames (int | None, optional): Maximum number of records to be selected.

    Returns:
        Selected records ordered by timestamp.
    """
    records = [record for record in t4.sample_data if record.channel == channel and record.is_valid]
    records.sort(key=lambda record: record.timestamp)

    if not records:
        return []

    time_range = TimeRange.resolve(
        start,
        end,
        duration=duration,
        first=records[0].timestamp,
        last=records[-1].timestamp,
    )

    selected = [record for record in records if time_range.contains(record.timestamp)]

    return selected if max_frames is None else selected[:max_frames]


def summarize_channels(t4: T4Devkit) -> list[ChannelSummary]:
    """Summarize sensor data recorded in each channel.

    Args:
        t4 (T4Devkit): T4Devkit instance.

    Returns:
        List of summaries ordered by modality and channel name.
    """
    # NOTE: Only the number of frames and the both ends of the time range are of interest,
    # hence the timestamps themselves are not accumulated.
    counts: dict[str, int] = {sensor.channel: 0 for sensor in t4.sensor}
    firsts: dict[str, int] = {}
    lasts: dict[str, int] = {}
    for record in t4.sample_data:
        if record.channel not in counts or not record.is_valid:
            continue
        channel, timestamp = record.channel, record.timestamp
        counts[channel] += 1
        firsts[channel] = min(firsts.get(channel, timestamp), timestamp)
        lasts[channel] = max(lasts.get(channel, timestamp), timestamp)

    summaries = [
        ChannelSummary(
            channel=sensor.channel,
            modality=sensor.modality,
            sensor_token=sensor.token,
            num_frames=counts[sensor.channel],
            first_timestamp=firsts.get(sensor.channel),
            last_timestamp=lasts.get(sensor.channel),
        )
        for sensor in t4.sensor
    ]

    return sorted(summaries, key=lambda summary: (summary.modality.value, summary.channel))
