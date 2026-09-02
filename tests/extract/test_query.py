from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from t4_devkit.extract import resolve_channels, select_sample_data, summarize_channels
from t4_devkit.schema import SensorModality

if TYPE_CHECKING:
    from t4_devkit import T4Devkit

FIRST = 1704067200000000
LAST = 1704067202000000
CAM_FRONT_TOKEN = "391237c9a8cb743864cdfd764e6c8627"


class TestResolveChannels:
    """Test cases for `resolve_channels`."""

    def test_resolve_all_channels(self, t4: T4Devkit) -> None:
        assert set(resolve_channels(t4)) == {"CAM_FRONT", "CAM_BACK", "LIDAR_TOP"}

    def test_resolve_by_modality(self, t4: T4Devkit) -> None:
        assert set(resolve_channels(t4, modality=SensorModality.CAMERA)) == {
            "CAM_FRONT",
            "CAM_BACK",
        }

    @pytest.mark.parametrize(
        "identifiers",
        ["CAM_FRONT", "cam_front", ["CAM_FRONT"], [CAM_FRONT_TOKEN]],
    )
    def test_resolve_by_identifier(self, t4: T4Devkit, identifiers) -> None:
        assert resolve_channels(t4, identifiers) == ["CAM_FRONT"]

    def test_resolve_deduplicates(self, t4: T4Devkit) -> None:
        assert resolve_channels(t4, ["CAM_FRONT", CAM_FRONT_TOKEN]) == ["CAM_FRONT"]

    def test_resolve_with_empty_identifiers(self, t4: T4Devkit) -> None:
        assert set(resolve_channels(t4, [], modality=SensorModality.CAMERA)) == {
            "CAM_FRONT",
            "CAM_BACK",
        }

    def test_resolve_with_unknown_identifier(self, t4: T4Devkit) -> None:
        with pytest.raises(ValueError):
            resolve_channels(t4, "CAM_UNKNOWN")

    def test_resolve_with_mismatched_modality(self, t4: T4Devkit) -> None:
        with pytest.raises(ValueError):
            resolve_channels(t4, "LIDAR_TOP", modality=SensorModality.CAMERA)

    def test_resolve_with_unregistered_modality(self, t4: T4Devkit) -> None:
        with pytest.raises(ValueError):
            resolve_channels(t4, modality=SensorModality.RADAR)


class TestSelectSampleData:
    """Test cases for `select_sample_data`."""

    def test_select_all(self, t4: T4Devkit) -> None:
        records = select_sample_data(t4, "CAM_FRONT")

        assert [record.timestamp for record in records] == [FIRST, FIRST + 1000000, LAST]

    def test_select_with_absolute_range(self, t4: T4Devkit) -> None:
        records = select_sample_data(t4, "CAM_FRONT", start=FIRST + 1000000, end=LAST)

        assert [record.timestamp for record in records] == [FIRST + 1000000, LAST]

    def test_select_with_relative_range(self, t4: T4Devkit) -> None:
        records = select_sample_data(t4, "CAM_FRONT", start="+1.0", end="-0.5")

        assert [record.timestamp for record in records] == [FIRST + 1000000]

    def test_select_with_duration(self, t4: T4Devkit) -> None:
        records = select_sample_data(t4, "CAM_FRONT", duration=1.0)

        assert [record.timestamp for record in records] == [FIRST, FIRST + 1000000]

    def test_select_with_max_frames(self, t4: T4Devkit) -> None:
        records = select_sample_data(t4, "CAM_FRONT", max_frames=2)

        assert [record.timestamp for record in records] == [FIRST, FIRST + 1000000]

    def test_select_out_of_range(self, t4: T4Devkit) -> None:
        records = select_sample_data(t4, "CAM_FRONT", start=LAST + 1, end=LAST + 100)

        assert records == []

    def test_select_unknown_channel(self, t4: T4Devkit) -> None:
        records = select_sample_data(t4, "CAM_UNKNOWN")

        assert records == []


class TestSummarizeChannels:
    """Test cases for `summarize_channels`."""

    def test_summarize(self, t4: T4Devkit) -> None:
        summaries = {summary.channel: summary for summary in summarize_channels(t4)}

        assert set(summaries.keys()) == {"CAM_FRONT", "CAM_BACK", "LIDAR_TOP"}

        summary = summaries["CAM_FRONT"]
        assert summary.modality == SensorModality.CAMERA
        assert summary.num_frames == 3
        assert (summary.first_timestamp, summary.last_timestamp) == (FIRST, LAST)
        assert summary.duration == pytest.approx(2.0)
        assert summary.fps == pytest.approx(1.0)

    def test_summarize_is_sorted_by_modality_and_channel(self, t4: T4Devkit) -> None:
        summaries = summarize_channels(t4)

        assert [summary.channel for summary in summaries] == [
            "CAM_BACK",
            "CAM_FRONT",
            "LIDAR_TOP",
        ]
