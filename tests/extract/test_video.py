from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from t4_devkit.extract import VideoFormat, extract_video
from t4_devkit.extract.video import estimate_fps, resolve_ffmpeg

if TYPE_CHECKING:
    from t4_devkit import T4Devkit

FIRST = 1704067200000000
LAST = 1704067202000000


def _has_ffmpeg() -> bool:
    try:
        resolve_ffmpeg()
    except FileNotFoundError:
        return False
    return True


class TestEstimateFps:
    """Test cases for `estimate_fps`."""

    @pytest.mark.parametrize(
        ("timestamps", "expect"),
        [
            ([0, 100000, 200000], 10.0),
            ([0, 100000, 200000, 400000], 10.0),  # NOTE: the median delta is used
            ([0], 10.0),  # NOTE: fallback to the default
            ([100, 100], 10.0),  # NOTE: fallback to the default
            ([0, 50000], 20.0),
        ],
    )
    def test_estimate(self, timestamps: list[int], expect: float) -> None:
        assert estimate_fps(timestamps) == pytest.approx(expect)


class TestExtractVideo:
    """Test cases for `extract_video`."""

    def test_extract_as_gif(self, t4: T4Devkit, tmp_path: Path) -> None:
        results = extract_video(t4, "CAM_FRONT", output_dir=tmp_path, video_format=VideoFormat.GIF)

        assert len(results) == 1

        result = results[0]
        assert result.channel == "CAM_FRONT"
        assert result.filepath == str(tmp_path / "CAM_FRONT.gif")
        assert result.num_frames == 3
        assert (result.width, result.height) == (800, 600)
        assert (result.start_timestamp, result.end_timestamp) == (FIRST, LAST)
        assert result.fps == pytest.approx(1.0)
        assert result.duration == pytest.approx(2.0)

        with Image.open(result.filepath) as image:
            # NOTE: Pillow merges consecutive frames which are identical to each other,
            # hence `n_frames` is not necessarily the same as the number of source frames.
            assert image.format == "GIF"
            assert image.size == (800, 600)

    def test_extract_every_camera(self, t4: T4Devkit, tmp_path: Path) -> None:
        results = extract_video(t4, output_dir=tmp_path, video_format="gif")

        assert {result.channel for result in results} == {"CAM_FRONT", "CAM_BACK"}

    def test_extract_with_time_range(self, t4: T4Devkit, tmp_path: Path) -> None:
        results = extract_video(
            t4, "CAM_FRONT", output_dir=tmp_path, start="+1.0", video_format="gif"
        )

        assert results[0].num_frames == 2
        assert results[0].start_timestamp == FIRST + 1000000

    def test_extract_with_scale(self, t4: T4Devkit, tmp_path: Path) -> None:
        results = extract_video(t4, "CAM_FRONT", output_dir=tmp_path, scale=0.5, video_format="gif")

        assert (results[0].width, results[0].height) == (400, 300)

        with Image.open(results[0].filepath) as image:
            assert image.size == (400, 300)

    def test_extract_with_explicit_fps(self, t4: T4Devkit, tmp_path: Path) -> None:
        results = extract_video(t4, "CAM_FRONT", output_dir=tmp_path, fps=30.0, video_format="gif")

        assert results[0].fps == pytest.approx(30.0)

    def test_extract_with_max_frames(self, t4: T4Devkit, tmp_path: Path) -> None:
        results = extract_video(
            t4, "CAM_FRONT", output_dir=tmp_path, max_frames=1, video_format="gif"
        )

        assert results[0].num_frames == 1

    def test_extract_out_of_range(self, t4: T4Devkit, tmp_path: Path) -> None:
        with pytest.warns(UserWarning):
            results = extract_video(
                t4,
                "CAM_FRONT",
                output_dir=tmp_path,
                start=LAST + 1,
                end=LAST + 100,
                video_format="gif",
            )

        assert results == []

    def test_extract_with_invalid_scale(self, t4: T4Devkit, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            extract_video(t4, "CAM_FRONT", output_dir=tmp_path, scale=0.0)

    def test_extract_with_lidar_channel(self, t4: T4Devkit, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            extract_video(t4, "LIDAR_TOP", output_dir=tmp_path, video_format="gif")

    @pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg is not installed.")
    def test_extract_as_mp4(self, t4: T4Devkit, tmp_path: Path) -> None:
        results = extract_video(t4, "CAM_FRONT", output_dir=tmp_path)

        assert len(results) == 1

        result = results[0]
        assert result.filepath == str(tmp_path / "CAM_FRONT.mp4")
        assert (tmp_path / "CAM_FRONT.mp4").stat().st_size > 0

    def test_extract_as_gif_keeps_odd_size(self, t4: T4Devkit, tmp_path: Path) -> None:
        results = extract_video(
            t4, "CAM_FRONT", output_dir=tmp_path, scale=0.999, video_format="gif"
        )

        assert (results[0].width, results[0].height) == (799, 599)

    @pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg is not installed.")
    def test_extract_as_mp4_with_odd_size(self, t4: T4Devkit, tmp_path: Path) -> None:
        """Test that the frame size is rounded down to even numbers for `yuv420p`."""
        results = extract_video(t4, "CAM_FRONT", output_dir=tmp_path, scale=0.999)

        assert (results[0].width, results[0].height) == (798, 598)

    @pytest.mark.skipif(not Path("/bin/false").is_file(), reason="`/bin/false` is not available.")
    def test_extract_with_failing_ffmpeg(self, t4: T4Devkit, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError):
            extract_video(t4, "CAM_FRONT", output_dir=tmp_path, ffmpeg="/bin/false")


class TestResolveFfmpeg:
    """Test cases for `resolve_ffmpeg`."""

    def test_resolve_with_unknown_executable(self) -> None:
        with pytest.raises(FileNotFoundError):
            resolve_ffmpeg("t4-devkit-unknown-ffmpeg")

    def test_resolve_with_existing_executable(self) -> None:
        executable = shutil.which("python3")

        assert resolve_ffmpeg(executable) == executable
