from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from t4_devkit.cli.extract import cli
from t4_devkit.extract.video import resolve_ffmpeg

runner = CliRunner()

DATA_ROOT = str(Path(__file__).parents[1] / "sample/t4dataset")


def _has_ffmpeg() -> bool:
    try:
        resolve_ffmpeg()
    except FileNotFoundError:
        return False
    return True


class TestT4Extract:
    """Test cases for the `t4extract` CLI."""

    def test_list(self) -> None:
        result = runner.invoke(cli, ["list", DATA_ROOT])

        assert result.exit_code == 0, result.output
        assert "CAM_FRONT" in result.output
        assert "LIDAR_TOP" in result.output

    def test_video(self, tmp_path: Path) -> None:
        result = runner.invoke(
            cli,
            [
                "video",
                DATA_ROOT,
                "-c",
                "CAM_FRONT",
                "-o",
                str(tmp_path),
                "--format",
                "gif",
                "-s",
                "+0.0",
                "-d",
                "1.0",
            ],
        )

        assert result.exit_code == 0, result.output
        assert (tmp_path / "CAM_FRONT.gif").exists()

    @pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg is not installed.")
    def test_video_as_mp4(self, tmp_path: Path) -> None:
        result = runner.invoke(cli, ["video", DATA_ROOT, "-c", "CAM_FRONT", "-o", str(tmp_path)])

        assert result.exit_code == 0, result.output
        assert (tmp_path / "CAM_FRONT.mp4").exists()

    def test_video_with_unknown_camera(self, tmp_path: Path) -> None:
        result = runner.invoke(cli, ["video", DATA_ROOT, "-c", "CAM_UNKNOWN", "-o", str(tmp_path)])

        assert result.exit_code != 0
