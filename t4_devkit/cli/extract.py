from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Callable

import typer
from tabulate import tabulate

from t4_devkit import T4Devkit
from t4_devkit.common.timestamp import microseconds2seconds
from t4_devkit.extract import VideoFormat, extract_video, summarize_channels

from .version import version_callback

cli = typer.Typer(
    name="t4extract",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)

# NOTE: Avoid square brackets in help messages, because `rich` interprets them as markup.
_TIMESTAMP_HELP = (
    "Unix time in seconds or microseconds, an ISO 8601 datetime such as "
    "`2024-01-01T00:00:00`, or an offset in seconds from the first/last frame "
    "such as `+1.5`/`-1.5`."
)

DataRootArgument = Annotated[str, typer.Argument(help="Root directory path to the dataset.")]

RevisionOption = Annotated[
    str | None,
    typer.Option(
        ..., "-rv", "--revision", help="Specify if you want to load the specific version."
    ),
]


@cli.command("video", help="Extract camera images in the specified time range as a video.")
def video(
    data_root: DataRootArgument,
    camera: Annotated[
        list[str] | None,
        typer.Option(
            ...,
            "-c",
            "--camera",
            help="Camera channel name or `sensor` token. "
            "This option can be specified multiple times. "
            "If not specified, every camera channel is extracted.",
        ),
    ] = None,
    output: Annotated[
        str,
        typer.Option(..., "-o", "--output", help="Directory path to save the extracted video(s)."),
    ] = "./output",
    start: Annotated[
        str | None,
        typer.Option(..., "-s", "--start", help=f"Start of the time range. {_TIMESTAMP_HELP}"),
    ] = None,
    end: Annotated[
        str | None,
        typer.Option(..., "-e", "--end", help=f"End of the time range. {_TIMESTAMP_HELP}"),
    ] = None,
    duration: Annotated[
        float | None,
        typer.Option(
            ...,
            "-d",
            "--duration",
            help="Time length in seconds from the start. This is ignored if `--end` is specified.",
        ),
    ] = None,
    fps: Annotated[
        float | None,
        typer.Option(
            ...,
            "-f",
            "--fps",
            help="Frame rate of the output video in Hz. "
            "If not specified, it is estimated from the timestamps of the source frames.",
        ),
    ] = None,
    video_format: Annotated[
        VideoFormat,
        typer.Option(..., "--format", help="Output video format."),
    ] = VideoFormat.MP4,
    scale: Annotated[
        float,
        typer.Option(..., "--scale", help="Scale factor to resize frames."),
    ] = 1.0,
    crf: Annotated[
        int,
        typer.Option(
            ...,
            "--crf",
            help="Constant rate factor of `libx264`, which is only used for MP4. "
            "The smaller value results in the better quality.",
        ),
    ] = 23,
    max_frames: Annotated[
        int | None,
        typer.Option(..., "--max-frames", help="Maximum number of frames to be extracted."),
    ] = None,
    revision: RevisionOption = None,
) -> None:
    t4 = T4Devkit(data_root, revision=revision, verbose=False)

    results = extract_video(
        t4,
        camera,
        output_dir=output,
        start=start,
        end=end,
        duration=duration,
        fps=fps,
        video_format=video_format,
        scale=scale,
        max_frames=max_frames,
        crf=crf,
        verbose=True,
    )

    if not results:
        raise typer.Exit(code=1)

    print(
        tabulate(
            [
                [
                    result.channel,
                    result.num_frames,
                    f"{result.fps:.3f}",
                    f"{result.width}x{result.height}",
                    f"{result.duration:.3f}",
                    result.filepath,
                ]
                for result in results
            ],
            headers=["Channel", "Frames", "FPS", "Resolution", "Duration[s]", "Output"],
            tablefmt="pretty",
        )
    )


@cli.command("list", help="List sensor channels and their time ranges.")
def list_channels(data_root: DataRootArgument, revision: RevisionOption = None) -> None:
    t4 = T4Devkit(data_root, revision=revision, verbose=False)

    summaries = summarize_channels(t4)

    print(
        tabulate(
            [
                [
                    summary.channel,
                    summary.modality.value,
                    summary.sensor_token,
                    summary.num_frames,
                    _or_dash(summary.first_timestamp),
                    _or_dash(summary.last_timestamp),
                    _or_dash(summary.first_timestamp, _format_datetime),
                    f"{summary.duration:.3f}",
                    f"{summary.fps:.3f}",
                ]
                for summary in summaries
            ],
            headers=[
                "Channel",
                "Modality",
                "SensorToken",
                "Frames",
                "First[us]",
                "Last[us]",
                "First(UTC)",
                "Duration[s]",
                "FPS",
            ],
            tablefmt="pretty",
        )
    )


def _or_dash(timestamp: int | None, formatter: Callable[[int], str] = str) -> str:
    """Format a timestamp, or `-` if it is not recorded.

    Args:
        timestamp (int | None): Unix time in [us].
        formatter (Callable[[int], str], optional): Formatter of a recorded timestamp.

    Returns:
        Formatted timestamp, or `-` if the input is None.
    """
    return "-" if timestamp is None else formatter(timestamp)


def _format_datetime(timestamp: int) -> str:
    """Format a unix time in [us] as an ISO 8601 datetime in UTC.

    Args:
        timestamp (int): Unix time in [us].

    Returns:
        Formatted datetime.
    """
    return datetime.fromtimestamp(microseconds2seconds(timestamp), tz=timezone.utc).isoformat()


@cli.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show the application version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    pass
