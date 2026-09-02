from __future__ import annotations

import os
import os.path as osp
import shutil
import subprocess
import warnings
from enum import Enum, unique
from tempfile import TemporaryFile
from typing import TYPE_CHECKING, Iterator, Sequence

import numpy as np
from attrs import define
from PIL import Image
from tqdm import tqdm

from t4_devkit.common.timestamp import microseconds2seconds
from t4_devkit.schema import SensorModality

from .query import resolve_channels, select_sample_data
from .timespec import TimeSpec

if TYPE_CHECKING:
    from t4_devkit.schema import SampleData
    from t4_devkit.tier4 import T4Devkit
    from t4_devkit.typing import PathLike

    from .timespec import TimeSpecLike

__all__ = [
    "VideoFormat",
    "VideoExtractionResult",
    "extract_video",
    "estimate_fps",
    "resolve_ffmpeg",
]

# Frame rate used when it cannot be estimated from timestamps.
_DEFAULT_FPS: float = 10.0


@unique
class VideoFormat(str, Enum):
    """An enum to represent supported video formats.

    Attributes:
        MP4: MP4 container encoded by `ffmpeg`.
        GIF: Animated GIF, which does not require `ffmpeg`.
    """

    MP4 = "mp4"
    GIF = "gif"

    def as_ext(self) -> str:
        """Return the value as file extension."""
        return f".{self.value}"


@define
class VideoExtractionResult:
    """A dataclass to represent the result of a video extraction.

    Attributes:
        channel (str): Camera channel name.
        filepath (str): File path of the extracted video.
        num_frames (int): The number of the encoded frames.
        fps (float): Frame rate of the extracted video in [Hz].
        width (int): Frame width in [px].
        height (int): Frame height in [px].
        start_timestamp (int): Unix time of the first frame in [us].
        end_timestamp (int): Unix time of the last frame in [us].
    """

    channel: str
    filepath: str
    num_frames: int
    fps: float
    width: int
    height: int
    start_timestamp: int
    end_timestamp: int

    @property
    def duration(self) -> float:
        """Return the time length of the source frames in [s]."""
        return microseconds2seconds(self.end_timestamp - self.start_timestamp)


def extract_video(
    t4: T4Devkit,
    camera: str | Sequence[str] | None = None,
    *,
    output_dir: PathLike,
    start: TimeSpecLike | None = None,
    end: TimeSpecLike | None = None,
    duration: float | None = None,
    fps: float | None = None,
    video_format: str | VideoFormat = VideoFormat.MP4,
    scale: float = 1.0,
    max_frames: int | None = None,
    crf: int = 23,
    ffmpeg: PathLike | None = None,
    verbose: bool = False,
) -> list[VideoExtractionResult]:
    """Extract camera images within the specified time range as a video file.

    A video file is generated for each camera channel, and saved as
    `<output_dir>/<CHANNEL>.<EXT>`.

    Args:
        t4 (T4Devkit): T4Devkit instance.
        camera (str | Sequence[str] | None, optional): Camera channel name(s) or `sensor` token(s).
            If None, every camera channel is extracted.
        output_dir (PathLike): Directory path to save the extracted video(s).
        start (TimeSpecLike | None, optional): Start of the time range.
            If None, the first frame of the channel is used.
        end (TimeSpecLike | None, optional): End of the time range.
            If None, `duration` or the last frame of the channel is used.
        duration (float | None, optional): Time length in [s] from `start`.
            This is ignored if `end` is specified.
        fps (float | None, optional): Frame rate of the output video in [Hz].
            If None, it is estimated from the timestamps of the source frames.
        video_format (str | VideoFormat, optional): Output video format.
        scale (float, optional): Scale factor to resize frames.
        max_frames (int | None, optional): Maximum number of frames to be encoded.
        crf (int, optional): Constant rate factor of `libx264`, which is only used for MP4.
            The smaller value results in the better quality.
        ffmpeg (PathLike | None, optional): Path to the `ffmpeg` executable.
            If None, it is searched automatically.
        verbose (bool, optional): Whether to display progress.

    Raises:
        ValueError: Expecting `scale` is positive and all cameras refer to an existing sensor.
        FileNotFoundError: Expecting the `ffmpeg` executable is found, if MP4 is specified.

    Returns:
        List of the results for each camera channel.
    """
    if scale <= 0.0:
        raise ValueError(f"`scale` must be positive, but got: {scale}")

    video_format = VideoFormat(video_format)
    is_mp4 = video_format == VideoFormat.MP4
    channels = resolve_channels(t4, camera, modality=SensorModality.CAMERA)

    # NOTE: Resolve the user inputs before loading any frame so that a malformed expression
    # or a missing ffmpeg is reported without doing useless work.
    start = TimeSpec.parse(start) if start is not None else None
    end = TimeSpec.parse(end) if end is not None else None
    executable = resolve_ffmpeg(ffmpeg) if is_mp4 else None

    os.makedirs(output_dir, exist_ok=True)

    results: list[VideoExtractionResult] = []
    for channel in channels:
        records = select_sample_data(
            t4, channel, start=start, end=end, duration=duration, max_frames=max_frames
        )

        if not records:
            warnings.warn(f"No frame is found in the specified time range: {channel}")
            continue

        filepath = osp.join(str(output_dir), f"{channel}{video_format.as_ext()}")
        frame_fps = fps if fps is not None else estimate_fps([r.timestamp for r in records])
        size = _resolve_size(t4, records[0], scale=scale, even=is_mp4)

        frames = _iter_frames(t4, records, size=size, channel=channel, verbose=verbose)
        if is_mp4:
            _save_as_mp4(frames, filepath, fps=frame_fps, size=size, crf=crf, ffmpeg=executable)
        else:
            _save_as_gif(frames, filepath, fps=frame_fps)

        results.append(
            VideoExtractionResult(
                channel=channel,
                filepath=filepath,
                num_frames=len(records),
                fps=frame_fps,
                width=size[0],
                height=size[1],
                start_timestamp=records[0].timestamp,
                end_timestamp=records[-1].timestamp,
            )
        )

    return results


def estimate_fps(timestamps: Sequence[int]) -> float:
    """Estimate a frame rate from the timestamps of consecutive frames.

    Args:
        timestamps (Sequence[int]): Unix times of the frames in [us].

    Returns:
        Estimated frame rate in [Hz], or 10.0 if it cannot be estimated.
    """
    deltas = np.diff(np.asarray(timestamps, dtype=np.float64))
    deltas = deltas[deltas > 0.0]

    if len(deltas) == 0:
        return _DEFAULT_FPS

    return round(1.0 / microseconds2seconds(float(np.median(deltas))), 3)


def resolve_ffmpeg(executable: PathLike | None = None) -> str:
    """Resolve the path to the `ffmpeg` executable.

    If `executable` is not specified, the binary bundled in `imageio-ffmpeg` is preferred,
    and then the one installed on the system is searched.

    Args:
        executable (PathLike | None, optional): Path to the `ffmpeg` executable.

    Raises:
        FileNotFoundError: Expecting the `ffmpeg` executable is found.

    Returns:
        Path to the `ffmpeg` executable.
    """
    if executable is not None:
        executable = str(executable)
        found = shutil.which(executable) or (executable if osp.isfile(executable) else None)
        if found is None:
            raise FileNotFoundError(f"ffmpeg is not found: {executable}")
        return found

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        pass

    found = shutil.which("ffmpeg")
    if found is None:
        raise FileNotFoundError(
            "ffmpeg is not found. Install it with `pip install 't4-devkit[video]'`, "
            "install ffmpeg on your system, or export a GIF instead of a MP4."
        )

    return found


def _resolve_size(
    t4: T4Devkit,
    record: SampleData,
    *,
    scale: float,
    even: bool,
) -> tuple[int, int]:
    """Resolve the frame size of an output video.

    Args:
        t4 (T4Devkit): T4Devkit instance.
        record (SampleData): The first `sample_data` record of the channel.
        scale (float): Scale factor to resize frames.
        even (bool): Whether to round the size down to even numbers, which is required by
            the `yuv420p` pixel format.

    Returns:
        Frame width and height in [px].
    """
    width, height = record.width, record.height
    if width <= 0 or height <= 0:
        # NOTE: Some datasets do not fill the resolution of camera data.
        with Image.open(t4.get_sample_data_path(record.token)) as image:
            width, height = image.size

    width, height = max(1, int(width * scale)), max(1, int(height * scale))

    if even:
        width, height = max(2, width - width % 2), max(2, height - height % 2)

    return width, height


def _iter_frames(
    t4: T4Devkit,
    records: Sequence[SampleData],
    *,
    size: tuple[int, int],
    channel: str,
    verbose: bool,
) -> Iterator[Image.Image]:
    """Load images of the specified records as RGB frames of the same size.

    Args:
        t4 (T4Devkit): T4Devkit instance.
        records (Sequence[SampleData]): `sample_data` records ordered by timestamp.
        size (tuple[int, int]): Frame width and height in [px].
        channel (str): Camera channel name, which is displayed on the progress bar.
        verbose (bool): Whether to display progress.

    Yields:
        Loaded frames.
    """
    for record in tqdm(records, desc=f"Extracting {channel}", disable=not verbose):
        with Image.open(t4.get_sample_data_path(record.token)) as image:
            # NOTE: `draft` lets a JPEG source be decoded at a reduced scale directly, which is
            # much faster than decoding it at the full resolution and downscaling afterwards.
            image.draft("RGB", size)
            frame = image if image.mode == "RGB" else image.convert("RGB")
            # NOTE: `resize`/`copy` detach the frame from the file, which is closed on exit.
            yield frame.resize(size, Image.BILINEAR) if frame.size != size else frame.copy()


def _save_as_mp4(
    frames: Iterator[Image.Image],
    filepath: str,
    *,
    fps: float,
    size: tuple[int, int],
    crf: int,
    ffmpeg: str,
) -> None:
    """Encode frames into a MP4 file by piping them into `ffmpeg`.

    Args:
        frames (Iterator[Image.Image]): Frames to be encoded.
        filepath (str): File path to save the video.
        fps (float): Frame rate in [Hz].
        size (tuple[int, int]): Frame width and height in [px].
        crf (int): Constant rate factor of `libx264`.
        ffmpeg (str): Path to the `ffmpeg` executable.

    Raises:
        RuntimeError: Expecting `ffmpeg` exits successfully.
    """
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{size[0]}x{size[1]}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        filepath,
    ]  # fmt: skip

    # NOTE: Redirect stderr to a temporary file so that ffmpeg never blocks on a full pipe
    # while frames are being written into its stdin.
    with TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
        )

        try:
            for frame in frames:
                process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            # NOTE: ffmpeg has already exited, and the reason is reported by its stderr.
            pass
        finally:
            if not process.stdin.closed:
                process.stdin.close()
            returncode = process.wait()

        stderr_file.seek(0)
        stderr = stderr_file.read().decode("utf-8", errors="replace").strip()

    if returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to encode {filepath} with the exit code {returncode}: {stderr}"
        )


def _save_as_gif(frames: Iterator[Image.Image], filepath: str, *, fps: float) -> None:
    """Encode frames into an animated GIF file.

    Args:
        frames (Iterator[Image.Image]): Frames to be encoded.
        filepath (str): File path to save the video.
        fps (float): Frame rate in [Hz].

    Raises:
        ValueError: Expecting at least a single frame is given.
    """
    first = next(frames, None)

    if first is None:
        raise ValueError("No frame is given to be encoded.")

    # NOTE: Pillow consumes `append_images` lazily, hence the frames are not buffered.
    first.save(
        filepath,
        save_all=True,
        append_images=frames,
        duration=max(1, int(round(1000.0 / fps))),
        loop=0,
    )
