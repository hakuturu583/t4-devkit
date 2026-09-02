`t4extract` extracts raw sensor data recorded in a dataset from command line.

It cuts out camera images in a particular time range as a video file.

```shell
$ t4extract -h

 Usage: t4extract [OPTIONS] COMMAND [ARGS]...

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --version             -v        Show the application version and exit.                                           │
│ --install-completion            Install completion for the current shell.                                        │
│ --show-completion               Show completion for the current shell, to copy it or customize the installation. │
│ --help                -h        Show this message and exit.                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ video   Extract camera images in the specified time range as a video.                                            │
│ list    List sensor channels and their time ranges.                                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Shell Completion

Run the following command to install completion, and reload shell.

```{ .shell .copy }
t4extract --install-completion
```

## Specifying Timestamps

The `video` command provides the following options to specify a time range,
which is inclusive at both ends:

| Option             | Description                                                         |
| :----------------- | :------------------------------------------------------------------ |
| `-s`, `--start`    | Start of the time range. Defaults to the first frame of a channel.  |
| `-e`, `--end`      | End of the time range. Defaults to the last frame of a channel.     |
| `-d`, `--duration` | Time length [s] from the start. This is ignored if `--end` is used. |
| `--max-frames`     | Maximum number of frames to be extracted.                           |

A timestamp can be specified in the following formats:

| Format               | Example                | Description                                     |
| :------------------- | :--------------------- | :---------------------------------------------- |
| Unix time [us]       | `1704067200000000`     | Timestamps stored in `sample_data.json`.        |
| Unix time [s]        | `1704067200.5`         | Values less than `1e12` are interpreted as [s]. |
| ISO 8601 datetime    | `2024-01-01T00:00:00Z` | A naive datetime is interpreted as UTC.         |
| Offset from the head | `+1.5`                 | 1.5[s] after the first frame of the channel.    |
| Offset from the tail | `-1.5`                 | 1.5[s] before the last frame of the channel.    |

## Usages

### List Sensor Channels

Before extracting data, this command shows the available sensor channels, their `sensor` tokens
and the recorded time range, which are useful to determine the arguments of the `video` command.

```shell
t4extract list <DATA_ROOT>
```

```shell
+-----------+----------+----------------------------------+--------+------------------+------------------+---------------------------+-------------+-------+
|  Channel  | Modality |           SensorToken            | Frames |    First[us]     |     Last[us]     |        First(UTC)         | Duration[s] |  FPS  |
+-----------+----------+----------------------------------+--------+------------------+------------------+---------------------------+-------------+-------+
| CAM_BACK  |  camera  | 99998d7e7be2bbca0f2ce80688d8a63a |   3    | 1704067200000000 | 1704067202000000 | 2024-01-01T00:00:00+00:00 |    2.000    | 1.000 |
| CAM_FRONT |  camera  | 391237c9a8cb743864cdfd764e6c8627 |   3    | 1704067200000000 | 1704067202000000 | 2024-01-01T00:00:00+00:00 |    2.000    | 1.000 |
| LIDAR_TOP |  lidar   | 0132c9748dfe2777a146807f8a95a6c4 |   3    | 1704067200000000 | 1704067202000000 | 2024-01-01T00:00:00+00:00 |    2.000    | 1.000 |
+-----------+----------+----------------------------------+--------+------------------+------------------+---------------------------+-------------+-------+
```

### Video

This command encodes camera images into a video file, which is saved as
`<OUTPUT>/<CHANNEL>.<EXT>`.

A camera is specified by its channel name, which is case insensitive, or its `sensor` token
with `-c [--camera]` option. If it is not specified, every camera channel is extracted.

For options, run `t4extract video -h`.

```shell
t4extract video <DATA_ROOT> -c <CAMERA> -o <OUTPUT_DIR> [OPTIONS]
```

For example, the following command cuts out 5.0[s] of `CAM_FRONT` starting from the specified
timestamp:

```shell
t4extract video <DATA_ROOT> -c CAM_FRONT -s 1704067200000000 -d 5.0 -o ./output
```

Multiple cameras can be specified at once, and a video file is generated for each of them:

```shell
t4extract video <DATA_ROOT> -c CAM_FRONT -c CAM_BACK -o ./output
```

The frame rate is estimated from the timestamps of the source frames.
You can overwrite it with `-f [--fps]` option, and resize frames with `--scale` option:

```shell
t4extract video <DATA_ROOT> -c CAM_FRONT -f 10.0 --scale 0.5 -o ./output
```

<!-- prettier-ignore-start -->
!!! NOTE
    Encoding a MP4 file requires `ffmpeg`. It is bundled in the `video` extra:

    <!-- markdownlint-disable MD046 -->
    ```shell
    pip install "t4-devkit[video] @ git+https://github.com/tier4/t4-devkit.git"
    ```

    Otherwise, install `ffmpeg` on your system, specify its path with `--ffmpeg` option,
    or export an animated GIF with `--format gif`, which does not require `ffmpeg`:

    <!-- markdownlint-disable MD046 -->
    ```shell
    t4extract video <DATA_ROOT> -c CAM_FRONT --format gif -o ./output
    ```
<!-- prettier-ignore-end -->

## API Usage

The same operations are available with [`T4Devkit`](../apis/tier4.md) as follows:

```python
from t4_devkit import T4Devkit

t4 = T4Devkit("data/tier4")

# Cut out 5.0[s] of `CAM_FRONT` as a video
t4.extract_video("CAM_FRONT", output_dir="./output", start="+1.0", duration=5.0)
```

For details, please refer to [`t4_devkit.extract`](../apis/extract.md).
