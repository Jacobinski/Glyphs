# Glyphs

> [!WARNING]  
> This project is a work in-progress and does not work perfectly. It sometimes
> gets stuck on large videos and also frequently creates artifacts in the output
> `.srt` files.

Glyphs is a command line application to extract hardcoded subtitles from videos.

## TODO

- [ ] Add user-friendly output to log where the `.srt` file was saved
- [ ] Add heuristics to detect subtitles vs random text in videos
- [ ] Add more logic/heuristics to guard against artifacts
- [ ] Add TUI showing location of subtitles and allowing for user clean-up.
- [ ] Add documentation showing how to run tool on multiple files simultaneously
- [ ] Add contributor guide

## Dependencies

This project uses the Python [uv](https://github.com/astral-sh/uv) package
manager to install the necessary Python interpreter and dependencies.

## Usage

Install `glyphs` via the [uv tool](https://docs.astral.sh/uv/concepts/tools/) 
command. This adds the command to your `PATH` variable, allowing it to be called
from anywhere with your command line.

``` sh
$ uv tool install -e .
Installed 1 executable: glyphs
```

The `glyphs` command can be given a video file and will use OCR to watch the
video and convert all subtitles into an `.srt` file within the same directory
as the video. This `.srt` file can be used by any standard video player to
display subtitles for the video.

```
$ glyphs <path_to_video>
Processing video: 100%|██████████████████████████████████████| 781/781 [00:14<00:00, 53.79it/s]
```

## Development

### Running Tests

We use [pytest](https://docs.pytest.org/en/stable/) as our testing framework.
Run all unit tests using the following command:

``` sh
$ uv run pytest
======================================== test session starts ========================================platform darwin -- Python 3.12.8, pytest-8.3.5, pluggy-1.5.0
rootdir: /Users/jacobbudzis/Code/Glyphs
configfile: pyproject.toml
plugins: anyio-4.8.0
collected 10 items                                                                                  

src/glyphs/cli/test_args.py .....                                                             [ 50%]
src/glyphs/video/test_video.py .....                                                          [100%]

======================================== 10 passed in 0.28s =========================================
```

