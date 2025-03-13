# Glyphs

> [!WARNING]  
> This project is a work in-progress and only partially works.

Glyphs is a command line application to extract hardcoded subtitles from videos.

## Dependencies

This project uses the Python [uv](https://github.com/astral-sh/uv) package
manager to install the necessary Python interpreter and dependencies.

## Usage

Install this a [uv tool](https://docs.astral.sh/uv/concepts/tools/) that can be 
called from anywhere within your command line. The `-e` flag allows local
changes to be immediately usable with the tool without reinstallation.

``` sh
$ uv tool install -e
Installed 1 executable: glyphs

$ glyphs <path_to_video>
Processing video: 100%|████████████████████████████████████████████| 781/781 [00:14<00:00, 53.79it/s]
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

