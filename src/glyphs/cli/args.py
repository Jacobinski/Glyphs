import argparse

from collections import namedtuple

Arguments = namedtuple('Arguments', [
    'files',  # Array of paths to video files
    'verbose' # Boolean controlling output of verbose flags
])

def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(
        prog='glyphs',
        description='A tool to extract hardcoded subtitles from videos.',
    )
    parser.add_argument(
        "files",
        help="paths to video file(s)",
        nargs='+',  # Allow multiple file arguments
        type=str,
    )
    parser.add_argument(
        "--verbose",
        help="Enable additional logs",
        default=False,
        action=argparse.BooleanOptionalAction  # Allows using --verbose for true and --no-verbose for false
    )
    args = vars(parser.parse_args())
    return Arguments(args["files"], args["verbose"])
