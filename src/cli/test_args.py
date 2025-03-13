import pytest
import sys
import argparse

from unittest.mock import patch
from .args import Arguments, parse_arguments

PROGRAM_NAME_ARGV0 = ["glyphs"]

cases = [
    (["/foo/bar"], [], Arguments(["/foo/bar"], False)), # Single file
    (["/foo/bar", "/foo/baz"], [], Arguments(["/foo/bar", "/foo/baz"], False)), # Multiple files
    (["/foo/bar"], ["--no-verbose"], Arguments(["/foo/bar"], False)), # Explicit non-verbose
    (["/foo/bar"], ["--verbose"], Arguments(["/foo/bar"], True)), # Explicit verbose
]

@pytest.mark.parametrize("files,verbose,want", cases)
def test_parse_arguments(files, verbose, want):
    test_args = PROGRAM_NAME_ARGV0 + files + verbose
    with patch.object(sys, 'argv', test_args):
        assert want == parse_arguments()

@patch("sys.argv")
def test_parse_arguments_requires_file(mock_argv):
    mock_argv.return_value = PROGRAM_NAME_ARGV0 + [] + ["--verbose"]
    with pytest.raises(SystemExit) as error:
        parse_arguments()
    assert error.value.code == 2  # Invalid CLI command code
