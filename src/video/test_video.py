from datetime import timedelta
import cv2
import pytest

from .video import Video
from unittest.mock import patch, MagicMock

@patch("cv2.VideoCapture")
def test_video_init(mock_video_capture):
    mock_video = MagicMock()
    mock_video_capture.return_value = mock_video

    path = "/foo/bar/video.mp4"
    start_idx, stop_idx = 100, 200
    Video(path, start_idx, stop_idx)

    mock_video_capture.assert_called_once_with(path)
    mock_video.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, start_idx)


@patch("cv2.VideoCapture")
def test_video_delete(mock_video_capture):
    mock_video = MagicMock()
    mock_video_capture.return_value = mock_video

    path = "/foo/bar/video.mp4"
    start_idx, stop_idx = 100, 200
    video = Video(path, start_idx, stop_idx)
    del video

    mock_video.release.assert_called_once_with()

@patch("cv2.VideoCapture")
def test_video_iteration(mock_video_capture):
    mock_video = MagicMock()
    mock_video_capture.return_value = mock_video
    mock_video.read.return_value = (True, MagicMock())

    path = "/foo/bar/video.mp4"
    start_idx, stop_idx = 100, 200
    video = Video(path, start_idx, stop_idx)

    for i in range(start_idx, stop_idx):
        assert video.frame_number() == i
        next(video)

    with pytest.raises(StopIteration):
        next(video)

@patch("cv2.VideoCapture")
def test_video_time(mock_video_capture):
    mock_video = MagicMock()
    mock_video_capture.return_value = mock_video
    mock_video.get.return_value = 1000

    path = "/foo/bar/video.mp4"
    start_idx, stop_idx = 100, 200
    video = Video(path, start_idx, stop_idx)
    assert video.time() == timedelta(milliseconds=1000)

@patch("cv2.VideoCapture")
def test_video_frame_height(mock_video_capture):
    mock_video = MagicMock()
    mock_video_capture.return_value = mock_video
    mock_video.get.return_value = 400.0

    path = "/foo/bar/video.mp4"
    start_idx, stop_idx = 100, 200
    video = Video(path, start_idx, stop_idx)

    assert video.frame_height() == 400
