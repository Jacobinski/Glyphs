from .util import count_frames, REWIND_FRAME_COUNT
from unittest.mock import patch, MagicMock

@patch("cv2.VideoCapture")
def test_count_frames(mock_video_capture):
    num_frames = 200

    mock_video = MagicMock()
    mock_video_capture.return_value = mock_video
    mock_video.get.return_value = num_frames
    mock_video.read.side_effect = [(True,)] * REWIND_FRAME_COUNT + [(False,)]

    assert num_frames == count_frames("/path/to/video.mkv")
