import cv2

from datetime import timedelta

from glyphs.timestamp import timestamp

class Video:
    __frame_number: int = 0
    __stop_index: int = 0

    def __init__(self, file_path: str, start_idx: int, stop_idx: int):
        v = cv2.VideoCapture(file_path)
        self.__video = v
        self.__frame_number = start_idx
        self.__stop_index = stop_idx
        v.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    def __del__(self):
        self.__video.release()

    def __iter__(self):
        return self

    def __next__(self):
        if self.__frame_number >= self.__stop_index:
            raise StopIteration
        success, frame = self.__video.read()
        assert success  # If this fails, our __stop_index is invalid
        self.__frame_number += 1
        return frame

    # TODO: Convert this to a timestamp and propogate it through the system
    def time(self) -> timestamp:
        ms = self.__video.get(cv2.CAP_PROP_POS_MSEC)
        return timestamp(milliseconds=ms)

    def frame_number(self) -> int:
        return self.__frame_number

    def frame_height(self) -> int:
        return int(self.__video.get(cv2.CAP_PROP_FRAME_HEIGHT))
