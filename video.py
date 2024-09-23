import cv2

from datetime import timedelta

class Video:
    __video = None
    __frame_number: int = 0
    __fps: int = 0
    __frame_count: int = 0  # This is approximate. Need to scan the video for true number

    def __init__(self, file_path: str):
        v = cv2.VideoCapture(file_path)
        self.__video = v
        self.__fps = v.get(cv2.CAP_PROP_FPS)
        self.__frame_count = v.get(cv2.CAP_PROP_FRAME_COUNT)

    def __iter__(self):
        return self

    def __next__(self):
        success, frame = self.__video.read()
        if success:
            self.__frame_number += 1
            return frame
        else:
            raise StopIteration

    def fps(self) -> int:
        return self.__fps

    def time(self) -> timedelta:
        ms = self.__video.get(cv2.CAP_PROP_POS_MSEC)
        return timedelta(milliseconds=ms)

    def frame_number(self) -> int:
        return self.__frame_number

    def progress(self) -> float:
        return (100.0 * self.__frame_number) / self.__frame_count

    def frame_height(self) -> int:
        return self.__video.get(cv2.CAP_PROP_FRAME_HEIGHT)