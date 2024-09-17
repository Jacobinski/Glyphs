import cv2

from dataclasses import dataclass

Image = cv2.typing.MatLike

@dataclass
class Frame:
    """Contains information about a frame"""
    index: int
    image: Image

@dataclass
class CurrentAndPreviousFrame:
    """Contains information about a frame and its predecessor."""
    current_frame: Frame
    previous_frame: Frame