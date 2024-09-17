import paddleocr

from dataclasses import dataclass 
from classes import Frame
from typing import Optional

@dataclass
class Point:
    """A point on a plane."""
    x: int
    y: int

@dataclass
class Result:
    """Contains OCR information from PaddleOCR."""
    bounding_box: list[Point]
    confidence: float
    text: str

class OCRWorker:
    def __init__(self):
        self.model = paddleocr.PaddleOCR(
            use_angle_cls=False,
            lang="ch",
            show_log=False,
        )

    def process_frame(self, frame: Optional[Frame]) -> list[Optional[Result]]:
        if frame is None:
            # Signifies that the frame is nil.
            # SRT generator should ignore such results.
            return None
        results = self.model.ocr(frame.image, cls=False)[0]
        if results is None:
            # Signfies that the frame has data, but no subtitles.
            # SRT generator uses these results to determine when subtitles stop.
            return []
        frame_results = []
        for res in results:
            bounding_box, character_tuple = res
            characters, confidence = character_tuple
            frame_results.append(
                Result(
                    bounding_box = [Point(b[0], b[1]) for b in bounding_box],
                    confidence = confidence,
                    text = characters,
                )
            )
        return frame_results
