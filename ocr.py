import paddleocr

from dataclasses import dataclass 

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

class OCR:
    """Wrapper around PaddleOCR engine."""
    model = None

    def __init__(self):
        self.model = paddleocr.PaddleOCR(
            use_angle_cls=False, 
            lang="ch", 
            show_log=False,
        )

    def run(self, image_or_path) -> list[Result]:
        results = self.model.ocr(image_or_path, cls=False)[0]
        if results is None:
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