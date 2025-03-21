import threading

from dataclasses import dataclass

# This asynchronously loads the PaddleOCR library to save O(seconds) during startup
def load_paddleocr():
    global paddleocr
    paddleocr = __import__("paddleocr")
import_thread = threading.Thread(target=load_paddleocr)
import_thread.start()

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
        import_thread.join()
        self.model = paddleocr.PaddleOCR(
            use_angle_cls=False,
            lang="ch",
            show_log=False,
        )

    def run(self, image_or_path) -> list[Result]:
        results = self.model.ocr(image_or_path, cls=False)[0]
        if results is None:
            return []
        # print(results)
        # [[[[530.0, 76.0], [1390.0, 76.0], [1390.0, 140.0], [530.0, 140.0]], ('这个机器。一分钟能够前进一百米', 0.9675779342651367)]]
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