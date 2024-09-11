import cv2
import paddleocr
import functools

from dataclasses import dataclass 
from subtitle import SubtitleGenerator
from datetime import timedelta
from statistics import mean

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

def merge_results(results: list[Result]) -> str:
    """Combines 'Result' containers, sorting by increasing average-x values for the bounding box. This is L-to-R reading order."""
    make_tuples = lambda res: (res.text, mean(point.x for point in res.bounding_box))
    sort_tuples = lambda tup: tup[1]
    reduce_tuples = lambda sum, tup: sum + tup[0]
    tuples = map(make_tuples, results)
    tuples_sorted = sorted(tuples, key=sort_tuples)
    return functools.reduce(reduce_tuples, tuples_sorted, "")

def ocr(image_or_path) -> list[Result]:
    # TODO: Download and save model to directory
    model = paddleocr.PaddleOCR(
        use_angle_cls=False, 
        lang="ch", 
        show_log=False,
    )
    results = model.ocr(image_or_path, cls=False)[0]
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

def frame_rate(video):
    return video.get(cv2.CAP_PROP_FPS)

def milliseconds(video):
    return timedelta(milliseconds=video.get(cv2.CAP_PROP_POS_MSEC))

def progress(video, current_frame):
    return (100.0 * current_frame) / video.get(cv2.CAP_PROP_FRAME_COUNT)

def crop_subtitle(image, height):
    return image[3*height//4:height, :]

if __name__ == "__main__":
    # TODO: Convert this into a user-passable flag
    cap = cv2.VideoCapture("/Users/jacobbudzis/Code/PythonSubtitles/examples/example.mkv")
    success, img = cap.read()
    height, _width, _channels = img.shape
    prev = None
    frame_num = 0
    rate = frame_rate(cap)
    subtitle_generator = SubtitleGenerator()
    while success:
        img = crop_subtitle(img, height)
        if prev is None:
            prev = img
        pct = progress(cap, frame_num)
        tm =cv2.matchTemplate(img, prev, cv2.TM_CCORR_NORMED)
        # TODO: We can speed up this matching code by applying some pre-processing (such as contour detection) to the
        #       current and previous image before doing the convolution.
        # TODO: This is a very sensitive parameter. The difference between 0.99 and 0.999 could mean missing lots of subs
        if tm < 0.999:
            print(f"frame: {frame_num} [{round(pct, 3)}%]")
            frame_results = ocr(img)
            subtitle_generator.add_subtitle(
                time=milliseconds(cap), 
                content=merge_results(frame_results)
            )
        prev = img
        success, img = cap.read()
        frame_num += 1
    print(subtitle_generator.create_srt())