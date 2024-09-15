import cv2
import paddleocr
import functools
import argparse
import os

from dataclasses import dataclass 
from subtitle import SubtitleGenerator
from frame_selector import FrameSelector
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
    parser = argparse.ArgumentParser(
        prog='ChineseSubtitleExtractor',
        description='Extracts hardcoded subtitles from videos into SRT files',
    )
    # TODO: Support glob file expansion
    parser.add_argument(
        'file',
        help='path to a video file',
        type=os.path.abspath,
    )
    args = vars(parser.parse_args())
    video_file = args['file']
    srt_file = os.path.splitext(video_file)[0] + '.srt'

    cap = cv2.VideoCapture(video_file)
    success, img = cap.read()
    height, _width, _channels = img.shape
    frame_num = 0
    rate = frame_rate(cap)
    subtitle_generator = SubtitleGenerator()
    frame_selector = FrameSelector()
    while success:
        img = crop_subtitle(img, height)
        pct = progress(cap, frame_num)
        if frame_selector.select(img):
            print(f"frame: {frame_num} [{round(pct, 3)}%]")
            frame_results = ocr(img)
            subtitle_generator.add_subtitle(
                time=milliseconds(cap), 
                content=merge_results(frame_results)
            )
        success, img = cap.read()
        frame_num += 1
    with open(srt_file, 'w') as f:
        f.write(subtitle_generator.create_srt())