import cv2
import functools
import argparse
import os
import math

from subtitle import SubtitleGenerator
from frame_selector import FrameSelector
from datetime import timedelta
from statistics import mean
from ocr import Result, OCR

def merge_results(results: list[Result]) -> str:
    """Combines 'Result' containers, sorting by increasing average-x values for the bounding box. This is L-to-R reading order."""
    make_tuples = lambda res: (res.text, mean(point.x for point in res.bounding_box))
    sort_tuples = lambda tup: tup[1]
    reduce_tuples = lambda sum, tup: sum + tup[0]
    tuples = map(make_tuples, results)
    tuples_sorted = sorted(tuples, key=sort_tuples)
    return functools.reduce(reduce_tuples, tuples_sorted, "")

def merged_bounding_box(results: list[Result]):
    points = functools.reduce(lambda pts, res: pts + res.bounding_box, results, [])
    min_x = functools.reduce(lambda m, pt: min(m, pt.x), points, math.inf)
    min_y = functools.reduce(lambda m, pt: min(m, pt.y), points, math.inf)
    max_x = functools.reduce(lambda m, pt: max(m, pt.x), points, -math.inf)
    max_y = functools.reduce(lambda m, pt: max(m, pt.y), points, -math.inf)
    return min_x, min_y, max_x, max_y

def frame_rate(video):
    return video.get(cv2.CAP_PROP_FPS)

def milliseconds(video):
    return timedelta(milliseconds=video.get(cv2.CAP_PROP_POS_MSEC))

def progress(video, current_frame):
    return (100.0 * current_frame) / video.get(cv2.CAP_PROP_FRAME_COUNT)

def crop_subtitle(image, height):
    # TODO: These values can be dynamically updated by the OCR
    return image[3*height//4:height, :]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ChineseSubtitleExtractor',
        description='Extracts hardcoded subtitles from videos into SRT files',
    )
    parser.add_argument(
        "files",
        help="paths to video file(s)",
        nargs='+',  # Allow multiple file arguments
        type=str,
    )
    args = vars(parser.parse_args())
    video_files = args["files"]

    for video_file in video_files:
        print(f"PROCESSING: {video_file}")
        cap = cv2.VideoCapture(video_file)
        success, img = cap.read()
        height, _width, _channels = img.shape
        frame_num = 0
        rate = frame_rate(cap)
        subtitle_generator = SubtitleGenerator()
        frame_selector = FrameSelector()
        ocr = OCR()
        while success:
            img = crop_subtitle(img, height)
            pct = progress(cap, frame_num)
            if frame_selector.select(img):
                print(f"frame: {frame_num} [{round(pct, 3)}%]")
                frame_results = ocr.run(img)
                subtitle_generator.add_subtitle(
                    time=milliseconds(cap), content=merge_results(frame_results)
                )
                if len(frame_results) == 0:
                    frame_selector.remove_filter()
                else:
                    frame_selector.add_filter(
                        *merged_bounding_box(frame_results)
                    )
            success, img = cap.read()
            frame_num += 1
        srt_file = os.path.splitext(video_file)[0] + ".srt"
        with open(srt_file, "w") as f:
            f.write(subtitle_generator.create_srt())
