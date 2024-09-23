import cv2
import functools
import argparse
import os
import math
import time
import multiprocessing

from subtitle import SubtitleGenerator
from typing import Dict, List, Tuple
from frame_selector import FrameSelector
from statistics import mean
from datetime import timedelta
from ocr import Result, OCR
from video import Video
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import Value

@dataclass
class Subtitle:
    time: timedelta
    text: str

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

def split_into_segments(length: int, num_segments: int) -> List[Tuple]:
    """Splits the input (0, length) into N segments.

    This is used by the video parallelization code to determine the range for
    which each executor is responsible.

    Example: split_into_segments(20, 2) -> [(0, 10), (10, 20)]

    In order to deal with the overlap of indices, the executor should treat
    the tuple as an closed-open range tuple: [0, 10), [10, 20). This is quite
    natural since in the case of one-segment, the video scanner should read
    frames in range [0, len(video)).
    """
    make_segments = lambda s: (length * s // num_segments, length * (s+1) // num_segments)
    return list(map(make_segments, range(num_segments)))

def crop_subtitle(image, height):
    # TODO: These values can be dynamically updated by the OCR
    return image[13*height//16:height, :]

def count_frames(file):
    """Determine the number of frames in a video via a full scan.

    The other methods, such as video.get(cv2.CAP_PROP_FRAME_COUNT) are not
    accurate for some videos.
    """
    i = 0
    video = cv2.VideoCapture(file)
    while video.read()[0]:
        i += 1
    video.release()
    return i

def extract_video_subtitles(file: str, start_idx: int, stop_idx: int, progress: Value) -> Dict[int, Subtitle]:
    video = Video(video_file, start_idx, stop_idx)
    frame_selector = FrameSelector()
    height = video.frame_height()
    ocr = OCR()
    sub_dict: Dict[int, Subtitle] = {}
    for frame in video:
        frame = crop_subtitle(frame, height)
        if frame_selector.select(frame):
            frame_results = ocr.run(frame)
            sub_dict[video.frame_number()] = Subtitle(
                time = video.time(),
                text = merge_results(frame_results),
            )
            if len(frame_results) == 0:
                frame_selector.remove_filter()
            else:
                frame_selector.add_filter(
                    *merged_bounding_box(frame_results)
                )
        with progress.get_lock():
            progress.value += 1
    return sub_dict

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

        num_frames = count_frames(video_file)
        num_workers = os.cpu_count() - 1
        segments = split_into_segments(num_frames, num_workers)

        # TODO: Create N of these and have them just communicate with the main thread
        #       to reduce lock contention. We can also set Lock=False in these cases.
        progress = Value('i', 0)

        subs = [None] * num_frames
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(extract_video_subtitles, video_file, start, stop, progress)
                for start, stop in segments
            ]

            with tqdm(total=num_frames, desc="Processing video") as pbar:
                while True:
                    val = progress.value
                    pbar.n = val
                    pbar.refresh()
                    if val >= num_frames:
                        break
                    time.sleep(0.100)

            for future in as_completed(futures):
                for idx, sub in future.result().items():
                    subs[idx] = sub

        subtitle_generator = SubtitleGenerator()
        for sub in subs:
            if sub is None:
                continue
            subtitle_generator.add_subtitle(
                time = sub.time,
                content = sub.text
            )
        srt_file = os.path.splitext(video_file)[0] + ".srt"
        with open(srt_file, "w", encoding='utf-8') as f:
            f.write(subtitle_generator.create_srt())
