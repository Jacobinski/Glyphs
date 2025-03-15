import cv2
import functools
import math
import multiprocessing
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from statistics import mean
from tqdm import tqdm
from typing import Dict, List, Tuple

import glyphs.cli as cli
from glyphs.frame_selector import FrameSelector
from glyphs.ocr import Result, OCR
from glyphs.subtitle import SubtitleGenerator
from glyphs.video import Video, count_frames

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

def process_video_segment(
        file: str,
        start_idx: int,
        stop_idx: int,
        progress: multiprocessing.Value,
        results_queue: multiprocessing.Queue,
    ) -> Dict[int, Subtitle]:
    video = Video(file, start_idx, stop_idx)
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
    results_queue.put(sub_dict)

def process_video(file: str, verbose=False) -> str:
    num_frames = count_frames(file)
    cpus = os.cpu_count()
    if cpus is None:
        raise Exception("cannot determine number of CPUs available to process")
    num_workers = cpus
    segments = split_into_segments(num_frames, num_workers)

    progress = multiprocessing.Value('I', 0)
    result_queue = multiprocessing.Queue()

    workers = [
        multiprocessing.Process(
            target=process_video_segment,
            args=(file, start, stop, progress, result_queue),
            daemon=True,
        )
        for start, stop in segments
    ]
    for p in workers:
        p.start()

    with tqdm(total=num_frames, desc="Processing video") as pbar:
        while progress.value < num_frames:
            pbar.n = progress.value
            pbar.refresh()
            # sleep for a short time to avoid busy waiting
            time.sleep(0.1)
        pbar.n = progress.value
        pbar.refresh()

    subs = [None] * num_frames
    for _ in workers:
        for idx, sub in result_queue.get().items():
            subs[idx] = sub

    subtitle_generator = SubtitleGenerator(verbose=verbose)
    for sub in subs:
        if sub is None:
            continue
        subtitle_generator.add_subtitle(
            time = sub.time,
            content = sub.text
        )
    return subtitle_generator.create_srt()

def main():
    args = cli.parse_arguments()

    for video_file in args.files:
        print(f"PROCESSING: {video_file}")
        subtitles = process_video(video_file, verbose=args.verbose)
        srt_file = os.path.splitext(video_file)[0] + ".srt"
        with open(srt_file, "w", encoding='utf-8') as f:
            f.write(subtitles)

if __name__ == "__main__":
    main()
