import concurrent.futures
import cv2
import functools
import argparse
import os
import math
import concurrent

from typing import List, Optional
from classes import Frame, CurrentAndPreviousFrame
from subtitle import SubtitleGenerator
from frame_selector import filter_frames
from datetime import timedelta
from statistics import mean
from ocr import Result, OCRWorker

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

def milliseconds(frame, fps):
    return timedelta(milliseconds=(1000.0 * frame / fps))

def total_number_frames(video):
    return int(video.get(cv2.CAP_PROP_FRAME_COUNT))

def crop_subtitle(image, height):
    # TODO: These values can be dynamically updated by the OCR
    return image[13*height//16:height, :]

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

        subtitle_generator = SubtitleGenerator()

        i = 0
        cap = cv2.VideoCapture(video_file)
        success, img = cap.read()
        height, _width, _channels = img.shape
        previous_frame = None
        fps = frame_rate(cap)
        total_frames = total_number_frames(cap)

        raw_frames = []
        while success:
            frame = Frame(index = i, image = crop_subtitle(img, height))
            raw_frames.append(CurrentAndPreviousFrame(
                current_frame=frame,
                previous_frame=previous_frame,
            ))
            previous_frame = frame
            i += 1
            success, img = cap.read()
        cap.release()

        # Determine frames that can be skipped due to similarity
        pruned_frames: List[Optional[Frame]] = [None] * len(raw_frames)
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # TODO: Since we pass idx here explicitly, we probably don't need it in frame type?
            futures = {executor.submit(filter_frames, frame): idx for idx, frame in enumerate(raw_frames)}

            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                frame_or_none = future.result()
                pruned_frames[idx] = frame_or_none

        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            workers = [OCRWorker() for _ in range(executor._max_workers)]
            futures = [executor.submit(workers[i % len(workers)].process_frame, frame) for i, frame in enumerate(pruned_frames)]
            ocr_results = [future.result() for future in futures]

        for idx, results in enumerate(ocr_results):
            print(f"frame: {idx} [{round(100.0 * idx / total_frames, 3)}%]")
            if results is None:
                continue
            subtitle_generator.add_subtitle(
                time=milliseconds(idx, fps), content=merge_results(results)
            )

        srt_file = os.path.splitext(video_file)[0] + ".srt"
        with open(srt_file, "w", encoding='utf-8') as f:
            f.write(subtitle_generator.create_srt())

