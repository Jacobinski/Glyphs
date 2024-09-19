import cv2
import functools
import argparse
import os
import math
import time

from tqdm import tqdm
from classes import Frame, CurrentAndPreviousFrame
from subtitle import SubtitleGenerator
from frame_selector import filter_frames
from datetime import timedelta
from statistics import mean
from ocr import Result, initialize_ocr_model, process_frame
from multiprocessing import Process, Queue, cpu_count, Manager, Value

MAX_PRUNE_WORKERS = 4
MAX_OCR_WORKERS = cpu_count() - 4

# Define a unique sentinel value to indicate the end of the queue
STOP_SIGNAL = "STOP"

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

def prune_worker(input_queue, output_queue, counter):
    while True:
        idx, frame = input_queue.get()
        if frame == STOP_SIGNAL:
            break
        result_frame = filter_frames(frame)
        output_queue.put((idx, result_frame))
        with counter.get_lock():
            counter.value += 1

def ocr_worker(input_queue, output_queue, counter):
    initialize_ocr_model()
    while True:
        idx, frame = input_queue.get()
        if frame == STOP_SIGNAL:
            break
        result = process_frame(frame)
        output_queue.put((idx, result))
        with counter.get_lock():
            counter.value += 1

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

        cap = cv2.VideoCapture(video_file)
        success, img = cap.read()
        height, _width, _channels = img.shape
        previous_frame = None
        fps = frame_rate(cap)
        total_frames = total_number_frames(cap)

        load_pbar = tqdm(total=total_frames, desc="Loading video frames")
        prune_pbar = tqdm(total=total_frames, desc="Filtering duplicate frames")
        ocr_pbar = tqdm(total=total_frames, desc="Running OCR on frames")

        prune_queue = Queue()
        ocr_queue = Queue()
        result_queue = Queue()

        prune_counter = Value('i', 0)
        ocr_counter = Value('i', 0)

        prune_processes = [
            Process(target=prune_worker, args=(prune_queue, ocr_queue, prune_counter))
            for _ in range(MAX_PRUNE_WORKERS)
        ]
        ocr_processes = [
            Process(target=ocr_worker, args=(ocr_queue, result_queue, ocr_counter))
            for _ in range(MAX_OCR_WORKERS)
        ]
        for p in prune_processes:
            p.start()
        for p in ocr_processes:
            p.start()

        i = 0
        while success:
            frame = Frame(index=i, image=crop_subtitle(img, height))
            prune_queue.put(
                (
                    i,
                    CurrentAndPreviousFrame(
                        current_frame=frame,
                        previous_frame=previous_frame,
                    )
                )
            )
            previous_frame = frame
            i += 1
            success, img = cap.read()
            load_pbar.update(1)

            # Update the UI while we're in a loop
            prune_pbar.n = prune_counter.value
            prune_pbar.refresh()
            ocr_pbar.n = ocr_counter.value
            ocr_pbar.refresh()

        # Update progress bars with correct total
        load_pbar.total = i - 1
        prune_pbar.total = i - 1
        ocr_pbar.total = i - 1

        cap.release()

        # Update progress bars in the main process
        while prune_counter.value < i-1 or ocr_counter.value < i-1:
            time.sleep(0.1)
            prune_pbar.n = prune_counter.value
            prune_pbar.refresh()
            ocr_pbar.n = ocr_counter.value
            ocr_pbar.refresh()

        # TODO: Read until the queue is empty, not some hard coded number
        # results = []
        # for _ in tqdm(range(i), desc="Collecting OCR results"):
        #     idx, result = result_queue.get()
        #     results[idx] = result
        results = [None] * i
        for _ in tqdm(range(i), desc="Collecting OCR results"):
            idx, result = result_queue.get()
            results[idx] = result

        for _ in prune_processes:
            print("B1")
            prune_queue.put((None, STOP_SIGNAL))
        for _ in ocr_processes:
            print("B2")
            ocr_queue.put((None, STOP_SIGNAL))

        for p in prune_processes:
            p.join()
        for p in ocr_processes:
            p.join()

        for idx, results in tqdm(enumerate(results), desc="Creating SRT file"):
            if results is None:
                continue
            subtitle_generator.add_subtitle(
                time=milliseconds(idx, fps), content=merge_results(results)
            )

        srt_file = os.path.splitext(video_file)[0] + ".srt"
        with open(srt_file, "w", encoding='utf-8') as f:
            f.write(subtitle_generator.create_srt())

        load_pbar.close()
        prune_pbar.close()
        ocr_pbar.close()
