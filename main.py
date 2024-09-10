import cv2
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
    return video.get(cv2.CAP_PROP_POS_MSEC)

def progress(video, current_frame):
    return (100.0 * current_frame) / video.get(cv2.CAP_PROP_FRAME_COUNT)

def crop_subtitle(image, height):
    return image[3*height//4:height, :]

if __name__ == "__main__":
    cap = cv2.VideoCapture("/Users/jacobbudzis/Code/PythonSubtitles/example.mkv")
    success, img = cap.read()
    height, _width, _channels = img.shape
    img = crop_subtitle(img, height)
    prev = img
    frame_num = 0
    rate = frame_rate(cap)
    subtitles = []
    current_start = None
    current_end = None
    current_content = None
    while success:
        pct = progress(cap, frame_num)
        tm =cv2.matchTemplate(img, prev, cv2.TM_CCORR_NORMED)
        if tm < 0.98:
            print(f"frame: {frame_num} [{pct}%]")
            print(f"  x-corr: {tm}")
            frame_results = ocr(img)
            for i, result in enumerate(frame_results):
                print(f"  result #{i}: {result}")
        prev = img
        success, img = cap.read()
        img = crop_subtitle(img, height)
        frame_num += 1