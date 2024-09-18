import cv2

from typing import Optional
from classes import Frame, CurrentAndPreviousFrame
from skimage.metrics import structural_similarity as ssim

def _preprocess(frame: Frame) -> cv2.typing.MatLike:
    """Applies filters to image to increase signal-to-noise ratio"""
    gray_image = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return blurred_image

def _similar(image1: Frame, image2: Frame) -> int:
    """Applies heuristics to determine if two frames are 'similar' enough to skip OCR."""
    p1 = _preprocess(image1)
    p2 = _preprocess(image2)
    ssim_score, _ = ssim(p1, p2, full=True)
    e1 = cv2.Canny(p1, 50, 150)
    e2 = cv2.Canny(p2, 50, 150)
    match_score = cv2.matchTemplate(e1, e2, cv2.TM_CCORR_NORMED)[0][0]
    # If either heuristic thinks the images are non-similar, return true.
    return ssim_score < 0.95 or match_score < 0.95

def filter_frames(bundle_bytes) -> Optional[Frame]:
    bundle = CurrentAndPreviousFrame.from_bytes(bundle_bytes)
    """Filters out similar frames based on heuristics"""
    if bundle.previous_frame is None or _similar(bundle.current_frame, bundle.previous_frame):
        return bundle.current_frame.to_bytes()
    return None
