import cv2

# TODO: Should file names have underscores?
# TODO: Install a Python linter, such as Black, to standardize conventions.
class FrameSelector:
    """Applies heuristics to determine if OCR should be run for a frame."""
    previous_frame = None

    # TODO: Add Python type to `frame`
    # TODO: We currently crop the image to just the subtitle in main.py.
    #       Maybe this logic should be moved in here?
    # TODO: We can speed up this matching code by applying some pre-processing (such as contour detection) to the
    #       current and previous image before doing the convolution.
    # TODO: This is a very sensitive parameter. The difference between 0.99 and 0.999 could mean missing lots of subs
    def select(self, frame) -> bool:
        if self.previous_frame is None:
            self.previous_frame = frame
            return True
        tm =cv2.matchTemplate(frame, self.previous_frame, cv2.TM_CCORR_NORMED)
        self.previous_frame = frame
        return tm < 0.999