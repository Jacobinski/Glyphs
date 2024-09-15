import cv2

# TODO: Should file names have underscores?
# TODO: Install a Python linter, such as Black, to standardize conventions.
class FrameSelector:
    """Applies heuristics to determine if OCR should be run for a frame."""
    previous_frame = None

    def _preprocess(self, image):
        """Applies filters to image to increase signal-to-noise ratio"""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred_image, 50, 150)
        return edges

    # TODO: Add Python type to `frame`
    # TODO: We currently crop the image to just the subtitle in main.py.
    #       Maybe this logic should be moved in here?
    # TODO: If there is a previous subtitle bounding box, it can be used to limit
    #       the area which we search for a subtitle. This works nicely with the above
    #       comment about cropping the frame.
    # TODO: This is a very sensitive parameter. The difference between 0.99 and 0.999 could mean missing lots of subs
    def select(self, frame) -> bool:
        frame = self._preprocess(frame)
        if self.previous_frame is None:
            self.previous_frame = frame
            return True
        tm =cv2.matchTemplate(frame, self.previous_frame, cv2.TM_CCORR_NORMED)
        self.previous_frame = frame
        return tm < 0.99