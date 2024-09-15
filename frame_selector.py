import cv2

# TODO: Should file names have underscores?
# TODO: Install a Python linter, such as Black, to standardize conventions.
class FrameSelector:
    """Applies heuristics to determine if OCR should be run for a frame."""
    previous_frame = None
    min_x = None
    min_y = None
    max_x = None
    max_y = None

    def add_filter(self, min_x, min_y, max_x, max_y):
        self.min_x = int(min_x)
        self.min_y = int(min_y)
        self.max_x = int(max_x)
        self.max_y = int(max_y)

    def remove_filter(self):
        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None

    def _crop(self, image):
        return image[
            self.min_y : self.max_y,
            self.min_x : self.max_x,
        ]

    def _preprocess(self, image):
        """Applies filters to image to increase signal-to-noise ratio"""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred_image, 50, 150)
        return edges

    # TODO: Add Python type to `frame`
    # TODO: This is a very sensitive parameter. The difference between 0.99 and 0.999 could mean missing lots of subs
    def select(self, frame) -> bool:
        frame = self._preprocess(frame)
        if self.previous_frame is None:
            self.previous_frame = frame
            return True
        tm = cv2.matchTemplate(
            self._crop(frame),
            self._crop(self.previous_frame),
            cv2.TM_CCORR_NORMED
        )
        self.previous_frame = frame
        return tm < 0.99