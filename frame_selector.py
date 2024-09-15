import cv2

# TODO: Should file names have underscores?
# TODO: Install a Python linter, such as Black, to standardize conventions.
class FrameSelector:
    """Applies heuristics to determine if OCR should be run for a frame."""
    previous_frame = None

    def _preprocess(self, image):
        """Applies filters to image to increase signal-to-noise ratio"""
        # MINIMUM_COLOR = 180
        # MAXIMUM_COLOR = 255 
        # image_thresholded = cv2.inRange(image, (MINIMUM_COLOR, MINIMUM_COLOR, MINIMUM_COLOR), (MAXIMUM_COLOR, MAXIMUM_COLOR, MAXIMUM_COLOR))
        image_contours = cv2.Laplacian(image, cv2.CV_8U)
        return image_contours

    # TODO: Add Python type to `frame`
    # TODO: We currently crop the image to just the subtitle in main.py.
    #       Maybe this logic should be moved in here?
    # TODO: If there is a previous subtitle bounding box, it can be used to limit
    #       the area which we search for a subtitle. This works nicely with the above
    #       comment about cropping the frame.
    # TODO: We can speed up this matching code by applying some pre-processing (such as contour detection) to the
    #       current and previous image before doing the convolution.
    # TODO: This is a very sensitive parameter. The difference between 0.99 and 0.999 could mean missing lots of subs
    def select(self, frame) -> bool:
        frame = self._preprocess(frame)
        if self.previous_frame is None:
            self.previous_frame = frame
            return True
        tm =cv2.matchTemplate(frame, self.previous_frame, cv2.TM_CCORR_NORMED)
        self.previous_frame = frame
        return tm < 0.99