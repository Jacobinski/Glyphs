import cv2

# Number of frames to rewind from the end of the video in count_frames().
REWIND_FRAME_COUNT = 50

def count_frames(file):
    """Determine the number of frames in a video via an approximate scan."""
    video = cv2.VideoCapture(file)
    # The video metadata is typically quite close to the real value.
    # Take the estimate, rewind a bit, and then count to obtain real value.
    estimated_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    count = estimated_count - REWIND_FRAME_COUNT
    video.set(cv2.CAP_PROP_POS_FRAMES, count)
    while video.read()[0]:
        count += 1
    video.release()
    return int(count)
