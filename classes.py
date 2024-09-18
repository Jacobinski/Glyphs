import numpy as np
import pickle

from dataclasses import dataclass

@dataclass
class Frame:
    """Contains information about a frame"""
    index: int
    image: np.ndarray

    def to_bytes(self) -> bytes:
        """Serialize Frame object to bytes."""
        image_bytes = self.image.tobytes()
        shape = self.image.shape
        dtype = str(self.image.dtype)
        return pickle.dumps((self.index, image_bytes, shape, dtype))

    @staticmethod
    def from_bytes(data: bytes):
        """Deserialize bytes back to a Frame object."""
        index, image_bytes, shape, dtype = pickle.loads(data)
        image = np.frombuffer(image_bytes, dtype=np.dtype(dtype)).reshape(shape)
        return Frame(index=index, image=image)

@dataclass
class CurrentAndPreviousFrame:
    """Contains information about a frame and its predecessor."""
    current_frame: Frame
    previous_frame: Frame

    def to_bytes(self) -> bytes:
        """Serialize CurrentAndPreviousFrame object to bytes."""
        current_frame_bytes = self.current_frame.to_bytes()
        previous_frame_bytes = self.previous_frame.to_bytes() if self.previous_frame else None
        return pickle.dumps((current_frame_bytes, previous_frame_bytes))

    @staticmethod
    def from_bytes(data: bytes):
        """Deserialize bytes back to a CurrentAndPreviousFrame object."""
        current_frame_bytes, previous_frame_bytes = pickle.loads(data)
        current_frame = Frame.from_bytes(current_frame_bytes)
        previous_frame = Frame.from_bytes(previous_frame_bytes) if previous_frame_bytes else None
        return CurrentAndPreviousFrame(current_frame=current_frame, previous_frame=previous_frame)
