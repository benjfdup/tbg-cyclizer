import dill
import torch

class TrajCamera:
    def __init__(self, frame_period: float = 0.0):
        self._frame_period = frame_period # should be between 0.0 and 1.0
        self._last_t = 0.0 # last time the trajectory camera recorded a frame

        self._frames = []
    
    @property
    def frame_period(self) -> float:
        return self._frame_period
    
    @property
    def last_t(self) -> float:
        return self._last_t
    
    def record(self, xs: torch.Tensor, t: float) -> None:
        pass