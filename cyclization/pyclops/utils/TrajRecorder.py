from typing import Optional, List, Tuple

import dill
import torch

class TrajCamera:
    def __init__(self, save_loc: str, frame_period: float = 0.0, max_frames: Optional[int] = None):
        self._save_loc = save_loc
        self._frame_period = frame_period  # Should be between 0.0 and 1.0
        self._max_frames = max_frames  # Maximum number of frames to store on the device.
        self._last_t = 0.0  # Last time a frame was recorded, will be between 0.0 and 1.0
        self._frames = List[Tuple[float, torch.Tensor]] = []  # Stores (t, xs)

    @property
    def save_loc(self) -> str:
        return self._save_loc
    
    @property
    def frame_period(self) -> float:
        return self._frame_period
    
    @property
    def max_frames(self) -> Optional[int]:
        return self._max_frames
    
    @property
    def last_t(self) -> float:
        return self._last_t
    
    @property
    def frames(self) -> list:
        return self._frames
    
    def set_last_t(self, t) -> None:
        self._last_t = t
    
    def record(self, xs: torch.Tensor, t: float) -> None:
        delta_t = t - self.last_t
        if delta_t > self.frame_period:  # Check if we should record
            
            if self._max_frames and len(self._frames) >= self._max_frames:
                pass
                
            self._frames.append((t, xs.clone().detach()))  # Store time & tensor snapshot
            self._last_t = t  # Update last recorded time
    
    def save(self) -> None:
        """Save recorded frames to a file using dill."""
        filename = self.save_loc

        with open(filename, "wb") as f:
            dill.dump(self._frames, f)
    
    def close(self) -> None:
        pass # save everything pertaining to the recording in a big pickle.