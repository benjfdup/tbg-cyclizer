from typing import List, Tuple
import dill
import torch
import numpy as np

class TrajCamera:
    def __init__(self, save_loc: str, frame_period: float = 0.0):
        """
        A trajectory camera to record model snapshots at different timesteps.

        :param save_loc: Path to save recorded frames.
        :param frame_period: Minimum time interval between consecutive frames.
        """
        self._save_loc = save_loc
        self._frame_period = frame_period  # Should be between 0.0 and 1.0
        self._last_t = 0.0  # Last time a frame was recorded
        self._frames: List[Tuple[float, torch.Tensor]] = []  # Stores (t, xs as numpy array)

    @property
    def save_loc(self) -> str:
        return self._save_loc
    
    @property
    def frame_period(self) -> float:
        return self._frame_period
    
    @property
    def last_t(self) -> float:
        return self._last_t
    
    @property
    def frames(self) -> List[Tuple[float, torch.Tensor]]:
        return self._frames

    def wipe(self) -> None:
        """Clears all stored frames from memory."""
        self._frames.clear()
    
    def set_last_t(self, t: float) -> None:
        """Manually update the last recorded time."""
        self._last_t = t
    
    def record(self, xs: torch.Tensor, t: float) -> None:
        """
        Records a snapshot of the model state at time `t`.

        :param xs: A torch tensor of shape [n_batches, n_atoms, 3] representing the current state.
        :param t: A float representing the time of the snapshot.
        """
        delta_t = t - self.last_t
        if delta_t > self.frame_period:  # Check if we should record
            self._frames.append((t, xs.clone().detach()))  # Store (t, xs as torch.tensor)
            self.set_last_t(t)  # Update last recorded time
    
    def save(self) -> None:
        """Save recorded frames to a file using dill, converting tensors to NumPy arrays first."""
        with open(self.save_loc, "wb") as f:
            dill.dump([(t, pos.cpu().numpy()) for t, pos in self._frames], f)

    def close(self) -> None:
        """Finalizes the recording process by saving everything and clearing memory."""
        self.save()
        self.wipe()
