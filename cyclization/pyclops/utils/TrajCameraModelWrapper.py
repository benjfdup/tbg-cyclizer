import torch
from pyclops.utils.TrajCamera import TrajCamera

class TrajCameraModelWrapper():
    def __init__(self, model: torch.nn.Module, camera: TrajCamera):
        """
        A wrapper for any EGNN dynamics model that records trajectory snapshots.
        
        :param model: The base model to wrap.
        :param camera: An optional TrajCamera instance to record snapshots.
        """
        self.model = model
        self.camera = camera

    def forward(self, t, xs):
        """Modified forward pass that records trajectory before running the base model."""
        self.camera.record(t, xs)  # Store positions before dynamics update
        return self.model.forward(t, xs)  # Call the original model's forward pass

    def __getattr__(self, name):
        """
        Delegate all other attributes/methods to the original model.
        Ensures torch.nn.Module attributes are handled correctly.
        """
        return getattr(self.model, name)