import torch
from pyclops.utils.TrajCamera import TrajCamera

class TrajCameraModelWrapper(torch.nn.Module):  # <-- Inherit from torch.nn.Module
    def __init__(self, model: torch.nn.Module, camera: TrajCamera):
        """
        A wrapper for any EGNN dynamics model that records trajectory snapshots.
        
        :param model: The base model to wrap.
        :param camera: An optional TrajCamera instance to record snapshots.
        """
        super().__init__()  # <-- Ensure torch.nn.Module initialization
        self.model = model
        self.camera = camera

    def forward(self, t, xs):
        """Modified forward pass that records trajectory before running the base model."""
        self.camera.record(t, xs)  # Store positions before dynamics update
        return self.model.forward(t, xs)  # Call the original model's forward pass

    def state_dict(self, *args, **kwargs):
        """
        Ensure PyTorch can save/load this model correctly by delegating to the wrapped model.
        """
        return self.model.state_dict(*args, **kwargs)  # <-- Delegate state dict access

    def load_state_dict(self, state_dict, strict=True):
        """
        Ensure PyTorch loads state dict correctly by passing it to the wrapped model.
        """
        return self.model.load_state_dict(state_dict, strict=strict)  # <-- Delegate loading state dict

    def __getattr__(self, name):
        """
        Delegate all other attributes/methods to the original model.
        Ensures torch.nn.Module attributes are handled correctly.
        """
        if name in self.__dict__:  # Ensure no recursion errors
            return self.__dict__[name]
        return getattr(self.model, name)
