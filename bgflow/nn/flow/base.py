
from typing import Union

import torch
import numpy as np


__all__ = ["Flow", "Parameters"]


class Flow(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def _forward(self, *xs, **kwargs):
        raise NotImplementedError()

    def _inverse(self, *xs, **kwargs):
        raise NotImplementedError()

    def forward(self, *xs, inverse=False, **kwargs):
        if inverse:
            return self._inverse(*xs, **kwargs)
        else:
            return self._forward(*xs, **kwargs)


class Transform(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for name, value in kwargs.items():
            setattr(self, name, self.as_module(value))

    @staticmethod
    def as_module():
        pass

    def compute_parameters(self):
        pass


class AffineTransform(Transform):
    def __init__(self, sigma=torch.tensor, mu=torch.tensor):
        super().__init__(sigma=sigma, mu=mu)


class Parameters(torch.nn.Module):
    def __init__(self, *values: Union[torch.Tensor, float, np.ndarray], names=None, constant=True):
        super().__init__()
        names = (f"c{i}" for i, _ in enumerate(values)) if names is None else names
        tensors = (torch.as_tensor(tensor) for tensor in values)
        for name, tensor in zip(names, tensors):
            if constant:
                self.register_buffer(name, tensor)
            else:
                self.register_parameter(name, torch.nn.Parameter(tensor))
        self.is_constant = True

    def forward(self, *args):
        if self.is_constant:
            return tuple(self.buffers(recurse=False))
        else:
            return tuple(self.parameters(recurse=False))
