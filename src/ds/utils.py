import numpy as np
import torch

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DATATYPE = torch.float32


def default_tensor(x: np.ndarray, device: str = None, dtype: torch.Type = None) -> torch.Tensor:
    return torch.tensor(
        x,
        dtype=DEFAULT_DATATYPE if dtype is None else dtype,
        device=DEFAULT_DEVICE if device is None else device,
    )
