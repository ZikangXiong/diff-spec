import os

from contextlib import contextmanager
import numpy as np


COLORED = False
IMPLIES_TRICK = False
HARDNESS = 100.0  # Reduce hardness of softmax to propagate gradients more easily


@contextmanager
def set_hardness(hardness: float):
    """Set the hardness of the softmax function for the duration of the context.
    Useful for making evaluation strict while allowing gradients to pass through during training.

    :param hardness: hardness of the softmax function
    :type hardness: float
    """
    global HARDNESS
    old_hardness = HARDNESS
    HARDNESS = hardness
    yield
    HARDNESS = old_hardness


if COLORED:
    from termcolor import colored
else:

    def colored(text, color):
        return text

# if JAX_BACKEND is set the import will be from jax.numpy
if os.environ.get("DIFF_STL_BACKEND") == "jax":
    # print("Using JAX backend")

    import jax
    from jax import numpy as jnp

    print(jnp.ones(3).devices())

    DEFAULT_DEVICE = jax.devices()[0]
    DEFAULT_DATATYPE = jnp.float32


    def default_tensor(x: np.ndarray, device: str = None, dtype=None) -> jnp.array:
        return jax.device_put(jnp.asarray(
            x,
            dtype=DEFAULT_DATATYPE if dtype is None else dtype,
        ), DEFAULT_DEVICE if device is None else device)


else:
    # print("Using PyTorch backend")

    import torch

    DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEFAULT_DATATYPE = torch.float32


    def default_tensor(x: np.ndarray, device: str = None, dtype=None) -> torch.Tensor:
        return torch.tensor(
            x,
            dtype=DEFAULT_DATATYPE if dtype is None else dtype,
            device=DEFAULT_DEVICE if device is None else device,
        )
