import numpy as np
import os
from contextlib import contextmanager

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

from stlpy.STL import LinearPredicate, NonlinearPredicate, STLTree


def inside_rectangle_formula(bounds, y1_index, y2_index, d, name=None):
    """
    Create an STL formula representing being inside a
    rectangle with the given bounds:

    ::

       y2_max   +-------------------+
                |                   |
                |                   |
                |                   |
       y2_min   +-------------------+
                y1_min              y1_max

    :param bounds:      Tuple ``(y1_min, y1_max, y2_min, y2_max)`` containing
                        the bounds of the rectangle.
    :param y1_index:    index of the first (``y1``) dimension
    :param y2_index:    index of the second (``y2``) dimension
    :param d:           dimension of the overall signal
    :param name:        (optional) string describing this formula

    :return inside_rectangle:   An ``STLFormula`` specifying being inside the
                                rectangle at time zero.
    """
    assert y1_index < d, "index must be less than signal dimension"
    assert y2_index < d, "index must be less than signal dimension"

    # Unpack the bounds
    y1_min, y1_max, y2_min, y2_max = bounds

    # Create predicates a*y >= b for each side of the rectangle
    a1 = np.zeros((1, d))
    a1[:, y1_index] = 1
    right = LinearPredicate(a1, y1_min)
    left = LinearPredicate(-a1, -y1_max)

    a2 = np.zeros((1, d))
    a2[:, y2_index] = 1
    top = LinearPredicate(a2, y2_min)
    bottom = LinearPredicate(-a2, -y2_max)

    # Take the conjuction across all the sides
    inside_rectangle = right & left & top & bottom

    # set the names
    if name is not None:
        inside_rectangle.__str__ = lambda: name
        inside_rectangle.__repr__ = lambda: name

    return inside_rectangle


def outside_rectangle_formula(bounds, y1_index, y2_index, d, name=None):
    """
    Create an STL formula representing being outside a
    rectangle with the given bounds:

    ::

       y2_max   +-------------------+
                |                   |
                |                   |
                |                   |
       y2_min   +-------------------+
                y1_min              y1_max

    :param bounds:      Tuple ``(y1_min, y1_max, y2_min, y2_max)`` containing
                        the bounds of the rectangle.
    :param y1_index:    index of the first (``y1``) dimension
    :param y2_index:    index of the second (``y2``) dimension
    :param d:           dimension of the overall signal
    :param name:        (optional) string describing this formula

    :return outside_rectangle:   An ``STLFormula`` specifying being outside the
                                 rectangle at time zero.
    """
    assert y1_index < d, "index must be less than signal dimension"
    assert y2_index < d, "index must be less than signal dimension"

    # Unpack the bounds
    y1_min, y1_max, y2_min, y2_max = bounds

    # Create predicates a*y >= b for each side of the rectangle
    a1 = np.zeros((1, d))
    a1[:, y1_index] = 1
    right = LinearPredicate(a1, y1_max)
    left = LinearPredicate(-a1, -y1_min)

    a2 = np.zeros((1, d))
    a2[:, y2_index] = 1
    top = LinearPredicate(a2, y2_max)
    bottom = LinearPredicate(-a2, -y2_min)

    # Take the disjuction across all the sides
    outside_rectangle = right | left | top | bottom

    # set the names
    if name is not None:
        outside_rectangle.__str__ = lambda: name
        outside_rectangle.__repr__ = lambda: name

    return outside_rectangle


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
