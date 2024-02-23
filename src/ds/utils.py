import os

import numpy as np

# if JAX_BACKEND is set the import will be from jax.numpy
if os.environ.get("JAX_STL_BACKEND") == "jax":
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
