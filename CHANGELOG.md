#### 2024-03-21

##### Bug Fixes

* **jax:**
  * fix evaluation bug in predicates, jnp.min and torch.min had different semantics. Fix examples to allow different
    batch sizes as input to STL.eval (885e4d3c)
  * add evaluation shape test (which is failing). TODO: Debug (47e75ae5)
  * rename os env, reload module to make tests work together (18cdc0ec)
  * Using the optax (25687960)

##### Tests

* **jax:**  basic jit jax optimization (a930a76e)

#### 2024-02-24

##### Tests

* **jax:**  basic jit jax optimization (a930a76e)

##### Features

- start trying jax optimizations
- replace np with jnp, get working base, now to optimize as needed
- init jax src and tests
- fix typos