# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import optax

# if JAX_BACKEND is set the import will be from jax.numpy
if os.environ.get("JAX_STL_BACKEND") == "jax":
    print("Using JAX backend")
    from ds.stl_jax import STL, RectAvoidPredicate, RectReachPredicate
    from ds.utils import default_tensor
    import jax
else:
    print("Using PyTorch backend")
    from ds.stl import STL, RectAvoidPredicate, RectReachPredicate
    from ds.utils import default_tensor
    import torch
    from torch.optim import Adam


def eval_reach_avoid(mute=False):
    """
    The evaluation of a formula
    """

    # Define the formula predicates
    # goal is a rectangle area centered in [0, 0] with width and height 1
    goal = STL(RectReachPredicate(np.array([0, 0]), np.array([1, 1]), "goal"))
    # obs is a rectangle area centered in [3, 2] with width and height 1
    obs = STL(RectAvoidPredicate(np.array([3, 2]), np.array([1, 1]), "obs"))
    # form is the formula goal eventually in 0 to 10 and avoid obs always in 0 to 10
    form = goal.eventually(0, 10) & obs.always(0, 10)

    # Define 2 initial paths in batch
    path_1 = default_tensor(
        np.array(
            [
                [
                    [9, 9],
                    [8, 8],
                    [7, 7],
                    [6, 6],
                    [5, 5],
                    [4, 4],
                    [3, 3],
                    [2, 2],
                    [1, 1],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                [
                    [9, 9],
                    [8, 8],
                    [7, 7],
                    [6, 6],
                    [5, 5],
                    [4, 4],
                    [3, 3],
                    [2, 2],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                ],
            ]
        )
    )

    # eval the formula, default at time 0
    res1 = form.eval(path=path_1)
    if not mute:
        print("eval result at time 0: ", res1)

    # eval the formula at time 2
    res2 = form.eval(path=path_1, t=2)
    if not mute:
        print("eval result at time 2: ", res2)

    return res1, res2


def backward(mute=True):
    """
    Planning with gradient descent
    """

    # Define the formula predicates
    # goal_1 is a rectangle area centered in [0, 0] with width and height 1
    goal_1 = STL(RectReachPredicate(np.array([0, 0]), np.array([1, 1]), "goal_1"))
    # goal_2 is a rectangle area centered in [2, 2] with width and height 1
    goal_2 = STL(RectReachPredicate(np.array([2, 2]), np.array([1, 1]), "goal_2"))

    # form is the formula goal_1 eventually in 0 to 5 and goal_2 eventually in 0 to 5
    # and that holds always in 0 to 8
    # In other words, the path will repeatedly visit goal_1 and goal_2 in 0 to 13
    form = (goal_1.eventually(0, 5) & goal_2.eventually(0, 5)).always(0, 8)
    path = default_tensor(
        np.array(
            [
                [
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [1, 0],
                    [1, 0],
                ],
            ]
        )
    )
    loss = None
    lr = 0.1
    num_iterations = 1000

    if os.environ.get("JAX_STL_BACKEND") == "jax":

        solver = optax.adam(lr)
        var_solver_state = solver.init(path)

        @jax.jit
        def train_step(params, solver_state):
            # Performs a one step update.
            (loss), grad = jax.value_and_grad(form.eval)(
                params
            )
            updates, solver_state = solver.update(-grad, solver_state)
            params = optax.apply_updates(params, updates)
            return params, solver_state, loss

        for _ in range(num_iterations):
            path, var_solver_state, train_loss = train_step(
                path, var_solver_state
            )

        loss = form.eval(path)
    else:
        # PyTorch backend (slower when num_iterations is high)
        path.requires_grad = True
        opt = Adam(params=[path], lr=lr)

        for _ in range(num_iterations):
            loss = -torch.mean(form.eval(path))
            opt.zero_grad()
            loss.backward()
            opt.step()

        if not mute:
            print(f"final loss: {loss.item()}")
            print(path)

            plt.plot(path[0, :, 0].numpy(force=True), path[0, :, 1].numpy(force=True))
            plt.show()

    return path, loss


if __name__ == "__main__":
    eval_reach_avoid()
    backward()

# %%
