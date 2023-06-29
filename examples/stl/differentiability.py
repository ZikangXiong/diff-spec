# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam

from ds.stl import STL, RectAvoidPredicte, RectReachPredicte
from ds.utils import default_tensor


def eval_reach_avoid():
    """
    The evaluation of a formula
    """

    # Define the formula predicates
    # goal is a rectangle area centered in [0, 0] with width and height 1
    goal = STL(RectReachPredicte(np.array([0, 0]), np.array([1, 1]), "goal"))
    # obs is a rectangle area centered in [3, 2] with width and height 1
    obs = STL(RectAvoidPredicte(np.array([3, 2]), np.array([1, 1]), "obs"))
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
    res = form.eval(path=path_1)
    print("eval result at time 0: ", res)

    # eval the formula at time 2
    res = form.eval(path=path_1, t=2)
    print("eval result at time 2: ", res)


def backward():
    """
    Planning with gradient descent
    """

    # Define the formula predicates
    # goal_1 is a rectangle area centered in [0, 0] with width and height 1
    goal_1 = STL(RectReachPredicte(np.array([0, 0]), np.array([1, 1]), "goal_1"))
    # goal_2 is a rectangle area centered in [2, 2] with width and height 1
    goal_2 = STL(RectReachPredicte(np.array([2, 2]), np.array([1, 1]), "goal_2"))

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

    path.requires_grad = True

    opt = Adam(params=[path], lr=0.1)

    for _ in range(100):
        loss = -torch.mean(form.eval(path))
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"final loss: {loss.item()}")
    print(path)

    plt.plot(path[0, :, 0].numpy(force=True), path[0, :, 1].numpy(force=True))
    plt.show()


if __name__ == "__main__":
    eval_reach_avoid()
    backward()

# %%
