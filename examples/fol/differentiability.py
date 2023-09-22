# %%
import matplotlib.pyplot as plt
import torch as th

from ds.fol import FOL, PredicateBase


# ===============================================================================
# Predicate Examples
# ===============================================================================
class InFirstQuadrant(PredicateBase):
    def __init__(self):
        super().__init__(name="InFirstQuadrant")

    def eval(self, xs: th.Tensor) -> th.Tensor:
        """
        Return positive values if the point is in the first quadrant.
        Return negative values if the point is not in the first quadrant.
        """
        return th.min(xs, dim=1).values


class XGreatThan(PredicateBase):
    def __init__(self, x):
        super().__init__(name=f"XGreatThan({x})")
        self.x = x

    def eval(self, xs: th.Tensor) -> th.Tensor:
        """
        Return positive values if the first coordinate is greater than x.
        Return negative values if the first coordinate is not greater than x.
        """
        return xs[..., 0] - self.x


class XLessThan(PredicateBase):
    def __init__(self, x):
        super().__init__(name=f"XLessThan({x})")
        self.x = x

    def eval(self, xs: th.Tensor) -> th.Tensor:
        """
        Return positive values if the first coordinate is less than x.
        Return negative values if the first coordinate is not less than x.
        """
        return -xs[..., 0] + self.x


class YGreatThan(PredicateBase):
    def __init__(self, y):
        super().__init__(name=f"YGreatThan({y})")
        self.y = y

    def eval(self, xs: th.Tensor) -> th.Tensor:
        """
        Return positive values if the second coordinate is greater than y.
        Return negative values if the second coordinate is not greater than y.
        """
        return xs[..., 1] - self.y


class YLessThan(PredicateBase):
    def __init__(self, y):
        super().__init__(name=f"YLessThan({y})")
        self.y = y

    def eval(self, xs: th.Tensor) -> th.Tensor:
        """
        Return positive values if the second coordinate is less than y.
        Return negative values if the second coordinate is not less than y.
        """
        return self.y - xs[..., 1]


# ===============================================================================
# Differentiability
# ===============================================================================
def backward():
    # Define a predicate
    p1 = InFirstQuadrant()
    p2 = XGreatThan(2.0)
    p3 = XLessThan(3.0)
    p4 = YLessThan(3.0)

    # Define a formula
    f = FOL(p1).forall() & FOL(p2).exists() & FOL(p3).forall() & FOL(p4).forall()
    print(f)

    # create plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # set x and y limits
    ax[0].set_xlim(-3, 3)
    ax[0].set_ylim(-3, 3)
    ax[1].set_xlim(-3, 3)
    ax[1].set_ylim(-3, 3)

    # Define two worlds, we support batch evaluation,
    # but the points in each world should be the same / padding to same.
    x = th.tensor(
        [
            [[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]],
            [[0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]],
        ],
        requires_grad=True,
    )

    ax[0].scatter(
        x.numpy(force=True)[0, :, 0], x.numpy(force=True)[0, :, 1], label="world 1"
    )
    ax[0].scatter(
        x.numpy(force=True)[1, :, 0], x.numpy(force=True)[1, :, 1], label="world 2"
    )

    # define optimizer
    optimizer = th.optim.Adam([x], lr=0.1)

    # optimize
    loss = None
    for i in range(10000):
        optimizer.zero_grad()
        loss = -f(x).mean()
        loss.backward()
        optimizer.step()

    print(loss)
    print(x)

    # plot
    ax[1].scatter(
        x.numpy(force=True)[0, :, 0], x.numpy(force=True)[0, :, 1], label="world 1"
    )
    ax[1].scatter(
        x.numpy(force=True)[1, :, 0], x.numpy(force=True)[1, :, 1], label="world 2"
    )

    ax[0].legend()
    ax[1].legend()
    plt.show()


if __name__ == "__main__":
    backward()

# %%
