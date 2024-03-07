import numpy as np

from ds.stl import STL, RectAvoidPredicate, RectReachPredicate, StlpySolver


def solve_reach_avoid():
    goal = STL(RectReachPredicate(np.array([0, 0]), np.array([1, 1]), "goal"))
    obs = STL(RectAvoidPredicate(np.array([3, 2]), np.array([1, 1]), "obs"))
    form = goal.eventually(0, 10) & obs.always(0, 10)

    stlpy_form = form.get_stlpy_form()
    solver = StlpySolver(space_dim=2)
    x_0 = np.array([0, 0])
    total_time = 10
    path, info = solver.solve_stlpy_formula(stlpy_form, x0=x_0, total_time=total_time)

    print(path)
    print(info)


if __name__ == "__main__":
    solve_reach_avoid()
