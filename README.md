# Differentiable Logic Specification

Connect differentiable components with logicial operators.

## Install

```bash
pip install git+https://github.com/ZikangXiong/diff-spec.git
```

## First Order Logic (On-going)
[First order logic](https://en.wikipedia.org/wiki/First-order_logic) is a logic formalism to describe the behavior of a system. It contrains basic logic operators such as `and`, `or`, `not`, and quantifiers like `forall`, `exists`. 

## Signal Temproal Logic
[Signal temproal logic](https://people.eecs.berkeley.edu/~sseshia/fmee/lectures/EECS294-98_Spring2014_STL_Lecture.pdf) is a formal language to describe the behavior of a dynamical system. It is widely used in formal verification of cyber-physical systems. 

We can use STL to describe the behavior of a system. For example, we can use STL to describe repeatedly visit `goal_1` and `goal_2` in timestep 0 to 13.

```python
# goal_1 is a rectangle area centered in [0, 0] with width and height 1
goal_1 = STL(RectReachPredicte(np.array([0, 0]), np.array([1, 1]), "goal_1"))
# goal_2 is a rectangle area centered in [2, 2] with width and height 1
goal_2 = STL(RectReachPredicte(np.array([2, 2]), np.array([1, 1]), "goal_2"))

# form is the formula goal_1 eventually in 0 to 5 and goal_2 eventually in 0 to 5
# and that holds always in 0 to 8
# In other words, the path will repeatedly visit goal_1 and goal_2 in 0 to 13
form = (goal_1.eventually(0, 5) & goal_2.eventually(0, 5)).always(0, 8)
```

We can synthesize a trace with [gradient](examples/stl/diffrentiablity.py) or [mixed-integer programming](examples/stl/solver.py).

## Probability Temproal Logic (On-going)
Probability temproal logic is a on-going work intergrating probability and random variables into temproal logic. It is useful in robot planning and control, reinforcement learning, and formal verification.

<!-- ## Citation
If you find this repository useful in your research, please cite:
```
@misc{xiong2023diffspec,
      title={DiffSpec: A Differentiable Logic Specification Framework}, 
      author={Zikang Xiong},
      year={2023},
}
``` -->