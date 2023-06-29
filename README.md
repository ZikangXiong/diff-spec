# Differentiable Logic Specification

Connect differentiable components with logical operators.

## Install

```bash
pip install git+https://github.com/ZikangXiong/diff-spec.git
```

## First Order Logic
[First-order logic](https://en.wikipedia.org/wiki/First-order_logic) is a logic formalism to describe the behavior of a system. It contains basic logic operators such as `and`, `or`, `not`, and quantifiers like `forall`, and `exists`. 

We can connect any differentiable components with logical operators, with the requirement that a component outputs a great-than-0 value representing 
`true` and a less-than-0 value representing `false`. 

```python
p1 = InFirstQuadrant()
p2 = XGreatThan(2.0)
p3 = XLessThan(3.0)
p4 = YLessThan(3.0)

# Define a formula
f = FOL(p1).forall() & FOL(p2).exists() & FOL(p3).forall() & FOL(p4).forall()
print(f)

# ((((∀ InFirstQuadrant) & (∃ XGreatThan(2.0))) & (∀ XLessThan(3.0))) & (∀ YLessThan(3.0)))
```

p1-4 can be any differentiable components, including neural networks. In the above example, p1 is a predicate that checks if the input is in the first quadrant. p2-4 are predicates that check if the input is greater than 2, less than 3, and less than 3 in the x and y-axis, respectively.

One can optimize a world of inputs to satisfy the formula with [gradient](examples/fol/differentiability.py).

## Signal Temporal Logic
[Signal temporal logic](https://people.eecs.berkeley.edu/~sseshia/fmee/lectures/EECS294-98_Spring2014_STL_Lecture.pdf) is a formal language to describe the behavior of a dynamical system. It is widely used in the formal verification of cyber-physical systems. 

We can use STL to describe the behavior of a system. For example, we can use STL to describe repeatedly visit `goal_1` and `goal_2` in timestep 0 to 13.

```python
# goal_1 is a rectangle area centered in [0, 0] with width and height 1
goal_1 = STL(RectReachPredicte(np.array([0, 0]), np.array([1, 1]), "goal_1"))
# goal_2 is a rectangle area centered in [2, 2] with width and height 1
goal_2 = STL(RectReachPredicte(np.array([2, 2]), np.array([1, 1]), "goal_2"))

# form is the formula goal_1 eventually in 0 to 5 and goal_2 eventually in 0 to 5
# and that always holds in 0 to 8
# In other words, the path will repeatedly visit goal_1 and goal_2 in 0 to 13
form = (goal_1.eventually(0, 5) & goal_2.eventually(0, 5)).always(0, 8)
```

We can synthesize a trace with [gradient](examples/stl/differentiability.py) or [mixed-integer programming](examples/stl/solver.py).

<!-- ## Probability Temporal Logic (Ongoing)
Probability temporal logic is an ongoing work integrating probability and random variables into temporal logic. It is useful in robot planning and control, reinforcement learning, and formal verification. -->

## Citation
If you find this repository useful in your research, considering to cite:
```bibtex
@misc{xiong2023diffspec,
      title={DiffSpec: A Differentiable Logic Specification Framework},
      url={https://github.com/ZikangXiong/diff-spec/},
      author={Zikang Xiong},
      year={2023},
}
```
