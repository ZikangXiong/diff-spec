import importlib
import os
import unittest

import jax
import numpy as np
from jax import jit

os.environ["DIFF_STL_BACKEND"] = "jax"  # So ds_utils does not require torch

import ds.utils as ds_utils
import examples.stl.differentiability as stl_diff_examples
from ds.stl_jax import STL, RectReachPredicate

from ds.stl import StlpySolver
class TestJAXExamples(unittest.TestCase):

    def setUp(self):
        os.environ["DIFF_STL_BACKEND"] = "jax"  # set the backend to JAX for all child processes
        importlib.reload(stl_diff_examples)  # Reload the module to reset the backend
        importlib.reload(ds_utils)  # Reload the module to reset the backend

        self.goal_1 = STL(RectReachPredicate(np.array([0, 0]), np.array([1, 1]), "goal_1"))
        # goal_2 is a rectangle area centered in [2, 2] with width and height 1
        self.goal_2 = STL(RectReachPredicate(np.array([2, 2]), np.array([1, 1]), "goal_2"))

        # form is the formula goal_1 eventually in 0 to 5 and goal_2 eventually in 0 to 5
        # and that holds always in 0 to 8
        # In other words, the path will repeatedly visit goal_1 and goal_2 in 0 to 13
        self.form = (self.goal_1.eventually(0, 5) & self.goal_2.eventually(0, 5)).always(0, 8)
        self.loop_form = (self.goal_1.eventually(0, 4) & self.goal_2.eventually(0, 4)).always(0, 8)
        self.cover_form = self.goal_1.eventually(0, 12) & self.goal_2.eventually(0, 12)
        self.seq_form = self.goal_1.eventually(0, 6) & self.goal_2.eventually(6, 12)

    def test_run(self):
        # TODO: Study jit decorator and see optimizations
        # jit(eval_reach_avoid)()

        final_result = []
        for _ in range(1000):
            # Magic of jax
            res = jit(stl_diff_examples.eval_reach_avoid)()
            final_result.append(res)

        print(final_result)

        # Test differentiability
        path = stl_diff_examples.backward()
        print(path)
        # (jax.lax.fori_loop(0, 1000, lambda i, _: jit(eval_reach_avoid)(), None)).block_until_ready()
        # for _ in range(1000):
        #     eval_reach_avoid()
        #
        # self.assertEqual(True, False)  # add assertion here

    def test_evaluations(self, num_tiles=3):
        """Run simple evaluations to test shapes and types"""
        path = ds_utils.default_tensor(
            np.array(
                [
                    [
                        [1, 0],
                        [1, 0],
                        [1, 0],
                        [0, 0],
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

        loss = self.form.eval(jax.numpy.tile(path, (num_tiles, 1, 1)))  # Make a batch of size num_tiles
        self.assertGreater(len(loss.shape), 0, f"Not returning correct shape")
        self.assertEqual(loss.shape[0], num_tiles, f"Not returning {num_tiles} values")

    def test_loop(self):
        # Test loop spec
        num_tiles = 4
        path = ds_utils.default_tensor(
            np.array(
                [
                    [
                        [0, 0],
                        [0, 2],
                        [2, 0],
                        [2, 2],
                    ] * 3
                ]))
        loss = self.loop_form.eval(jax.numpy.tile(path, (num_tiles, 1, 1)))  # Make a batch of size num_tiles
        self.assertGreater(len(loss.shape), 0, f"Not returning correct shape")
        self.assertEqual(loss.shape[0], num_tiles, f"Not returning {num_tiles} values")
        self.assertGreater(loss[0], 0, f"Loss is not greater than 0")

        unsat_path = path.at[0, -4].set([0, 2])  # Make the last point unsatisfiable
        loss = self.loop_form.eval(jax.numpy.tile(unsat_path, (num_tiles, 1, 1)))
        self.assertLess(loss[0], 0, f"Loss is not less than 0 for unsat path")

    def test_stlpy_solver(self):
        """Test the stlpy solver with different forms of STL formulas"""
        x_0 = np.array([0, 0])
        solver = StlpySolver(space_dim=2)
        total_time = 12 # Common total time for all formulas

        for form in [self.loop_form, self.cover_form, self.seq_form]:

            stlpy_form = form.get_stlpy_form()
            path, info = solver.solve_stlpy_formula(stlpy_form, x0=x_0, total_time=total_time)

            num_tiles = 4
            loss = form.eval(jax.numpy.tile(path, (num_tiles, 1, 1)))  # Make a batch of size num_tiles
            self.assertGreater(loss[0], 0, f"STLPY solved path loss is not greater than 0 for {form}")

if __name__ == '__main__':
    unittest.main()
