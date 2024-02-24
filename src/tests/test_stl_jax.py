import unittest

import os

os.environ["JAX_STL_BACKEND"] = "jax"  # set the backend to JAX for all child processes
from examples.stl.differentiability import eval_reach_avoid
from jax import jit
import jax

class TestJAXExamples(unittest.TestCase):

    def test_run(self):
        # TODO: Study jit decorator and see optimizations

        (jax.lax.fori_loop(0, 1000, lambda i, _: jit(eval_reach_avoid()), None)).block_until_ready()
        # for _ in range(1000):
        #     eval_reach_avoid()
        #
        # self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
