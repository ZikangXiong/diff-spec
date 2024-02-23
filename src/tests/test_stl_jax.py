import unittest

import os

os.environ["JAX_STL_BACKEND"] = "jax"  # set the backend to JAX for all child processes
from examples.stl.differentiability import eval_reach_avoid


class TestJAXExamples(unittest.TestCase):

    def test_run(self):
        eval_reach_avoid()
        #
        # self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
