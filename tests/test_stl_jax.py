import importlib
import os
import unittest

from jax import jit

import examples.stl.differentiability as stl_diff_examples


class TestJAXExamples(unittest.TestCase):

    def setUp(self):
        os.environ["DIFF_STL_BACKEND"] = "jax"  # set the backend to JAX for all child processes
        importlib.reload(stl_diff_examples)  # Reload the module to reset the backend

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


if __name__ == '__main__':
    unittest.main()
