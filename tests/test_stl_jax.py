import os
import unittest

os.environ["JAX_STL_BACKEND"] = "jax"  # set the backend to JAX for all child processes
from jax import jit

from examples.stl.differentiability import backward, eval_reach_avoid


class TestJAXExamples(unittest.TestCase):

    def test_run(self):
        # TODO: Study jit decorator and see optimizations
        # jit(eval_reach_avoid)()

        final_result = []
        for _ in range(1000):
            # Magic of jax
            res = jit(eval_reach_avoid)()
            final_result.append(res)

        print(final_result)

        # Test differentiability
        path = backward()
        print(path)
        # (jax.lax.fori_loop(0, 1000, lambda i, _: jit(eval_reach_avoid)(), None)).block_until_ready()
        # for _ in range(1000):
        #     eval_reach_avoid()
        #
        # self.assertEqual(True, False)  # add assertion here


if __name__ == "__main__":
    unittest.main()
