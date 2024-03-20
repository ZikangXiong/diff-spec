import importlib
import os
import unittest

import examples.stl.differentiability as stl_diff_examples


class TestExamples(unittest.TestCase):

    def setUp(self):
        os.environ["DIFF_STL_BACKEND"] = ""
        importlib.reload(stl_diff_examples)  # Reload the module to reset the backend

    def test_run(self):
        # Test Eval
        final_result = []
        for _ in range(1000):
            # Fair test with jax
            res = stl_diff_examples.eval_reach_avoid(mute=True)
            final_result.append(res)

        print(final_result)

        # Test differentiability
        path = stl_diff_examples.backward()
        print(path)

        #
        # self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
