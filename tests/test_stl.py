import unittest

from examples.stl.differentiability import backward, eval_reach_avoid


class TestExamples(unittest.TestCase):

    def test_run(self):
        # Test Eval
        final_result = []
        for _ in range(1000):
            # Fair test with jax
            res = eval_reach_avoid(mute=True)
            final_result.append(res)

        print(final_result)

        # Test differentiability
        path = backward()
        print(path)

        #
        # self.assertEqual(True, False)  # add assertion here


if __name__ == "__main__":
    unittest.main()
