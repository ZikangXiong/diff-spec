import unittest

from examples.stl.differentiability import eval_reach_avoid


class TestExamples(unittest.TestCase):

    def test_run(self):
        eval_reach_avoid()
        #
        # self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
