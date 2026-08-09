import unittest

from pfd.exploration.scheduler.sheduler import Scheduler


def _make_scheduler(consecutive=1, n_stages=3, max_iter=100):
    # stages are only counted via len(); their content is irrelevant here
    return Scheduler(
        explore_stages=[object() for _ in range(n_stages)],
        max_iter=max_iter,
        consecutive_clean_iters=consecutive,
    )


class TestConsecutiveCleanIters(unittest.TestCase):
    def test_default_advances_on_first_clean_iter(self):
        """Default (consecutive_clean_iters=1) keeps the old behavior."""
        s = _make_scheduler(consecutive=1)
        s.set_convergence(False)  # first iteration marker
        s.set_convergence(True)
        self.assertEqual(s.idx_stage, 1)
        self.assertFalse(s.convergence)

    def test_streak_required_before_advancing(self):
        s = _make_scheduler(consecutive=2)
        s.set_convergence(False)  # first iteration marker
        s.set_convergence(True)  # streak 1/2 -> no advance
        self.assertEqual(s.idx_stage, 0)
        s.set_convergence(True)  # streak 2/2 -> advance
        self.assertEqual(s.idx_stage, 1)
        # streak resets after advancing
        s.set_convergence(True)  # streak 1/2 at new stage
        self.assertEqual(s.idx_stage, 1)
        s.set_convergence(True)  # streak 2/2 -> advance
        self.assertEqual(s.idx_stage, 2)

    def test_failed_iter_resets_streak(self):
        s = _make_scheduler(consecutive=2)
        s.set_convergence(False)
        s.set_convergence(True)  # streak 1/2
        s.set_convergence(False)  # reset
        s.set_convergence(True)  # streak 1/2 again
        self.assertEqual(s.idx_stage, 0)
        s.set_convergence(True)  # streak 2/2
        self.assertEqual(s.idx_stage, 1)

    def test_converge_after_last_stage(self):
        s = _make_scheduler(consecutive=2, n_stages=1)
        s.set_convergence(False)
        s.set_convergence(True)  # streak 1/2
        self.assertFalse(s.convergence)
        s.set_convergence(True)  # streak 2/2, last stage -> converge
        self.assertTrue(s.convergence)

    def test_max_iter_still_hard_stop(self):
        s = _make_scheduler(consecutive=5, max_iter=2)
        s.set_convergence(False)
        s.set_convergence(False)  # iter 1
        s.set_convergence(False)  # iter 2 >= max_iter
        self.assertTrue(s.convergence)


if __name__ == "__main__":
    unittest.main()
