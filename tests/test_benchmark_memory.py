import os
import unittest

from scripts.benchmark_local_vision import rss_mb


class MemoryTests(unittest.TestCase):
    def test_current_process_memory_is_positive(self):
        self.assertGreater(rss_mb(os.getpid()), 0)

    def test_exited_process_is_zero(self):
        self.assertEqual(rss_mb(2147483647), 0)
