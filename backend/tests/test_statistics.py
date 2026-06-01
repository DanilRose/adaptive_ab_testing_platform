import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.microservices.ab_testing_core.statistics import (
    StatisticalAnalyzer,
    SRMChecker,
    SequentialTesting,
)


class TestStatistics(unittest.TestCase):
    def test_srm_checker_pass(self):
        result = SRMChecker.check_srm_by_variant({"A": 100, "B": 100})
        self.assertFalse(result.srm_detected)
        self.assertGreater(result.p_value, 0.001)

    def test_srm_checker_detect(self):
        result = SRMChecker.check_srm_by_variant({"A": 190, "B": 10})
        self.assertTrue(result.srm_detected)
        self.assertLess(result.p_value, 0.001)

    def test_sequential_success(self):
        seq = SequentialTesting(alpha=0.05, max_looks=1)
        should_stop, reason = seq.should_stop_for_success(p_value=0.001, effect=0.1)
        self.assertTrue(should_stop)
        self.assertIn("boundary", reason)

    def test_power_calculation(self):
        analyzer = StatisticalAnalyzer(alpha=0.05)
        power = analyzer.calculate_power(observed_effect=1.0, sample_size_per_variant=100, baseline_std=1.0, alpha=0.05)
        self.assertGreaterEqual(power, 0.8)


if __name__ == "__main__":
    unittest.main()
