import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.microservices.ab_testing_core.decision_engine import ABDecisionEngine


class TestDecisionPolicy(unittest.TestCase):
    def test_policy_allows_deploy(self):
        result = ABDecisionEngine.evaluate_decision_policy(
            analysis_validity="valid_for_inference",
            srm_passed=True,
            guardrails_passed=True,
            corrected_p_values={"B": 0.01},
            power=0.85,
            alpha=0.05,
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.status, "deploy")

    def test_policy_blocks_on_invalid_design(self):
        result = ABDecisionEngine.evaluate_decision_policy(
            analysis_validity="exploration_only",
            srm_passed=True,
            guardrails_passed=True,
            corrected_p_values={"B": 0.01},
            power=0.9,
            alpha=0.05,
        )
        self.assertFalse(result.allowed)
        self.assertIn("Невалидный дизайн", " ".join(result.reasons))

    def test_policy_blocks_on_power(self):
        result = ABDecisionEngine.evaluate_decision_policy(
            analysis_validity="valid_for_inference",
            srm_passed=True,
            guardrails_passed=True,
            corrected_p_values={"B": 0.01},
            power=0.5,
            alpha=0.05,
        )
        self.assertFalse(result.allowed)
        self.assertIn("мощность", " ".join(result.reasons))


if __name__ == "__main__":
    unittest.main()
