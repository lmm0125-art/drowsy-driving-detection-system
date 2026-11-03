"""Unit tests for the fatigue rule — no camera, model or display needed.

    python -m unittest discover -s tests -t .
"""

import unittest

from fatigue import FatigueRule


class FatigueRuleTest(unittest.TestCase):
    def test_does_not_fire_before_threshold(self):
        rule = FatigueRule(window=10, threshold=6)
        fired = [rule.update(["closed_eye"])[0] for _ in range(5)]
        self.assertEqual(fired, [False] * 5)

    def test_fires_on_the_threshold_frame(self):
        rule = FatigueRule(window=10, threshold=6)
        for _ in range(5):
            rule.update(["closed_eye"])
        fatigued, count = rule.update(["closed_eye"])
        self.assertTrue(fatigued)
        self.assertEqual(count, 6)

    def test_opens_eyes_clear_the_window(self):
        rule = FatigueRule(window=4, threshold=3)
        for _ in range(3):
            rule.update(["closed_eye"])
        self.assertTrue(rule.update(["closed_eye"])[0])

        # four alert frames push every drowsy frame out of the window
        for _ in range(4):
            rule.update(["open_eye", "closed_mouth"])
        self.assertEqual(rule.update(["open_eye"])[1], 0)

    def test_a_single_blink_does_not_fire(self):
        rule = FatigueRule(window=10, threshold=6)
        sequence = ["open_eye"] * 4 + ["closed_eye"] + ["open_eye"] * 5
        fired = [rule.update([label])[0] for label in sequence]
        self.assertFalse(any(fired))

    def test_open_mouth_is_drowsy_but_closed_mouth_is_not(self):
        rule = FatigueRule(window=3, threshold=1)
        self.assertFalse(rule.update(["closed_mouth"])[0])
        rule.reset()
        self.assertTrue(rule.update(["open_mouth"])[0])

    def test_reset_empties_the_window(self):
        rule = FatigueRule(window=5, threshold=3)
        for _ in range(5):
            rule.update(["closed_eye"])
        rule.reset()
        self.assertEqual(rule.frames_seen, 0)
        self.assertFalse(rule.update(["closed_eye"])[0])

    def test_window_is_bounded(self):
        rule = FatigueRule(window=4, threshold=4)
        for _ in range(20):
            rule.update(["closed_eye"])
        self.assertEqual(rule.frames_seen, 4)

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            FatigueRule(window=0)
        with self.assertRaises(ValueError):
            FatigueRule(window=5, threshold=0)
        with self.assertRaises(ValueError):
            FatigueRule(window=5, threshold=6)

    def test_still_image_judgement(self):
        self.assertTrue(FatigueRule.judge_image(["closed_eye", "open_mouth"]))
        self.assertTrue(FatigueRule.judge_image(["open_mouth"]))
        self.assertFalse(FatigueRule.judge_image(["open_eye", "closed_mouth"]))
        self.assertFalse(FatigueRule.judge_image([]))


if __name__ == "__main__":
    unittest.main()
