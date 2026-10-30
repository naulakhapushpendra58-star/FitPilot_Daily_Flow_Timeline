import unittest
from datetime import date
from planner import TimeBlock, free_windows, make_datetime, PersonalAssistantAgent, load_knowledge_base

class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.day = date(2026, 10, 28)
        self.agent = PersonalAssistantAgent(load_knowledge_base())

    def test_free_windows_splits_busy_block(self):
        start = make_datetime(self.day, "07:00")
        end = make_datetime(self.day, "12:00")
        busy = [TimeBlock(make_datetime(self.day, "09:00"), make_datetime(self.day, "10:00"))]
        windows = free_windows(start, end, busy)
        self.assertEqual([(a.strftime("%H:%M"), b.strftime("%H:%M")) for a,b in windows],
                         [("07:00","09:00"),("10:00","12:00")])

    def test_plan_is_within_budget_and_no_overlap(self):
        profile = {"day": self.day, "goal":"Build consistency", "preference":"Walking",
                   "available_minutes":30, "target_duration":20, "day_start":"07:00", "day_end":"12:00"}
        busy = [TimeBlock(make_datetime(self.day,"08:00"), make_datetime(self.day,"10:00"))]
        plan = self.agent.plan(profile, busy)
        result = self.agent.validate(plan, busy, 30)
        self.assertTrue(result["feasible"])
        self.assertLessEqual(result["total_minutes"], 30)

    def test_invalid_window_raises(self):
        profile = {"day": self.day, "goal":"Stay active", "preference":"Any suitable activity",
                   "available_minutes":20, "target_duration":20, "day_start":"18:00", "day_end":"07:00"}
        with self.assertRaises(ValueError):
            self.agent.plan(profile, [])

if __name__ == "__main__":
    unittest.main()
