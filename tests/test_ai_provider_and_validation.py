import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from ai_providers import generate_chat_completion
from app import app


class AiProviderAndValidationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_generate_chat_completion_uses_local_coach(self):
        response = generate_chat_completion(
            "Show my complete routine.",
            {"goal": "Build consistency", "preferred_activities": ["Walking"], "schedule": {"items": [{"activity_name": "Walking", "start_time": "07:30", "end_time": "08:00", "duration_minutes": 30, "completed": False}]}}
            ,
            [],
            provider="local",
        )
        self.assertIn("routine", response.lower())

    def test_generate_chat_completion_rejects_invalid_provider(self):
        with self.assertRaises(ValueError):
            generate_chat_completion("Hello", {}, [], provider="invalid")

    def test_generate_schedule_rejects_invalid_time_window(self):
        payload = {
            "day": "2026-11-01",
            "goal": "Build consistency",
            "preferred_activities": ["Walking"],
            "available_minutes": 90,
            "target_duration": 20,
            "wake_time": "23:00",
            "sleep_time": "07:00",
        }
        response = self.client.post("/api/schedules/generate", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Wake time must be earlier than sleep time", response.json()["detail"])

    def test_local_coach_rejects_unrelated_question(self):
        response = generate_chat_completion(
            "What is the capital of France?",
            {"goal": "Build consistency"},
            [],
            provider="local",
        )
        self.assertIn("FitPilot's project assistant", response)


if __name__ == "__main__":
    unittest.main()
