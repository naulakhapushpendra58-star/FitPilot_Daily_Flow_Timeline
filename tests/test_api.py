import unittest
from fastapi.testclient import TestClient

from app import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_generate_schedule(self):
        payload = {
            "day": "2026-10-30",
            "goal": "Build consistency",
            "preference": "Walking",
            "experience": "Beginner",
            "available_minutes": 90,
            "target_duration": 20,
            "wake_time": "07:00",
            "sleep_time": "22:00",
            "study_duration": 40,
            "break_duration": 15,
            "work_start": "09:00",
            "work_end": "17:00",
            "preferred_activities": ["Walking", "Gentle mobility and stretching"],
        }
        response = self.client.post("/api/schedules/generate", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("schedule", body)
        self.assertGreater(len(body["schedule"]["items"]), 0)


if __name__ == "__main__":
    unittest.main()
