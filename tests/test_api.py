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

    def test_custom_routine_activity_creation(self):
        payload = {
            "day": "2026-10-30",
            "goal": "Build consistency",
            "preference": "Walking",
            "experience": "Beginner",
            "available_minutes": 180,
            "target_duration": 20,
            "wake_time": "07:00",
            "sleep_time": "23:00",
            "study_duration": 45,
            "break_duration": 15,
            "work_start": "09:00",
            "work_end": "17:00",
            "preferred_activities": ["Walking"],
            "custom_activities": [
                {"activity_name": "Lunch", "start_time": "13:00", "end_time": "13:30", "category": "Lunch"},
                {"activity_name": "Study", "start_time": "18:00", "end_time": "18:45", "category": "Study"},
            ],
        }
        response = self.client.post("/api/schedules/generate", json=payload)
        self.assertEqual(response.status_code, 200)
        item_names = [item["activity_name"] for item in response.json()["schedule"]["items"]]
        self.assertTrue(any(name == "Lunch" for name in item_names))
        self.assertTrue(any(name == "Study" for name in item_names))

    def test_complete_and_skip_status(self):
        payload = {
            "day": "2026-10-31",
            "goal": "Stay active",
            "preferred_activities": ["Walking"],
            "available_minutes": 90,
            "target_duration": 20,
            "wake_time": "07:00",
            "sleep_time": "22:00",
            "work_start": "09:00",
            "work_end": "17:00",
        }
        generate = self.client.post("/api/schedules/generate", json=payload)
        item_id = generate.json()["schedule"]["items"][0]["id"] if "id" in generate.json()["schedule"]["items"][0] else None
        if item_id is None:
            self.fail("Schedule item missing id")
        complete_resp = self.client.patch(f"/api/schedule-items/{item_id}/complete")
        self.assertEqual(complete_resp.status_code, 200)
        self.assertEqual(complete_resp.json()["item"]["status"], "completed")

        skip_resp = self.client.patch(f"/api/schedule-items/{item_id}/skip")
        self.assertEqual(skip_resp.status_code, 200)
        self.assertEqual(skip_resp.json()["item"]["status"], "skipped")

    def test_dashboard_progress_and_clear_plan(self):
        payload = {"day": "2026-11-01", "goal": "Build consistency", "preferred_activities": ["Walking"], "available_minutes": 90, "target_duration": 20, "wake_time": "07:00", "sleep_time": "22:00", "work_start": "09:00", "work_end": "17:00"}
        self.client.post("/api/schedules/generate", json=payload)
        dashboard = self.client.get("/api/dashboard")
        self.assertEqual(dashboard.status_code, 200)
        stats = dashboard.json()["stats"]
        self.assertIn("completed_count", stats)
        self.assertIn("skipped_count", stats)
        self.assertIn("pending_count", stats)
        self.assertIn("total_minutes", stats)
        self.assertIn("progress_pct", stats)

        clear_resp = self.client.post("/api/schedules/clear")
        self.assertEqual(clear_resp.status_code, 200)
        self.assertIn("Plan cleared successfully", clear_resp.json()["message"])

    def test_clear_chat(self):
        chat_response = self.client.post("/api/chat", json={"question": "How can I improve my daily productivity?"})
        self.assertEqual(chat_response.status_code, 200)
        clear_resp = self.client.delete("/api/chat")
        self.assertEqual(clear_resp.status_code, 200)
        self.assertIn("successfully", clear_resp.json()["message"])

    def test_provider_status(self):
        providers = self.client.get("/api/chat/providers")
        self.assertEqual(providers.status_code, 200)
        payload = providers.json()
        self.assertIn("providers", payload)
        self.assertTrue(any(item["id"] == "local" for item in payload["providers"]))

    def test_static_endpoint(self):
        response = self.client.get("/static/.gitkeep")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
