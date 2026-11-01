import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from ai_providers import generate_chat_completion
from app import app


class AiProviderAndValidationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_generate_chat_completion_uses_openai_mock(self):
        with patch("ai_providers._call_openai", return_value="OpenAI answer") as mock_openai:
            response = generate_chat_completion(
                "How should I structure my day?",
                {"goal": "Build consistency", "preferred_activities": ["Walking"]},
                [],
                provider="openai",
            )
            self.assertEqual(response, "OpenAI answer")
            mock_openai.assert_called_once()

    def test_generate_chat_completion_uses_gemini_mock(self):
        with patch("ai_providers._call_gemini", return_value="Gemini answer") as mock_gemini:
            response = generate_chat_completion(
                "How should I structure my day?",
                {"goal": "Build consistency", "preferred_activities": ["Walking"]},
                [],
                provider="gemini",
            )
            self.assertEqual(response, "Gemini answer")
            mock_gemini.assert_called_once()

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


if __name__ == "__main__":
    unittest.main()
