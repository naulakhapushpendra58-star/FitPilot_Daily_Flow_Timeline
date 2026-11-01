import json
import os
from typing import Any, Dict, List, Optional

import requests

try:
    from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI, RateLimitError
except ImportError:  # pragma: no cover - optional dependency in minimal environments
    OpenAI = None
    APIConnectionError = APIStatusError = AuthenticationError = RateLimitError = Exception

try:
    from google import genai
except ImportError:  # pragma: no cover - optional dependency in minimal environments
    genai = None

DEFAULT_SYSTEM_INSTRUCTIONS = (
    "You are FitPilot's AI Coach. Help with daily routines, wellness, productivity, breaks, and fitness planning. "
    "Use the user profile and schedule context when available, keep answers concise, practical, and grounded in the provided data, "
    "and avoid diagnosing medical conditions or giving medical treatment advice."
)


def _normalize_history(history: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    if not history:
        return []
    normalized = []
    for message in history:
        role = str(message.get("role", "user"))
        content = str(message.get("content", "")).strip()
        if content:
            normalized.append({"role": role, "content": content})
    return normalized


def _build_context(question: str, profile: Optional[Dict[str, Any]], activity_library: List[Dict[str, Any]], history: Optional[List[Dict[str, str]]]) -> str:
    schedule = profile.get("schedule") if isinstance(profile, dict) else {}
    items = (schedule or {}).get("items") or []
    remaining = [item.get("activity_name") for item in items if not item.get("completed")][:5]
    preferred = profile.get("preferred_activities") or profile.get("preferred_activity") or profile.get("preference") or []
    preferred_text = ", ".join(str(item).strip() for item in preferred if str(item).strip()) if isinstance(preferred, list) else str(preferred)
    if not preferred_text:
        preferred_text = "your selected activities"

    summary_lines = [
        f"Question: {question}",
        f"Goal: {profile.get('goal') or 'Build consistency'}",
        f"Preferred activities: {preferred_text}",
        f"Wake time: {profile.get('wake_time') or '07:00'}",
        f"Sleep time: {profile.get('sleep_time') or '23:00'}",
        f"Work schedule: {profile.get('work_start') or 'unspecified'} to {profile.get('work_end') or 'unspecified'}",
        f"Lunch time: {profile.get('lunch_time') or '12:30'}",
        f"Remaining activities: {', '.join(remaining) if remaining else 'none currently listed'}",
        f"Activity library count: {len(activity_library)}",
    ]
    if history:
        summary_lines.append("Conversation history:")
        for item in history[-6:]:
            summary_lines.append(f"{item['role']}: {item['content']}")
    return "\n".join(summary_lines)


def _build_messages(question: str, profile: Dict[str, Any], activity_library: List[Dict[str, Any]], history: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    context = _build_context(question, profile, activity_library, history)
    messages = [{"role": "system", "content": DEFAULT_SYSTEM_INSTRUCTIONS}]
    for entry in _normalize_history(history):
        messages.append({"role": entry["role"], "content": entry["content"]})
    messages.append({"role": "user", "content": context})
    return messages


def _call_openai(question: str, profile: Dict[str, Any], activity_library: List[Dict[str, Any]], history: Optional[List[Dict[str, str]]]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI API key is not configured. Set OPENAI_API_KEY before using OpenAI.")
    if OpenAI is None:
        raise RuntimeError("The OpenAI SDK is not installed. Install the project dependencies and try again.")

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=_build_messages(question, profile, activity_library, history),
            temperature=0.4,
            max_tokens=400,
        )
    except AuthenticationError as exc:
        raise RuntimeError("OpenAI authentication failed. Check your OPENAI_API_KEY and account billing status.") from exc
    except RateLimitError as exc:
        raise RuntimeError("OpenAI quota or rate limit was reached. Add billing credit or wait and retry.") from exc
    except APIConnectionError as exc:
        raise RuntimeError("OpenAI connection failed. Check your network access and the OpenAI API status.") from exc
    except Exception as exc:  # pragma: no cover - provider error surface
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc

    text = getattr(response.choices[0].message, "content", None)
    if not text:
        raise RuntimeError("OpenAI returned an empty response.")
    return str(text).strip()


def _call_gemini(question: str, profile: Dict[str, Any], activity_library: List[Dict[str, Any]], history: Optional[List[Dict[str, str]]]) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Gemini API key is not configured. Set GEMINI_API_KEY before using Gemini.")
    if genai is None:
        raise RuntimeError("The official Google Gemini SDK is not installed. Install google-genai to enable Gemini.")

    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=[_build_context(question, profile, activity_library, history)],
        )
    except Exception as exc:
        message = str(exc).lower()
        if "api key" in message or "authentication" in message or "forbidden" in message:
            raise RuntimeError("Gemini authentication failed. Check your GEMINI_API_KEY and account access.") from exc
        if "quota" in message or "billing" in message or "limit" in message:
            raise RuntimeError("Gemini quota or billing limit was reached. Add credits or wait and retry.") from exc
        if "rate" in message:
            raise RuntimeError("Gemini rate limit was reached. Please retry in a moment.") from exc
        raise RuntimeError(f"Gemini request failed: {exc}") from exc

    text = getattr(response, "text", None)
    if not text:
        text = getattr(response, "candidates", None)
        if isinstance(text, list) and text:
            try:
                parts = text[0].content.parts
                text = "".join(getattr(part, "text", "") or "" for part in parts)
            except Exception:
                text = ""
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return str(text).strip()


def generate_chat_completion(question: str, profile: Optional[Dict[str, Any]], activity_library: Optional[List[Dict[str, Any]]] = None, provider: str = "openai", history: Optional[List[Dict[str, str]]] = None) -> str:
    activity_library = activity_library or []
    profile = profile or {}
    provider_name = (provider or "openai").lower()
    if provider_name == "openai":
        return _call_openai(question, profile, activity_library, history)
    if provider_name == "gemini":
        return _call_gemini(question, profile, activity_library, history)
    raise ValueError(f"Unsupported AI provider: {provider_name}")
