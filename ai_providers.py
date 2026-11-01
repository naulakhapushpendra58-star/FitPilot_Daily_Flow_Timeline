import json
import os
from typing import Any, Dict, List, Optional

import requests


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
        f"Remaining activities: {', '.join(remaining) if remaining else 'none currently listed'}",
        f"Activity library count: {len(activity_library)}",
    ]
    if history:
        summary_lines.append("Conversation history:")
        for item in history[-6:]:
            summary_lines.append(f"{item['role']}: {item['content']}")
    return "\n".join(summary_lines)


def _call_openai(question: str, profile: Dict[str, Any], activity_library: List[Dict[str, Any]], history: Optional[List[Dict[str, str]]]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI API key is not configured.")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = __import__("openai").OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_INSTRUCTIONS},
        *[{"role": "user", "content": item["content"]} if item["role"] == "user" else {"role": "assistant", "content": item["content"]} for item in _normalize_history(history)],
        {"role": "user", "content": _build_context(question, profile, activity_library, history)},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.4,
        max_tokens=400,
    )
    content = response.choices[0].message.content
    return content.strip() if content else "I could not produce a response from OpenAI."


def _call_gemini(question: str, profile: Dict[str, Any], activity_library: List[Dict[str, Any]], history: Optional[List[Dict[str, str]]]) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Gemini API key is not configured.")
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"{DEFAULT_SYSTEM_INSTRUCTIONS}\n\n{_build_context(question, profile, activity_library, history)}"}
                ]
            }
        ],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 400},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    response = requests.post(url, json=payload, timeout=30)
    if response.status_code >= 400:
        detail = response.text[:400]
        raise RuntimeError(f"Gemini request failed ({response.status_code}): {detail}")
    body = response.json()
    try:
        text = body["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("Gemini returned an unexpected response payload.")
    return text.strip() if text else "I could not produce a response from Gemini."


def generate_chat_completion(question: str, profile: Optional[Dict[str, Any]], activity_library: Optional[List[Dict[str, Any]]] = None, provider: str = "openai", history: Optional[List[Dict[str, str]]] = None) -> str:
    activity_library = activity_library or []
    profile = profile or {}
    provider_name = (provider or "openai").lower()
    if provider_name == "openai":
        return _call_openai(question, profile, activity_library, history)
    if provider_name == "gemini":
        return _call_gemini(question, profile, activity_library, history)
    raise ValueError(f"Unsupported AI provider: {provider_name}")
