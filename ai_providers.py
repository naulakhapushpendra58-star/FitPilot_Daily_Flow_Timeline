import os
from typing import Any, Dict, List, Optional

from local_ai_coach import generate_local_coach_response


def generate_chat_completion(
    question: str,
    profile: Optional[Dict[str, Any]],
    activity_library: Optional[List[Dict[str, Any]]] = None,
    provider: str = "local",
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Return a deterministic FitPilot-only assistant response without any external AI service."""
    provider_name = (provider or "local").lower()
    if provider_name not in {"local", "openai", "gemini"}:
        raise ValueError(f"Unsupported AI provider: {provider_name}")
    return generate_local_coach_response(
        question=question,
        profile=profile or {},
        activity_library=activity_library or [],
        history=history,
        provider=provider_name,
    )
