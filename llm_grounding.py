"""Optional grounded LLM explanation. Safe template fallback works without API access."""
import os
import json

def grounded_explanation(user_profile: dict, plan: list, activity_facts: list) -> str:
    """Use OpenAI if configured; otherwise provide a grounded deterministic explanation.

    The schedule itself is generated and validated by planner.py, never by the LLM.
    """
    evidence = {
        "user_profile": user_profile,
        "validated_plan": [
            {"activity": p.activity_name, "start": p.start.isoformat(),
             "end": p.end.isoformat(), "minutes": p.duration_minutes,
             "reason": p.reason} for p in plan
        ],
        "activity_library_facts": activity_facts
    }
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        if not plan:
            return "No plan was found in the supplied time window. Adjust the schedule and try again."
        names = ", ".join(f"{p.activity_name} ({p.duration_minutes} min)" for p in plan)
        return (f"Your validated plan includes {names}. These activities were selected from the "
                "provided activity library and fit the planner's checked time windows. "
                "This is a general activity suggestion, not medical advice.")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            instructions=(
                "You are a non-clinical activity planning assistant. Explain the validated plan "
                "using only the JSON evidence provided. Do not alter times or durations, invent "
                "availability, make medical claims, diagnose, or prescribe treatment. "
                "If evidence is insufficient, say so."
            ),
            input=json.dumps(evidence, ensure_ascii=False)
        )
        return response.output_text
    except Exception as exc:
        return ("LLM explanation unavailable; using grounded fallback. "
                f"Details: {type(exc).__name__}. " +
                grounded_explanation({**user_profile, "llm_disabled": True}, plan, activity_facts)
                if plan else "LLM explanation unavailable; no feasible plan was found.")
