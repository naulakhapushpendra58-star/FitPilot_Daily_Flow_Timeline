"""Grounded local explanation for FitPilot schedule generation.

This module never calls external AI services and uses only the validated
Plan + user profile data already stored in the app.
"""
import json


def grounded_explanation(user_profile: dict, plan: list, activity_facts: list) -> str:
    """Return a grounded, deterministic explanation of the current plan."""
    preferred_activities = user_profile.get("preferred_activities") or user_profile.get("preferred_activity") or user_profile.get("preference") or []
    if isinstance(preferred_activities, str):
        preferred_activities = [preferred_activities]
    preferred_text = ", ".join(str(item) for item in preferred_activities) if preferred_activities else "general movement"

    if not plan:
        return (
            f"No feasible activity plan was found for {preferred_text}. "
            "The schedule is constrained by your available time, busy blocks, and planning window. "
            "Try reducing the number of activities or widening your free time."
        )

    names = ", ".join(f"{p.activity_name} ({p.duration_minutes} min)" for p in plan)
    return (
        f"Your validated plan includes {names}. These activities were selected from your preferred list ({preferred_text}) "
        "and fit the available free windows after checking your busy blocks and time budget. "
        "Any activities not shown were left unscheduled because they did not fit the current feasibility constraints. "
        "This is a general activity suggestion, not medical advice."
    )
