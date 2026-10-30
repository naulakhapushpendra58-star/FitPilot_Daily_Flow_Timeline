import json
import os

RECOMMENDED_QUESTIONS = [
    "Schedule all my preferred activities",
    "Which activities fit into my available time?",
    "Replace walking with cycling",
    "Add yoga to my schedule",
    "Reduce my exercise duration",
    "Show my unscheduled activities",
    "Which activity is best for building consistency?",
    "How can I improve flexibility with a short routine?",
]


def _normalize_preferences(profile=None):
    profile = profile or {}
    preferred = profile.get("preferred_activities") or profile.get("preferred_activity") or profile.get("preference") or []
    if isinstance(preferred, str):
        return [item.strip() for item in preferred.split(",") if item.strip()]
    return [str(item).strip() for item in preferred if str(item).strip()]


def answer_fitness_question(question, knowledge_base, profile=None):
    activities = knowledge_base.get("activities", [])
    profile = profile or {}
    preferences = _normalize_preferences(profile)

    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            payload = {
                "activity_library": activities,
                "profile": profile,
                "preferred_activities": preferences,
                "question": question,
            }
            response = OpenAI().responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                instructions=(
                    "You are a helpful fitness planning chatbot. Ground activity facts in the supplied library. "
                    "Do not diagnose, prescribe treatment or make medical claims. Respect the selected preferred activities and current schedule context."
                ),
                input=json.dumps(payload, default=str),
            )
            return response.output_text
        except Exception:
            pass

    q = question.lower()
    prefs_text = ", ".join(preferences) if preferences else "your selected activities"

    if "schedule all" in q or "preferred activities" in q or "daily plan" in q:
        return f"Your preferred activities are: {prefs_text}. I can schedule a feasible subset based on your available time, busy blocks, and planning window. If time is tight, I will prioritize the activities that best match your goal and feasibility constraints."

    if "available time" in q or "fit into" in q or "remaining" in q:
        return f"For the activities you selected ({prefs_text}), the planner checks each activity's duration, the schedule window, and your busy blocks before choosing a feasible subset. Activities that do not fit are left unscheduled and explained clearly."

    if "replace" in q and "walking" in q and "cycling" in q:
        return "A cycling block can replace a walking block when you want more cardio or when walking is not the best fit for the available time window. The planner will check duration and free time before scheduling it."

    if "add yoga" in q or "yoga" in q:
        return "Yoga is a strong fit for flexibility, recovery, and shorter mindful sessions. It works well as a focused block near the middle or end of the day, depending on your time window."

    if "reduce" in q and "duration" in q:
        return "To reduce exercise duration, the planner can shorten each activity to its minimum supported length while still respecting the user goal and time budget. This keeps the schedule feasible without too much overlap."

    if "unscheduled" in q or "not scheduled" in q:
        return "Unscheduled activities are simply those that do not fit within the available free windows or do not fit the time budget. The planner explains the reason and keeps the schedule feasible."

    if "consistency" in q:
        return "For consistency, choose activities you can repeat comfortably and keep realistic. Walking, yoga, stretching, and mobility work are commonly chosen for repeatable routines."

    if "flexib" in q or "stretch" in q:
        return "Stretching and mobility exercises are a good fit for flexibility and recovery. They often work well in shorter blocks and can help reduce stiffness."

    if "equipment" in q:
        return "Equipment varies by activity. Walking, stretching, yoga, meditation, breathing exercises, and mobility work usually require no equipment, while cycling, swimming, and strength training may require more specific gear or a venue."

    if any(w in q for w in ["choose", "planner", "schedule"]):
        return "The planner ranks activities against your goal, checks your preferred activities, looks at available free windows, and validates the schedule against busy blocks and time budget. It then keeps only feasible activities in the final plan."

    if any(w in q for w in ["medical", "health", "pain", "injur"]):
        return "This is not medical advice, diagnosis, or treatment. Stop if you experience pain, dizziness, or unusual discomfort, and seek professional guidance when needed."

    if preferences:
        return (
            f"I can help with your selected activities: {prefs_text}. "
            "I can recommend which of them fit the available time, suggest substitutions, and explain why some are left out when the schedule is constrained."
        )

    return "I can help with activity choices, time planning, the app's rules, and the activity library. Try a recommended question above."
