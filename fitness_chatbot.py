import json
import re
from typing import Any, Dict, Iterable, List

RECOMMENDED_QUESTIONS = [
    "Create my complete daily routine",
    "Explain my current schedule",
    "Why did you recommend this activity?",
    "How can I improve my productivity?",
    "What can I do in 30 minutes?",
    "Suggest a better exercise routine",
    "Add more breaks to my schedule",
    "Explain the benefits of walking",
    "How can I maintain consistency?",
    "What activities are still remaining today?",
    "Change my schedule based on my available time",
    "Suggest activities based on my selected preferences",
]


def _normalize_preferences(profile=None):
    profile = profile or {}
    preferred = profile.get("preferred_activities") or profile.get("preferred_activity") or profile.get("preference") or []
    if isinstance(preferred, str):
        return [item.strip() for item in preferred.split(",") if item.strip()]
    if isinstance(preferred, list):
        return [str(item).strip() for item in preferred if str(item).strip()]
    return [str(preferred).strip()] if str(preferred).strip() else []


def _schedule_summary(profile: Dict[str, Any]) -> str:
    schedule = profile.get("schedule") or {}
    items = schedule.get("items") or []
    if not items:
        return "No schedule has been generated yet."

    planned = [item for item in items if not item.get("completed")]
    completed_items = [item for item in items if item.get("completed")]
    names = [item.get("activity_name") for item in planned[:3] if item.get("activity_name")]
    completed_names = [item.get("activity_name") for item in completed_items[:3] if item.get("activity_name")]

    summary = f"Your current schedule includes {len(items)} items."
    if planned:
        summary += f" Remaining activities include {', '.join(names) if names else 'several planned blocks'}."
    if completed_items:
        summary += f" Completed so far: {', '.join(completed_names) if completed_names else 'some blocks'}."
    return summary


def _question_tokens(question: str) -> List[str]:
    raw = re.sub(r"[^a-z0-9\s]", " ", question.lower())
    return [token for token in raw.split() if token]


def _contains_any(question: str, terms: Iterable[str]) -> bool:
    q = question.lower()
    return any(term.lower() in q for term in terms)


def _fallback_answer(question: str, knowledge_base: Dict[str, Any], profile: Dict[str, Any]) -> str:
    activities = knowledge_base.get("activities", [])
    preferred = _normalize_preferences(profile)
    prefs_text = ", ".join(preferred) if preferred else "your selected activities"
    schedule = profile.get("schedule") or {}
    items = schedule.get("items") or []
    remaining = [item.get("activity_name") for item in items if not item.get("completed")]
    available_minutes = profile.get("available_minutes") or 120
    goal = profile.get("goal") or "Build consistency"
    work_start = profile.get("work_start") or "09:00"
    work_end = profile.get("work_end") or "17:00"
    wake_time = profile.get("wake_time") or "07:00"
    sleep_time = profile.get("sleep_time") or "23:00"

    q = question.lower()

    if _contains_any(q, ["create my complete daily routine", "daily routine", "complete routine", "build my day", "schedule all"]):
        return (
            f"A balanced daily routine for your goal of {goal} could start with a wake-up block, then {prefs_text if preferred else 'a preferred activity'}, "
            f"followed by focused work or study from {work_start} to {work_end}, a short recovery break, and a wind-down period before {sleep_time}. "
            "The exact plan should fit your available time and preserve realistic recovery."
        )

    if _contains_any(q, ["explain my current schedule", "schedule", "current plan", "what is in my schedule", "show my schedule"]):
        summary = _schedule_summary(profile)
        if remaining:
            return f"{summary} Right now, your remaining work is concentrated on {', '.join(remaining[:3])}. The plan is designed to keep the most relevant blocks in your free windows while respecting your day structure."
        return f"{summary} Your plan is fully accounted for, so the next step is to review completion, keep consistency, and protect recovery time."

    if _contains_any(q, ["why did you recommend this activity", "why this activity", "recommend this activity", "recommended this activity"]):
        if preferred:
            return (
                f"I prioritize activities that align with your goal ({goal}) and your selected preferences: {prefs_text}. "
                "That helps keep the plan realistic, personally relevant, and sustainable without overloading your schedule."
            )
        return (
            f"I recommend activities based on your goal ({goal}), time window, and the constraints in your day such as work blocks and available minutes. "
            "This keeps the plan specific, feasible, and easier to maintain consistently."
        )

    if _contains_any(q, ["improve my productivity", "productivity", "focus", "better productivity", "be more productive"]):
        return (
            "To improve productivity, protect your deep-focus windows, use consistent work blocks, and keep activity intensity aligned with your available energy. "
            "Short movement breaks, hydration, and realistic scheduling often increase focus better than trying to do too much at once."
        )

    if _contains_any(q, ["30 minutes", "half hour", "in 30", "what can i do in 30 minutes"]):
        if preferred:
            return (
                f"With 30 minutes available, a strong FitPilot choice is a focused session using one of your selected activities such as {prefs_text}. "
                "For example, a brisk walk, a mobility block, or a short strength circuit with a brief warm-up and cool-down."
            )
        return "With 30 minutes available, a short walk, mobility session, or a focused bodyweight circuit is a strong option. Keep it efficient and realistic so consistency stays high."

    if _contains_any(q, ["suggest a better exercise routine", "better exercise routine", "exercise routine", "what routine", "routine suggestion"]):
        if preferred:
            return (
                f"A better routine for you is a repeating pattern built around {prefs_text} and your goal of {goal}. "
                "Aim for a realistic mix of movement, recovery, and focus, and keep the plan sustainable across the week rather than maximizing every minute."
            )
        return f"A better routine should align with your goal ({goal}) and the time you can consistently protect. A realistic mix of aerobic work, mobility, and recovery usually performs better than a highly intense plan."

    if _contains_any(q, ["add more breaks", "more breaks", "breaks in my schedule", "rest break", "increase breaks"]):
        return (
            "A healthy cadence usually includes a short recovery break every 60 to 90 minutes, especially during work or intense focus sessions. "
            "Adding a hydration break, a mobility reset, and a short wind-down period can improve energy and consistency without making the day feel crowded."
        )

    if _contains_any(q, ["benefits of walking", "walking", "why walking", "is walking good", "benefits of exercise"]):
        return (
            "Walking is accessible, low friction, and easy to repeat consistently. It supports cardiovascular health, boosts energy, improves recovery, and can be used as a stress-relief activity without needing special equipment."
        )

    if _contains_any(q, ["maintain consistency", "consistency", "stay consistent", "be more consistent"]):
        return (
            "Consistency works best when the plan is realistic and easy to repeat. Protect a stable wake-up and movement window, choose a few preferred activities you enjoy, and avoid overloading the day with too many intense blocks."
        )

    if _contains_any(q, ["remaining today", "what activities are still remaining", "remaining activities", "still remaining", "what is left"]):
        if remaining:
            return f"Today, the remaining activities are {', '.join(remaining[:5])}. These are still open in your plan, so you can review them against your available time and complete the highest-priority items first."
        return "Nothing appears to remain in the current schedule. That usually means your plan is fully accounted for and can be reviewed for recovery, consistency, and future planning."

    if _contains_any(q, ["change my schedule based on my available time", "adjust schedule", "based on my available time", "with my available time", "available time"]):
        return (
            f"With {available_minutes} minutes available, the best approach is to keep the plan focused on your highest-priority activities and trim or remove lower-value items. "
            f"The planner should prioritize {prefs_text if preferred else 'the most relevant activities'} while keeping the day realistic and sustainable."
        )

    if _contains_any(q, ["selected preferences", "based on my preferences", "my preferred activities", "preference"]):
        return (
            f"Your selected preferences are {prefs_text}. The planner uses these as a strong signal when choosing activity blocks, but it still validates against your time budget, work schedule, and free windows to keep the plan feasible."
        )

    if _contains_any(q, ["how can i improve", "suggest", "recommend", "help", "advice", "options"]) or activities:
        if preferred:
            return (
                f"Based on your goals and preferences ({prefs_text}), I would focus on a short, repeatable activity pattern, protect recovery time, and keep work blocks realistic. "
                "This tends to yield better long-term consistency than designing a schedule that is too intense for the day."
            )
        return (
            "I can help by aligning your schedule with your goal, available time, and selected preferences. The strongest plans usually combine realistic movement, short breaks, and enough recovery to stay sustainable."
        )

    if _contains_any(q, ["medical", "diagnosis", "pain", "injury", "treatment"]):
        return "This assistant is designed for daily planning, fitness, and productivity guidance. It does not diagnose medical conditions or recommend treatment. If you have pain or injury concerns, a qualified professional should guide that decision."

    if profile.get("goal"):
        return (
            f"For your goal of {goal}, I’d keep the plan realistic and repeatable. The current context suggests you have {available_minutes} minutes available, a wake time of {wake_time}, and a sleep time of {sleep_time}. "
            "I can help you refine the routine, explain the chosen activities, or adjust the schedule around your work and recovery needs."
        )

    return (
        "I’m designed to help with fitness planning, productivity, and daily routine guidance. Ask about your schedule, activity choices, time management, or wellness habits, and I’ll tailor the answer to your current FitPilot context."
    )


def answer_fitness_question(question, knowledge_base, profile=None):
    activities = knowledge_base.get("activities", [])
    profile = profile or {}
    question_text = str(question or "").strip()
    if not question_text:
        return "Please ask a question about your schedule, preferred activities, or wellness routine."

    return _fallback_answer(question_text, knowledge_base, profile)
