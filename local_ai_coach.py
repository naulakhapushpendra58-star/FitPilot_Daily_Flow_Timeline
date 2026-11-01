import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


def _to_minutes(value: Optional[str]) -> int:
    if not value:
        return 0
    try:
        text = str(value).strip()
        hour_text, minute_text = text.split(":", 1)
        return int(hour_text) * 60 + int(minute_text)
    except (TypeError, ValueError):
        return 0


def _normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _contains_any(question: str, terms: Iterable[str]) -> bool:
    q = question.lower()
    return any(term.lower() in q for term in terms)


def _sorted_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(items, key=lambda item: (_to_minutes(item.get("start_time") or "23:59"), str(item.get("activity_name") or "")))


def _count_pending(items: List[Dict[str, Any]]) -> int:
    return sum(1 for item in items if not (item.get("completed") or str(item.get("status") or "").lower() in {"completed", "skipped"}))


def _format_activity_list(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "You haven't generated a daily plan yet. Open the Planner, enter your preferences, and click Generate Plan to create your routine."

    lines = [
        f"{idx}. {item.get('activity_name') or 'Activity'} ({item.get('start_time') or '--:--'} to {item.get('end_time') or '--:--'}, {item.get('duration_minutes') or 0} min)"
        for idx, item in enumerate(_sorted_items(items), start=1)
    ]
    return "Your routine today: " + "; ".join(lines)


def _project_only_reply() -> str:
    return "I'm FitPilot's project assistant. I can help you with your daily plan, Planner, Dashboard, fitness activities, timeline, and FitPilot features. Please ask me something related to the project."


def _no_plan_reply() -> str:
    return "You haven't generated a daily plan yet. Open the Planner, enter your preferences, and click Generate Plan to create your routine."


def _describe_goal(profile: Dict[str, Any]) -> str:
    goal = profile.get("goal") or "Build consistency"
    experience = profile.get("experience") or "Beginner"
    return f"Your goal is {goal}. Your experience level is {experience}."


def _handle_next_activity(items: List[Dict[str, Any]], profile: Dict[str, Any]) -> str:
    if not items:
        return _no_plan_reply()
    pending = [item for item in _sorted_items(items) if not (item.get("completed") or str(item.get("status") or "").lower() in {"completed", "skipped"})]
    if not pending:
        return "There are no pending activities left in your current plan. You can mark items complete or add a new one in the Planner."

    next_item = pending[0]
    start = next_item.get("start_time") or "--:--"
    end = next_item.get("end_time") or "--:--"
    name = next_item.get("activity_name") or "Activity"
    return f"Your next activity is {name}, scheduled from {start} to {end}."


def generate_local_coach_response(question: str, profile: Optional[Dict[str, Any]] = None, activity_library: Optional[List[Dict[str, Any]]] = None, history: Optional[List[Dict[str, str]]] = None, provider: Optional[str] = None) -> str:
    profile = profile or {}
    activity_library = activity_library or []
    q = (question or "").strip()
    if not q:
        return "Please ask a question about your FitPilot plan, timeline, dashboard, or preferences."

    lowered = q.lower()
    project_keywords = [
        "fitpilot", "planner", "plan", "dashboard", "timeline", "routine", "schedule", "activity",
        "work", "study", "break", "sleep", "wake", "goal", "exercise", "fitness", "custom",
        "daily", "progress", "habit", "walking", "stretching", "fatigue", "coach", "workout",
        "run", "running", "cycling", "cycle", "yoga", "strength", "recovery", "motivation"
    ]
    if not any(keyword in lowered for keyword in project_keywords):
        return _project_only_reply()

    schedule = profile.get("schedule") or {}
    items = _sorted_items(schedule.get("items") or [])
    preferred = _normalize_list(profile.get("preferred_activities") or profile.get("preferred_activity") or profile.get("preference") or [])
    prefs_text = ", ".join(preferred) if preferred else "your selected activities"
    goal = profile.get("goal") or "Build consistency"
    wake_time = profile.get("wake_time") or "07:00"
    sleep_time = profile.get("sleep_time") or "23:00"
    work_start = profile.get("work_start") or "09:00"
    work_end = profile.get("work_end") or "17:00"

    if _contains_any(lowered, ["show my complete routine", "show my routine", "complete daily routine", "show my schedule", "routine dikhao", "mera routine", "daily routine", "what activities are scheduled today", "schedule today", "show my complete daily routine"]):
        return _format_activity_list(items)

    if _contains_any(lowered, ["next activity", "what is my next activity", "meri next activity", "next task", "what should i do next", "what is next", "what is my next task", "kya hai next activity", "next activity kya hai"]):
        return _handle_next_activity(items, profile)

    if _contains_any(lowered, ["how many activities are pending", "pending activities", "how many are pending", "remaining activities", "what is left", "kitne activities pending", "pending count"]):
        pending = _count_pending(items)
        if not items:
            return _no_plan_reply()
        return f"You currently have {pending} pending activity(s) in your routine."

    if _contains_any(lowered, ["when should i sleep", "what time should i sleep", "sleep time", "kab sona chahiye", "mujhe kab sona chahiye", "sleep"]):
        return f"Your current planned sleep time is {sleep_time}."

    if _contains_any(lowered, ["when does my work start", "work start", "when does work start", "work/study start", "study start", "kab kaam start", "work start time"]):
        return f"Your work or study window starts at {work_start}."

    if _contains_any(lowered, ["when does my work end", "work end", "when does work end", "work/study end", "study end", "kab kaam khatam hota hai", "work end time"]):
        return f"Your work or study window ends at {work_end}."

    if _contains_any(lowered, ["when is my lunch break", "lunch break", "lunch time", "kab lunch", "lunch", "lunch schedule"]):
        lunch_time = profile.get("lunch_time") or "12:30"
        return f"Your current planned lunch time is {lunch_time}."

    if _contains_any(lowered, ["what should i do after waking up", "after waking up", "what to do after wake up", "wake up routine", "morning routine"]):
        if not items:
            return _no_plan_reply()
        first_activity = _sorted_items(items)[0]
        name = first_activity.get("activity_name") or "Activity"
        time_value = first_activity.get("start_time") or wake_time
        return f"After waking up, your first planned block is {name} starting at {time_value}."

    if _contains_any(lowered, ["how do i generate a daily plan", "how do i generate a plan", "generate plan", "how do i make my routine", "why is my plan not generating", "planner not working", "generate daily plan"]):
        return (
            "Open the Planner, fill in your daily goal, experience level, wake and sleep times, work window, available fitness minutes, preferred activities, and custom items if needed. "
            "Then click Generate Plan. If the plan does not generate, check that wake time is earlier than sleep time and all required values are valid."
        )

    if _contains_any(lowered, ["what is my daily goal", "my daily goal", "what is my goal", "goal"]):
        return f"Your daily goal is {goal}."

    if _contains_any(lowered, ["what does my experience level mean", "experience level", "beginner", "intermediate", "advanced", "my experience"]):
        experience = profile.get("experience") or "Beginner"
        lookup = {
            "beginner": "A beginner plan focuses on consistency, easy-to-repeat movement, and recovery.",
            "intermediate": "An intermediate plan balances consistency with a bit more intensity and structure.",
            "advanced": "An advanced plan can handle more training load, higher intensity, and shorter recovery windows.",
        }
        return f"Your experience level is {experience}. {lookup.get(str(experience).lower(), 'This helps the planner balance intensity with recovery.') }"

    if _contains_any(lowered, ["how does my fitness duration work", "fitness duration", "available minutes", "available fitness duration", "target duration"]):
        available = profile.get("available_minutes") or 120
        target = profile.get("target_duration") or 20
        return f"Your planner uses {available} minutes as the total available fitness window and targets activity blocks of about {target} minutes when possible."

    if _contains_any(lowered, ["how can i change my routine", "change my routine", "adjust my routine", "improve my routine", "how to adjust schedule"]):
        return (
            "You can change your routine from the Planner by editing your wake/sleep times, work window, preferred activities, and custom activity rows. "
            "After updating the form, click Generate Plan again to rebuild the schedule."
        )

    if _contains_any(lowered, ["what is the purpose of walking", "walking purpose", "why walking", "benefits of walking", "walking is good"]):
        return "Walking is a simple, repeatable activity that supports cardiovascular health, energy, recovery, and stress management while fitting into a realistic routine."

    if _contains_any(lowered, ["what is the purpose of stretching", "stretching purpose", "benefits of stretching", "why stretching", "stretching"]):
        return "Stretching helps improve mobility, reduces stiffness, supports recovery, and is a good complement to work, long sitting periods, and exercise blocks."

    if _contains_any(lowered, ["how can i include more breaks", "more breaks", "add more breaks", "include more breaks", "breaks"]):
        return "Add or increase your break duration in the Planner and regenerate the daily plan. A short recovery break after focused work or exercise helps reduce fatigue and protects consistency."

    if _contains_any(lowered, ["how can i improve consistency", "improve consistency", "maintain consistency", "be more consistent"]):
        return "Consistency is strongest when the routine is repeatable and realistic. Keep the same wake-up and sleep times, use a few preferred activities you enjoy, and protect your recovery windows before adding more intensity."

    if _contains_any(lowered, ["why was this activity included", "why was this activity included in my routine", "why this activity", "included in my plan", "recommended activity"]):
        if not items:
            return _no_plan_reply()
        first = _sorted_items(items)[0]
        return f"{first.get('activity_name') or 'This activity'} was included because it matches your goal ({goal}), preferences ({prefs_text}), and the available time window while staying realistic and non-overlapping."

    if _contains_any(lowered, ["explain my dashboard", "what does my dashboard mean", "dashboard significance", "what does overall progress mean", "dashboard"]):
        return "The Dashboard shows your latest generated plan, totals, completion counts, and progress. It uses the same saved schedule as the Planner, so your timeline and stats stay synchronized after each generation."

    if _contains_any(lowered, ["how are completed and pending activities calculated", "how is progress calculated", "completed and pending", "overall progress", "count progress"]):
        completed = sum(1 for item in items if item.get("completed") or str(item.get("status") or "").lower() == "completed")
        skipped = sum(1 for item in items if str(item.get("status") or "").lower() == "skipped")
        pending = max(len(items) - completed - skipped, 0)
        return f"The Dashboard counts completed items, skipped items, and pending items from your schedule. Right now, your routine has {completed} completed, {skipped} skipped, and {pending} pending activities."

    if _contains_any(lowered, ["how can i mark an activity as completed", "mark activity complete", "completed activity", "mark as completed"]):
        return "Use the Complete button on any timeline item. The Dashboard updates the schedule status and reflects the change immediately."

    if _contains_any(lowered, ["why is my timeline empty", "dashboard empty", "timeline not updating", "why is the timeline empty", "empty timeline", "timeline empty"]):
        return "The timeline is empty when no plan has been generated or the current schedule has been cleared. Generate a plan in the Planner to populate the dashboard and timeline with your current day."

    if _contains_any(lowered, ["how can i add an activity", "add an activity", "add custom activity", "custom activity", "how do i add a custom activity", "add activity in planner"]):
        return "Open the Planner, use the Custom routine activities section, add a name, start and end time, and category, and then save by generating a plan. Custom activities are included in the timeline when they fit the day."

    if _contains_any(lowered, ["how can i edit my routine", "edit my routine", "edit an activity", "remove activity", "delete activity"]):
        return "Use the Planner to change your selected preferences and custom-row values, or use the activity controls in the timeline to mark items complete or skip them. Regenerate the plan after major changes."

    if _contains_any(lowered, ["how does the local ai coach work", "local ai coach", "how does this work", "what is this assistant"]):
        return "FitPilot's local coach is a deterministic project assistant. It checks your question for intent, reads the current plan and preferences from the app, and answers using the actual schedule and project rules instead of an external API."

    if _contains_any(lowered, ["why is generate plan not working", "generate plan not working", "why is planner not working", "not generating", "plan not generate"]):
        return "If Generate Plan is not working, confirm the wake time is earlier than sleep time, work and study windows are valid, available minutes are greater than zero, and required fields are filled. The planner will reject invalid time windows before creating a schedule."

    if _contains_any(lowered, ["what is my next activity", "what is my next", "whats my next activity", "mujhe next activity kya hai", "next activity kya hai", "mera next activity"]):
        return _handle_next_activity(items, profile)

    if _contains_any(lowered, ["show my complete routine", "show routine", "show my daily routine", "show my complete daily routine"]):
        return _format_activity_list(items)

    if _contains_any(lowered, ["what activities are scheduled today", "scheduled today", "what are my activities", "list my activities", "my activities"]):
        return _format_activity_list(items)

    if _contains_any(lowered, ["what is my daily goal", "what is my goal", "goal of the day"]):
        return f"Your daily goal is {goal}."

    if _contains_any(lowered, ["how can i add more breaks", "add break", "should i take a break"]):
        return "Use the Planner to increase your break duration and regenerate the plan. The schedule should leave enough recovery time to keep the day sustainable."

    if _contains_any(lowered, ["what time should i wake up", "wake time", "when should i wake"]):
        return f"Your current wake-up time is {wake_time}."

    if _contains_any(lowered, ["how much time is remaining in my day", "remaining in my day", "time remaining", "kitna time bacha hai"]):
        if not items:
            return _no_plan_reply()
        total_schedule_minutes = sum(int(item.get("duration_minutes") or 0) for item in items)
        remaining = max((int(profile.get("available_minutes") or 0) - total_schedule_minutes), 0)
        return f"Based on your current schedule, about {remaining} minutes remain in your fitness window."

    if _contains_any(lowered, ["what is my next activity?", "what is my next activity", "mera next activity"]):
        return _handle_next_activity(items, profile)

    if _contains_any(lowered, ["show my complete routine.", "show my complete routine", "show my routine.", "show my routine"]):
        return _format_activity_list(items)

    if _contains_any(lowered, ["why is dashboard empty", "why my dashboard empty", "why is my dashboard empty"]):
        return "The Dashboard is empty when no generated plan has been saved yet. Return to the Planner, create a valid routine, and click Generate Plan to populate the dashboard and timeline."

    if _contains_any(lowered, ["what does my goal mean", "explain my goal"]) :
        return f"Your goal is {goal}. The planner uses it to choose activities that fit your desired outcome and keep the schedule realistic."

    if _contains_any(lowered, ["how do i use fitpilot", "what can i do in fitpilot", "what is fitpilot"]):
        return "FitPilot helps you plan daily fitness, study, work, breaks, and recovery around your preferences and time constraints. Use the Planner to generate a day plan and the Dashboard to view your timeline and progress."

    if _contains_any(lowered, ["what should i do today", "what should i do", "how should i structure my day", "structure my day"]):
        if not items:
            return _no_plan_reply()
        return "Your current routine already structures the day with the planned flow from wake-up to sleep. Review the timeline and start with the earliest uncompleted activity."

    if _contains_any(lowered, ["what is my schedule", "schedule details", "plan details", "show plan"]):
        return _format_activity_list(items)

    if _contains_any(lowered, ["how can i remove an activity", "remove activity", "delete activity from plan"]):
        return "Use the planner to update or regenerate the schedule, or remove the unwanted custom row before generating the plan again. The timeline also supports marking items as skipped when you decide not to do them."

    if _contains_any(lowered, ["what workout should i do", "what activity should i do", "workout suggestion", "fitness suggestion", "exercise suggestion", "suggest a workout", "what should i do for exercise", "best workout", "what exercises can i do"]):
        if not items:
            return _no_plan_reply()
        preferred_summary = prefs_text if prefs_text != "your selected activities" else "low-impact movement"
        return f"Based on your goal ({goal}) and your selected activities ({preferred_summary}), a strong option is a short warm-up, 20–30 minutes of focused movement, and a brief cooldown or stretch. Keep it within your available time and leave recovery breaks in the plan."

    if _contains_any(lowered, ["walking", "walk", "brisk walk", "nature walk", "walking routine"]):
        if not items:
            return _no_plan_reply()
        return f"Walking is a strong low-impact option for consistency, recovery, and energy. Given your available window of {profile.get('available_minutes') or 120} minutes, a 20–30 minute walk fits well alongside {prefs_text}."

    if _contains_any(lowered, ["running", "run", "jog", "interval run", "sprint"]):
        return "Running works best when your plan already includes recovery and a manageable pace. Start easy, keep sessions controlled, and add breaks or lighter days if you're building consistency."

    if _contains_any(lowered, ["cycling", "cycle", "bike", "bicycle"]):
        return "Cycling is a good cardio option when you want lower joint impact than running. A steady 20–35 minute ride can work well if your schedule has enough time and your goal is endurance or general fitness."

    if _contains_any(lowered, ["yoga", "mobility", "mobility workout", "stretch flow"]):
        return "Yoga or mobility work is useful for flexibility, stress reduction, and recovery. It is especially helpful after long study or work blocks and works well as a short evening routine."

    if _contains_any(lowered, ["strength training", "strength", "weights", "gym", "resistance", "lifting"]):
        return "Strength training is best when paired with recovery. Keep sessions focused and realistic, and allow enough rest between hard efforts so the plan remains sustainable."

    if _contains_any(lowered, ["study work balance", "work study balance", "balance work and study", "study break", "work life balance", "balance my day"]):
        return "A realistic balance keeps deep work or study blocks focused, then adds movement or recovery before the next demanding period. Protect your breaks and activity windows so the day stays sustainable."

    if _contains_any(lowered, ["how can i improve consistency", "improve consistency", "maintain consistency", "be more consistent", "fitness consistency", "motivation"]):
        return "Consistency gets stronger when the routine is realistic. Keep your wake and sleep times steady, use a few preferred activities you enjoy, and protect your recovery breaks before increasing intensity."

    if _contains_any(lowered, ["how can i recover after workout", "recovery after workout", "rest and recovery", "how to recover", "recovery tips"]):
        return "Recovery is part of the plan. Keep a short stretch, hydration, and a break after exercise, and avoid piling too much intensity on back-to-back days."

    if _contains_any(lowered, ["what should i do today", "what should i do", "how should i structure my day", "structure my day", "what is my plan for today"]):
        if not items:
            return _no_plan_reply()
        return "Your current routine already structures the day from wake-up to sleep. Review the timeline and start with the earliest uncompleted activity."

    if _contains_any(lowered, ["what is my selected activity", "selected activities", "my preferences", "preferred activities", "what activities do i like"]):
        return f"Your current preferences include {prefs_text}."

    return _project_only_reply()
