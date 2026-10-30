"""Constraint-aware activity planner and personal assistant agent."""
from dataclasses import dataclass, asdict
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Optional, Iterable, Union
import json
from pathlib import Path

KB_PATH = Path(__file__).with_name("knowledge_base.json")


@dataclass
class Activity:
    id: str
    name: str
    category: str
    min_minutes: int
    max_minutes: int
    intensity: str
    equipment: str
    tags: List[str]
    description: str


@dataclass
class TimeBlock:
    start: datetime
    end: datetime
    label: str = "Busy"


@dataclass
class PlannedActivity:
    activity_id: str
    activity_name: str
    start: datetime
    end: datetime
    duration_minutes: int
    reason: str
    intensity: str


def load_knowledge_base(path: Path = KB_PATH) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_time(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


def make_datetime(day: date, value: str) -> datetime:
    return datetime.combine(day, parse_time(value))


def overlaps(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    return a_start < b_end and b_start < a_end


def normalize_preferred_activities(preferred_activities: Optional[Union[str, Iterable[str]]]) -> List[str]:
    if preferred_activities is None:
        return []
    if isinstance(preferred_activities, str):
        raw = [preferred_activities]
    else:
        raw = list(preferred_activities)

    cleaned = []
    for item in raw:
        if not item:
            continue
        value = str(item).strip()
        if value:
            cleaned.append(value)
    return cleaned


def free_windows(day_start: datetime, day_end: datetime, busy_blocks: List[TimeBlock]) -> List[tuple]:
    """Return free intervals after clipping and merging busy blocks."""
    clipped = []
    for block in busy_blocks:
        start, end = max(day_start, block.start), min(day_end, block.end)
        if start < end:
            clipped.append((start, end))
    clipped.sort(key=lambda x: x[0])
    merged = []
    for start, end in clipped:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    result, cursor = [], day_start
    for start, end in merged:
        if cursor < start:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < day_end:
        result.append((cursor, day_end))
    return result


def rank_activity(activity: dict, goal: str, preferred_activities: List[str]) -> int:
    score = 0
    goal_tag = {"Build consistency": "consistency", "Improve flexibility": "flexibility",
                "Stay active": "stay_active", "Improve endurance": "endurance"}.get(goal, "stay_active")
    if goal_tag in activity.get("tags", []):
        score += 5

    names = [item.lower() for item in preferred_activities]
    activity_name = activity["name"].lower()
    for pref in names:
        if pref in activity_name or pref.replace(" ", "") in activity_name.replace(" ", ""):
            score += 12
        if pref in activity.get("category", "").lower():
            score += 4

    if not preferred_activities:
        score += 1
    return score


def resolve_activity_duration(activity: dict, preferred_activities: List[str], target_duration: int, activity_durations: Optional[Dict[str, int]] = None) -> int:
    lookup = activity.get("name", "").lower()
    lookup_id = activity.get("id", "").lower()
    if isinstance(activity_durations, dict):
        for key in [activity.get("name"), activity.get("id"), activity.get("category")]:
            if key and str(key).lower() in {k.lower(): v for k, v in activity_durations.items()}:
                value = {k.lower(): v for k, v in activity_durations.items()}.get(str(key).lower())
                return max(activity["min_minutes"], min(value, activity["max_minutes"]))
        for pref in preferred_activities:
            if pref.lower() in lookup:
                if pref.lower() in {k.lower(): v for k, v in activity_durations.items()}:
                    value = {k.lower(): v for k, v in activity_durations.items()}[pref.lower()]
                    return max(activity["min_minutes"], min(value, activity["max_minutes"]))
    preferred_target = max(activity["min_minutes"], min(int(target_duration), activity["max_minutes"]))
    if not preferred_activities:
        return preferred_target
    return max(activity["min_minutes"], min(preferred_target, activity["max_minutes"]))


def search_schedule(activities: List[dict], goal: str, preference: Union[str, List[str], None],
                    day_start: datetime, day_end: datetime, busy_blocks: List[TimeBlock],
                    available_minutes: int, target_duration: int = 20,
                    activity_durations: Optional[Dict[str, int]] = None) -> List[PlannedActivity]:
    """Greedy search over free windows; returns a feasible plan with a hard time cap."""
    preferred_activities = normalize_preferred_activities(preference)
    windows = free_windows(day_start, day_end, busy_blocks)
    ranked = sorted(activities, key=lambda a: rank_activity(a, goal, preferred_activities), reverse=True)
    remaining = max(0, available_minutes)
    plan = []
    used_names = set()

    for window_start, window_end in windows:
        if remaining <= 0:
            break
        window_minutes = int((window_end - window_start).total_seconds() // 60)
        if window_minutes < 5:
            continue

        for activity in ranked:
            if activity["name"] in used_names:
                continue
            duration = resolve_activity_duration(activity, preferred_activities, target_duration, activity_durations)
            duration = min(duration, remaining, window_minutes)
            if duration < activity["min_minutes"]:
                continue
            end = window_start + timedelta(minutes=duration)
            if end > window_end:
                continue
            reason = (
                f"Matches your goal '{goal}' and preferred activities "
                f"({', '.join(preferred_activities) if preferred_activities else 'general fitness'})."
            )
            plan.append(PlannedActivity(activity["id"], activity["name"], window_start, end,
                                        duration, reason, activity["intensity"]))
            used_names.add(activity["name"])
            remaining -= duration
            break
    return plan


class PersonalAssistantAgent:
    """Simple observe-plan-validate-explain assistant model."""

    def __init__(self, knowledge_base: Optional[dict] = None):
        self.kb = knowledge_base or load_knowledge_base()

    def observe(self, user_profile: dict) -> dict:
        preferred = normalize_preferred_activities(user_profile.get("preferred_activities") or user_profile.get("preferred_activity") or user_profile.get("preference"))
        required = ["goal", "available_minutes", "day_start", "day_end"]
        missing = [key for key in required if key not in user_profile]
        return {"profile": user_profile, "preferred_activities": preferred, "missing_fields": missing}

    def plan(self, user_profile: dict, busy_blocks: List[TimeBlock]) -> List[PlannedActivity]:
        day = user_profile.get("day", date.today())
        start = make_datetime(day, user_profile["day_start"])
        end = make_datetime(day, user_profile["day_end"])
        if end <= start:
            raise ValueError("Planning window end must be later than its start.")

        activity_durations = user_profile.get("activity_durations", {}) or {}
        preferred_activities = user_profile.get("preferred_activities") or user_profile.get("preferred_activity") or user_profile.get("preference")
        return search_schedule(
            self.kb["activities"],
            user_profile["goal"],
            preferred_activities,
            start,
            end,
            busy_blocks,
            int(user_profile["available_minutes"]),
            int(user_profile.get("target_duration", 20)),
            activity_durations,
        )

    def validate(self, plan: List[PlannedActivity], busy_blocks: List[TimeBlock],
                 available_minutes: int) -> dict:
        total = sum(item.duration_minutes for item in plan)
        no_busy_overlap = all(
            not overlaps(item.start, item.end, block.start, block.end)
            for item in plan for block in busy_blocks
        )
        ordered = all(plan[i].end <= plan[i + 1].start for i in range(len(plan) - 1))
        unique_activities = len({item.activity_name for item in plan})
        return {
            "feasible": total <= available_minutes and no_busy_overlap and ordered,
            "total_minutes": total,
            "within_time_budget": total <= available_minutes,
            "no_busy_overlap": no_busy_overlap,
            "chronological": ordered,
            "unique_activities": unique_activities,
        }

    def explain(self, plan: List[PlannedActivity], validation: dict) -> str:
        if not plan:
            return ("No feasible activity block was found for the current planning window. "
                    "Try increasing the available time or adjusting the window.")
        items = "; ".join(f"{p.activity_name} for {p.duration_minutes} minutes" for p in plan)
        return (f"Suggested plan: {items}. The schedule validator reports "
                f"{'feasible' if validation['feasible'] else 'not fully feasible'} "
                f"with {validation['total_minutes']} planned minutes. "
                "This is general, non-clinical activity planning.")


def plan_to_dict(plan: List[PlannedActivity]) -> List[dict]:
    return [{**asdict(item), "start": item.start.isoformat(), "end": item.end.isoformat()} for item in plan]
