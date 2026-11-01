from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from database import Base, SessionLocal, engine
from fitness_chatbot import RECOMMENDED_QUESTIONS, answer_fitness_question
from llm_grounding import grounded_explanation
from models import ChatMessage, Preference, Schedule, ScheduleItem
from planner import PersonalAssistantAgent, TimeBlock, make_datetime
from schemas import (
    ActivityCreate,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    PreferenceUpdate,
    ScheduleUpdate,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Ensure directories exist across local development and Render deployments.
for directory in (STATIC_DIR, TEMPLATES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="FitPilot – AI-Powered Personal Activity Planner", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists() and STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

KB = __import__("knowledge").load_knowledge_base()
AGENT = PersonalAssistantAgent(KB)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def parse_list_field(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _is_valid_time(value: Optional[str]) -> bool:
    if not value:
        return False
    try:
        datetime.strptime(str(value), "%H:%M")
        return True
    except ValueError:
        return False


def _to_minutes(value: Optional[str]) -> int:
    if not value:
        return 0
    try:
        hour, minute = str(value).split(":", 1)
        return int(hour) * 60 + int(minute)
    except ValueError:
        return 0


def _minutes_to_time(total_minutes: int) -> str:
    total_minutes = max(0, total_minutes)
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"


def validate_schedule_inputs(payload: Dict[str, Any]) -> None:
    available_minutes = int(payload.get("available_minutes") or 0)
    target_duration = int(payload.get("target_duration") or 0)
    if available_minutes <= 0:
        raise ValueError("Available minutes must be greater than zero.")
    if target_duration <= 0:
        raise ValueError("Target duration must be greater than zero.")

    wake_time = payload.get("wake_time") or "07:00"
    sleep_time = payload.get("sleep_time") or "23:00"
    if not _is_valid_time(wake_time) or not _is_valid_time(sleep_time):
        raise ValueError("Wake time and sleep time must be valid times in HH:MM format.")
    if _to_minutes(wake_time) >= _to_minutes(sleep_time):
        raise ValueError("Wake time must be earlier than sleep time.")

    work_start = payload.get("work_start")
    work_end = payload.get("work_end")
    if bool(work_start) != bool(work_end):
        raise ValueError("Work start and work end must be provided together.")
    if work_start and work_end and not _is_valid_time(work_start):
        raise ValueError("Work start must be a valid time in HH:MM format.")
    if work_start and work_end and not _is_valid_time(work_end):
        raise ValueError("Work end must be a valid time in HH:MM format.")
    if work_start and work_end and _to_minutes(work_start) >= _to_minutes(work_end):
        raise ValueError("Work end must be later than work start.")

    if int(payload.get("study_duration") or 0) < 0:
        raise ValueError("Study duration cannot be negative.")
    if int(payload.get("break_duration") or 0) < 0:
        raise ValueError("Break duration cannot be negative.")


def find_free_slot(
    scheduled_items: List[Dict[str, Any]],
    busy_blocks: List[TimeBlock],
    window_start: str,
    window_end: str,
    duration_minutes: int,
) -> Optional[tuple[int, int]]:
    if duration_minutes <= 0:
        return None

    start_limit = _to_minutes(window_start)
    end_limit = _to_minutes(window_end)
    if end_limit <= start_limit:
        return None

    occupied_intervals = []
    for item in scheduled_items:
        start_time = item.get("start_time")
        end_time = item.get("end_time")
        if start_time and end_time:
            occupied_intervals.append((_to_minutes(start_time), _to_minutes(end_time)))
    for block in busy_blocks:
        occupied_intervals.append((int(block.start.strftime("%H%M")) // 100 * 60 + int(block.start.strftime("%H%M")) % 100,
                                 int(block.end.strftime("%H%M")) // 100 * 60 + int(block.end.strftime("%H%M")) % 100))

    occupied = sorted(interval for interval in occupied_intervals if interval[1] > interval[0])
    candidates = [(start_limit, end_limit)]

    for block_start, block_end in occupied:
        if block_end <= start_limit or block_start >= end_limit:
            continue
        next_candidates = []
        for candidate_start, candidate_end in candidates:
            if block_end <= candidate_start or block_start >= candidate_end:
                next_candidates.append((candidate_start, candidate_end))
                continue
            if candidate_start < block_start:
                next_candidates.append((candidate_start, min(block_start, candidate_end)))
            if block_end < candidate_end:
                next_candidates.append((max(block_end, candidate_start), candidate_end))
        candidates = next_candidates

    for candidate_start, candidate_end in sorted(candidates):
        if candidate_end - candidate_start >= duration_minutes:
            return candidate_start, candidate_start + duration_minutes
    return None


def schedule_block(
    items: List[Dict[str, Any]],
    name: str,
    category: str,
    reason: str,
    duration_minutes: int,
    window_start: str,
    window_end: str,
    busy_blocks: List[TimeBlock],
) -> None:
    if duration_minutes <= 0:
        return
    slot = find_free_slot(items, busy_blocks, window_start, window_end, duration_minutes)
    if slot is None:
        return
    start_minutes, end_minutes = slot
    items.append(
        {
            "activity_name": name,
            "category": category,
            "start_time": _minutes_to_time(start_minutes),
            "end_time": _minutes_to_time(end_minutes),
            "duration_minutes": duration_minutes,
            "reason": reason,
            "completed": False,
            "status": "planned",
        }
    )


def dedupe_and_sort_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique = {}
    for item in items:
        key = (item.get("activity_name"), item.get("start_time"), item.get("end_time"))
        if key in unique:
            continue
        unique[key] = item
    return sorted(unique.values(), key=lambda entry: (_to_minutes(entry.get("start_time") or "00:00"), entry.get("activity_name", "")))


def get_or_create_preference(db: Session) -> Preference:
    pref = db.query(Preference).first()
    if pref is None:
        pref = Preference()
        db.add(pref)
        db.commit()
        db.refresh(pref)
    return pref


def serialize_preference(pref: Preference) -> Dict[str, Any]:
    preferred_activities = parse_list_field(pref.preferred_activities or pref.preferred_activity)
    if not preferred_activities and pref.preferred_activity:
        preferred_activities = parse_list_field(pref.preferred_activity)
    return {
        "user_name": pref.user_name,
        "goal": pref.goal,
        "preferred_activity": pref.preferred_activity,
        "preferred_activities": preferred_activities,
        "experience": pref.experience,
        "wake_time": pref.wake_time,
        "sleep_time": pref.sleep_time,
        "work_start": pref.work_start,
        "work_end": pref.work_end,
        "study_duration": pref.study_duration,
        "break_duration": pref.break_duration,
        "available_minutes": pref.available_minutes,
        "target_duration": pref.target_duration,
        "activity_duration": pref.activity_duration,
        "productivity_goal": pref.productivity_goal,
        "priorities": pref.priorities,
    }


def normalize_profile_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw_day = payload.get("day") or date.today().isoformat()
    day = datetime.fromisoformat(raw_day).date() if isinstance(raw_day, str) else raw_day
    preferred = payload.get("preferred_activities")
    if preferred is None:
        preferred = payload.get("preferred_activity") or payload.get("preference") or []
    preferred_activities = parse_list_field(preferred)

    return {
        "day": day,
        "goal": payload.get("goal", "Build consistency"),
        "preference": payload.get("preference", "Walking"),
        "preferred_activities": preferred_activities,
        "preferred_activity": payload.get("preferred_activity") or (preferred_activities[0] if preferred_activities else "Walking"),
        "experience": payload.get("experience", "Beginner"),
        "available_minutes": int(payload.get("available_minutes", 120)),
        "target_duration": int(payload.get("target_duration", 20)),
        "day_start": payload.get("wake_time") or "07:00",
        "day_end": payload.get("sleep_time") or "23:00",
        "study_duration": int(payload.get("study_duration", 45)),
        "break_duration": int(payload.get("break_duration", 15)),
        "work_start": payload.get("work_start"),
        "work_end": payload.get("work_end"),
        "productivity_goal": payload.get("productivity_goal", "Stay focused and balanced"),
        "priorities": payload.get("priorities", "Health, focus, recovery"),
        "activity_durations": payload.get("activity_durations") or {},
    }


def build_busy_blocks(payload: Dict[str, Any]) -> List[TimeBlock]:
    profile = normalize_profile_payload(payload)
    day = profile["day"]
    busy_blocks: List[TimeBlock] = []
    work_start = payload.get("work_start")
    work_end = payload.get("work_end")
    if work_start and work_end:
        try:
            start_dt = make_datetime(day, work_start)
            end_dt = make_datetime(day, work_end)
            if end_dt > start_dt:
                busy_blocks.append(TimeBlock(start_dt, end_dt, "Work / college commitment"))
        except ValueError:
            pass

    for idx in range(1, 3):
        start_value = payload.get(f"busy_start_{idx}")
        end_value = payload.get(f"busy_end_{idx}")
        if start_value and end_value:
            try:
                start_dt = make_datetime(day, start_value)
                end_dt = make_datetime(day, end_value)
                if end_dt > start_dt:
                    busy_blocks.append(TimeBlock(start_dt, end_dt, f"Busy block {idx}"))
            except ValueError:
                continue

    return busy_blocks


def schedule_item_to_dict(item: ScheduleItem) -> Dict[str, Any]:
    return {
        "id": item.id,
        "schedule_id": item.schedule_id,
        "activity_name": item.activity_name,
        "category": item.category,
        "start_time": item.start_time,
        "end_time": item.end_time,
        "duration_minutes": item.duration_minutes,
        "reason": item.reason,
        "completed": bool(item.completed),
        "status": getattr(item, "status", "completed" if item.completed else "planned"),
    }


def schedule_to_dict(schedule: Schedule, items: List[ScheduleItem]) -> Dict[str, Any]:
    return {
        "id": schedule.id,
        "title": schedule.title,
        "date": schedule.date.isoformat() if schedule.date else None,
        "total_minutes": schedule.total_minutes,
        "status": schedule.status,
        "user_goal": schedule.user_goal,
        "notes": schedule.notes,
        "items": [schedule_item_to_dict(item) for item in items],
        "created_at": schedule.created_at.isoformat() if schedule.created_at else None,
        "updated_at": schedule.updated_at.isoformat() if schedule.updated_at else None,
    }


def generate_schedule_from_profile(profile: Dict[str, Any], busy_blocks: List[TimeBlock]) -> Dict[str, Any]:
    day = profile["day"]
    start_dt = make_datetime(day, profile["day_start"])
    end_dt = make_datetime(day, profile["day_end"])
    if end_dt <= start_dt:
        raise ValueError("Planning window end must be later than the start.")

    plan = AGENT.plan(profile, busy_blocks)
    validation = AGENT.validate(plan, busy_blocks, int(profile["available_minutes"]))
    if not plan:
        return {"schedule": {"items": []}, "validation": validation, "feasible": False}

    items = []
    for entry in sorted(plan, key=lambda item: item.start):
        items.append(
            {
                "activity_name": entry.activity_name,
                "category": "activity",
                "start_time": entry.start.strftime("%H:%M"),
                "end_time": entry.end.strftime("%H:%M"),
                "duration_minutes": entry.duration_minutes,
                "reason": entry.reason,
                "completed": False,
                "status": "planned",
            }
        )

    daily_routine = build_daily_routine(profile, plan, busy_blocks)
    return {
        "schedule": {
            "title": f"{profile['goal']} plan",
            "date": day.isoformat(),
            "total_minutes": sum(item["duration_minutes"] for item in daily_routine),
            "status": "generated",
            "user_goal": profile["goal"],
            "notes": "AI-generated plan based on constraints and preferences.",
            "items": daily_routine,
        },
        "validation": validation,
        "feasible": validation["feasible"],
    }


def get_latest_schedule(db: Session) -> Optional[Schedule]:
    return db.query(Schedule).order_by(Schedule.created_at.desc()).first()


def compute_dashboard_stats(schedule: Optional[Schedule], items: List[ScheduleItem]) -> Dict[str, Any]:
    total_minutes = sum(item.duration_minutes for item in items)
    completed_count = sum(1 for item in items if item.completed)
    remaining_count = max(len(items) - completed_count, 0)
    break_minutes = sum(
        item.duration_minutes for item in items if item.category.lower() in {"break", "recovery", "relaxation", "rest"}
    )
    progress_pct = round((completed_count / len(items) * 100), 1) if items else 0.0
    return {
        "total_minutes": total_minutes,
        "completed_count": completed_count,
        "remaining_count": remaining_count,
        "break_minutes": break_minutes,
        "progress_pct": progress_pct,
        "activity_count": len(items),
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok", message="FitPilot API is running.")


@app.get("/api/preferences")
async def get_preferences_api(db: Session = Depends(get_db)):
    pref = get_or_create_preference(db)
    return serialize_preference(pref)


@app.put("/api/preferences")
async def save_preferences(payload: PreferenceUpdate, db: Session = Depends(get_db)):
    pref = get_or_create_preference(db)
    updates = payload.model_dump(exclude_unset=True)
    for field, value in updates.items():
        if field == "preferred_activities" and isinstance(value, list):
            setattr(pref, field, ", ".join(str(item).strip() for item in value if str(item).strip()))
            continue
        if field == "preferred_activity" and value is not None:
            pref.preferred_activity = str(value)
            continue
        setattr(pref, field, value)
    db.commit()
    db.refresh(pref)
    return {"message": "Preferences saved successfully.", "preferences": serialize_preference(pref)}


@app.post("/api/schedules/clear")
async def clear_current_schedule(db: Session = Depends(get_db)):
    latest = get_latest_schedule(db)
    if not latest:
        return {"message": "No generated plan to clear."}

    db.query(ScheduleItem).filter(ScheduleItem.schedule_id == latest.id).delete()
    db.delete(latest)
    db.commit()
    return {"message": "Plan cleared successfully."}


@app.post("/api/schedules/generate")
async def generate_schedule(payload: Dict[str, Any], db: Session = Depends(get_db)):
    try:
        validate_schedule_inputs(payload)
        profile = normalize_profile_payload(payload)
        busy_blocks = build_busy_blocks(payload)
        result = generate_schedule_from_profile(profile, busy_blocks)

        if not result["feasible"] and not result["schedule"]["items"]:
            raise HTTPException(
                status_code=400,
                detail="No feasible schedule could be generated for the provided constraints.",
            )

        plan = AGENT.plan(profile, busy_blocks)
        routine = build_daily_routine(profile, plan, busy_blocks)
        generated_items = routine if routine else result["schedule"]["items"]

        schedule_record = Schedule(
            title=f"{profile['goal']} plan",
            date=profile["day"],
            total_minutes=sum(item["duration_minutes"] for item in generated_items),
            status="generated",
            user_goal=profile["goal"],
            notes="AI-generated daily routine based on preferences, time window, and work constraints.",
        )
        db.add(schedule_record)
        db.commit()
        db.refresh(schedule_record)

        for item in generated_items:
            db.add(
                ScheduleItem(
                    schedule_id=schedule_record.id,
                    activity_name=item["activity_name"],
                    category=item["category"],
                    start_time=item["start_time"],
                    end_time=item["end_time"],
                    duration_minutes=item["duration_minutes"],
                    reason=item["reason"],
                    completed=item.get("completed", False),
                )
            )
        db.commit()

        items = db.query(ScheduleItem).filter(ScheduleItem.schedule_id == schedule_record.id).all()
        explanation = grounded_explanation(profile, plan, KB["activities"])
        return {
            "message": "Schedule generated successfully.",
            "schedule": schedule_to_dict(schedule_record, items),
            "validation": result["validation"],
            "explanation": explanation,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/schedules")
async def list_schedules(db: Session = Depends(get_db)):
    schedules = db.query(Schedule).order_by(Schedule.created_at.desc()).all()
    result = []
    for schedule in schedules:
        items = db.query(ScheduleItem).filter(ScheduleItem.schedule_id == schedule.id).all()
        result.append(schedule_to_dict(schedule, items))
    return result


@app.get("/api/schedules/{schedule_id}")
async def get_schedule(schedule_id: int, db: Session = Depends(get_db)):
    schedule = db.query(Schedule).filter(Schedule.id == schedule_id).first()
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found.")
    items = db.query(ScheduleItem).filter(ScheduleItem.schedule_id == schedule.id).all()
    return schedule_to_dict(schedule, items)


@app.put("/api/schedules/{schedule_id}")
async def update_schedule(schedule_id: int, payload: ScheduleUpdate, db: Session = Depends(get_db)):
    schedule = db.query(Schedule).filter(Schedule.id == schedule_id).first()
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found.")
    if payload.title is not None:
        schedule.title = payload.title
    if payload.status is not None:
        schedule.status = payload.status
    if payload.notes is not None:
        schedule.notes = payload.notes
    if payload.items is not None:
        db.query(ScheduleItem).filter(ScheduleItem.schedule_id == schedule.id).delete()
        for item in payload.items:
            db.add(
                ScheduleItem(
                    schedule_id=schedule.id,
                    activity_name=item.activity_name,
                    category=item.category,
                    start_time=item.start_time,
                    end_time=item.end_time,
                    duration_minutes=item.duration_minutes,
                    reason=item.reason,
                    completed=item.completed,
                )
            )
    db.commit()
    items = db.query(ScheduleItem).filter(ScheduleItem.schedule_id == schedule.id).all()
    return {"message": "Schedule updated successfully.", "schedule": schedule_to_dict(schedule, items)}


@app.delete("/api/schedules/{schedule_id}")
async def delete_schedule(schedule_id: int, db: Session = Depends(get_db)):
    schedule = db.query(Schedule).filter(Schedule.id == schedule_id).first()
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found.")
    db.query(ScheduleItem).filter(ScheduleItem.schedule_id == schedule.id).delete()
    db.delete(schedule)
    db.commit()
    return {"message": "Schedule deleted successfully."}


@app.post("/api/activities")
async def create_activity(payload: ActivityCreate, db: Session = Depends(get_db)):
    schedule_id = getattr(payload, "schedule_id", None)
    if schedule_id is None:
        latest = get_latest_schedule(db)
        schedule_id = latest.id if latest else 1

    item = ScheduleItem(
        schedule_id=schedule_id,
        activity_name=payload.name,
        category=payload.category,
        start_time=payload.start_time,
        end_time=payload.end_time,
        duration_minutes=payload.duration_minutes,
        reason=payload.reason,
        completed=payload.completed,
        status="completed" if payload.completed else "planned",
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return {"message": "Activity added successfully.", "activity": schedule_item_to_dict(item)}


@app.put("/api/activities/{activity_id}")
async def update_activity(activity_id: int, payload: Dict[str, Any], db: Session = Depends(get_db)):
    item = db.query(ScheduleItem).filter(ScheduleItem.id == activity_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Activity not found.")
    for key, value in payload.items():
        setattr(item, key, value)
    db.commit()
    return {"message": "Activity updated successfully.", "activity": schedule_item_to_dict(item)}


@app.delete("/api/activities/{activity_id}")
async def delete_activity(activity_id: int, db: Session = Depends(get_db)):
    item = db.query(ScheduleItem).filter(ScheduleItem.id == activity_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Activity not found.")
    db.delete(item)
    db.commit()
    return {"message": "Activity deleted successfully."}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, db: Session = Depends(get_db)):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    pref = get_or_create_preference(db)
    schedule = None
    if payload.schedule_id:
        schedule = db.query(Schedule).filter(Schedule.id == payload.schedule_id).first()

    schedule_payload: Dict[str, Any] = {}
    if schedule:
        items = db.query(ScheduleItem).filter(ScheduleItem.schedule_id == schedule.id).all()
        schedule_payload = {
            "title": schedule.title,
            "status": schedule.status,
            "items": [schedule_item_to_dict(item) for item in items],
        }

    profile = {
        "user_name": pref.user_name,
        "goal": pref.goal,
        "preferred_activity": pref.preferred_activity,
        "preferred_activities": parse_list_field(pref.preferred_activities or pref.preferred_activity),
        "experience": pref.experience,
        "wake_time": pref.wake_time,
        "sleep_time": pref.sleep_time,
        "work_start": pref.work_start,
        "work_end": pref.work_end,
        "schedule": schedule_payload,
    }

    answer = answer_fitness_question(question, KB, profile)
    db.add(ChatMessage(role="user", content=question, schedule_id=payload.schedule_id))
    db.add(ChatMessage(role="assistant", content=answer, schedule_id=payload.schedule_id))
    db.commit()
    return ChatResponse(
        answer=answer,
        provider="openai" if os.getenv("OPENAI_API_KEY") else "grounded-fallback",
    )


@app.get("/api/ai/paradigms")
async def ai_paradigms():
    return {
        "paradigms": [
            {
                "name": "Rule-based AI",
                "how_it_works": "Explicit business and scheduling rules are applied deterministically.",
                "advantages": ["Explainable", "Reliable"],
                "limitations": ["Needs domain rules"],
                "example": "Constraint checking for time windows and busy blocks.",
            },
            {
                "name": "Search-based planning",
                "how_it_works": "The system searches for feasible time windows and validates overlaps.",
                "advantages": ["Respects hard constraints", "Efficient for modest schedules"],
                "limitations": ["Complexity increases with many constraints"],
                "example": "Find available blocks and rank activities.",
            },
            {
                "name": "Machine learning",
                "how_it_works": "Models can learn user preference patterns over time.",
                "advantages": ["Personalization", "Adaptive recommendations"],
                "limitations": ["Needs data and monitoring"],
                "example": "Preference tuning from historical completions.",
            },
            {
                "name": "Large language models",
                "how_it_works": "LLMs convert conversational requests into explanations and recommendations.",
                "advantages": ["Natural language", "Helpful follow-ups"],
                "limitations": ["Can hallucinate without grounding"],
                "example": "Answering schedule questions and productivity guidance.",
            },
            {
                "name": "Hybrid AI",
                "how_it_works": "The planner uses explicit rules while the LLM offers grounded explanations and conversational assistance.",
                "advantages": ["Balanced control and flexibility"],
                "limitations": ["More system complexity"],
                "example": "Schedule generation plus chatbot guidance.",
            },
        ]
    }


@app.get("/api/knowledge-base")
async def knowledge_base():
    return KB


@app.get("/api/health-check")
async def health_check_alias() -> Dict[str, str]:
    return {"status": "ok", "message": "FitPilot API is running."}


@app.get("/api/dashboard")
async def dashboard_data(db: Session = Depends(get_db)):
    pref = get_or_create_preference(db)
    schedule = get_latest_schedule(db)
    items = db.query(ScheduleItem).filter(ScheduleItem.schedule_id == schedule.id).all() if schedule else []
    stats = compute_dashboard_stats(schedule, items)
    return {
        "preferences": serialize_preference(pref),
        "schedule": schedule_to_dict(schedule, items) if schedule else None,
        "stats": stats,
    }


@app.patch("/api/schedule-items/{item_id}/complete")
async def mark_activity_complete(item_id: int, db: Session = Depends(get_db)):
    item = db.query(ScheduleItem).filter(ScheduleItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Activity not found.")
    item.completed = True
    item.status = "completed"
    db.commit()
    return {"message": "Activity marked as completed.", "item": schedule_item_to_dict(item)}


@app.patch("/api/schedule-items/{item_id}/skip")
async def mark_activity_skipped(item_id: int, db: Session = Depends(get_db)):
    item = db.query(ScheduleItem).filter(ScheduleItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Activity not found.")
    item.completed = False
    item.status = "skipped"
    db.commit()
    return {"message": "Activity marked as skipped.", "item": schedule_item_to_dict(item)}


@app.put("/api/schedule-items/{item_id}")
async def update_schedule_item(item_id: int, payload: Dict[str, Any], db: Session = Depends(get_db)):
    item = db.query(ScheduleItem).filter(ScheduleItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Activity not found.")
    for key, value in payload.items():
        if hasattr(item, key):
            setattr(item, key, value)
    if payload.get("completed") is not None:
        item.status = "completed" if payload["completed"] else "planned"
    db.commit()
    return {"message": "Activity updated successfully.", "item": schedule_item_to_dict(item)}


@app.delete("/api/schedule-items/{item_id}")
async def delete_schedule_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(ScheduleItem).filter(ScheduleItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Activity not found.")
    db.delete(item)
    db.commit()
    return {"message": "Activity deleted successfully."}


def to_minutes(value: str) -> int:
    if not value:
        return 0
    try:
        hour, minute = str(value).split(":", 1)
        return int(hour) * 60 + int(minute)
    except ValueError:
        return 0


def add_time_slot(items: List[Dict[str, Any]], start: str, end: str, name: str, category: str, reason: str) -> None:
    start_min = to_minutes(start)
    end_min = to_minutes(end)
    if start_min >= end_min:
        return
    items.append(
        {
            "activity_name": name,
            "category": category,
            "start_time": start,
            "end_time": end,
            "duration_minutes": max(0, end_min - start_min),
            "reason": reason,
            "completed": False,
            "status": "planned",
        }
    )


def build_daily_routine(profile: Dict[str, Any], plan: List[Any], busy_blocks: List[TimeBlock]) -> List[Dict[str, Any]]:
    wake = profile.get("day_start") or "07:00"
    sleep = profile.get("day_end") or "23:00"
    base_items: List[Dict[str, Any]] = []

    for entry in plan:
        base_items.append(
            {
                "activity_name": entry.activity_name,
                "category": "activity",
                "start_time": entry.start.strftime("%H:%M"),
                "end_time": entry.end.strftime("%H:%M"),
                "duration_minutes": entry.duration_minutes,
                "reason": entry.reason,
                "completed": False,
                "status": "planned",
            }
        )

    study_duration = max(int(profile.get("study_duration") or 0), 0)
    if study_duration > 0:
        schedule_block(
            base_items,
            "Study block",
            "focus",
            "Focused study time using the duration from your planner settings.",
            study_duration,
            wake,
            sleep,
            busy_blocks,
        )

    break_duration = max(int(profile.get("break_duration") or 0), 0)
    if break_duration > 0:
        schedule_block(
            base_items,
            "Recovery break",
            "break",
            "Short reset to keep your plan sustainable and realistic.",
            break_duration,
            wake,
            sleep,
            busy_blocks,
        )

    return dedupe_and_sort_items(base_items)
