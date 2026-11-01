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

from ai_providers import generate_chat_completion
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

    sorted_items = sorted(unique.values(), key=lambda entry: (_to_minutes(entry.get("start_time") or "00:00"), entry.get("activity_name", "")))
    cleaned = []
    for item in sorted_items:
        if not cleaned:
            cleaned.append(item)
            continue
        previous = cleaned[-1]
        prev_end = _to_minutes(previous.get("end_time") or previous.get("start_time") or "00:00")
        current_start = _to_minutes(item.get("start_time") or "00:00")
        if prev_end > current_start:
            previous_end_safe = _to_minutes(previous.get("end_time") or previous.get("start_time") or "00:00")
            current_start_safe = _to_minutes(item.get("start_time") or "00:00")
            if current_start_safe < previous_end_safe:
                previous["end_time"] = _minutes_to_time(max(previous_end_safe, current_start_safe))
                previous["duration_minutes"] = max(0, _to_minutes(previous["end_time"]) - _to_minutes(previous["start_time"]))
                continue
        cleaned.append(item)
    return cleaned


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
        "lunch_time": getattr(pref, "lunch_time", "12:30"),
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
        "lunch_time": payload.get("lunch_time") or "12:30",
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
    completed_count = sum(1 for item in items if item.completed or item.status == "completed")
    skipped_count = sum(1 for item in items if item.status == "skipped")
    pending_count = max(len(items) - completed_count - skipped_count, 0)
    break_minutes = sum(
        item.duration_minutes for item in items if (item.category or "").lower() in {"break", "recovery", "relaxation", "rest"}
    )
    progress_pct = round((completed_count / len(items) * 100), 1) if items else 0.0
    return {
        "total_minutes": total_minutes,
        "completed_count": completed_count,
        "skipped_count": skipped_count,
        "pending_count": pending_count,
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
        generated_items = merge_generated_schedule(payload, routine)
        if not generated_items:
            generated_items = merge_generated_schedule(payload, result["schedule"]["items"])

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
                    status=item.get("status", "planned"),
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

    provider = (payload.provider or "openai").lower()
    if provider not in {"openai", "gemini"}:
        raise HTTPException(status_code=400, detail="Unsupported AI provider selected.")

    pref = get_or_create_preference(db)
    schedule = None
    schedule_id = payload.schedule_id
    if schedule_id:
        schedule = db.query(Schedule).filter(Schedule.id == schedule_id).first()

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

    history = get_chat_history(db, schedule_id)

    try:
        answer = generate_chat_completion(
            question,
            profile,
            KB.get("activities", []),
            provider=provider,
            history=history,
        )
    except Exception as exc:
        fallback = answer_fitness_question(question, KB, profile)
        answer = f"AI provider unavailable ({provider}): {str(exc)}. Using local FitPilot guidance instead: {fallback}"
        provider = "grounded-fallback"

    db.add(ChatMessage(role="user", content=question, schedule_id=schedule_id))
    db.add(ChatMessage(role="assistant", content=answer, schedule_id=schedule_id))
    db.commit()
    return ChatResponse(answer=answer, provider=provider)


def get_chat_history(db: Session, schedule_id: Optional[int] = None) -> List[Dict[str, str]]:
    query = db.query(ChatMessage).order_by(ChatMessage.created_at.asc())
    if schedule_id is not None:
        query = query.filter(ChatMessage.schedule_id == schedule_id)
    history = query.all()
    return [{"role": item.role, "content": item.content} for item in history]


@app.get("/api/chat/history")
async def get_chat_history_api(schedule_id: Optional[int] = None, db: Session = Depends(get_db)):
    records = get_chat_history(db, schedule_id)
    return {"messages": records}


@app.delete("/api/chat")
async def delete_chat_history(schedule_id: Optional[int] = None, db: Session = Depends(get_db)):
    query = db.query(ChatMessage)
    if schedule_id is not None:
        query = query.filter(ChatMessage.schedule_id == schedule_id)
    deleted = query.delete(synchronize_session=False)
    db.commit()
    return {"message": "Chat history cleared successfully.", "deleted": deleted}


@app.get("/api/chat/providers")
async def available_providers():
    return {
        "providers": [
            {"id": "openai", "name": "OpenAI", "configured": bool(os.getenv("OPENAI_API_KEY"))},
            {"id": "gemini", "name": "Google Gemini", "configured": bool(os.getenv("GEMINI_API_KEY"))},
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
    lunch_time = profile.get("lunch_time") or "12:30"
    base_items: List[Dict[str, Any]] = []

    if wake:
        base_items.append({
            "activity_name": "Wake-up",
            "category": "routine",
            "start_time": wake,
            "end_time": _minutes_to_time(_to_minutes(wake) + 30),
            "duration_minutes": 30,
            "reason": "Morning reset and start of your daily flow.",
            "completed": False,
            "status": "planned",
        })

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

    if lunch_time:
        lunch_start = lunch_time
        lunch_end = _minutes_to_time(_to_minutes(lunch_time) + 30)
        base_items.append({
            "activity_name": "Lunch",
            "category": "meal",
            "start_time": lunch_start,
            "end_time": lunch_end,
            "duration_minutes": 30,
            "reason": "Scheduled meal break during your midday window.",
            "completed": False,
            "status": "planned",
        })

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

    if sleep:
        sleep_minutes = _to_minutes(sleep)
        base_items.append({
            "activity_name": "Sleep",
            "category": "sleep",
            "start_time": sleep,
            "end_time": _minutes_to_time((sleep_minutes + 30) % (24 * 60)),
            "duration_minutes": 30,
            "reason": "Sleep window and wind-down period for recovery.",
            "completed": False,
            "status": "planned",
        })

    return dedupe_and_sort_items(base_items)


def normalize_custom_activity(item: Dict[str, Any], wake_time: Optional[str] = None, sleep_time: Optional[str] = None) -> Dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("Each custom activity must be an object.")

    name = str(item.get("activity_name") or item.get("name") or "").strip()
    if not name:
        raise ValueError("Every custom activity requires a name.")

    category = str(item.get("category") or "activity").strip() or "activity"
    description = str(item.get("description") or item.get("reason") or "").strip()
    start_time = item.get("start_time")
    end_time = item.get("end_time")
    duration_minutes = int(item.get("duration_minutes") or item.get("duration") or 0)

    if start_time and not _is_valid_time(start_time):
        raise ValueError(f"Custom activity '{name}' has an invalid start time.")
    if end_time and not _is_valid_time(end_time):
        raise ValueError(f"Custom activity '{name}' has an invalid end time.")

    if start_time and end_time:
        if _to_minutes(end_time) <= _to_minutes(start_time):
            raise ValueError(f"Custom activity '{name}' must end after it starts.")
        end_minutes = _to_minutes(end_time)
        start_minutes = _to_minutes(start_time)
        duration = max(0, end_minutes - start_minutes)
    elif start_time and duration_minutes > 0:
        start_minutes = _to_minutes(start_time)
        duration = duration_minutes
        end_time = _minutes_to_time(start_minutes + duration)
    else:
        raise ValueError(f"Custom activity '{name}' must include a start time and end time or duration.")

    if wake_time and _to_minutes(start_time) < _to_minutes(wake_time):
        raise ValueError(f"Custom activity '{name}' cannot start before the wake-up time.")
    if sleep_time and _to_minutes(end_time) > _to_minutes(sleep_time):
        raise ValueError(f"Custom activity '{name}' cannot extend past the sleep time.")

    return {
        "activity_name": name,
        "category": category,
        "start_time": start_time,
        "end_time": end_time,
        "duration_minutes": duration,
        "reason": description,
        "completed": False,
        "status": "planned",
    }


def normalize_custom_activities(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_items = payload.get("custom_activities") or payload.get("routine_activities") or []
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    wake_time = payload.get("wake_time") or "07:00"
    sleep_time = payload.get("sleep_time") or "23:00"
    return [normalize_custom_activity(item, wake_time, sleep_time) for item in raw_items if item]


def sort_and_validate_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sorted_items = sorted(items, key=lambda item: (_to_minutes(item.get("start_time") or "00:00"), item.get("activity_name", "")))
    for previous, current in zip(sorted_items, sorted_items[1:]):
        previous_end = _to_minutes(previous.get("end_time") or previous.get("start_time") or "00:00")
        current_start = _to_minutes(current.get("start_time") or "00:00")
        if previous_end > current_start:
            raise ValueError(f"Activity overlap detected between '{previous.get('activity_name')}' and '{current.get('activity_name')}'.")
    return sorted_items


def merge_generated_schedule(payload: Dict[str, Any], generated_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    custom_items = normalize_custom_activities(payload)
    merged = list(generated_items) + custom_items
    if not merged:
        return []
    final_items = sort_and_validate_items(merged)
    return final_items
