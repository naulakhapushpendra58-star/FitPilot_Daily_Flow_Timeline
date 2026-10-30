from __future__ import annotations

import os
from datetime import date, datetime
from typing import List

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

Base.metadata.create_all(bind=engine)

app = FastAPI(title="FitPilot – AI-Powered Personal Activity Planner", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

KB = __import__("knowledge").load_knowledge_base()
AGENT = PersonalAssistantAgent(KB)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_or_create_preference(db: Session) -> Preference:
    pref = db.query(Preference).first()
    if pref is None:
        pref = Preference()
        db.add(pref)
        db.commit()
        db.refresh(pref)
    return pref


def normalize_profile_payload(payload: dict) -> dict:
    day = payload.get("day") or date.today().isoformat()
    return {
        "day": datetime.fromisoformat(day).date() if isinstance(day, str) else day,
        "goal": payload.get("goal", "Build consistency"),
        "preference": payload.get("preference", "Walking"),
        "experience": payload.get("experience", "Beginner"),
        "available_minutes": int(payload.get("available_minutes", 120)),
        "target_duration": int(payload.get("target_duration", 20)),
        "day_start": payload.get("wake_time") or "07:00",
        "day_end": payload.get("sleep_time") or "23:00",
        "study_duration": int(payload.get("study_duration", 45)),
        "break_duration": int(payload.get("break_duration", 15)),
        "work_start": payload.get("work_start"),
        "work_end": payload.get("work_end"),
        "preferred_activities": payload.get("preferred_activities", []),
        "productivity_goal": payload.get("productivity_goal", "Stay focused and balanced"),
        "priorities": payload.get("priorities", "Health, focus, recovery"),
    }


def build_busy_blocks(payload: dict) -> List[TimeBlock]:
    busy_blocks: List[TimeBlock] = []
    day = normalize_profile_payload(payload)["day"]
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


def schedule_item_to_dict(item: ScheduleItem) -> dict:
    return {
        "id": item.id,
        "schedule_id": item.schedule_id,
        "activity_name": item.activity_name,
        "category": item.category,
        "start_time": item.start_time,
        "end_time": item.end_time,
        "duration_minutes": item.duration_minutes,
        "reason": item.reason,
        "completed": item.completed,
    }


def schedule_to_dict(schedule: Schedule, items: List[ScheduleItem]) -> dict:
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


def generate_schedule_from_profile(profile: dict, busy_blocks: List[TimeBlock]) -> dict:
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
    for entry in plan:
        items.append(
            {
                "activity_name": entry.activity_name,
                "category": "activity",
                "start_time": entry.start.strftime("%H:%M"),
                "end_time": entry.end.strftime("%H:%M"),
                "duration_minutes": entry.duration_minutes,
                "reason": entry.reason,
                "completed": False,
            }
        )

    return {
        "schedule": {
            "title": f"{profile['goal']} plan",
            "date": day.isoformat(),
            "total_minutes": validation["total_minutes"],
            "status": "generated",
            "user_goal": profile["goal"],
            "notes": "AI-generated plan based on constraints and preferences.",
            "items": items,
        },
        "validation": validation,
        "feasible": validation["feasible"],
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return {"status": "ok", "message": "FitPilot API is running."}


@app.get("/api/preferences")
async def get_preferences_api(db: Session = Depends(get_db)):
    pref = get_or_create_preference(db)
    return {
        "user_name": pref.user_name,
        "goal": pref.goal,
        "preferred_activity": pref.preferred_activity,
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


@app.put("/api/preferences")
async def save_preferences(payload: PreferenceUpdate, db: Session = Depends(get_db)):
    pref = get_or_create_preference(db)
    for field, value in payload.model_dump().items():
        setattr(pref, field, value)
    db.commit()
    db.refresh(pref)
    preferences = await get_preferences_api(db)
    return {"message": "Preferences saved successfully.", "preferences": preferences}


@app.post("/api/schedules/generate")
async def generate_schedule(payload: dict, db: Session = Depends(get_db)):
    try:
        profile = normalize_profile_payload(payload)
        busy_blocks = build_busy_blocks(payload)
        result = generate_schedule_from_profile(profile, busy_blocks)
        if not result["feasible"] and not result["schedule"]["items"]:
            raise HTTPException(status_code=400, detail="No feasible schedule could be generated for the provided constraints.")

        schedule_record = Schedule(
            title=f"{profile['goal']} plan",
            date=profile["day"],
            total_minutes=result["validation"]["total_minutes"],
            status="generated",
            user_goal=profile["goal"],
            notes="AI-generated schedule based on user preferences and constraints.",
        )
        db.add(schedule_record)
        db.commit()
        db.refresh(schedule_record)

        for item in result["schedule"]["items"]:
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
        return {
            "message": "Schedule generated successfully.",
            "schedule": schedule_to_dict(schedule_record, items),
            "validation": result["validation"],
            "explanation": grounded_explanation(profile, AGENT.plan(profile, busy_blocks), KB["activities"]),
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
    item = ScheduleItem(
        schedule_id=getattr(payload, "schedule_id", 1),
        activity_name=payload.name,
        category=payload.category,
        start_time=payload.start_time,
        end_time=payload.end_time,
        duration_minutes=payload.duration_minutes,
        reason=payload.reason,
        completed=payload.completed,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return {"message": "Activity added successfully.", "activity": schedule_item_to_dict(item)}


@app.put("/api/activities/{activity_id}")
async def update_activity(activity_id: int, payload: dict, db: Session = Depends(get_db)):
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
    schedule_payload = {}
    if schedule:
        items = db.query(ScheduleItem).filter(ScheduleItem.schedule_id == schedule.id).all()
        schedule_payload = {"title": schedule.title, "status": schedule.status, "items": [schedule_item_to_dict(item) for item in items]}

    profile = {
        "user_name": pref.user_name,
        "goal": pref.goal,
        "preferred_activity": pref.preferred_activity,
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
    return {"answer": answer, "provider": "openai" if os.getenv("OPENAI_API_KEY") else "grounded-fallback"}


@app.get("/api/ai/paradigms")
async def ai_paradigms():
    return {
        "paradigms": [
            {"name": "Rule-based AI", "how_it_works": "Explicit business and scheduling rules are applied deterministically.", "advantages": ["Explainable", "Reliable"], "limitations": ["Needs domain rules"], "example": "Constraint checking for time windows and busy blocks."},
            {"name": "Search-based planning", "how_it_works": "The system searches for feasible time windows and validates overlaps.", "advantages": ["Respects hard constraints", "Efficient for modest schedules"], "limitations": ["Complexity increases with many constraints"], "example": "Find available blocks and rank activities."},
            {"name": "Machine learning", "how_it_works": "Models can learn user preference patterns over time.", "advantages": ["Personalization", "Adaptive recommendations"], "limitations": ["Needs data and monitoring"], "example": "Preference tuning from historical completions."},
            {"name": "Large language models", "how_it_works": "LLMs convert conversational requests into explanations and recommendations.", "advantages": ["Natural language", "Helpful follow-ups"], "limitations": ["Can hallucinate without grounding"], "example": "Answering schedule questions and productivity guidance."},
            {"name": "Hybrid AI", "how_it_works": "The planner uses explicit rules while the LLM offers grounded explanations and conversational assistance.", "advantages": ["Balanced control and flexibility"], "limitations": ["More system complexity"], "example": "Schedule generation plus chatbot guidance."},
        ]
    }


@app.get("/api/knowledge-base")
async def knowledge_base():
    return KB


@app.get("/api/health-check")
async def health_check_alias():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
