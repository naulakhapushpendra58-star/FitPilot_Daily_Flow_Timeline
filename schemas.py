from datetime import date as date_type, datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class PreferenceBase(BaseModel):
    user_name: str = "User"
    goal: str = "Build consistency"
    preferred_activity: str = "Walking"
    preferred_activities: List[str] = Field(default_factory=lambda: ["Walking", "Stretching"])
    experience: str = "Beginner"
    wake_time: str = "07:00"
    sleep_time: str = "23:00"
    lunch_time: Optional[str] = "12:30"
    work_start: Optional[str] = None
    work_end: Optional[str] = None
    study_duration: int = 60
    break_duration: int = 15
    available_minutes: int = 120
    target_duration: int = 20
    activity_duration: int = 30
    productivity_goal: str = "Stay focused and balanced"
    priorities: str = "Health, focus, recovery"


class PreferenceUpdate(PreferenceBase):
    pass


class ScheduleItemCreate(BaseModel):
    activity_name: str
    category: str = "activity"
    start_time: str
    end_time: str
    duration_minutes: int
    reason: str = ""
    completed: bool = False


class ScheduleCreate(BaseModel):
    title: str = "Daily plan"
    date: date_type = Field(default_factory=lambda: datetime.utcnow().date())
    user_goal: str = "Build consistency"
    notes: str = ""
    items: List[ScheduleItemCreate] = Field(default_factory=list)


class ScheduleUpdate(BaseModel):
    title: Optional[str] = None
    status: Optional[str] = None
    notes: Optional[str] = None
    items: Optional[List[ScheduleItemCreate]] = None


class ActivityCreate(BaseModel):
    activity_name: Optional[str] = None
    name: Optional[str] = None
    category: str = "activity"
    start_time: str
    end_time: str
    duration_minutes: int
    reason: str = ""
    completed: bool = False

    @property
    def resolved_name(self) -> str:
        return self.activity_name or self.name or "Custom activity"


class ChatRequest(BaseModel):
    question: str
    schedule_id: Optional[int] = None
    provider: Optional[str] = "local"


class ChatResponse(BaseModel):
    answer: str
    provider: str


class HealthResponse(BaseModel):
    status: str
    message: str
