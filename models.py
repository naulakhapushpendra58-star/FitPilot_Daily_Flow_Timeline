from datetime import datetime

from sqlalchemy import Boolean, Column, Date, DateTime, Float, Integer, String, Text

from database import Base


class Preference(Base):
    __tablename__ = "preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_name = Column(String(255), default="User")
    goal = Column(String(255), default="Build consistency")
    preferred_activity = Column(String(255), default="Walking")
    preferred_activities = Column(Text, default="Walking,Stretching")
    experience = Column(String(50), default="Beginner")
    wake_time = Column(String(20), default="07:00")
    sleep_time = Column(String(20), default="23:00")
    lunch_time = Column(String(20), default="12:30")
    work_start = Column(String(20), nullable=True)
    work_end = Column(String(20), nullable=True)
    study_duration = Column(Integer, default=60)
    break_duration = Column(Integer, default=15)
    available_minutes = Column(Integer, default=120)
    target_duration = Column(Integer, default=20)
    activity_duration = Column(Integer, default=30)
    productivity_goal = Column(String(255), default="Stay focused and balanced")
    priorities = Column(Text, default="Health, focus, recovery")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Schedule(Base):
    __tablename__ = "schedules"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), default="Daily plan")
    date = Column(Date, default=datetime.utcnow().date)
    total_minutes = Column(Integer, default=0)
    status = Column(String(50), default="draft")
    user_goal = Column(String(255), default="Build consistency")
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ScheduleItem(Base):
    __tablename__ = "schedule_items"

    id = Column(Integer, primary_key=True, index=True)
    schedule_id = Column(Integer, nullable=False)
    activity_name = Column(String(255), nullable=False)
    category = Column(String(100), default="activity")
    start_time = Column(String(20), nullable=False)
    end_time = Column(String(20), nullable=False)
    duration_minutes = Column(Integer, default=0)
    reason = Column(Text, default="")
    completed = Column(Boolean, default=False)
    status = Column(String(20), default="planned")
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    schedule_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
