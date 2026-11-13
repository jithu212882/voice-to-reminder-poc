from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field

CategoryType = Literal["Birthday", "Anniversary", "To-Do", "Meeting", "Custom"]
RepeatType = Literal["None", "Yearly", "Monthly", "Weekly"]


class ReminderSchema(BaseModel):
    title: str = Field(description="Concise, clean title of the reminder")
    category: CategoryType = Field(default="To-Do", description="Reminder category")
    date: str = Field(description="Date in YYYY-MM-DD format")
    time: str = Field(default="09:00", description="Time in HH:MM (24-hour) format")
    repeat: RepeatType = Field(default="None", description="Recurrence frequency")
    advance_alert: str = Field(
        default="Morning of event", description="Advance reminder alert timing"
    )


class BaseReminderParser(ABC):
    provider_name: str

    @abstractmethod
    async def parse(
        self, transcript: str, current_datetime: datetime
    ) -> ReminderSchema:
        """Parse reminder details from transcript text."""
        pass
