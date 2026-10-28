"""Schemas shared by future alert endpoints."""
from enum import Enum
from pydantic import BaseModel


class AlertStatus(str, Enum):
    OPEN = "Open"
    UNDER_INSPECTION = "Under inspection"
    VERIFIED = "Verified"
    FALSE_ALARM = "False alarm"


class AlertStatusUpdate(BaseModel):
    status: AlertStatus
