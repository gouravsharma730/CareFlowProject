"""Typed data models for CareFlow."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

CareFlowSpecialty = Literal[
    "Cardiology",
    "Orthopaedics",
    "Dermatology",
    "Endocrinology",
    "Gastroenterology",
    "Pulmonology",
    "Neurology",
    "ENT",
]

CareFlowUrgency = Literal["Routine", "Priority", "Urgent", "Emergent"]


class PatientRequest(BaseModel):
    """A patient message turned into validated, structured fields."""

    patient_ref: str
    channel: str
    raw_text: str

    specialty: Optional[CareFlowSpecialty] = None
    insurance_id: Optional[str] = None

    urgency: CareFlowUrgency

    seeks_clinical_advice: bool
    requires_human_review: bool

    missing_fields: list[str] = Field(default_factory=list)