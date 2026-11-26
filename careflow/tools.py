"""The tools CareFlow's agent can call.

Every tool returns a dictionary:
  success: {"success": True, ...data}
  failure: {"success": False, "error_code": "...", "message": "..."}
so a problem reaches the model as a readable message instead of a crash.

Business limits (such as the maximum number of slots returned) are
enforced HERE in code, not only in the prompt.
"""

import inspect
import json
import logging
from dataclasses import dataclass
from typing import Callable

from careflow import config
from careflow.mock_data import find_row, load_table

logger = logging.getLogger(__name__)

URGENCY_LEVELS = ("Routine", "Priority", "Urgent", "Emergent")
APPOINTMENT_MODES = ("in_person", "telehealth")


def _error(error_code: str, message: str) -> dict:
    return {"success": False, "error_code": error_code, "message": message}


def check_insurance_eligibility(patient_ref: str) -> dict:
    """Look up a patient's insurance eligibility and plan details.

    Use when the patient asks whether their insurance is active or what
    their plan covers. Do not use it to estimate the cost of a procedure
    (use calculate_copay) or to check pre-authorization.

    Args:
        patient_ref: Patient reference such as "MHP-P-00000".

    Returns:
        On success: eligibility_status, plan_code, plan_name, deductible
        figures in USD, coinsurance_rate and copays.
        On failure: error_code PATIENT_NOT_FOUND or PLAN_NOT_FOUND.
    """
    patient = find_row("patients", patient_ref=patient_ref)
    if patient is None:
        return _error("PATIENT_NOT_FOUND", f"No patient with reference {patient_ref}.")
    plan = find_row("plans", plan_code=patient["plan_code"])
    if plan is None:
        return _error("PLAN_NOT_FOUND", f"No plan {patient['plan_code']} on file.")
    return {
        "success": True,
        "patient_ref": patient["patient_ref"],
        "eligibility_status": patient["eligibility_status"],
        "plan_code": plan["plan_code"],
        "plan_name": plan["plan_name"],
        "annual_deductible_usd": float(plan["annual_deductible_usd"]),
        "deductible_met_usd": float(patient["deductible_met_usd"]),
        "coinsurance_rate": float(plan["coinsurance_rate"]),
        "specialist_copay_usd": float(plan["specialist_copay_usd"]),
        "telehealth_copay_usd": float(plan["telehealth_copay_usd"]),
    }


def find_appointment_slots(
    specialty: str, mode: str | None = None, limit: int = 3
) -> dict:
    """Find the earliest open appointment slots for a specialty.

    Use when the patient wants to book or see availability. This only
    searches; it does not book anything.

    Args:
        specialty: One of the clinic's specialties, e.g. "Cardiology".
        mode: Optional. "in_person" or "telehealth".
        limit: How many slots to return, 1 to the configured maximum.

    Returns:
        On success: a list of slots (slot_id, provider_name, site_name,
        city, starts_at, duration_minutes, mode), earliest first.
        On failure: error_code UNKNOWN_SPECIALTY or INVALID_ARGUMENT.
    """
    match = next(
        (
            row
            for row in load_table("specialties")
            if row["specialty"].lower() == str(specialty).strip().lower()
        ),
        None,
    )
    if match is None:
        return _error("UNKNOWN_SPECIALTY", f"'{specialty}' is not a known specialty.")
    if mode is not None and mode not in APPOINTMENT_MODES:
        return _error("INVALID_ARGUMENT", "mode must be 'in_person' or 'telehealth'.")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        return _error("INVALID_ARGUMENT", "limit must be a whole number of 1 or more.")

    # Business limit enforced in code, whatever the model asks for.
    limit = min(limit, config.MAX_SLOTS_RETURNED)

    open_slots = [
        row
        for row in load_table("appointment_slots")
        if row["specialty"] == match["specialty"]
        and row["status"].lower() == "open"
        and (mode is None or row["mode"] == mode)
    ]
    open_slots.sort(key=lambda row: row["starts_at"])

    slots = []
    for row in open_slots[:limit]:
        provider = find_row("providers", provider_id=row["provider_id"]) or {}
        site = find_row("sites", site_id=row["site_id"]) or {}
        slots.append(
            {
                "slot_id": row["slot_id"],
                "provider_name": provider.get("name", "Unknown"),
                "site_name": site.get("site_name", "Unknown"),
                "city": site.get("city", "Unknown"),
                "starts_at": row["starts_at"],
                "duration_minutes": int(row["duration_minutes"]),
                "mode": row["mode"],
            }
        )
    return {"success": True, "specialty": match["specialty"], "slots": slots}


def calculate_copay(patient_ref: str, procedure_code: str) -> dict:
    """Estimate a patient's out-of-pocket cost for one procedure.

    Use when the patient asks what a visit or procedure will cost them.
    The figure is an estimate, never a guarantee of coverage.

    Args:
        patient_ref: Patient reference such as "MHP-P-00000".
        procedure_code: Procedure code such as "CARD-CONS".

    Returns:
        On success: estimated_out_of_pocket_usd and a short explanation.
        On failure: error_code PATIENT_NOT_FOUND, PROCEDURE_NOT_FOUND,
        PLAN_NOT_FOUND, COPAY_NOT_CONFIGURED or UNSUPPORTED_COST_BASIS.
    """
    patient = find_row("patients", patient_ref=patient_ref)
    if patient is None:
        return _error("PATIENT_NOT_FOUND", f"No patient with reference {patient_ref}.")
    procedure = find_row("procedures", procedure_code=procedure_code)
    if procedure is None:
        return _error("PROCEDURE_NOT_FOUND", f"No procedure {procedure_code}.")
    plan = find_row("plans", plan_code=patient["plan_code"])
    if plan is None:
        return _error("PLAN_NOT_FOUND", f"No plan {patient['plan_code']} on file.")

    list_price = float(procedure["list_price_usd"])
    basis = procedure["cost_share_basis"]

    if basis == "copay":
        copay_row = find_row(
            "plan_copays",
            plan_code=plan["plan_code"],
            service_category=procedure["category"],
        )
        if copay_row is None:
            return _error(
                "COPAY_NOT_CONFIGURED",
                f"No copay set for {plan['plan_code']} / {procedure['category']}.",
            )
        estimate = float(copay_row["copay_usd"])
        explanation = f"${estimate:.2f} {procedure['category']} copay"
    elif basis == "deductible_then_coinsurance":
        remaining = max(
            0.0,
            float(plan["annual_deductible_usd"]) - float(patient["deductible_met_usd"]),
        )
        deductible_part = min(list_price, remaining)
        coinsurance_part = (list_price - deductible_part) * float(plan["coinsurance_rate"])
        estimate = deductible_part + coinsurance_part
        explanation = (
            f"${deductible_part:.2f} deductible + ${coinsurance_part:.2f} coinsurance"
        )
    else:
        return _error("UNSUPPORTED_COST_BASIS", f"Unknown cost basis '{basis}'.")

    return {
        "success": True,
        "patient_ref": patient["patient_ref"],
        "procedure_code": procedure["procedure_code"],
        "procedure_name": procedure["procedure_name"],
        "plan_name": plan["plan_name"],
        "list_price_usd": list_price,
        "estimated_out_of_pocket_usd": round(estimate, 2),
        "explanation": explanation,
        "note": "Estimate only. Not a guarantee of coverage.",
    }


def check_preauthorization(patient_ref: str, procedure_code: str) -> dict:
    """Check whether a procedure is covered and needs pre-authorization.

    Use before booking an MRI or other procedure to find out if the
    patient's plan requires approval first. It only reads the rules; it
    does not request an approval.

    Args:
        patient_ref: Patient reference such as "MHP-P-00000".
        procedure_code: Procedure code such as "CARD-CONS".

    Returns:
        On success: covered, preauth_required and, when approval is
        needed, approval_window_business_days.
        On failure: error_code PATIENT_NOT_FOUND or RULE_NOT_FOUND.
    """
    patient = find_row("patients", patient_ref=patient_ref)
    if patient is None:
        return _error("PATIENT_NOT_FOUND", f"No patient with reference {patient_ref}.")
    rule = find_row(
        "preauth_rules", procedure_code=procedure_code, plan_code=patient["plan_code"]
    )
    if rule is None:
        return _error(
            "RULE_NOT_FOUND",
            f"No rule for {procedure_code} under plan {patient['plan_code']}.",
        )
    window = rule["approval_window_business_days"].strip()
    return {
        "success": True,
        "patient_ref": patient["patient_ref"],
        "procedure_code": rule["procedure_code"],
        "plan_code": rule["plan_code"],
        "covered": rule["covered"].strip().lower() == "true",
        "preauth_required": rule["preauth_required"].strip().lower() == "true",
        "approval_window_business_days": int(float(window)) if window else None,
    }


def flag_for_human(patient_ref: str, reason: str, urgency: str = "Routine") -> dict:
    """Hand a request over to a human staff member.

    Use whenever the patient asks for medical advice (symptoms,
    medication, dosage, test results, "should I ...") or when you are
    unsure. This is a mock hand-off: it records the request but contacts
    no one.

    Args:
        patient_ref: Patient reference such as "MHP-P-00000".
        reason: Short plain-language reason for the hand-off.
        urgency: One of "Routine", "Priority", "Urgent", "Emergent".

    Returns:
        On success: status "flagged_for_human_review".
        On failure: error_code INVALID_ARGUMENT or PATIENT_NOT_FOUND.
    """
    if urgency not in URGENCY_LEVELS:
        return _error("INVALID_ARGUMENT", f"urgency must be one of {list(URGENCY_LEVELS)}.")
    if not str(reason).strip():
        return _error("INVALID_ARGUMENT", "A reason is required for human review.")
    if find_row("patients", patient_ref=patient_ref) is None:
        return _error("PATIENT_NOT_FOUND", f"No patient with reference {patient_ref}.")
    logger.info("Flagged %s for human review (%s)", patient_ref, urgency)
    return {
        "success": True,
        "status": "flagged_for_human_review",
        "patient_ref": patient_ref,
        "urgency": urgency,
        "reason": str(reason).strip(),
    }


@dataclass(frozen=True)
class Tool:
    """One tool: its function plus the JSON schema the LLM sees."""

    function: Callable
    parameters: dict

    @property
    def name(self) -> str:
        return self.function.__name__

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": inspect.getdoc(self.function),
                "parameters": self.parameters,
            },
        }


_PATIENT_REF = {
    "type": "string",
    "description": "Patient reference, e.g. MHP-P-00000.",
}
_PROCEDURE_CODE = {
    "type": "string",
    "description": "Procedure code, e.g. CARD-CONS.",
}

TOOLS: dict[str, Tool] = {
    tool.name: tool
    for tool in [
        Tool(
            check_insurance_eligibility,
            {
                "type": "object",
                "properties": {"patient_ref": _PATIENT_REF},
                "required": ["patient_ref"],
            },
        ),
        Tool(
            find_appointment_slots,
            {
                "type": "object",
                "properties": {
                    "specialty": {
                        "type": "string",
                        "description": "Specialty, e.g. Cardiology.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": list(APPOINTMENT_MODES),
                        "description": "Optional appointment mode.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Number of slots wanted (capped by the system).",
                    },
                },
                "required": ["specialty"],
            },
        ),
        Tool(
            calculate_copay,
            {
                "type": "object",
                "properties": {
                    "patient_ref": _PATIENT_REF,
                    "procedure_code": _PROCEDURE_CODE,
                },
                "required": ["patient_ref", "procedure_code"],
            },
        ),
        Tool(
            check_preauthorization,
            {
                "type": "object",
                "properties": {
                    "patient_ref": _PATIENT_REF,
                    "procedure_code": _PROCEDURE_CODE,
                },
                "required": ["patient_ref", "procedure_code"],
            },
        ),
        Tool(
            flag_for_human,
            {
                "type": "object",
                "properties": {
                    "patient_ref": _PATIENT_REF,
                    "reason": {"type": "string", "description": "Why a human is needed."},
                    "urgency": {
                        "type": "string",
                        "enum": list(URGENCY_LEVELS),
                        "description": "How quickly a human should respond.",
                    },
                },
                "required": ["patient_ref", "reason"],
            },
        ),
    ]
}


def tool_schemas() -> list[dict]:
    """The tool definitions to pass to the LLM."""
    return [tool.schema() for tool in TOOLS.values()]


def execute_tool(name: str, arguments: dict) -> dict:
    """Run one tool by name. Never raises: problems come back as errors."""
    tool = TOOLS.get(name)
    if tool is None:
        return _error("UNKNOWN_TOOL", f"There is no tool called '{name}'.")
    if not isinstance(arguments, dict):
        return _error("INVALID_ARGUMENT", "Tool arguments must be a JSON object.")
    try:
        inspect.signature(tool.function).bind(**arguments)
    except TypeError as error:
        return _error("INVALID_ARGUMENT", str(error))
    try:
        result = tool.function(**arguments)
        json.dumps(result)  # must be JSON-safe to send back to the model
        return result
    except Exception:
        logger.exception("Tool %s crashed", name)
        return _error("TOOL_FAILED", f"Tool '{name}' failed unexpectedly.")