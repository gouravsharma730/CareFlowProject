"""Procedural memory: the clinic's step-by-step procedures.

Each procedure is stored in long-term memory with a version number.
When a request matches a procedure (by keyword, decided by code), the
agent is given the steps and told to follow them in order.

The steps below are taken from the clinic's own policy documents; the
source of each one is recorded next to it.
"""

import re
from dataclasses import dataclass

from careflow.memory_store import MemoryStore


@dataclass(frozen=True)
class ProcedureDefinition:
    """A procedure: its name, when it applies, its steps and its source."""

    name: str
    patterns: tuple[str, ...]
    steps: tuple[str, ...]
    source: str


DEFINITIONS = (
    ProcedureDefinition(
        name="mri_booking_preauth",
        patterns=(r"\bmri\b",),
        steps=(
            "Check the patient's plan and eligibility status.",
            "Look up the pre-authorisation rule for ORTH-MRI under that plan.",
            "If pre-authorisation is required, collect the documents: "
            "orthopaedic consultation notes, proof of at least 6 weeks of "
            "physical therapy, and plain radiograph reports.",
            "Submit the request and allow the approval window for the plan "
            "(3 business days for Silver, 5 business days for Bronze or "
            "CIVIC-BASE, none for Gold).",
            "If the request is denied, start an appeal: Level 1 Medical Director "
            "Peer Review, then External Independent Medical Review.",
            "Book the MRI slot once approval is recorded, or at once if no "
            "pre-authorisation is required.",
        ),
        source="preauth-matrix.md, section 3.2.2 (ORTH-MRI)",
    ),
)


def find_procedure(request_text: str, store: MemoryStore | None = None) -> dict | None:
    """Return the stored procedure that matches a request, or None.

    Matching is done by code with keyword patterns. The memory store is
    only opened when a procedure matches. A new or changed definition is
    saved as a new version.

    Returns:
        {"name", "version", "steps"} or None.
    """
    text = request_text.lower()
    for definition in DEFINITIONS:
        if any(re.search(pattern, text) for pattern in definition.patterns):
            store = store or MemoryStore()
            stored = store.get_procedure(definition.name)
            if stored is None or stored["steps"] != list(definition.steps):
                store.save_procedure(definition.name, list(definition.steps))
                stored = store.get_procedure(definition.name)
            return stored
    return None


def procedure_label(procedure: dict) -> str:
    """A short label such as 'mri_booking_preauth v1'."""
    return f"{procedure['name']} v{procedure['version']}"


def format_procedure(procedure: dict) -> str:
    """The procedure as numbered steps, ready to add to the prompt."""
    lines = [
        f"Clinic procedure ({procedure_label(procedure)}). "
        "Follow these steps in order for this task. Use the tools to look up "
        "the patient's real plan and rules; never guess them:"
    ]
    for number, step in enumerate(procedure["steps"], start=1):
        lines.append(f"{number}. {step}")
    return "\n".join(lines)