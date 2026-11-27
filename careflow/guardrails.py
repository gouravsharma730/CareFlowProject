"""Guardrails enforced in code, independent of what any model says.

Input (every patient message, before any LLM sees it):
  - size check: empty or longer than MAX_MESSAGE_CHARS is rejected
  - prompt-injection scan: weighted patterns; a score at or above
    INJECTION_BLOCK_SCORE blocks the request
  - clinical-advice detection (MHP-SOP-401 categories A to E, including a
    self-declared clinician asking about doses): the request goes to the
    human-approval path, whatever the intake model concluded
  - emergency detection: the request gets the emergency message

Retrieved content (every policy chunk, before an LLM sees it):
  - the same injection scan; flagged chunks are dropped

Output (every final reply, before it reaches the patient):
  - schema: a non-empty string no longer than MAX_REPLY_CHARS
  - no clinical advice (doses, medicine instructions, diagnoses, test
    interpretation), no other patient's reference, no internal prompt text
  - a failing reply is replaced by a safe hand-off message

All thresholds live in careflow/config.py.
"""

import re
from dataclasses import dataclass, field

from careflow import config

# ---- prompt injection ----
# (pattern, weight, flag). Weight 2 = a clear attempt on its own;
# weight 1 = suspicious, blocked only together with another signal.
INJECTION_PATTERNS = [
    (r"\b(ignore|disregard|forget|override|bypass)\b.{0,30}\b(previous|prior|above|earlier|all|your|the|these|system)\b.{0,20}\b(instructions?|rules?|prompts?|policy|policies|guidelines?)",
     2, "override_instructions"),
    (r"\b(you are now|from now on you are|act as|pretend to be|roleplay as)\b", 2, "role_change"),
    (r"\b(system prompt|developer mode|jailbreak|dan mode|hidden instructions?)\b", 2, "prompt_attack"),
    (r"(</?\s*(system|assistant|untrusted_document)\b|\[/?(system|inst)\]|<<\s*sys\s*>>)", 2, "fake_markup"),
    (r"\b(reveal|print|show|repeat|output)\b.{0,25}\b(your|the|system)\b.{0,15}\b(prompt|instructions?|rules)\b",
     2, "prompt_extraction"),
    (r"\b(approve|authori[sz]e|confirm|sign off|escalate|book)\b.{0,30}\b(immediately|right now|without (review|checks?|approval))\b",
     1, "pressure_to_act"),
    (r"\b(do not|don't|never)\b.{0,20}\b(tell|inform|flag|escalate to|involve)\b.{0,20}\b(staff|human|doctor|clinician|anyone)\b",
     1, "avoid_human"),
    (r"\b(new|updated|real) (instructions?|rules?|policy)\b\s*[:\-]", 1, "instruction_injection"),
]

# ---- clinical advice requested (SOP-401 categories A-E) ----
CLINICAL_REQUEST_PATTERNS = [
    (r"\b(dose|dosage|dosing|mg|mcg|milligrams?|units? of)\b", "dosage"),
    (r"\bhow (much|many)\b.{0,40}\b(take|tablets?|pills?|insulin|medicine|medication)\b", "dosage"),
    (r"\b(should|can|could|may) i\b.{0,30}\b(take|stop|start|double|skip|increase|reduce|lower|mix|switch)\b",
     "medication_should_i"),
    (r"\bshould i\b.{0,40}\b(worry|go to|come in|see a|be concerned|get it checked)\b", "should_i"),
    (r"\b(is it|is this|could (it|this) be|do i have|what (is|could) (causing|cause))\b.{0,50}\b(serious|normal|infection|cancer|disease|condition|symptom|pain|rash|fever|lump|bleeding|headaches?|dizz\w*)\b",
     "symptom_interpretation"),
    (r"\b(test|lab|blood|scan|mri|x-?ray|biopsy|ecg)\b.{0,20}\b(results?|report)\b.{0,40}\b(mean|normal|high|low|bad|worried|concern\w*)\b",
     "test_result_meaning"),
    (r"\b(side effects?|interact\w*|safe to take|together with)\b", "medication_question"),
]
CLINICIAN_CLAIM = re.compile(
    r"\b(i am|i'm|as) (a |an )?(doctor|physician|nurse|pharmacist|gp|surgeon|clinician|consultant)\b"
    r"|\bclinician[- ]to[- ]clinician\b", re.I)

# ---- emergencies ----
EMERGENCY_PATTERNS = [
    r"\b(can'?t|cannot|unable to|struggling to) breathe\b",
    r"\bchest pain\b.{0,40}\b(left arm|jaw|sweat\w*|collaps\w*|faint\w*)\b",
    r"\b(overdos\w*|took too many)\b",
    r"\b(suicid\w*|kill myself|end my life)\b",
    r"\b(unconscious|not breathing|seizure|stroke)\b",
    r"\b(severe|heavy) bleeding\b",
]

# ---- unsafe replies ----
UNSAFE_REPLY_PATTERNS = [
    (r"\b\d+(\.\d+)?\s?(mg|mcg|µg|ml|units?)\b", "reply_gives_dose"),
    (r"\b(take|stop|start|increase|decrease|double|skip)\s+(taking\s+)?(your|the|this|that)?\s*"
     r"(medication|medicine|tablets?|pills?|dose|drug)", "reply_gives_medicine_instruction"),
    (r"\byou (probably |likely |may |might )?have (a |an )?[a-z]+ (infection|disease|condition|syndrome)\b",
     "reply_suggests_diagnosis"),
    (r"\b(your|the) ((test|lab|blood|scan|mri|x-?ray|biopsy) )?(results?|test|scan|report|x-?ray|mri) "
     r"(shows?|means?|is normal|are normal|looks?|indicates?|suggests?)\b", "reply_interprets_results"),
    (r"(untrusted_document|SYSTEM_PROMPT|You are CareFlow's)", "reply_leaks_internal_text"),
]
PATIENT_REF = re.compile(r"\bMHP-P-\d{5}\b")

SAFE_REPLY = ("I'm sorry, I can't complete this request automatically. "
              "A member of our staff will follow up with you.")


@dataclass
class InputCheck:
    """What the input guardrail decided about one message."""

    action: str                          # "allow", "block", "clinical", "emergency"
    flags: list[str] = field(default_factory=list)
    injection_score: int = 0
    reason: str = ""

    def to_dict(self) -> dict:
        return {"action": self.action, "flags": self.flags,
                "injection_score": self.injection_score, "reason": self.reason}


def injection_scan(text: str) -> tuple[int, list[str]]:
    """Weighted prompt-injection score and the flags that matched."""
    score, flags = 0, []
    for pattern, weight, flag in INJECTION_PATTERNS:
        if re.search(pattern, text, re.I | re.S):
            score += weight
            flags.append(flag)
    return score, flags


def check_input(text) -> InputCheck:
    """Decide what to do with a patient message before any model sees it."""
    if not isinstance(text, str) or not text.strip():
        return InputCheck("block", ["empty_message"], reason="The message is empty.")
    if len(text) > config.MAX_MESSAGE_CHARS:
        return InputCheck("block", ["message_too_long"],
                          reason=f"The message is longer than {config.MAX_MESSAGE_CHARS} characters.")

    score, flags = injection_scan(text)
    if score >= config.INJECTION_BLOCK_SCORE:
        return InputCheck("block", flags, score,
                          "The message tries to change the assistant's instructions.")
    if any(re.search(p, text, re.I) for p in EMERGENCY_PATTERNS):
        return InputCheck("emergency", flags + ["possible_emergency"], score,
                          "The message describes a possible emergency.")
    clinical = sorted({flag for pattern, flag in CLINICAL_REQUEST_PATTERNS
                       if re.search(pattern, text, re.I | re.S)})
    if CLINICIAN_CLAIM.search(text):
        flags.append("claims_to_be_clinician")  # no exception (SOP-401 5.2.2)
    if clinical:
        return InputCheck("clinical", flags + clinical, score,
                          "The message asks for clinical advice, which only staff may give.")
    return InputCheck("allow", flags, score)


def scan_documents(chunks: list[dict]) -> tuple[list[dict], list[str]]:
    """Drop retrieved chunks that contain injection attempts. Returns (kept, dropped ids)."""
    kept, dropped = [], []
    for chunk in chunks:
        score, _ = injection_scan(chunk.get("text", ""))
        if score >= config.INJECTION_BLOCK_SCORE:
            dropped.append(chunk.get("chunk_id", "?"))
        else:
            kept.append(chunk)
    return kept, dropped


def check_output(reply, patient_ref: str | None = None) -> tuple[bool, list[str]]:
    """Is this reply safe to send? Returns (ok, flags)."""
    if not isinstance(reply, str) or not reply.strip():
        return False, ["reply_empty"]
    flags = [flag for pattern, flag in UNSAFE_REPLY_PATTERNS if re.search(pattern, reply, re.I)]
    if len(reply) > config.MAX_REPLY_CHARS:
        flags.append("reply_too_long")
    others = {ref for ref in PATIENT_REF.findall(reply) if ref != patient_ref}
    if others:
        flags.append("reply_mentions_another_patient")
    return (not flags), flags