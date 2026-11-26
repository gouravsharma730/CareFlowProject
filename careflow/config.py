"""Central configuration for CareFlow.

Every model name, path and tunable value lives here and is read from
environment variables (or the .env file), with safe defaults.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# The project root is the folder that contains the "careflow" package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load variables from the .env file (if it exists).
load_dotenv(PROJECT_ROOT / ".env")

# --- LLM settings ---
LLM_MODEL = os.getenv("CAREFLOW_LLM_MODEL", "gemini/gemini-flash-lite-latest")
LLM_TEMPERATURE = float(os.getenv("CAREFLOW_LLM_TEMPERATURE", "0.0"))
LLM_TIMEOUT_SECONDS = int(os.getenv("CAREFLOW_LLM_TIMEOUT_SECONDS", "30"))

# --- Folders ---
DATA_DIR = Path(os.getenv("CAREFLOW_DATA_DIR", PROJECT_ROOT / "data"))


def summary() -> dict:
    """Return the current settings (never includes secret keys)."""
    return {
        "llm_model": LLM_MODEL,
        "llm_temperature": LLM_TEMPERATURE,
        "llm_timeout_seconds": LLM_TIMEOUT_SECONDS,
        "data_dir": str(DATA_DIR),
    }


if __name__ == "__main__":
    for key, value in summary().items():
        print(f"{key}: {value}")