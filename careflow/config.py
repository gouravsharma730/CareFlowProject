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
MAX_REPAIR_ATTEMPTS = int(os.getenv("CAREFLOW_MAX_REPAIR_ATTEMPTS", "2"))
LLM_MAX_RETRIES = int(os.getenv("CAREFLOW_LLM_MAX_RETRIES", "5"))
LLM_RETRY_BASE_SECONDS = float(os.getenv("CAREFLOW_LLM_RETRY_BASE_SECONDS", "2"))
LLM_RETRY_MAX_SECONDS = float(os.getenv("CAREFLOW_LLM_RETRY_MAX_SECONDS", "60"))

# --- Folders ---
DATA_DIR = Path(os.getenv("CAREFLOW_DATA_DIR", PROJECT_ROOT / "data"))
MOCK_API_DIR = Path(
    os.getenv("CAREFLOW_MOCK_API_DIR", DATA_DIR / "careflow" / "mock_api")
)

# Most appointment slots a tool may return, whatever the model asks for.
MAX_SLOTS_RETURNED = int(os.getenv("CAREFLOW_MAX_SLOTS_RETURNED", "5"))

# Most tool-calling rounds the agent may run for one request.
MAX_AGENT_STEPS = int(os.getenv("CAREFLOW_MAX_AGENT_STEPS", "5"))
# Where long-term memory is stored (survives restarts).
MEMORY_DB_PATH = Path(
    os.getenv("CAREFLOW_MEMORY_DB", DATA_DIR / "memory" / "careflow_memory.sqlite")
)

def summary() -> dict:
    """Return the current settings (never includes secret keys)."""
    return {
        "llm_model": LLM_MODEL,
        "llm_temperature": LLM_TEMPERATURE,
        "llm_timeout_seconds": LLM_TIMEOUT_SECONDS,
        "max_repair_attempts": MAX_REPAIR_ATTEMPTS,
        "data_dir": str(DATA_DIR),
    }


if __name__ == "__main__":
    for key, value in summary().items():
        print(f"{key}: {value}")