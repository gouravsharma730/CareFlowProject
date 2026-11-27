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

# --- Agent and tool limits ---
# Most appointment slots a tool may return, whatever the model asks for.
MAX_SLOTS_RETURNED = int(os.getenv("CAREFLOW_MAX_SLOTS_RETURNED", "5"))

# Most tool-calling rounds the agent may run for one request.
MAX_AGENT_STEPS = int(os.getenv("CAREFLOW_MAX_AGENT_STEPS", "5"))

# --- Memory ---
# Longest value the agent may store for one patient preference.
MAX_PREFERENCE_VALUE_LENGTH = int(
    os.getenv("CAREFLOW_MAX_PREFERENCE_VALUE_LENGTH", "100")
)

# Most recent events loaded into the prompt for one patient.
MEMORY_CONTEXT_MAX_EVENTS = int(os.getenv("CAREFLOW_MEMORY_CONTEXT_MAX_EVENTS", "5"))

# Where long-term memory is stored (survives restarts).
MEMORY_DB_PATH = Path(
    os.getenv("CAREFLOW_MEMORY_DB", DATA_DIR / "memory" / "careflow_memory.sqlite")
)

# Working memory: how much of the current conversation is sent to the model.
MAX_HISTORY_TURNS = int(os.getenv("CAREFLOW_MAX_HISTORY_TURNS", "6"))
MAX_HISTORY_CHARS = int(os.getenv("CAREFLOW_MAX_HISTORY_CHARS", "4000"))

# --- Retrieval (M4) ---
# Folder with the policy documents (index.json, markdown/, pdf/).
CORPUS_DIR = Path(os.getenv("CAREFLOW_CORPUS_DIR", DATA_DIR / "careflow" / "corpus"))

# Largest chunk, in characters. A longer section is split between
# paragraphs; a single paragraph or table is never cut in the middle.
CHUNK_MAX_CHARS = int(os.getenv("CAREFLOW_CHUNK_MAX_CHARS", "1200"))

# Embedding model (runs locally through fastembed, no API key needed).
EMBEDDING_MODEL = os.getenv("CAREFLOW_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Where the Qdrant vector index is saved (survives restarts).
QDRANT_PATH = Path(os.getenv("CAREFLOW_QDRANT_PATH", DATA_DIR / "qdrant_store"))
POLICY_COLLECTION = os.getenv("CAREFLOW_POLICY_COLLECTION", "careflow_policies")

# How many chunks a search returns by default.
RETRIEVAL_TOP_K = int(os.getenv("CAREFLOW_RETRIEVAL_TOP_K", "5"))

# Retrieval mode: "dense", "hybrid" (dense + BM25 fused with RRF) or
# "hybrid_rerank" (hybrid, then reordered by a cross-encoder reranker).
RETRIEVAL_MODE = os.getenv("CAREFLOW_RETRIEVAL_MODE", "hybrid_rerank")

# How many candidates each search method returns before fusion.
RETRIEVAL_CANDIDATES = int(os.getenv("CAREFLOW_RETRIEVAL_CANDIDATES", "10"))

# Reciprocal Rank Fusion constant (60 is the usual value from the paper).
RRF_K = int(os.getenv("CAREFLOW_RRF_K", "60"))

# Cross-encoder reranker (runs locally through fastembed).
RERANKER_MODEL = os.getenv("CAREFLOW_RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

def summary() -> dict:
    """Return the current settings (never includes secret keys)."""
    return {
        "llm_model": LLM_MODEL,
        "llm_temperature": LLM_TEMPERATURE,
        "llm_timeout_seconds": LLM_TIMEOUT_SECONDS,
        "max_repair_attempts": MAX_REPAIR_ATTEMPTS,
        "max_agent_steps": MAX_AGENT_STEPS,
        "max_history_turns": MAX_HISTORY_TURNS,
        "data_dir": str(DATA_DIR),
        "memory_db_path": str(MEMORY_DB_PATH),
    }


if __name__ == "__main__":
    for key, value in summary().items():
        print(f"{key}: {value}")