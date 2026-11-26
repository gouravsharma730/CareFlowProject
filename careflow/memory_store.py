"""Long-term memory for CareFlow, stored in a SQLite file.

Three kinds of long-term memory live here:
  - facts       (semantic):   stable preferences, one value per key
  - events      (episodic):   things that happened, with time and session
  - procedures  (procedural): named lists of steps, versioned

Every patient memory is stored under a patient_ref and is only ever
read back with that same patient_ref, so patients never see each
other's data. Procedures belong to the clinic, not to a patient.
"""

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from careflow import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    patient_ref TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    session_id  TEXT,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (patient_ref, key)
);
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_ref TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    summary     TEXT NOT NULL,
    session_id  TEXT,
    occurred_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_by_patient ON events (patient_ref, occurred_at);
CREATE TABLE IF NOT EXISTS procedures (
    name       TEXT PRIMARY KEY,
    version    INTEGER NOT NULL,
    steps      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryStore:
    """Reads and writes long-term memory in one SQLite file."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else config.MEMORY_DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    # ---- semantic memory: facts and preferences ----

    def remember_fact(
        self, patient_ref: str, key: str, value: str, session_id: str | None = None
    ) -> None:
        """Store a fact. A new value for the same key replaces the old one."""
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO facts (patient_ref, key, value, session_id, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (patient_ref, key) DO UPDATE SET "
                "value = excluded.value, session_id = excluded.session_id, "
                "updated_at = excluded.updated_at",
                (patient_ref, key, value, session_id, _now()),
            )

    def get_facts(self, patient_ref: str) -> dict[str, str]:
        """Return all facts for one patient as {key: value}."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT key, value FROM facts WHERE patient_ref = ? ORDER BY key",
                (patient_ref,),
            ).fetchall()
        return {row["key"]: row["value"] for row in rows}

    # ---- episodic memory: events ----

    def add_event(
        self,
        patient_ref: str,
        event_type: str,
        summary: str,
        session_id: str | None = None,
    ) -> None:
        """Record something that happened, stamped with the time."""
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO events (patient_ref, event_type, summary, session_id, "
                "occurred_at) VALUES (?, ?, ?, ?, ?)",
                (patient_ref, event_type, summary, session_id, _now()),
            )

    def recent_events(self, patient_ref: str, limit: int = 5) -> list[dict]:
        """Return a patient's most recent events, newest first."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT event_type, summary, session_id, occurred_at FROM events "
                "WHERE patient_ref = ? ORDER BY occurred_at DESC, id DESC LIMIT ?",
                (patient_ref, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    # ---- procedural memory: step lists ----

    def save_procedure(self, name: str, steps: list[str]) -> int:
        """Store or update a procedure. Returns its new version number."""
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT version FROM procedures WHERE name = ?", (name,)
            ).fetchone()
            version = (row["version"] + 1) if row else 1
            connection.execute(
                "INSERT INTO procedures (name, version, steps, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT (name) DO UPDATE SET version = excluded.version, "
                "steps = excluded.steps, updated_at = excluded.updated_at",
                (name, version, json.dumps(steps), _now()),
            )
        return version

    def get_procedure(self, name: str) -> dict | None:
        """Return {"name", "version", "steps"} or None if it does not exist."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT name, version, steps FROM procedures WHERE name = ?", (name,)
            ).fetchone()
        if row is None:
            return None
        return {"name": row["name"], "version": row["version"],
                "steps": json.loads(row["steps"])}

    # ---- privacy ----

    def delete_patient_data(self, patient_ref: str) -> None:
        """Erase every fact and event stored for one patient."""
        with closing(self._connect()) as connection, connection:
            connection.execute("DELETE FROM facts WHERE patient_ref = ?", (patient_ref,))
            connection.execute("DELETE FROM events WHERE patient_ref = ?", (patient_ref,))