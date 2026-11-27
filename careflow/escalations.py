"""Mock clinical triage queue: where approved clinical escalations go.

In a real clinic this would create a ticket in the EHR clinical inbox
("Priority-Clinical Routing", MHP-SOP-401 section 4.3). Here it is a
SQLite table, so the restart test can check that an escalation happened
exactly once.

Each workflow thread can create at most one escalation: thread_id is
UNIQUE, so if the node ever runs twice (for example after a crash and
resume) the second call finds the existing row instead of adding another.
"""

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from careflow import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS clinical_escalations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id   TEXT NOT NULL UNIQUE,
    patient_ref TEXT NOT NULL,
    urgency     TEXT NOT NULL,
    reason      TEXT NOT NULL,
    approved_by TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""


class EscalationQueue:
    """Reads and writes the mock clinical triage queue."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else config.OPERATIONS_DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def escalate(
        self, thread_id: str, patient_ref: str, urgency: str, reason: str, approved_by: str
    ) -> dict:
        """Send a case to the clinical triage queue (once per thread).

        Returns the escalation row and whether it was newly created.
        """
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO clinical_escalations "
                "(thread_id, patient_ref, urgency, reason, approved_by, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    thread_id, patient_ref, urgency, reason, approved_by,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )
            created = cursor.rowcount == 1
            row = connection.execute(
                "SELECT * FROM clinical_escalations WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        return {**dict(row), "newly_created": created}

    def for_thread(self, thread_id: str) -> list[dict]:
        """All escalations created by one workflow thread (0 or 1)."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM clinical_escalations WHERE thread_id = ?", (thread_id,)
            ).fetchall()
        return [dict(row) for row in rows]