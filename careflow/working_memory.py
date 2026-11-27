"""Working memory: the recent history of one conversation.

Only the last few turns are sent to the model, within a size budget.
Older turns are dropped (trimmed). Anything that must outlive the
window, such as a patient's preferences, is kept in long-term memory
and loaded into the prompt on every turn instead.

Budget policy (both limits apply, newest turns win):
  - at most MAX_HISTORY_TURNS turns
  - at most MAX_HISTORY_CHARS characters in total
  - the latest turn is always kept
"""

from careflow import config


class WorkingMemory:
    """The trimmed message history of one conversation (one session)."""

    def __init__(self, max_turns: int | None = None, max_chars: int | None = None):
        self.max_turns = config.MAX_HISTORY_TURNS if max_turns is None else max_turns
        self.max_chars = config.MAX_HISTORY_CHARS if max_chars is None else max_chars
        self._turns: list[tuple[str, str]] = []
        self.trimmed_turns = 0

    def add(self, user_message: str, assistant_message: str) -> None:
        """Record one finished turn, then trim to the budget."""
        self._turns.append((user_message, assistant_message))
        self._trim()

    def messages(self) -> list[dict]:
        """The kept turns as chat messages, oldest first."""
        messages = []
        for user_message, assistant_message in self._turns:
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": assistant_message})
        return messages

    def __len__(self) -> int:
        return len(self._turns)

    def _size(self) -> int:
        return sum(len(user) + len(assistant) for user, assistant in self._turns)

    def _trim(self) -> None:
        while len(self._turns) > self.max_turns and len(self._turns) > 1:
            self._turns.pop(0)
            self.trimmed_turns += 1
        while len(self._turns) > 1 and self._size() > self.max_chars:
            self._turns.pop(0)
            self.trimmed_turns += 1