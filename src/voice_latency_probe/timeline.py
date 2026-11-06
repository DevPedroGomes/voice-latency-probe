"""The live state of one call, kept apart from LiveKit so it can be unit tested."""

from __future__ import annotations

from dataclasses import dataclass, field

LISTENING = "listening"
SPEAKING = "speaking"


@dataclass
class Timeline:
    agent_voice: list[float] = field(default_factory=list)
    states: list[tuple[float, str]] = field(default_factory=list)

    @property
    def state(self) -> str:
        return self.states[-1][1] if self.states else ""

    def first_voice_after(self, moment: float) -> float | None:
        return next((t for t in self.agent_voice if t > moment), None)

    def voice_between(self, start: float, end: float) -> bool:
        return any(start < t < end for t in self.agent_voice)

    def agent_finished(
        self,
        now: float,
        since: float,
        quiet_with_state_s: float = 0.8,
        quiet_without_state_s: float = 2.5,
    ) -> bool:
        """True once the agent spoke after `since` and went quiet.

        A pause inside a sentence must not read as the end, so silence alone needs a
        long wait. The `listening` state shortens it when the agent publishes one.
        """
        if not self.agent_voice or self.agent_voice[-1] <= since:
            return False
        quiet = now - self.agent_voice[-1]
        if self.state == LISTENING and quiet >= quiet_with_state_s:
            return True
        return quiet >= quiet_without_state_s
