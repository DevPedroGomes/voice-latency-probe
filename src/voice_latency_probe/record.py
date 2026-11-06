"""What one call leaves on disk: one JSON object per line, all times on one clock.

Every instant is in seconds since the caller joined, read from the caller's monotonic
clock. Nothing is compared across machines, so there is no clock to synchronize.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Turn:
    utterance: str
    speech_s: float
    t_speech_end: float
    t_agent_voice: float | None
    overlap: bool
    agent_finished: bool


@dataclass(frozen=True)
class Transcript:
    t: float
    identity: str
    final: bool
    text: str | None = None


@dataclass(frozen=True)
class DataEvent:
    t: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class StateChange:
    t: float
    state: str


@dataclass
class Call:
    room: str
    caller_identity: str
    started_at: str
    greeted: bool
    duration_s: float = 0.0
    turns: list[Turn] = field(default_factory=list)
    states: list[StateChange] = field(default_factory=list)
    data: list[DataEvent] = field(default_factory=list)
    transcripts: list[Transcript] = field(default_factory=list)
    schema: int = SCHEMA_VERSION


def call_from_dict(raw: dict[str, Any]) -> Call:
    schema = raw.get("schema")
    if schema != SCHEMA_VERSION:
        raise ValueError(f"unsupported record schema: {schema!r}")
    return Call(
        room=raw["room"],
        caller_identity=raw["caller_identity"],
        started_at=raw["started_at"],
        greeted=raw["greeted"],
        duration_s=raw["duration_s"],
        turns=[Turn(**item) for item in raw["turns"]],
        states=[StateChange(**item) for item in raw["states"]],
        data=[DataEvent(**item) for item in raw["data"]],
        transcripts=[Transcript(**item) for item in raw["transcripts"]],
        schema=schema,
    )


def append_call(path: Path, call: Call) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(asdict(call), ensure_ascii=False) + "\n")


def read_calls(paths: Iterable[Path]) -> list[Call]:
    calls = []
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                calls.append(call_from_dict(json.loads(line)))
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
    return calls
