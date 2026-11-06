import json
from dataclasses import asdict
from pathlib import Path

import pytest

from voice_latency_probe.liveparse import parse_data_packet
from voice_latency_probe.record import (
    Call,
    DataEvent,
    StateChange,
    Transcript,
    Turn,
    append_call,
    read_calls,
)


def sample() -> Call:
    return Call(
        room="r",
        caller_identity="probe-1",
        started_at="2026-11-06T00:00:00+00:00",
        greeted=True,
        duration_s=9.5,
        turns=[
            Turn("u1", 1.2, 3.0, 4.1, False, True),
            Turn("u2", 1.0, 8.0, None, True, False),
        ],
        states=[StateChange(4.0, "speaking")],
        data=[DataEvent(4.2, {"total_ms": 900})],
        transcripts=[Transcript(3.4, "probe-1", True)],
    )


def test_a_call_survives_the_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    append_call(path, sample())
    append_call(path, sample())
    assert read_calls([path]) == [sample(), sample()]


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text("\n" + json.dumps(asdict(sample())) + "\n\n", encoding="utf-8")
    assert len(read_calls([path])) == 1


def test_another_schema_is_refused_with_the_line_number(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    raw = asdict(sample())
    raw["schema"] = 99
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"run\.jsonl:1: unsupported record schema"):
        read_calls([path])


def test_a_truncated_line_is_refused_with_the_line_number(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text('{"schema": 1, "room"', encoding="utf-8")
    with pytest.raises(ValueError, match=r"run\.jsonl:1"):
        read_calls([path])


@pytest.mark.parametrize("raw", [b"\xff\xfe", b"not json", b"[1, 2]", b"3"])
def test_a_bad_data_packet_is_dropped(raw: bytes) -> None:
    assert parse_data_packet(raw) is None


def test_a_json_object_passes() -> None:
    assert parse_data_packet(b'{"total_ms": 1}') == {"total_ms": 1}
