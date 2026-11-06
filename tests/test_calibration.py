"""The probe against an agent whose delays are known. Needs a LiveKit server.

Locally: `livekit-server --dev`, then
LIVEKIT_URL=ws://127.0.0.1:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret pytest
"""

import asyncio
import os
import uuid
from pathlib import Path

import pytest

from voice_latency_probe.analyze import summarize
from voice_latency_probe.audio import tone, write_wav
from voice_latency_probe.caller import CallOptions, load_utterances, run_call
from voice_latency_probe.fake_agent import FakeAgentOptions, run_fake_agent
from voice_latency_probe.record import Call, append_call

pytestmark = pytest.mark.integration

TOLERANCE_MS = 120
ARTIFACTS = os.environ.get("PROBE_ARTIFACTS")


def server() -> tuple[str, str, str]:
    url = os.environ.get("LIVEKIT_URL")
    if not url:
        pytest.skip("LIVEKIT_URL is not set")
    return url, os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"]


async def measure(
    tmp_path: Path, end_of_turn_ms: int, agent_ms: int, report_every: int, count: int
) -> Call:
    url, key, secret = server()
    room = f"calibration-{uuid.uuid4().hex[:8]}"
    for number in range(count):
        write_wav(tmp_path / f"u{number}.wav", tone(1.0, 180.0))
    agent = asyncio.ensure_future(
        run_fake_agent(
            FakeAgentOptions(
                url,
                key,
                secret,
                room,
                end_of_turn_ms=end_of_turn_ms,
                agent_ms=agent_ms,
                report_every=report_every,
            )
        )
    )
    try:
        await asyncio.sleep(1.0)
        record = await run_call(
            CallOptions(url, key, secret, room=room, settle_s=1.0, turn_timeout_s=15.0),
            load_utterances([tmp_path]),
        )
        if ARTIFACTS:
            directory = Path(ARTIFACTS)
            directory.mkdir(parents=True, exist_ok=True)
            append_call(directory / f"test-{end_of_turn_ms}-{agent_ms}.jsonl", record)
        return record
    finally:
        agent.cancel()
        await asyncio.gather(agent, return_exceptions=True)


async def test_the_probe_recovers_known_delays(tmp_path: Path) -> None:
    record = await measure(tmp_path, 500, 700, report_every=1, count=5)
    summary = summarize([record])
    assert record.greeted
    assert summary.unanswered == 0
    assert summary.overlaps == 0
    assert summary.end_of_turn is not None
    assert summary.reply is not None
    assert summary.total is not None
    assert summary.end_of_turn.p50 == pytest.approx(500, abs=TOLERANCE_MS)
    assert summary.reply.p50 == pytest.approx(700, abs=TOLERANCE_MS)
    assert summary.total.p50 == pytest.approx(1200, abs=TOLERANCE_MS)


async def test_the_probe_sees_what_the_agents_number_leaves_out(tmp_path: Path) -> None:
    record = await measure(tmp_path, 500, 300, report_every=2, count=6)
    summary = summarize([record])
    assert summary.reported_turns == 3
    assert summary.reported_gap is not None
    assert summary.reported_gap.p50 == pytest.approx(500, abs=2 * TOLERANCE_MS)
