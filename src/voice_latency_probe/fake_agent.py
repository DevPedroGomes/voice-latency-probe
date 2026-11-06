"""An agent with known delays, used to check the probe against the truth.

It has no STT, LLM or TTS, so it needs no paid key: it waits for the caller's audio to
go quiet, holds for exactly the configured times and answers with a tone. If the probe
is right, its numbers come back close to these delays plus local network time.

It also repeats a common way a voice stack misreports itself: its reported total starts
at the end of turn, and it can stay silent on some turns.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from livekit import api, rtc

from .audio import (
    DEFAULT_VOICE_RMS,
    FRAME_SAMPLES,
    FRAME_SECONDS,
    SAMPLE_RATE,
    frames,
    rms,
    tone,
)
from .caller import FINAL_ATTRIBUTE, STATE_ATTRIBUTE, TRANSCRIPTION_TOPIC

MIN_END_OF_TURN_MS = 50


@dataclass(frozen=True)
class FakeAgentOptions:
    url: str
    api_key: str
    api_secret: str
    room: str
    end_of_turn_ms: int = 400
    agent_ms: int = 300
    report_every: int = 1
    reply_s: float = 0.8
    greeting_s: float = 0.5
    voice_rms: float = DEFAULT_VOICE_RMS

    def __post_init__(self) -> None:
        if self.end_of_turn_ms < MIN_END_OF_TURN_MS:
            raise ValueError(f"end_of_turn_ms must be at least {MIN_END_OF_TURN_MS}")
        if self.agent_ms < 0:
            raise ValueError("agent_ms cannot be negative")


class _FakeAgent:
    def __init__(self, options: FakeAgentOptions) -> None:
        self.options = options
        self.room = rtc.Room()
        self.source = rtc.AudioSource(SAMPLE_RATE, 1, queue_size_ms=50)
        self.outgoing: asyncio.Queue[list[np.ndarray]] = asyncio.Queue()
        self.states: asyncio.Queue[str] = asyncio.Queue()
        self.last_voice: float | None = None
        self.caller_identity: str | None = None
        self.turns = 0
        self.tasks: set[asyncio.Task[None]] = set()
        self.done = asyncio.Event()

    def spawn(self, coroutine: Any) -> None:
        task: asyncio.Task[None] = asyncio.ensure_future(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def state_sender(self) -> None:
        """Sends states in order. A slow update must not delay the audio."""
        while True:
            state = await self.states.get()
            await self.room.local_participant.set_attributes({STATE_ATTRIBUTE: state})

    async def voice_out(self) -> None:
        """Sends silence until there is something to say, so the track stays warm."""
        silence = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        sent = 0
        started = time.monotonic()
        pending: list[np.ndarray] = []
        while not self.done.is_set():
            if not pending and not self.outgoing.empty():
                pending = list(self.outgoing.get_nowait())
            block = pending.pop(0) if pending else silence
            await self.source.capture_frame(
                rtc.AudioFrame(block.tobytes(), SAMPLE_RATE, 1, FRAME_SAMPLES)
            )
            sent += 1
            wait = started + sent * FRAME_SECONDS - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)

    async def ears(self, track: rtc.Track) -> None:
        stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1)
        async for event in stream:
            samples = np.frombuffer(event.frame.data, dtype=np.int16)
            if rms(samples) > self.options.voice_rms:
                self.last_voice = time.monotonic()

    async def say(self, seconds: float) -> None:
        self.states.put_nowait("speaking")
        self.outgoing.put_nowait(frames(tone(seconds)))
        await asyncio.sleep(seconds + 0.1)
        self.states.put_nowait("listening")

    async def publish_final_transcript(self) -> None:
        writer = await self.room.local_participant.stream_text(
            topic=TRANSCRIPTION_TOPIC,
            attributes={FINAL_ATTRIBUTE: "true"},
            sender_identity=self.caller_identity,
        )
        await writer.write("(fake transcript)")
        await writer.aclose()

    async def report(self) -> None:
        self.turns += 1
        every = self.options.report_every
        if every and self.turns % every == 0:
            # Starts counting at the end of turn, so it leaves that wait out.
            payload = {"type": "latency", "total_ms": self.options.agent_ms}
            await self.room.local_participant.publish_data(json.dumps(payload))

    async def conversation(self) -> None:
        if self.options.greeting_s > 0:
            await self.say(self.options.greeting_s)
        wait_s = self.options.end_of_turn_ms / 1000
        hold_s = self.options.agent_ms / 1000
        while not self.done.is_set():
            await asyncio.sleep(0.002)
            heard = self.last_voice
            if heard is None or time.monotonic() - heard < wait_s:
                continue
            # Forget this turn now. Clearing it after the reply could erase the next
            # one, if the caller has already started and finished speaking by then.
            self.last_voice = None
            self.spawn(self.publish_final_transcript())
            self.states.put_nowait("thinking")
            # Deadlines from the moment of the last voice, so the time spent sending
            # these messages does not add to the delays this agent promises.
            await asyncio.sleep(max(0.0, heard + wait_s + hold_s - time.monotonic()))
            self.spawn(self.report())
            await self.say(self.options.reply_s)


def fake_agent_token(options: FakeAgentOptions) -> str:
    return (
        api.AccessToken(options.api_key, options.api_secret)
        .with_identity("fake-agent")
        .with_kind("agent")
        .with_attributes({STATE_ATTRIBUTE: "listening"})
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=options.room,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .to_jwt()
    )


async def run_fake_agent(options: FakeAgentOptions) -> None:
    """Runs until the caller leaves, or forever if no caller ever came."""
    agent = _FakeAgent(options)
    room = agent.room
    ready = asyncio.Event()

    @room.on("track_subscribed")
    def on_track(
        track: rtc.Track,
        _publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            agent.caller_identity = participant.identity
            agent.spawn(agent.ears(track))
            ready.set()

    @room.on("participant_disconnected")
    def on_left(participant: rtc.RemoteParticipant) -> None:
        if participant.identity == agent.caller_identity:
            agent.done.set()

    await room.connect(options.url, fake_agent_token(options))
    track = rtc.LocalAudioTrack.create_audio_track("agent", agent.source)
    await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )
    voice = asyncio.ensure_future(agent.voice_out())
    states = asyncio.ensure_future(agent.state_sender())
    try:
        await ready.wait()
        talk = asyncio.ensure_future(agent.conversation())
        await agent.done.wait()
        talk.cancel()
    finally:
        voice.cancel()
        states.cancel()
        await room.disconnect()
        for task in list(agent.tasks):
            task.cancel()
