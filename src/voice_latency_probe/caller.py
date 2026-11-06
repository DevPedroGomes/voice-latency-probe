"""A synthetic caller: speaks in real time and timestamps what comes back."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from livekit import api, rtc

from .audio import (
    DEFAULT_VOICE_RMS,
    FRAME_SAMPLES,
    FRAME_SECONDS,
    SAMPLE_RATE,
    Samples,
    last_voiced_sample,
    load_wav,
    rms,
)
from .liveparse import parse_data_packet
from .record import Call, DataEvent, StateChange, Transcript, Turn
from .timeline import Timeline

STATE_ATTRIBUTE = "lk.agent.state"
TRANSCRIPTION_TOPIC = "lk.transcription"
FINAL_ATTRIBUTE = "lk.transcription_final"
TOKEN_TTL = timedelta(minutes=30)


@dataclass(frozen=True)
class Utterance:
    name: str
    samples: Samples
    last_voiced: int


def load_utterances(
    paths: Sequence[Path], threshold: float = DEFAULT_VOICE_RMS
) -> list[Utterance]:
    files: list[Path] = []
    for path in paths:
        files.extend(sorted(path.glob("*.wav")) if path.is_dir() else [path])
    if not files:
        raise ValueError("no WAV files to speak")
    utterances = []
    for file in files:
        samples = load_wav(file)
        utterances.append(
            Utterance(file.stem, samples, last_voiced_sample(samples, threshold))
        )
    return utterances


@dataclass(frozen=True)
class CallOptions:
    url: str
    api_key: str
    api_secret: str
    room: str | None = None
    room_prefix: str = "probe-"
    agent_name: str | None = None
    attributes: dict[str, str] = field(default_factory=dict)
    greeting_timeout_s: float = 30.0
    turn_timeout_s: float = 45.0
    gap_s: float = 0.6
    settle_s: float = 3.0
    voice_rms: float = DEFAULT_VOICE_RMS
    keep_text: bool = False


def caller_token(options: CallOptions, room: str, identity: str) -> str:
    token = (
        api.AccessToken(options.api_key, options.api_secret)
        .with_identity(identity)
        .with_attributes(options.attributes)
        .with_ttl(TOKEN_TTL)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=False,
            )
        )
    )
    if options.agent_name:
        token = token.with_room_config(
            api.RoomConfiguration(
                agents=[api.RoomAgentDispatch(agent_name=options.agent_name)]
            )
        )
    return token.to_jwt()


class _LiveCall:
    def __init__(self, options: CallOptions, identity: str) -> None:
        self.options = options
        self.identity = identity
        self.room = rtc.Room()
        self.source = rtc.AudioSource(SAMPLE_RATE, 1, queue_size_ms=50)
        self.queue: asyncio.Queue[Utterance] = asyncio.Queue()
        self.speech_end: dict[str, float] = {}
        self.timeline = Timeline()
        self.data: list[DataEvent] = []
        self.transcripts: list[Transcript] = []
        self.t0 = time.monotonic()
        self.disconnected = asyncio.Event()
        self.tasks: set[asyncio.Task[None]] = set()

    def now(self) -> float:
        return time.monotonic() - self.t0

    def spawn(self, coroutine: Any) -> None:
        task: asyncio.Task[None] = asyncio.ensure_future(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def microphone(self) -> None:
        """Sends a frame every 10 ms, silence between utterances, like an open mic."""
        silence = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        sent = 0
        started = time.monotonic()
        current: Utterance | None = None
        position = 0
        while not self.disconnected.is_set():
            if current is None and not self.queue.empty():
                current, position = self.queue.get_nowait(), 0
            if current is None:
                block = silence
            else:
                block = current.samples[position : position + FRAME_SAMPLES]
                if len(block) < FRAME_SAMPLES:
                    block = np.pad(block, (0, FRAME_SAMPLES - len(block)))
                if position <= current.last_voiced < position + FRAME_SAMPLES:
                    # The frame carrying the last voiced sample enters the send queue
                    # now, which is at most queue_size_ms ahead of the wire.
                    self.speech_end[current.name] = (
                        self.now() + (current.last_voiced - position) / SAMPLE_RATE
                    )
                position += FRAME_SAMPLES
                if position >= len(current.samples):
                    current = None
            await self.source.capture_frame(
                rtc.AudioFrame(block.tobytes(), SAMPLE_RATE, 1, FRAME_SAMPLES)
            )
            sent += 1
            wait = started + sent * FRAME_SECONDS - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)

    async def listen(self, track: rtc.Track) -> None:
        stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1)
        async for event in stream:
            samples = np.frombuffer(event.frame.data, dtype=np.int16)
            if rms(samples) > self.options.voice_rms:
                self.timeline.agent_voice.append(self.now())

    def note_state(self, state: str) -> None:
        self.timeline.states.append((self.now(), state))

    def bind_events(self) -> None:
        room = self.room

        @room.on("track_subscribed")
        def on_track(
            track: rtc.Track,
            _publication: rtc.RemoteTrackPublication,
            _participant: rtc.RemoteParticipant,
        ) -> None:
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                self.spawn(self.listen(track))

        @room.on("participant_attributes_changed")
        def on_attributes(
            changed: dict[str, str], _participant: rtc.Participant
        ) -> None:
            if STATE_ATTRIBUTE in changed:
                self.note_state(changed[STATE_ATTRIBUTE])

        @room.on("participant_connected")
        def on_participant(participant: rtc.RemoteParticipant) -> None:
            state = participant.attributes.get(STATE_ATTRIBUTE)
            if state:
                self.note_state(state)

        @room.on("data_received")
        def on_data(packet: rtc.DataPacket) -> None:
            payload = parse_data_packet(packet.data)
            if payload is not None:
                self.data.append(DataEvent(round(self.now(), 3), payload))

        @room.on("disconnected")
        def on_disconnected(*_args: object) -> None:
            self.disconnected.set()

        def on_text(reader: rtc.TextStreamReader, identity: str) -> None:
            arrived = self.now()

            async def read() -> None:
                text = await reader.read_all()
                attributes = dict(reader.info.attributes or {})
                self.transcripts.append(
                    Transcript(
                        t=round(arrived, 3),
                        identity=identity,
                        final=attributes.get(FINAL_ATTRIBUTE) == "true",
                        text=text if self.options.keep_text else None,
                    )
                )

            self.spawn(read())

        room.register_text_stream_handler(TRANSCRIPTION_TOPIC, on_text)

    async def wait_for_agent_to_finish(self, since: float, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not self.disconnected.is_set():
            if self.timeline.agent_finished(self.now(), since):
                return True
            await asyncio.sleep(0.05)
        return False

    async def speak(self, utterance: Utterance) -> Turn:
        queued = self.now()
        await self.queue.put(utterance)
        while utterance.name not in self.speech_end:
            if self.disconnected.is_set():
                raise ConnectionError("the room closed during an utterance")
            await asyncio.sleep(0.005)
        end = self.speech_end.pop(utterance.name)
        finished = await self.wait_for_agent_to_finish(end, self.options.turn_timeout_s)
        voice = self.timeline.first_voice_after(end)
        return Turn(
            utterance=utterance.name,
            speech_s=round(utterance.last_voiced / SAMPLE_RATE, 2),
            t_speech_end=round(end, 3),
            t_agent_voice=round(voice, 3) if voice is not None else None,
            overlap=self.timeline.voice_between(queued, end),
            agent_finished=finished,
        )


async def run_call(options: CallOptions, utterances: Sequence[Utterance]) -> Call:
    """One call: join, hear the greeting if there is one, then speak each utterance."""
    room_name = options.room or f"{options.room_prefix}{uuid.uuid4()}"
    identity = f"probe-{uuid.uuid4().hex[:8]}"
    live = _LiveCall(options, identity)
    live.bind_events()
    await live.room.connect(options.url, caller_token(options, room_name, identity))
    for participant in live.room.remote_participants.values():
        state = participant.attributes.get(STATE_ATTRIBUTE)
        if state:
            live.note_state(state)
    track = rtc.LocalAudioTrack.create_audio_track("microphone", live.source)
    await live.room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )
    microphone = asyncio.ensure_future(live.microphone())
    call = Call(
        room=room_name,
        caller_identity=identity,
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        greeted=False,
    )
    try:
        # An agent that waits for the caller to speak first is fine: the call goes on.
        call.greeted = await live.wait_for_agent_to_finish(
            0.0, options.greeting_timeout_s
        )
        for utterance in utterances:
            await asyncio.sleep(options.gap_s)
            if live.disconnected.is_set():
                break
            call.turns.append(await live.speak(utterance))
        # Late packets and transcripts still belong to the last turn.
        await asyncio.sleep(options.settle_s)
    finally:
        microphone.cancel()
        await live.room.disconnect()
        for task in list(live.tasks):
            task.cancel()
    call.states = [StateChange(round(t, 3), state) for t, state in live.timeline.states]
    call.data = live.data
    call.transcripts = live.transcripts
    call.duration_s = round(live.now(), 1)
    return call
