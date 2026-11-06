"""Audio helpers shared by the caller and the fake agent. No LiveKit imports."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 48_000
FRAME_SAMPLES = SAMPLE_RATE // 100  # 10 ms
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE

# Decoded Opus silence stays close to zero, so a low bar separates voice from it.
DEFAULT_VOICE_RMS = 300.0

Samples = NDArray[np.int16]


def rms(samples: Samples) -> float:
    if samples.size == 0:
        return 0.0
    as_float = samples.astype(np.float64)
    return float(np.sqrt(np.mean(as_float * as_float)))


def last_voiced_sample(samples: Samples, threshold: float = DEFAULT_VOICE_RMS) -> int:
    """Index of the last sample above the threshold: where the speech really ends.

    A WAV usually has trailing silence, and counting it would shorten every latency
    measured from the end of the file.
    """
    voiced = np.nonzero(np.abs(samples.astype(np.int32)) > threshold)[0]
    if voiced.size == 0:
        raise ValueError("the audio has no sample above the voice threshold")
    return int(voiced[-1])


def load_wav(path: Path) -> Samples:
    """Reads a 48 kHz mono 16-bit WAV. Anything else is refused, not resampled."""
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz")
        if wav.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono")
        if wav.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit samples")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)


def write_wav(path: Path, samples: Samples) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(samples.tobytes())


def tone(
    seconds: float, hertz: float = 220.0, amplitude: int = 8_000, fade_s: float = 0.01
) -> Samples:
    """A tone with short fades, for the fake agent. No STT can read it."""
    count = int(seconds * SAMPLE_RATE)
    time = np.arange(count) / SAMPLE_RATE
    wave_form = np.sin(2 * np.pi * hertz * time) * amplitude
    fade = min(int(fade_s * SAMPLE_RATE), count // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade)
        wave_form[:fade] *= ramp
        wave_form[-fade:] *= ramp[::-1]
    return wave_form.astype(np.int16)


def frames(samples: Samples) -> list[Samples]:
    """Cuts audio into 10 ms frames, padding the last one with silence."""
    pieces = []
    for start in range(0, len(samples), FRAME_SAMPLES):
        piece = samples[start : start + FRAME_SAMPLES]
        if len(piece) < FRAME_SAMPLES:
            piece = np.pad(piece, (0, FRAME_SAMPLES - len(piece)))
        pieces.append(piece)
    return pieces
