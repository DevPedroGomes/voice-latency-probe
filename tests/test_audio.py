import wave
from pathlib import Path

import numpy as np
import pytest

from voice_latency_probe.audio import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    frames,
    last_voiced_sample,
    load_wav,
    rms,
    tone,
    write_wav,
)
from voice_latency_probe.caller import load_utterances


def test_the_end_of_speech_ignores_trailing_silence() -> None:
    samples = np.concatenate(
        [np.full(1000, 5000, dtype=np.int16), np.zeros(9000, dtype=np.int16)]
    )
    assert last_voiced_sample(samples) == 999


def test_audio_with_no_voice_is_refused() -> None:
    with pytest.raises(ValueError, match="no sample above"):
        last_voiced_sample(np.zeros(100, dtype=np.int16))


def test_rms_handles_the_int16_edge_and_empty_input() -> None:
    assert rms(np.array([], dtype=np.int16)) == 0.0
    assert rms(np.full(4, -32768, dtype=np.int16)) == 32768.0


def test_a_tone_is_loud_enough_to_count_as_voice() -> None:
    assert rms(tone(0.2)) > 300


def test_frames_are_all_10_ms_with_the_last_one_padded() -> None:
    pieces = frames(np.ones(FRAME_SAMPLES + 5, dtype=np.int16))
    assert [len(piece) for piece in pieces] == [FRAME_SAMPLES, FRAME_SAMPLES]
    assert pieces[1][5] == 0


def test_a_wav_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    samples = tone(0.1)
    write_wav(path, samples)
    assert np.array_equal(load_wav(path), samples)


def _wav(path: Path, rate: int, channels: int, width: int) -> None:
    with wave.open(str(path), "wb") as output:
        output.setframerate(rate)
        output.setnchannels(channels)
        output.setsampwidth(width)
        output.writeframes(b"\0\0" * channels * width * 10)


@pytest.mark.parametrize(
    ("rate", "channels", "width", "message"),
    [
        (16_000, 1, 2, "48000 Hz"),
        (SAMPLE_RATE, 2, 2, "mono"),
        (SAMPLE_RATE, 1, 1, "16-bit"),
    ],
)
def test_other_formats_are_refused(
    tmp_path: Path, rate: int, channels: int, width: int, message: str
) -> None:
    path = tmp_path / "bad.wav"
    _wav(path, rate, channels, width)
    with pytest.raises(ValueError, match=message):
        load_wav(path)


def test_a_directory_of_utterances_is_read_in_name_order(tmp_path: Path) -> None:
    write_wav(tmp_path / "b.wav", tone(0.1))
    write_wav(tmp_path / "a.wav", tone(0.1))
    assert [item.name for item in load_utterances([tmp_path])] == ["a", "b"]


def test_no_utterances_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no WAV files"):
        load_utterances([tmp_path])
