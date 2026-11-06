import json
from pathlib import Path

import pytest

from voice_latency_probe.audio import load_wav
from voice_latency_probe.cli import load_rules, main

SAMPLE = Path(__file__).parent / "fixtures" / "run.jsonl"


def test_analyze_runs_on_the_sample_with_no_keys(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LIVEKIT_URL", raising=False)
    assert main(["analyze", str(SAMPLE)]) == 0
    output = capsys.readouterr().out
    assert "caller: voice to voice" in output
    assert "stage: end of turn" in output
    assert "stage: reply" in output
    assert "caller minus agent" in output


def test_analyze_can_print_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["analyze", str(SAMPLE), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["calls"] >= 1


def test_compare_runs(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["compare", str(SAMPLE), str(SAMPLE)]) == 0
    assert "+0" in capsys.readouterr().out


def test_a_broken_record_is_a_message_and_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{", encoding="utf-8")
    assert main(["analyze", str(path)]) == 1
    assert "bad.jsonl:1" in capsys.readouterr().err


def test_make_tones_writes_files_the_caller_accepts(tmp_path: Path) -> None:
    assert main(["make-tones", str(tmp_path), "--count", "2"]) == 0
    assert load_wav(tmp_path / "tone-01.wav").size > 0
    assert len(list(tmp_path.glob("*.wav"))) == 2


def test_call_without_credentials_names_what_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit, match="missing LIVEKIT_URL"):
        main(["call", str(tmp_path), "-o", str(tmp_path / "o.jsonl")])


def test_a_bad_attribute_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "ws://x")
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")
    with pytest.raises(SystemExit, match="key=value"):
        main(["call", str(tmp_path), "-o", str(tmp_path / "o"), "--attr", "nope"])


def test_label_rules_load_from_json(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text('[{"field": "f", "prefix": "p", "label": "l"}]', encoding="utf-8")
    assert load_rules(path)[0].label == "l"
    assert load_rules(None) == []
