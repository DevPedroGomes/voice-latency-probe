"""Command line: `call`, `analyze`, `compare`, `fake-agent`, `make-tones`."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .analyze import (
    LabelRule,
    Summary,
    compare,
    render,
    render_comparison,
    summarize,
)
from .audio import DEFAULT_VOICE_RMS, tone, write_wav
from .record import append_call, read_calls

ENV_HINT = "set LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET in the environment"


def _credentials() -> tuple[str, str, str]:
    try:
        return (
            os.environ["LIVEKIT_URL"],
            os.environ["LIVEKIT_API_KEY"],
            os.environ["LIVEKIT_API_SECRET"],
        )
    except KeyError as error:
        raise SystemExit(f"missing {error.args[0]}: {ENV_HINT}") from error


def _attributes(items: Sequence[str]) -> dict[str, str]:
    attributes = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise SystemExit(f"--attr needs key=value, got {item!r}")
        attributes[key] = value
    return attributes


def load_rules(path: Path | None) -> list[LabelRule]:
    """A JSON list of {"field", "prefix", "label"}, first match wins."""
    if path is None:
        return []
    return [LabelRule(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


def _cmd_call(args: argparse.Namespace) -> int:
    from .caller import CallOptions, load_utterances, run_call

    url, key, secret = _credentials()
    options = CallOptions(
        url=url,
        api_key=key,
        api_secret=secret,
        room=args.room,
        room_prefix=args.room_prefix,
        agent_name=args.agent_name,
        attributes=_attributes(args.attr),
        greeting_timeout_s=args.greeting_timeout,
        turn_timeout_s=args.turn_timeout,
        voice_rms=args.voice_rms,
        keep_text=args.keep_text,
    )
    utterances = load_utterances(args.utterances, args.voice_rms)

    async def calls() -> None:
        for number in range(args.calls):
            if number:
                await asyncio.sleep(args.pause)
            call = await run_call(options, utterances)
            append_call(args.output, call)
            heard = [
                round((turn.t_agent_voice - turn.t_speech_end) * 1000)
                if turn.t_agent_voice is not None
                else None
                for turn in call.turns
            ]
            print(f"{call.room}: greeted={call.greeted} caller_ms={heard}", flush=True)

    asyncio.run(calls())
    return 0


def _cmd_fake_agent(args: argparse.Namespace) -> int:
    from .fake_agent import FakeAgentOptions, run_fake_agent

    url, key, secret = _credentials()
    asyncio.run(
        run_fake_agent(
            FakeAgentOptions(
                url=url,
                api_key=key,
                api_secret=secret,
                room=args.room,
                end_of_turn_ms=args.end_of_turn_ms,
                agent_ms=args.agent_ms,
                report_every=args.report_every,
            )
        )
    )
    return 0


def _summary(paths: Sequence[Path], field: str, rules: Sequence[LabelRule]) -> Summary:
    return summarize(read_calls(paths), field, rules)


def _cmd_analyze(args: argparse.Namespace) -> int:
    summary = _summary(args.records, args.reported_field, load_rules(args.labels))
    print(json.dumps(asdict(summary), indent=1) if args.json else render(summary))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    before = _summary([args.before], args.reported_field, [])
    after = _summary([args.after], args.reported_field, [])
    print(render_comparison(compare(before, after)))
    return 0


def _cmd_make_tones(args: argparse.Namespace) -> int:
    """WAVs for the fake agent. A real agent's STT needs real speech."""
    args.directory.mkdir(parents=True, exist_ok=True)
    for number in range(args.count):
        seconds = 1.0 + 0.25 * (number % 4)
        write_wav(args.directory / f"tone-{number + 1:02}.wav", tone(seconds, 180.0))
    print(f"{args.count} WAVs in {args.directory}")
    return 0


def parser() -> argparse.ArgumentParser:
    main_parser = argparse.ArgumentParser(prog="voice-probe", description=__doc__)
    main_parser.add_argument("--version", action="version", version=__version__)
    commands = main_parser.add_subparsers(dest="command", required=True)

    call = commands.add_parser("call", help="call an agent and record what comes back")
    call.add_argument(
        "utterances", nargs="+", type=Path, help="WAV files or directories"
    )
    call.add_argument(
        "-o", "--output", type=Path, required=True, help="JSONL, appended"
    )
    call.add_argument("--calls", type=int, default=1)
    call.add_argument("--pause", type=float, default=5.0, help="seconds between calls")
    call.add_argument("--room", help="fixed room name; default is a new room per call")
    call.add_argument(
        "--room-prefix", default="probe-", help="for prefix dispatch rules"
    )
    call.add_argument("--agent-name", help="explicit agent dispatch")
    call.add_argument("--attr", action="append", default=[], metavar="KEY=VALUE")
    call.add_argument("--greeting-timeout", type=float, default=30.0)
    call.add_argument("--turn-timeout", type=float, default=45.0)
    call.add_argument("--voice-rms", type=float, default=DEFAULT_VOICE_RMS)
    call.add_argument(
        "--keep-text", action="store_true", help="store transcripts' text"
    )
    call.set_defaults(run=_cmd_call)

    analyze = commands.add_parser("analyze", help="print the numbers of recorded calls")
    analyze.add_argument("records", nargs="+", type=Path)
    analyze.add_argument("--reported-field", default="total_ms")
    analyze.add_argument("--labels", type=Path, help="JSON rules that name turn kinds")
    analyze.add_argument("--json", action="store_true")
    analyze.set_defaults(run=_cmd_analyze)

    versus = commands.add_parser("compare", help="compare two runs at the median")
    versus.add_argument("before", type=Path)
    versus.add_argument("after", type=Path)
    versus.add_argument("--reported-field", default="total_ms")
    versus.set_defaults(run=_cmd_compare)

    fake = commands.add_parser("fake-agent", help="an agent with known delays, no keys")
    fake.add_argument("--room", required=True)
    fake.add_argument("--end-of-turn-ms", type=int, default=400)
    fake.add_argument("--agent-ms", type=int, default=300)
    fake.add_argument("--report-every", type=int, default=1, help="0 reports nothing")
    fake.set_defaults(run=_cmd_fake_agent)

    tones = commands.add_parser("make-tones", help="WAVs to use with the fake agent")
    tones.add_argument("directory", type=Path)
    tones.add_argument("--count", type=int, default=6)
    tones.set_defaults(run=_cmd_make_tones)
    return main_parser


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.run(args))
    except (ValueError, OSError) as error:
        print(f"voice-probe: {error}", file=sys.stderr)
        return 1
