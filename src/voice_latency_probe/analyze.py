"""Turns records into numbers. Pure functions, no audio and no network.

For each turn, on the caller's clock:

    t_speech_end   last voiced frame the caller sent
    t_final        caller's final transcript came back (the agent closed the turn)
    t_speaking     the agent's `lk.agent.state` became `speaking`
    t_agent_voice  first voiced frame received from the agent

    total        = t_agent_voice - t_speech_end
    end_of_turn  = t_final - t_speech_end
    reply        = t_agent_voice - t_final   (agent + delivery, needs no state)
    agent        = t_speaking - t_final      (LLM, tools, TTS)
    delivery     = t_agent_voice - t_speaking

The state travels on the signaling path and the voice on the media path, so the state
can arrive after the voice. Delivery is then negative and agent reads high. Reply is the
number to trust for that split: it uses only the transcript and the audio.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .record import Call
from .timeline import SPEAKING

# A final transcript can land just before the last voiced frame leaves the send queue.
FINAL_SLACK_S = 0.3
SMALL_SAMPLE = 20

REPORTED = "reported by the agent"
NOT_REPORTED = "not reported"


@dataclass(frozen=True)
class LabelRule:
    """Names a kind of turn when a data packet's `field` starts with `prefix`."""

    field: str
    prefix: str
    label: str

    def matches(self, payload: dict[str, Any]) -> bool:
        return str(payload.get(self.field, "")).startswith(self.prefix)


@dataclass(frozen=True)
class TurnResult:
    room: str
    utterance: str
    label: str
    total_ms: int
    end_of_turn_ms: int | None
    reply_ms: int | None
    agent_ms: int | None
    delivery_ms: int | None
    reported_ms: float | None
    overlap: bool


@dataclass(frozen=True)
class Stats:
    count: int
    p50: float
    p90: float
    p95: float
    low: float
    high: float


def percentile(values: Sequence[float], rank: float) -> float:
    """Linear interpolation between closest ranks."""
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * rank / 100
    below = math.floor(position)
    above = math.ceil(position)
    weight = position - below
    return ordered[below] + (ordered[above] - ordered[below]) * weight


def stats(values: Sequence[float]) -> Stats | None:
    if not values:
        return None
    return Stats(
        count=len(values),
        p50=percentile(values, 50),
        p90=percentile(values, 90),
        p95=percentile(values, 95),
        low=min(values),
        high=max(values),
    )


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


def _number(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def decompose(
    call: Call,
    reported_field: str = "total_ms",
    rules: Sequence[LabelRule] = (),
) -> list[TurnResult]:
    """One result per answered turn. Unanswered turns are counted elsewhere."""
    finals = sorted(
        item.t
        for item in call.transcripts
        if item.final and item.identity == call.caller_identity
    )
    results = []
    for index, turn in enumerate(call.turns):
        if turn.t_agent_voice is None:
            continue
        end, voice = turn.t_speech_end, turn.t_agent_voice
        next_end = (
            call.turns[index + 1].t_speech_end
            if index + 1 < len(call.turns)
            else math.inf
        )
        final = next((t for t in finals if end - FINAL_SLACK_S < t < voice), None)
        speaking = next(
            (
                change.t
                for change in call.states
                if change.state == SPEAKING and end < change.t < next_end
            ),
            None,
        )
        events = [
            event for event in call.data if end - FINAL_SLACK_S < event.t < next_end
        ]
        reported = next(
            (
                value
                for event in events
                if (value := _number(event.payload, reported_field)) is not None
            ),
            None,
        )
        label = next(
            (
                rule.label
                for rule in rules
                for event in events
                if rule.matches(event.payload)
            ),
            REPORTED if reported is not None else NOT_REPORTED,
        )
        results.append(
            TurnResult(
                room=call.room,
                utterance=turn.utterance,
                label=label,
                total_ms=_ms(voice - end),
                end_of_turn_ms=_ms(final - end) if final is not None else None,
                reply_ms=_ms(voice - final) if final is not None else None,
                agent_ms=_ms(speaking - final)
                if speaking is not None and final is not None
                else None,
                delivery_ms=_ms(voice - speaking) if speaking is not None else None,
                reported_ms=reported,
                overlap=turn.overlap,
            )
        )
    return results


@dataclass(frozen=True)
class Summary:
    calls: int
    turns_sent: int
    unanswered: int
    overlaps: int
    total: Stats | None
    by_label: dict[str, Stats]
    end_of_turn: Stats | None
    reply: Stats | None
    agent: Stats | None
    delivery: Stats | None
    reported_turns: int
    reported: Stats | None
    caller_on_reported_turns: Stats | None
    reported_gap: Stats | None


def summarize(
    calls: Sequence[Call],
    reported_field: str = "total_ms",
    rules: Sequence[LabelRule] = (),
) -> Summary:
    results = [
        result for call in calls for result in decompose(call, reported_field, rules)
    ]
    turns_sent = sum(len(call.turns) for call in calls)

    def present(values: Iterable[int | None]) -> list[float]:
        return [float(value) for value in values if value is not None]

    labels = sorted({result.label for result in results})
    with_report = [result for result in results if result.reported_ms is not None]
    return Summary(
        calls=len(calls),
        turns_sent=turns_sent,
        unanswered=turns_sent - len(results),
        overlaps=sum(result.overlap for result in results),
        total=stats([float(result.total_ms) for result in results]),
        by_label={
            label: summary
            for label in labels
            if (
                summary := stats(
                    [float(r.total_ms) for r in results if r.label == label]
                )
            )
        },
        end_of_turn=stats(present(r.end_of_turn_ms for r in results)),
        reply=stats(present(r.reply_ms for r in results)),
        agent=stats(present(r.agent_ms for r in results)),
        delivery=stats(present(r.delivery_ms for r in results)),
        reported_turns=len(with_report),
        reported=stats(
            [r.reported_ms for r in with_report if r.reported_ms is not None]
        ),
        caller_on_reported_turns=stats([float(r.total_ms) for r in with_report]),
        reported_gap=stats(
            [
                r.total_ms - r.reported_ms
                for r in with_report
                if r.reported_ms is not None
            ]
        ),
    )


def _row(name: str, item: Stats | None) -> str:
    if item is None:
        return f"{name:<36} {'n=0':>6}"
    return (
        f"{name:<36} {'n=' + str(item.count):>6} {item.p50:>7.0f} {item.p90:>7.0f} "
        f"{item.p95:>7.0f} {item.low:>7.0f} {item.high:>7.0f}"
    )


def render(summary: Summary) -> str:
    header = (
        f"{'milliseconds':<36} {'':>6} {'p50':>7} {'p90':>7} "
        f"{'p95':>7} {'min':>7} {'max':>7}"
    )
    lines = [
        f"{summary.calls} calls, {summary.turns_sent} turns sent, "
        f"{summary.unanswered} unanswered, {summary.overlaps} with the agent talking "
        "over the caller",
        "",
        header,
        _row("caller: voice to voice", summary.total),
    ]
    lines += [_row(f"  {label}", item) for label, item in summary.by_label.items()]
    lines += [
        "",
        _row("stage: end of turn", summary.end_of_turn),
        _row("stage: reply (agent + delivery)", summary.reply),
        _row("  by state: agent (LLM, tools, TTS)", summary.agent),
        _row("  by state: delivery", summary.delivery),
    ]
    answered = summary.turns_sent - summary.unanswered
    lines += [
        "",
        f"agent reported a total on {summary.reported_turns} of {answered} turns",
    ]
    if summary.reported_turns:
        lines += [
            _row("agent's own number", summary.reported),
            _row("caller, same turns", summary.caller_on_reported_turns),
            _row("caller minus agent", summary.reported_gap),
        ]
    if summary.total and summary.total.count < SMALL_SAMPLE:
        lines += [
            "",
            f"small sample ({summary.total.count} turns): p90 and p95 are close to the "
            "largest value measured.",
        ]
    return "\n".join(lines)


@dataclass(frozen=True)
class Comparison:
    name: str
    before: Stats | None
    after: Stats | None

    @property
    def p50_delta(self) -> float | None:
        if self.before is None or self.after is None:
            return None
        return self.after.p50 - self.before.p50


def compare(before: Summary, after: Summary) -> list[Comparison]:
    return [
        Comparison("voice to voice", before.total, after.total),
        Comparison("end of turn", before.end_of_turn, after.end_of_turn),
        Comparison("reply", before.reply, after.reply),
    ]


def render_comparison(rows: Sequence[Comparison]) -> str:
    lines = [
        f"{'p50, milliseconds':<20} {'before':>8} {'after':>8} {'change':>8} {'n':>10}"
    ]
    for row in rows:
        before = f"{row.before.p50:.0f}" if row.before else "-"
        after = f"{row.after.p50:.0f}" if row.after else "-"
        delta = f"{row.p50_delta:+.0f}" if row.p50_delta is not None else "-"
        counts = (
            f"{row.before.count if row.before else 0}/"
            f"{row.after.count if row.after else 0}"
        )
        lines.append(f"{row.name:<20} {before:>8} {after:>8} {delta:>8} {counts:>10}")
    return "\n".join(lines)
