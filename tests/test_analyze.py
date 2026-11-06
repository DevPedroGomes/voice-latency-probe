import math

import pytest

from voice_latency_probe.analyze import (
    NOT_REPORTED,
    REPORTED,
    LabelRule,
    compare,
    decompose,
    percentile,
    render,
    render_comparison,
    stats,
    summarize,
)
from voice_latency_probe.record import Call, DataEvent, StateChange, Transcript, Turn

CALLER = "probe-1"


def turn(end: float, voice: float | None, overlap: bool = False) -> Turn:
    return Turn("u", 1.0, end, voice, overlap, voice is not None)


def call(
    turns: list[Turn],
    finals: tuple[float, ...] = (),
    speaking: tuple[float, ...] = (),
    data: tuple[tuple[float, dict[str, object]], ...] = (),
) -> Call:
    return Call(
        room="r",
        caller_identity=CALLER,
        started_at="2026-11-06T00:00:00+00:00",
        greeted=True,
        turns=turns,
        states=[StateChange(t, "speaking") for t in speaking],
        data=[DataEvent(t, dict(payload)) for t, payload in data],
        transcripts=[Transcript(t, CALLER, True) for t in finals],
    )


def test_a_turn_breaks_into_three_stages_that_add_up() -> None:
    result = decompose(call([turn(10.0, 11.5)], finals=(10.4,), speaking=(11.3,)))[0]
    assert result.total_ms == 1500
    assert result.end_of_turn_ms == 400
    assert result.reply_ms == 1100
    assert result.agent_ms == 900
    assert result.delivery_ms == 200


def test_a_state_that_arrives_after_the_voice_gives_negative_delivery() -> None:
    result = decompose(call([turn(10.0, 11.0)], finals=(10.4,), speaking=(11.1,)))[0]
    assert result.delivery_ms == -100
    assert result.reply_ms == 600


def test_a_state_from_the_next_turn_does_not_count() -> None:
    first, _ = decompose(
        call([turn(10.0, 11.0), turn(20.0, 21.0)], finals=(10.4,), speaking=(20.9,))
    )
    assert first.delivery_ms is None
    assert first.reply_ms == 600


def test_a_final_slightly_before_the_recorded_end_is_kept_as_negative() -> None:
    result = decompose(call([turn(10.0, 11.0)], finals=(9.9,), speaking=(10.8,)))[0]
    assert result.end_of_turn_ms == -100


def test_a_final_from_the_previous_turn_does_not_count() -> None:
    result = decompose(call([turn(10.0, 11.0)], finals=(9.0,), speaking=(10.8,)))[0]
    assert result.end_of_turn_ms is None
    assert result.agent_ms is None
    assert result.delivery_ms == 200


def test_the_agents_own_transcript_is_not_the_end_of_turn() -> None:
    record = call([turn(10.0, 11.0)], speaking=(10.8,))
    record.transcripts = [Transcript(10.5, "the-agent", True)]
    assert decompose(record)[0].end_of_turn_ms is None


def test_an_interim_transcript_is_not_the_end_of_turn() -> None:
    record = call([turn(10.0, 11.0)], speaking=(10.8,))
    record.transcripts = [Transcript(10.5, CALLER, False)]
    assert decompose(record)[0].end_of_turn_ms is None


def test_an_unanswered_turn_has_no_numbers_and_is_counted() -> None:
    summary = summarize([call([turn(10.0, None), turn(20.0, 21.0)])])
    assert summary.turns_sent == 2
    assert summary.unanswered == 1
    assert summary.total is not None
    assert summary.total.count == 1


def test_a_reported_number_belongs_to_the_turn_it_arrives_in() -> None:
    record = call(
        [turn(10.0, 11.0), turn(20.0, 21.0)],
        data=((11.2, {"total_ms": 700}), (21.3, {"other": 1})),
    )
    first, second = decompose(record)
    assert first.reported_ms == 700
    assert first.label == REPORTED
    assert second.reported_ms is None
    assert second.label == NOT_REPORTED


def test_the_gap_between_caller_and_agent_is_measured_on_the_same_turns() -> None:
    record = call(
        [turn(10.0, 11.5), turn(20.0, 25.0)], data=((11.6, {"total_ms": 500}),)
    )
    summary = summarize([record])
    assert summary.reported_turns == 1
    assert summary.reported_gap is not None
    assert summary.reported_gap.p50 == 1000
    assert summary.caller_on_reported_turns is not None
    assert summary.caller_on_reported_turns.p50 == 1500


@pytest.mark.parametrize("bad", [True, "700", None])
def test_a_reported_value_must_be_a_number(bad: object) -> None:
    record = call([turn(10.0, 11.0)], data=((11.1, {"total_ms": bad}),))
    assert decompose(record)[0].reported_ms is None


def test_the_reported_field_is_configurable() -> None:
    record = call([turn(10.0, 11.0)], data=((11.1, {"e2e": 300}),))
    assert decompose(record, reported_field="e2e")[0].reported_ms == 300


def test_label_rules_name_a_turn_kind_and_the_first_rule_wins() -> None:
    record = call(
        [turn(10.0, 11.0)], data=((10.5, {"rule": "CONFIRM_BY_RULE", "total_ms": 1}),)
    )
    rules = [
        LabelRule("rule", "CONFIRM", "rule, no LLM"),
        LabelRule("rule", "CONF", "other"),
    ]
    assert decompose(record, rules=rules)[0].label == "rule, no LLM"


def test_percentile_interpolates_like_the_usual_tools() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4, 5], 90) == pytest.approx(4.6)
    assert percentile([7], 95) == 7
    assert math.isnan(percentile([], 50))


def test_stats_of_nothing_is_none() -> None:
    assert stats([]) is None


def test_the_report_warns_about_a_small_sample_and_shows_coverage() -> None:
    text = render(
        summarize(
            [
                call(
                    [turn(10.0, 11.0), turn(20.0, 21.0)],
                    data=((11.1, {"total_ms": 9}),),
                )
            ]
        )
    )
    assert "small sample (2 turns)" in text
    assert "reported a total on 1 of 2 turns" in text


def test_an_empty_run_renders_without_failing() -> None:
    text = render(summarize([]))
    assert "0 calls" in text
    assert "small sample" not in text


def test_compare_shows_the_change_at_the_median() -> None:
    slow = summarize([call([turn(10.0, 12.0)])])
    fast = summarize([call([turn(10.0, 11.5)])])
    rows = compare(slow, fast)
    assert rows[0].p50_delta == -500
    assert "-500" in render_comparison(rows)
    assert rows[1].p50_delta is None
