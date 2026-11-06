from voice_latency_probe.timeline import Timeline


def test_first_voice_after_ignores_the_previous_answer() -> None:
    timeline = Timeline(agent_voice=[1.0, 1.01, 5.0])
    assert timeline.first_voice_after(2.0) == 5.0
    assert timeline.first_voice_after(6.0) is None


def test_voice_while_the_caller_speaks_is_an_overlap() -> None:
    timeline = Timeline(agent_voice=[3.0])
    assert timeline.voice_between(2.0, 4.0)
    assert not timeline.voice_between(3.5, 4.0)


def test_silence_before_the_agent_answers_is_not_the_end() -> None:
    timeline = Timeline(agent_voice=[1.0], states=[(1.5, "listening")])
    assert not timeline.agent_finished(now=9.0, since=2.0)


def test_the_listening_state_shortens_the_wait() -> None:
    timeline = Timeline(agent_voice=[5.0], states=[(5.1, "listening")])
    assert not timeline.agent_finished(now=5.5, since=2.0)
    assert timeline.agent_finished(now=5.9, since=2.0)


def test_a_pause_inside_a_sentence_is_not_the_end() -> None:
    timeline = Timeline(agent_voice=[5.0], states=[(4.0, "speaking")])
    assert not timeline.agent_finished(now=6.0, since=2.0)


def test_an_agent_without_states_ends_after_long_silence() -> None:
    timeline = Timeline(agent_voice=[5.0])
    assert not timeline.agent_finished(now=7.0, since=2.0)
    assert timeline.agent_finished(now=7.6, since=2.0)
