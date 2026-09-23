from agent.barge_in import BargeInTracker


def _speaking(t: BargeInTracker, at: float = 0.0) -> None:
    t.on_agent_state("thinking", "speaking", at)


def test_clean_interruption():
    t = BargeInTracker()
    _speaking(t)
    t.on_user_state("listening", "speaking", 1.0)
    t.on_agent_state("speaking", "listening", 1.52)  # paused after min_duration
    t.on_user_state("speaking", "listening", 2.4)
    assert t.on_user_turn(2.6).outcome == "interrupted"
    m = t.pop_turn_metrics()
    assert m.interrupted and m.barge_in_success and not m.false_interruption
    assert m.interruption_handled_ms == 520.0


def test_overlap_ignored_is_failure():
    t = BargeInTracker()
    _speaking(t)
    t.on_user_state("listening", "speaking", 1.0)
    ep = t.on_user_state("speaking", "listening", 1.3)  # too short to interrupt
    assert ep.outcome == "ignored"
    m = t.pop_turn_metrics()
    assert not m.interrupted
    assert m.barge_in_success is False
    assert m.interruption_handled_ms is None


def test_backchannel_false_interruption_resumed():
    t = BargeInTracker()
    _speaking(t)
    t.on_user_state("listening", "speaking", 1.0)
    t.on_agent_state("speaking", "listening", 1.5)
    t.on_user_state("speaking", "listening", 1.7)
    t.on_agent_state("listening", "speaking", 3.7)  # LiveKit restores state, then emits
    assert t.on_false_interruption(resumed=True, at=3.7).outcome == "false_interruption"
    assert t.on_user_turn(4.0) is None
    m = t.pop_turn_metrics()
    assert not m.interrupted
    assert m.false_interruption and m.false_interruption_resumed is True


def test_no_overlap_means_not_interrupted():
    t = BargeInTracker()
    _speaking(t)
    t.on_agent_state("speaking", "listening", 3.0)
    t.on_user_state("listening", "speaking", 4.0)  # user speaks after the agent finished
    m = t.pop_turn_metrics()
    assert not m.interrupted
    assert m.barge_in_success is None
    assert not t.episode_open


def test_state_resets_between_turns():
    t = BargeInTracker()
    _speaking(t)
    t.on_user_state("listening", "speaking", 1.0)
    t.on_agent_state("speaking", "listening", 1.5)
    t.on_user_turn(2.0)
    assert t.pop_turn_metrics().interrupted
    assert t.pop_turn_metrics().interrupted is False
    assert not t.awaiting_user_turn

