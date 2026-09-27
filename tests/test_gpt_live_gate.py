import numpy as np
import pytest

from joinly.voice.gpt_live.gate import AddressGate, NameMatcher, OutputRouter


class FakeClock:
    """Manually advanced monotonic clock."""

    def __init__(self) -> None:
        """Initialize the fake."""
        self.now = 1000.0

    def __call__(self) -> float:
        """Return the current fake time."""
        return self.now

    def advance(self, seconds: float) -> None:
        """Advance the fake time."""
        self.now += seconds


ALIASES = ["helius", "helio's", "elios", "hey leos"]


@pytest.fixture
def matcher() -> NameMatcher:
    """Name matcher for Helios with ASR aliases."""
    return NameMatcher("Helios", ALIASES, "@helios")


@pytest.mark.parametrize(
    "text",
    [
        "Helios, can you summarize?",
        "hey helios what did Priya say",
        "Okay HELIOS.",
        "Hélios, one question",
        "helius can you check",
        "so, hey leos, quick one",
        "ask Helio's opinion",
        "what do you think... helios?",
    ],
)
def test_name_matches_speech(matcher: NameMatcher, text: str) -> None:
    """The name and its ASR variants are recognized in speech."""
    assert matcher.in_speech(text)


@pytest.mark.parametrize(
    "text",
    [
        "the heliosphere is large",
        "we use heliostat mirrors",
        "Let's talk about the budget.",
        "@helios in chat syntax is not speech",
        "",
    ],
)
def test_name_does_not_match_speech(matcher: NameMatcher, text: str) -> None:
    """Words containing the name, and unrelated text, do not match."""
    assert not matcher.in_speech(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("@helios what's the status?", True),
        ("@Helios summarize please", True),
        ("hey @HELIOS", True),
        ("Helios hey", True),  # Teams @mention chip renders as the plain name
        ("mail me at bob@helios.com", False),
        ("@heliosbot ping", False),
        ("the heliosphere is large", False),
    ],
)
def test_chat_trigger(matcher: NameMatcher, text: str, *, expected: bool) -> None:
    """Chat requires the explicit @helios token."""
    assert matcher.in_chat(text) is expected


def test_gate_followup_window() -> None:
    """The gate opens when addressed and closes after the follow-up window."""
    clock = FakeClock()
    gate = AddressGate(followup_seconds=20, clock=clock)
    assert not gate.is_open
    gate.open("name")
    assert gate.is_open
    clock.advance(19.9)
    assert gate.is_open
    clock.advance(0.2)
    assert not gate.is_open


def test_gate_extend_only_when_open() -> None:
    """Agent speech extends an open gate but never opens a closed one."""
    clock = FakeClock()
    gate = AddressGate(followup_seconds=10, clock=clock)
    gate.extend()
    assert not gate.is_open
    gate.open("name")
    clock.advance(8)
    gate.extend()
    clock.advance(8)
    assert gate.is_open


def test_gate_holds() -> None:
    """Holds keep the gate open; releasing starts a normal follow-up window."""
    clock = FakeClock()
    gate = AddressGate(followup_seconds=5, clock=clock)
    gate.hold("d1")
    clock.advance(100)
    assert gate.is_open
    gate.release("d1")
    clock.advance(4)
    assert gate.is_open
    clock.advance(2)
    assert not gate.is_open
    gate.release("unknown")  # no-op
    assert not gate.is_open


BPS = 48000  # 24 kHz PCM16


def _speech(tag: int, seconds: float = 0.1) -> bytes:
    """Loud PCM16 audio, distinguishable by `tag`."""
    n = int(BPS * seconds) // 2
    return np.full(n, 3000 + tag, dtype="<i2").tobytes()


def _silence(seconds: float = 0.1) -> bytes:
    """Digital silence."""
    return b"\x00" * (int(BPS * seconds) // 2 * 2)


def _router(
    clock: FakeClock, hold: float = 1.5
) -> tuple[AddressGate, OutputRouter, list[bytes]]:
    gate = AddressGate(followup_seconds=20, clock=clock)
    played: list[bytes] = []
    router = OutputRouter(
        gate,
        played.append,
        hold_seconds=hold,
        utterance_gap_seconds=0.5,
        bytes_per_second=BPS,
        clock=clock,
    )
    return gate, router, played


def test_router_plays_when_open() -> None:
    """Speech passes straight through while the gate is open."""
    clock = FakeClock()
    gate, router, played = _router(clock)
    gate.open("name")
    router.on_audio(_speech(1))
    router.on_audio(_speech(2))
    assert played == [_speech(1), _speech(2)]


def test_router_releases_held_speech_when_name_arrives_late() -> None:
    """Speech produced just before the name transcript arrives is released."""
    clock = FakeClock()
    gate, router, played = _router(clock)
    router.on_audio(_speech(1))
    clock.advance(0.1)
    router.on_audio(_speech(2))
    assert played == []
    clock.advance(0.8)
    gate.open("name heard")
    assert router.tick() is False
    assert played == [_speech(1), _speech(2)]
    router.on_audio(_speech(3))
    assert played[-1] == _speech(3)


def test_router_discards_unaddressed_speech() -> None:
    """Held speech is discarded after the hold time.

    The rest of that utterance is dropped, even if the gate opens later in it.
    """
    clock = FakeClock()
    gate, router, played = _router(clock, hold=1.5)
    for _ in range(5):
        router.on_audio(_speech(1))
        assert router.tick() is False
        clock.advance(0.3)
    clock.advance(0.1)
    router.on_audio(_speech(1))
    assert router.tick() is True
    assert router.discarded_utterances == 1
    clock.advance(0.2)
    gate.open("name")
    router.on_audio(_speech(2))
    assert played == []
    # a new utterance (after a gap in delivery) plays normally
    clock.advance(1.0)
    router.on_audio(_speech(3))
    assert played == [_speech(3)]


def test_router_continuous_stream_recovers_after_discard() -> None:
    """GPT-Live streams silence between replies without delivery gaps.

    Silence must end the discarded utterance so the next reply can be heard.
    """
    clock = FakeClock()
    gate, router, played = _router(clock, hold=0.3)
    router.on_audio(_speech(1))
    clock.advance(0.4)
    assert router.tick() is True  # unaddressed opening line discarded
    for _ in range(8):  # 0.8 s of streamed silence, no delivery gap
        clock.advance(0.1)
        router.on_audio(_silence())
    gate.open("name heard")
    clock.advance(0.1)
    router.on_audio(_speech(2))
    assert _speech(2) in played


def test_router_silence_is_not_held_or_discarded() -> None:
    """Silence while the gate is closed is dropped quietly (no discard note)."""
    clock = FakeClock()
    _gate, router, played = _router(clock)
    for _ in range(30):
        router.on_audio(_silence())
        clock.advance(0.1)
        assert router.tick() is False
    assert router.discarded_utterances == 0
    assert played == []


def test_router_barge_in_suppresses_rest_of_utterance() -> None:
    """After a barge-in, speech is dropped until the model utterance ends."""
    clock = FakeClock()
    gate, router, played = _router(clock)
    gate.open("name")
    router.on_audio(_speech(1))
    router.barge_in()
    clock.advance(0.1)
    router.on_audio(_speech(2))
    assert played == [_speech(1)]
    # continuous stream: 0.6 s of silence ends the interrupted utterance
    for _ in range(6):
        clock.advance(0.1)
        router.on_audio(_silence())
    clock.advance(0.1)
    router.on_audio(_speech(3))
    assert played[-1] == _speech(3)
