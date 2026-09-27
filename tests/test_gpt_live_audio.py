import numpy as np
import pytest

from joinly.voice.gpt_live.audio import Pcm16Carry, PlaybackBuffer, StreamResampler
from joinly.voice.gpt_live.bargein import SpeechActivity


def _tone(freq: float, rate: int, seconds: float = 1.0, amp: float = 0.5) -> bytes:
    t = np.arange(int(rate * seconds)) / rate
    return (np.sin(2 * np.pi * freq * t) * amp * 32767).astype("<i2").tobytes()


def _amplitude(pcm: bytes) -> float:
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    mid = x[len(x) // 4 : 3 * len(x) // 4]  # skip filter warm-up
    return float(np.sqrt(2) * mid.std() / 32767)


RATE_PAIRS = [(48000, 24000), (48000, 16000), (24000, 16000), (16000, 24000)]


@pytest.mark.parametrize(("src", "dst"), RATE_PAIRS)
def test_resampler_output_length(src: int, dst: int) -> None:
    """One second of input yields ~one second of output (minus filter lookahead)."""
    out = StreamResampler(src, dst).process(_tone(1000, src))
    n_out = len(out) // 2
    assert dst - 50 <= n_out <= dst


@pytest.mark.parametrize(("src", "dst"), RATE_PAIRS)
def test_resampler_preserves_passband(src: int, dst: int) -> None:
    """A 1 kHz tone keeps its amplitude."""
    out = StreamResampler(src, dst).process(_tone(1000, src))
    assert _amplitude(out) == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize(
    ("src", "dst", "freq"),
    [(48000, 24000, 15000), (48000, 16000, 12000), (24000, 16000, 10000)],
)
def test_resampler_rejects_aliases(src: int, dst: int, freq: float) -> None:
    """Content above the target Nyquist frequency is attenuated by >= 60 dB."""
    out = StreamResampler(src, dst).process(_tone(freq, src))
    attenuation_db = 20 * np.log10(max(_amplitude(out), 1e-9) / 0.5)
    assert attenuation_db < -60  # noqa: PLR2004


@pytest.mark.parametrize(("src", "dst"), RATE_PAIRS)
def test_resampler_chunking_is_seamless(src: int, dst: int) -> None:
    """Random chunk sizes (including odd byte counts) match one-pass output."""
    data = _tone(440, src) + _tone(3000, src, 0.3)
    whole = StreamResampler(src, dst).process(data)

    chunked = StreamResampler(src, dst)
    rng = np.random.default_rng(1)
    parts, i = [], 0
    while i < len(data):
        n = int(rng.integers(1, 777))
        parts.append(chunked.process(data[i : i + n]))
        i += n
    assert b"".join(parts) == whole


def test_resampler_same_rate_passthrough() -> None:
    """Equal rates return the input unchanged (aligned to whole samples)."""
    r = StreamResampler(24000, 24000)
    assert r.process(b"\x01\x02\x03") == b"\x01\x02"
    assert r.process(b"\x04") == b"\x03\x04"


def test_pcm16_carry_keeps_odd_byte() -> None:
    """Odd trailing bytes are carried to the next chunk."""
    carry = Pcm16Carry()
    assert carry.push(b"\x01") == b""
    assert carry.push(b"\x02\x03") == b"\x01\x02"
    assert carry.push(b"\x04\x05\x06") == b"\x03\x04\x05\x06"


def test_resampler_rejects_invalid_rates() -> None:
    """Non-positive rates are rejected."""
    with pytest.raises(ValueError, match="positive"):
        StreamResampler(0, 16000)


async def test_playback_buffer_pop_and_clear() -> None:
    """Buffer pops in order across chunk boundaries and clears fully."""
    buf = PlaybackBuffer()
    buf.push(b"abc")
    buf.push(b"defg")
    assert len(buf) == len(b"abcdefg")
    assert buf.pop(2) == b"ab"
    assert buf.pop(3) == b"cde"
    assert buf.clear() == len(b"fg")
    assert len(buf) == 0
    assert buf.pop(10) == b""


def test_speech_activity_detects_speech_and_silence() -> None:
    """Voice-like audio triggers `started`; trailing silence triggers `stopped`."""
    rate = 16000
    t = np.arange(rate) / rate
    # harmonic-rich, amplitude-modulated signal that webrtcvad classifies as speech
    voiced = sum(
        np.sin(2 * np.pi * f0 * k * t) / k for f0 in (140,) for k in range(1, 12)
    )
    voiced = voiced * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
    speech = (voiced / np.abs(voiced).max() * 0.6 * 32767).astype("<i2").tobytes()
    silence = b"\x00\x00" * rate

    activity = SpeechActivity(sample_rate=rate, min_speech_ms=300, min_silence_ms=300)
    started, stopped = activity.feed(silence[: rate // 2])
    assert (started, stopped) == (False, False)
    started, _ = activity.feed(speech)
    assert started
    assert activity.speaking
    _, stopped = activity.feed(silence)
    assert stopped
    assert not activity.speaking


def test_speech_activity_ignores_short_bursts() -> None:
    """A burst shorter than `min_speech_ms` does not count as a barge-in."""
    rate = 16000
    t = np.arange(rate // 10) / rate  # 100 ms
    burst = (np.sin(2 * np.pi * 200 * t) * 0.6 * 32767).astype("<i2").tobytes()
    activity = SpeechActivity(sample_rate=rate, min_speech_ms=300)
    started, _ = activity.feed(burst + b"\x00\x00" * rate)
    assert not started


def test_speech_activity_rejects_unsupported_rate() -> None:
    """Webrtcvad only supports 8/16/32/48 kHz."""
    with pytest.raises(ValueError, match="24000"):
        SpeechActivity(sample_rate=24000)
