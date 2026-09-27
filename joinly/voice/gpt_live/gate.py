import logging
import re
import time
import unicodedata
from collections.abc import Callable, Iterable

import numpy as np

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Lowercase, strip accents and collapse punctuation to spaces."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w@']+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


class NameMatcher:
    """Detects when the agent is addressed by name in speech or chat."""

    def __init__(
        self, name: str, aliases: Iterable[str] = (), chat_trigger: str | None = None
    ) -> None:
        """Initialize the matcher.

        Args:
            name: The agent's name, e.g. "Helios".
            aliases: Additional spellings that ASR may produce for the name.
            chat_trigger: Literal chat token, e.g. "@helios". Defaults to "@<name>".
        """
        variants = {_normalize(v) for v in (name, *aliases) if v.strip()}
        pattern = "|".join(
            re.escape(v) for v in sorted(variants, key=len, reverse=True)
        )
        self._speech = re.compile(rf"(?<![\w@])(?:{pattern})(?!\w)")
        trigger = chat_trigger or f"@{name}"
        self._chat = re.compile(rf"(?<!\w){re.escape(trigger.lower())}(?!\w)")

    def in_speech(self, text: str) -> bool:
        """Return True if the (transcribed) text addresses the agent by name."""
        return bool(self._speech.search(_normalize(text)))

    def in_chat(self, text: str) -> bool:
        """Return True if a chat message addresses the agent.

        Matches the explicit trigger (e.g. "@helios") or the name as a word.
        Teams renders real @-mentions as the plain display name ("Helios hey"),
        so the name alone must count in chat.
        """
        return bool(self._chat.search(text.lower())) or self.in_speech(text)


class AddressGate:
    """Decides whether the agent may currently be heard in the meeting.

    The gate is open for a follow-up window after the agent was addressed or last
    spoke, and while holds (e.g. running delegations) are active.
    """

    def __init__(
        self,
        *,
        followup_seconds: float = 20.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize a closed gate.

        Args:
            followup_seconds: How long the gate stays open after the agent was
                addressed or last spoke.
            clock: Monotonic time source in seconds (injectable for tests).
        """
        self.followup_seconds = followup_seconds
        self._clock = clock
        self._open_until = float("-inf")
        self._holds: set[str] = set()

    @property
    def is_open(self) -> bool:
        """Whether agent audio may currently be played."""
        return bool(self._holds) or self._clock() < self._open_until

    def open(self, reason: str, seconds: float | None = None) -> None:
        """Open the gate for `seconds` (default: the follow-up window)."""
        until = self._clock() + (self.followup_seconds if seconds is None else seconds)
        if until > self._open_until:
            if not self.is_open:
                logger.info("Gate opened (%s)", reason)
            self._open_until = until

    def extend(self) -> None:
        """Extend an open gate by the follow-up window (agent is still talking)."""
        if self.is_open:
            self._open_until = max(
                self._open_until, self._clock() + self.followup_seconds
            )

    def close(self) -> None:
        """Close the gate immediately, ignoring the follow-up window."""
        self._open_until = float("-inf")

    def hold(self, key: str) -> None:
        """Keep the gate open until `release(key)` is called."""
        self._holds.add(key)

    def release(self, key: str) -> None:
        """Release a hold and continue with a normal follow-up window."""
        if key in self._holds:
            self._holds.discard(key)
            self.open(f"release {key}")


class OutputRouter:
    """Routes model output audio to playback, a short hold, or the bin.

    GPT-Live has no control over when it speaks, so speech produced while the
    gate is closed is held for `hold_seconds` (the name often arrives in a
    transcript delta just after the model starts talking) and discarded if the
    gate does not open in time. After a barge-in, output is suppressed until
    the current model utterance ends.

    GPT-Live streams output audio continuously (silence included), so utterance
    boundaries are detected from audio energy: `utterance_gap_seconds` of
    silent audio (or of no audio at all) ends an utterance. Silence is never
    held or counted as discarded speech.
    """

    def __init__(  # noqa: PLR0913
        self,
        gate: AddressGate,
        play: Callable[[bytes], None],
        *,
        hold_seconds: float = 1.5,
        utterance_gap_seconds: float = 0.5,
        bytes_per_second: int = 48000,
        silence_rms: float = 200.0,
        on_speech_start: Callable[[], None] = lambda: None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize the router.

        Args:
            gate: The address gate to consult.
            play: Callback that enqueues audio for playback.
            hold_seconds: Maximum time to hold speech while the gate is closed.
            utterance_gap_seconds: Silence that ends a model utterance.
            bytes_per_second: Byte rate of the PCM16 output stream.
            silence_rms: RMS level (int16 scale) below which audio is silence.
            on_speech_start: Called when an utterance's first speech is played.
            clock: Monotonic time source in seconds.
        """
        self._gate = gate
        self._play = play
        self.hold_seconds = hold_seconds
        self.utterance_gap_seconds = utterance_gap_seconds
        self._bps = bytes_per_second
        self._silence_rms = silence_rms
        self._on_speech_start = on_speech_start
        self._clock = clock
        self._held: list[bytes] = []
        self._held_since: float | None = None
        self._last_delta = float("-inf")
        self._silence_run = 0.0
        self._in_utterance = False
        self._suppressed = False
        self._discarded_in_utterance = False
        self._played_speech = False
        self.last_speech_played = float("-inf")
        self.discarded_utterances = 0

    def _is_silent(self, data: bytes) -> bool:
        samples = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2")
        if not len(samples):
            return True
        return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) < (
            self._silence_rms
        )

    def _end_utterance(self) -> None:
        self._in_utterance = False
        self._suppressed = False
        self._discarded_in_utterance = False
        self._played_speech = False

    def on_audio(self, data: bytes) -> None:
        """Handle an output audio delta from the model."""
        now = self._clock()
        if now - self._last_delta >= self.utterance_gap_seconds:
            self._end_utterance()
        self._last_delta = now

        if self._is_silent(data):
            self._silence_run += len(data) / self._bps
            if self._silence_run >= self.utterance_gap_seconds:
                self._end_utterance()
            if self._gate.is_open and not self._suppressed and self._in_utterance:
                self._play(data)  # keep pauses inside an utterance
            return

        self._silence_run = 0.0
        self._in_utterance = True
        if self._suppressed or self._discarded_in_utterance:
            return
        if self._gate.is_open:
            self._release_held()
            self._play_speech(data)
            self._gate.extend()
            return
        if self._held_since is None:
            self._held_since = now
        self._held.append(data)

    def tick(self) -> bool:
        """Release or discard held speech; call periodically and on gate changes.

        Returns:
            bool: True if held speech was discarded in this call.
        """
        if self._held_since is None:
            return False
        if self._gate.is_open:
            self._release_held()
            return False
        if self._clock() - self._held_since >= self.hold_seconds:
            self._held.clear()
            self._held_since = None
            self._discarded_in_utterance = True
            self.discarded_utterances += 1
            logger.info("Discarded unaddressed model speech")
            return True
        return False

    def barge_in(self) -> None:
        """Drop held audio and ignore the rest of the current model utterance."""
        self._held.clear()
        self._held_since = None
        self._suppressed = True

    def _play_speech(self, data: bytes) -> None:
        if not self._played_speech:
            self._played_speech = True
            self._on_speech_start()
        self.last_speech_played = self._clock()
        self._play(data)

    def _release_held(self) -> None:
        if self._held:
            for chunk in self._held:
                self._play_speech(chunk)
            self._held.clear()
        self._held_since = None
