import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Self

from joinly.core import (
    STT,
    TTS,
    VAD,
    AudioReader,
    AudioWriter,
    SpeechController,
    TranscriptionController,
)
from joinly.types import SpeakerRole, Transcript, TranscriptSegment
from joinly.utils.clock import Clock
from joinly.utils.events import EventBus

logger = logging.getLogger(__name__)


class GptLiveTranscriptionController(TranscriptionController):
    """Transcription sink for GPT-Live mode.

    GPT-Live transcribes the meeting itself, so this controller runs no VAD/STT.
    It drives the meeting clock from wall time and lets the voice agent add
    segments to the shared transcript, publishing the usual events.
    """

    reader: AudioReader
    vad: VAD
    stt: STT

    def __init__(self, *, clock_interval: float = 0.05) -> None:
        """Initialize the controller.

        Args:
            clock_interval: How often the meeting clock is advanced, in seconds.
        """
        self.clock_interval = clock_interval
        self._no_speech_event = asyncio.Event()
        self._no_speech_event.set()
        self._clock: Clock | None = None
        self._transcript: Transcript | None = None
        self._event_bus: EventBus | None = None
        self._t0_ns = 0
        self._clock_task: asyncio.Task[None] | None = None

    @property
    def no_speech_event(self) -> asyncio.Event:
        """Always set: barge-in is handled by the GPT-Live agent."""
        return self._no_speech_event

    @property
    def active(self) -> bool:
        """Whether a meeting is joined."""
        return self._transcript is not None

    async def __aenter__(self) -> Self:
        """Enter the controller."""
        return self

    async def __aexit__(self, *_exc: object) -> None:
        """Stop the controller."""
        await self.stop()

    async def start(
        self, clock: Clock, transcript: Transcript, event_bus: EventBus
    ) -> None:
        """Start driving the clock and accepting segments."""
        self._clock = clock
        self._transcript = transcript
        self._event_bus = event_bus
        self._t0_ns = time.monotonic_ns() - clock.now_ns
        self._clock_task = asyncio.create_task(self._tick())

    async def stop(self) -> None:
        """Stop the controller."""
        if self._clock_task is not None:
            self._clock_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._clock_task
            self._clock_task = None
        self._clock = None
        self._transcript = None
        self._event_bus = None

    @property
    def now_s(self) -> float:
        """Seconds since the meeting was joined."""
        self._advance()
        return self._clock.now_s if self._clock is not None else 0.0

    def add_segment(self, segment: TranscriptSegment) -> None:
        """Add a segment to the transcript and notify subscribers."""
        if self._transcript is None or self._event_bus is None:
            return
        self._transcript.add_segment(segment)
        logger.info(
            "%s: %s",
            segment.speaker
            or (
                "assistant" if segment.role == SpeakerRole.assistant else "participant"
            ),
            segment.text,
        )
        self._event_bus.publish("segment")
        if segment.role == SpeakerRole.participant:
            self._event_bus.publish("utterance")

    def _advance(self) -> None:
        if self._clock is not None:
            self._clock.update(
                max(self._clock.now_ns, time.monotonic_ns() - self._t0_ns)
            )

    async def _tick(self) -> None:
        while True:
            self._advance()
            await asyncio.sleep(self.clock_interval)


class GptLiveSpeechController(SpeechController):
    """Speech controller for GPT-Live mode.

    Speech is produced by GPT-Live; `speak_text` forwards to the voice agent's
    `say` handler (set once the agent is running).
    """

    writer: AudioWriter
    tts: TTS
    no_speech_event: asyncio.Event

    def __init__(self) -> None:
        """Initialize the controller."""
        self.say: Callable[[str], Awaitable[None]] | None = None

    async def start(
        self,
        clock: Clock,
        transcript: Transcript,
        event_bus: EventBus,
    ) -> None:
        """Nothing to start; speech output is owned by the voice agent."""

    async def stop(self) -> None:
        """Nothing to stop."""

    async def speak_text(self, text: str) -> None:
        """Ask GPT-Live to say `text`."""
        if self.say is None:
            msg = "GPT-Live voice agent is not running"
            raise RuntimeError(msg)
        await self.say(text)


class TranscriptAccumulator:
    """Groups GPT-Live transcript deltas into transcript segments.

    Deltas carry `start_ms`/`end_ms` relative to the GPT-Live session start and
    no turn boundaries. A segment is closed when the gap to the next delta
    exceeds `gap_seconds`, the speaker changes, or it grows beyond `max_seconds`.
    """

    def __init__(
        self,
        sink: Callable[[TranscriptSegment], None],
        *,
        gap_seconds: float = 0.8,
        max_seconds: float = 20.0,
    ) -> None:
        """Initialize the accumulator.

        Args:
            sink: Receives completed segments.
            gap_seconds: Silence gap that ends a segment.
            max_seconds: Maximum segment duration.
        """
        self._sink = sink
        self.gap_seconds = gap_seconds
        self.max_seconds = max_seconds
        self.offset_seconds = 0.0
        self._open: dict[SpeakerRole, dict] = {}

    def add(
        self,
        role: SpeakerRole,
        delta: str,
        start_ms: float,
        end_ms: float,
        speaker: str | None,
    ) -> None:
        """Add a transcript delta."""
        start = self.offset_seconds + start_ms / 1000
        end = self.offset_seconds + end_ms / 1000
        cur = self._open.get(role)
        if cur is not None and (
            start - cur["end"] > self.gap_seconds
            or speaker != cur["speaker"]
            or end - cur["start"] > self.max_seconds
        ):
            self.flush(role)
            cur = None
        if cur is None:
            self._open[role] = {
                "text": delta,
                "start": start,
                "end": end,
                "speaker": speaker,
                "arrived": time.monotonic(),
            }
        else:
            cur["text"] += delta
            cur["end"] = max(cur["end"], end)
            cur["arrived"] = time.monotonic()

    def flush(self, role: SpeakerRole | None = None) -> None:
        """Emit open segments (all roles, or just `role`)."""
        for r in [role] if role is not None else list(self._open):
            cur = self._open.pop(r, None)
            if cur is None or not cur["text"].strip():
                continue
            self._sink(
                TranscriptSegment(
                    text=cur["text"].strip(),
                    start=cur["start"],
                    end=max(cur["end"], cur["start"]),
                    speaker=cur["speaker"],
                    role=r,
                )
            )

    def flush_stale(self, idle_seconds: float = 3.0) -> None:
        """Emit segments that received no delta for `idle_seconds` (wall time).

        Deltas arrive with some lag, so this uses a longer threshold than
        `gap_seconds` to avoid splitting segments that are still growing.
        """
        now = time.monotonic()
        for role, cur in list(self._open.items()):
            if now - cur["arrived"] > idle_seconds:
                self.flush(role)
