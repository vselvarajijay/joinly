import asyncio
import base64
import contextlib
import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from joinly.types import (
    AudioChunk,
    AudioFormat,
    MeetingChatHistory,
    MeetingChatMessage,
    MeetingParticipant,
    SpeakerRole,
    Transcript,
    VideoSnapshot,
)
from joinly.utils.clock import Clock
from joinly.utils.events import EventBus
from joinly.voice.gpt_live.config import GptLiveConfig
from joinly.voice.gpt_live.controllers import (
    GptLiveSpeechController,
    GptLiveTranscriptionController,
)
from joinly.voice.gpt_live.loop import GptLiveVoiceAgent
from tests.utils.fake_gpt_live import FakeGptLiveServer

if TYPE_CHECKING:
    from joinly.session import MeetingSession

IN_RATE = 48000
OUT_RATE = 24000
# 4 s of agent audio is queued; playback must stop soon after ~0.3 s of speech
BARGE_IN_MAX_PLAYED_S = 1.5


def _voiced(seconds: float, rate: int = IN_RATE) -> bytes:
    """Harmonic, amplitude-modulated signal that webrtcvad detects as speech."""
    t = np.arange(int(rate * seconds)) / rate
    x = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 12))
    x = x * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
    return (x / np.abs(x).max() * 0.6 * 32767).astype("<i2").tobytes()


class FakeReader:
    """Meeting audio at 48 kHz PCM16, 20 ms chunks, paced in real time."""

    audio_format = AudioFormat(sample_rate=IN_RATE, byte_depth=2)

    def __init__(self) -> None:
        """Initialize the fake."""
        self._chunk = IN_RATE // 50 * 2
        self._pending = bytearray()
        self.speaker: str | None = None
        self._t = 0

    def say(self, pcm: bytes) -> None:
        """Queue audio to be spoken into the meeting."""
        self._pending.extend(pcm)

    async def read(self) -> AudioChunk:
        """Return the next 20 ms chunk."""
        await asyncio.sleep(0.02)
        data = bytes(self._pending[: self._chunk])
        del self._pending[: self._chunk]
        data += b"\x00" * (self._chunk - len(data))
        self._t += 20_000_000
        return AudioChunk(data=data, time_ns=self._t, speaker=self.speaker)


class FakeWriter:
    """Virtual mic at 24 kHz PCM16, paced in real time."""

    audio_format = AudioFormat(sample_rate=OUT_RATE, byte_depth=2)
    chunk_size = OUT_RATE // 100 * 2

    def __init__(self) -> None:
        """Initialize the fake."""
        self.written = bytearray()

    async def write(self, data: bytes) -> None:
        """Record audio and wait for its real-time duration."""
        # like VirtualMicrophone: a partial trailing block is zero-padded
        pad = -len(data) % self.chunk_size
        self.written.extend(data + b"\x00" * pad)
        await asyncio.sleep(len(data) / (OUT_RATE * 2))

    @property
    def seconds(self) -> float:
        """Seconds of audio written."""
        return len(self.written) / (OUT_RATE * 2)


class FakeProvider:
    """Meeting provider with scriptable chat."""

    def __init__(self) -> None:
        """Initialize the fake."""
        self.audio_reader = FakeReader()
        self.audio_writer = FakeWriter()
        self.chat: list[MeetingChatMessage] = []
        self.sent: list[str] = []

    async def get_chat_history(self) -> MeetingChatHistory:
        """Return the fake chat history."""
        return MeetingChatHistory(messages=list(self.chat))

    async def send_chat_message(self, message: str) -> None:
        """Record a chat message."""
        self.sent.append(message)
        self.chat.append(MeetingChatMessage(text=message, sender=None, timestamp="me"))

    async def get_participants(self) -> list[MeetingParticipant]:
        """Return fake participants."""
        return [MeetingParticipant(name="Priya")]


class FakeMeeting:
    """Joined meeting session for the agent."""

    def __init__(self) -> None:
        """Initialize the fake."""
        self.meeting_provider = FakeProvider()
        self.transcription_controller = GptLiveTranscriptionController()
        self.speech_controller = GptLiveSpeechController()
        self._transcript = Transcript()

    async def start(self) -> None:
        """Start the transcription controller."""
        await self.transcription_controller.start(Clock(), self._transcript, EventBus())

    @property
    def transcript(self) -> Transcript:
        """The meeting transcript."""
        return self._transcript

    @property
    def meeting_seconds(self) -> float:
        """Seconds since join."""
        return self.transcription_controller.now_s

    async def get_chat_history(self) -> MeetingChatHistory:
        """Return the fake chat history."""
        return await self.meeting_provider.get_chat_history()

    async def send_chat_message(self, message: str) -> None:
        """Record a chat message."""
        await self.meeting_provider.send_chat_message(message)

    async def get_participants(self) -> list[MeetingParticipant]:
        """Return fake participants."""
        return await self.meeting_provider.get_participants()

    async def get_video_snapshot(self) -> VideoSnapshot:
        """Return a fake snapshot."""
        return VideoSnapshot(data=b"img", media_type="image/jpeg")


def _config(url: str, **overrides: Any) -> GptLiveConfig:  # noqa: ANN401
    values: dict[str, Any] = {
        "openai_api_key": "sk-test",
        "gpt_live_url": url,
        "gpt_live_hold_seconds": 0.3,
        "gpt_live_chat_poll_seconds": 0.1,
        "gpt_live_disclosure": False,
        "gpt_live_followup_seconds": 5.0,
    } | overrides
    return GptLiveConfig(**values)


@contextlib.asynccontextmanager
async def running_agent(
    **overrides: Any,  # noqa: ANN401
) -> AsyncIterator[tuple[GptLiveVoiceAgent, FakeMeeting, FakeGptLiveServer]]:
    """Run a GptLiveVoiceAgent against a fake meeting and fake GPT-Live."""
    async with FakeGptLiveServer() as server:
        meeting = FakeMeeting()
        await meeting.start()
        agent = GptLiveVoiceAgent(
            cast("MeetingSession", meeting),
            _config(server.url, **overrides),
            name="Helios",
            language="en",
            meeting_url="https://teams.microsoft.com/l/meetup-join/x",
        )
        task = asyncio.create_task(agent.run())
        await server.wait_type("session.start")
        await asyncio.sleep(0.05)
        try:
            yield agent, meeting, server
        finally:
            await agent.stop()
            await asyncio.wait_for(task, 5)
            await meeting.transcription_controller.stop()


def _audio_delta(seconds: float) -> dict[str, Any]:
    samples = (np.sin(np.arange(int(OUT_RATE * seconds)) / 5) * 8000).astype("<i2")
    return {
        "type": "session.output_audio.delta",
        "delta": base64.b64encode(samples.tobytes()).decode(),
    }


def _input_text(text: str, start_ms: int, end_ms: int) -> dict[str, Any]:
    return {
        "type": "session.input_transcript.delta",
        "delta": text,
        "start_ms": start_ms,
        "end_ms": end_ms,
    }


async def test_session_config() -> None:
    """session.start uses gpt-live-1 + Responses delegation to gpt-5.6-terra."""
    async with running_agent() as (_agent, _meeting, server):
        session = server.of_type("session.start")[0]["session"]
    assert session["model"] == "gpt-live-1"
    assert session["audio"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert "Helios" in session["instructions"]
    responses = session["delegation"]["responses"]
    assert session["delegation"]["type"] == "responses"
    assert responses["model"] == "gpt-5.6-terra"
    assert responses["reasoning"] == {"effort": "low"}
    assert {t["name"] for t in responses["tools"]} == {
        "get_transcript",
        "get_chat_history",
        "send_chat_message",
        "get_participants",
        "get_video_snapshot",
    }
    assert server.auth_headers[0] == "Bearer sk-test"
    assert server.of_type("session.close")


async def test_streams_meeting_audio_at_session_rate() -> None:
    """Meeting audio is resampled 48k -> 24k and sent as input_audio.append."""
    async with running_agent() as (_agent, _meeting, server):
        await asyncio.sleep(0.5)
        appends = server.of_type("session.input_audio.append")
    total = sum(len(base64.b64decode(e["audio"])) for e in appends)
    assert appends
    assert all(len(base64.b64decode(e["audio"])) % 2 == 0 for e in appends)
    # ~0.5 s at 24 kHz PCM16, generous bounds for scheduling jitter
    assert 0.25 * OUT_RATE * 2 < total < 0.8 * OUT_RATE * 2


async def test_unaddressed_speech_is_not_played() -> None:
    """Model audio without the name being said is held, then discarded."""
    async with running_agent() as (_agent, meeting, server):
        await server.send(_audio_delta(0.5))
        await asyncio.sleep(0.8)
        writer = meeting.meeting_provider.audio_writer
        assert writer.seconds == 0
        note = await server.wait_for(
            lambda e: e["type"] == "session.thinking.append"
            and "not played" in e["content"]
        )
        assert "Helios" in note["content"]


async def test_addressed_speech_is_played_and_transcribed() -> None:
    """Saying the name opens the gate (even if the transcript arrives late)."""
    async with running_agent() as (_agent, meeting, server):
        await server.send(_audio_delta(0.3))  # model starts before name transcript
        await server.send(_input_text("Hey Heli", 1000, 1300))
        await server.send(_input_text("os, what did Priya say?", 1300, 2400))
        await asyncio.sleep(0.8)
        writer = meeting.meeting_provider.audio_writer
        assert writer.seconds == pytest.approx(0.3, abs=0.05)

        await server.send(
            {
                "type": "session.output_transcript.delta",
                "delta": "Priya said to ship Friday.",
                "start_ms": 3000,
                "end_ms": 4500,
            }
        )
        await asyncio.sleep(3.5)  # stale flush of open segments
        segments = meeting.transcript.segments
        participant = [s for s in segments if s.role == SpeakerRole.participant]
        assistant = [s for s in segments if s.role == SpeakerRole.assistant]
        assert participant[0].text == "Hey Helios, what did Priya say?"
        assert assistant[0].text == "Priya said to ship Friday."
        assert assistant[0].speaker == "Helios"


async def test_barge_in_stops_playback() -> None:
    """Human speech while the agent talks flushes queued agent audio."""
    async with running_agent(gpt_live_bargein_ms=300) as (_agent, meeting, server):
        await server.send(_input_text("Helios, tell me a long story", 0, 1500))
        await asyncio.sleep(0.1)
        await server.send(_audio_delta(4.0))
        await asyncio.sleep(0.5)
        meeting.meeting_provider.audio_reader.say(_voiced(1.0))
        await asyncio.sleep(1.5)
        # playback stopped shortly after the ~300 ms barge-in threshold
        assert meeting.meeting_provider.audio_writer.seconds < BARGE_IN_MAX_PLAYED_S
        # rest of the interrupted utterance is dropped
        await server.send(_audio_delta(0.3))
        await asyncio.sleep(0.4)
        assert meeting.meeting_provider.audio_writer.seconds < BARGE_IN_MAX_PLAYED_S


async def test_chat_mention_triggers_backend_request() -> None:
    """A new @helios chat message is queued for the backend and continued."""
    async with running_agent() as (_agent, meeting, server):
        provider = meeting.meeting_provider
        provider.chat.append(
            MeetingChatMessage(text="old @helios", sender="Sam", timestamp="1")
        )
        await asyncio.sleep(0.3)  # first poll only marks history as seen
        provider.chat.append(
            MeetingChatMessage(text="no mention here", sender="Sam", timestamp="2")
        )
        provider.chat.append(
            MeetingChatMessage(
                text="@Helios summarize please", sender="Priya", timestamp="3"
            )
        )
        item = await server.wait_for(
            lambda e: e["type"] == "response.item.create"
            and e["item"].get("type") == "message"
        )
        await server.wait_type("response.create")
        text = item["item"]["content"][0]["text"]
        assert "Priya" in text
        assert "@Helios summarize please" in text
        creates = server.of_type("response.item.create")
        assert len(creates) == 1


async def test_disclosure_in_chat_and_voice() -> None:
    """On join the agent posts a chat disclosure and speaks one."""
    async with running_agent(gpt_live_disclosure=True) as (_agent, meeting, server):
        instr = await server.wait_type("session.instructions.append")
        assert "AI assistant" in instr["content"]
        assert "exactly" in instr["content"]
        provider = meeting.meeting_provider
        assert "@helios" in provider.sent[0]
        # the gate is open for the disclosure itself
        await server.send(_audio_delta(0.2))
        await asyncio.sleep(0.5)
        assert provider.audio_writer.seconds == pytest.approx(0.2, abs=0.05)
        # own disclosure message must not trigger a chat request
        await asyncio.sleep(0.3)
        assert not server.of_type("response.create")


async def test_delegated_tool_call_reads_live_transcript() -> None:
    """A backend get_transcript call returns what GPT-Live transcribed."""
    async with running_agent() as (_agent, _meeting, server):
        await server.send(_input_text("We decided to ship on Friday.", 0, 1500))
        await server.send(
            {
                "type": "session.delegation.created",
                "delegation": {"id": "item_1", "target": "responses"},
            }
        )
        for event in (
            {"type": "response.created", "response": {"id": "r1"}},
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_transcript",
                    "arguments": json.dumps({"mode": "full", "minutes": 0}),
                },
            },
            {"type": "response.completed"},
        ):
            await server.send(
                {"type": "response.event", "delegation_id": "item_1", "event": event}
            )
        output = await server.wait_type("response.item.create")
        await server.wait_type("response.create")
        segments = json.loads(output["item"]["output"])["segments"]
        assert segments[0]["text"] == "We decided to ship on Friday."


async def test_speak_text_uses_instructions() -> None:
    """The MCP speak_text path asks GPT-Live to say the text verbatim."""
    async with running_agent() as (_agent, meeting, server):
        await meeting.speech_controller.speak_text("The meeting ends in five minutes.")
        instr = server.of_type("session.instructions.append")[-1]
        assert "The meeting ends in five minutes." in instr["content"]


async def test_reconnects_with_seeded_transcript() -> None:
    """After a dropped connection a new session is seeded with the transcript."""
    async with running_agent() as (_agent, _meeting, server):
        await server.send(_input_text("Budget is approved.", 0, 1500))
        await asyncio.sleep(0.1)
        await server.drop()
        await server.wait_for(
            lambda e: e["type"] == "session.start" and "input" in e["session"],
            timeout=5,
        )
        assert server.connections == 2  # noqa: PLR2004
        start = server.of_type("session.start")[-1]["session"]
        seed = start["input"][0]["content"][0]["text"]
        assert "Budget is approved." in seed
        assert "do not do it again" in seed


async def test_unaddressed_delegation_stays_silent() -> None:
    """Backend work the model starts on its own does not open the gate."""
    async with running_agent() as (_agent, meeting, server):
        await server.send(_input_text("Priya, can you check the budget?", 0, 1500))
        await server.send(
            {
                "type": "session.delegation.created",
                "delegation": {"id": "item_u", "target": "responses"},
            }
        )
        await server.send(
            {
                "type": "response.event",
                "delegation_id": "item_u",
                "event": {"type": "response.created", "response": {"id": "r"}},
            }
        )
        await server.send(_audio_delta(0.3))
        await asyncio.sleep(0.8)
        assert meeting.meeting_provider.audio_writer.seconds == 0


async def test_addressed_delegation_keeps_gate_open() -> None:
    """Requested backend work keeps the answer audible past the follow-up window."""
    async with running_agent(gpt_live_followup_seconds=0.5) as (_a, meeting, server):
        await server.send(_input_text("Helios, summarize the meeting", 0, 1500))
        await asyncio.sleep(0.1)
        await server.send(
            {
                "type": "session.delegation.created",
                "delegation": {"id": "item_a", "target": "responses"},
            }
        )
        await asyncio.sleep(1.0)  # longer than the follow-up window
        await server.send(_audio_delta(0.3))
        await asyncio.sleep(0.6)
        assert meeting.meeting_provider.audio_writer.seconds == pytest.approx(
            0.3, abs=0.05
        )


async def test_playback_has_no_spliced_silence() -> None:
    """Odd-sized deltas are played contiguously (no zero-padding mid-speech)."""
    async with running_agent() as (_agent, meeting, server):
        await server.send(_input_text("Helios, say something", 0, 1000))
        await asyncio.sleep(0.1)
        rng = np.random.default_rng(0)
        sent = bytearray()
        # stream ~2 s just ahead of real time with network-like jitter, in
        # odd-sized pieces that don't align with the mic's 10 ms blocks
        for _ in range(50):
            n = int(rng.integers(801, 1101)) | 1
            pcm = (rng.integers(1000, 8000, n) * rng.choice([-1, 1], n)).astype("<i2")
            sent.extend(pcm.tobytes())
            await server.send(
                {
                    "type": "session.output_audio.delta",
                    "delta": base64.b64encode(pcm.tobytes()).decode(),
                }
            )
            await asyncio.sleep(n / OUT_RATE * 0.8 + float(rng.uniform(0, 0.012)))
        await asyncio.sleep(1.0)
        written = bytes(meeting.meeting_provider.audio_writer.written)
        # everything sent is played in order and contiguously; only the very end
        # may be padded to a whole block
        assert written[: len(sent)] == bytes(sent)
        assert set(written[len(sent) :]) <= {0}
