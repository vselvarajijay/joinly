import logging

import webrtcvad

logger = logging.getLogger(__name__)

_SUPPORTED_RATES = (8000, 16000, 32000, 48000)


class SpeechActivity:
    """Frame-level webrtcvad over a PCM16 stream with speech on/off debouncing.

    Emits a `started` signal after `min_speech_ms` of continuous speech (used for
    barge-in) and a `stopped` signal after `min_silence_ms` of silence (used to
    measure response latency from the end of human speech).
    """

    def __init__(
        self,
        *,
        sample_rate: int = 48000,
        frame_ms: int = 30,
        aggressiveness: int = 3,
        min_speech_ms: int = 300,
        min_silence_ms: int = 300,
    ) -> None:
        """Initialize the detector.

        Args:
            sample_rate: Sample rate of the PCM16 input (8/16/32/48 kHz).
            frame_ms: VAD frame length (10, 20 or 30 ms).
            aggressiveness: webrtcvad aggressiveness (0-3).
            min_speech_ms: Continuous speech required before `started` fires.
            min_silence_ms: Continuous silence required before `stopped` fires.
        """
        if sample_rate not in _SUPPORTED_RATES:
            msg = f"webrtcvad does not support {sample_rate} Hz"
            raise ValueError(msg)
        if frame_ms not in (10, 20, 30):
            msg = f"webrtcvad does not support {frame_ms} ms frames"
            raise ValueError(msg)
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self._frame_bytes = sample_rate * frame_ms // 1000 * 2
        self._vad = webrtcvad.Vad(aggressiveness)
        self._speech_frames_needed = max(1, min_speech_ms // frame_ms)
        self._silence_frames_needed = max(1, min_silence_ms // frame_ms)
        self._buf = bytearray()
        self._speech_run = 0
        self._silence_run = 0
        self.speaking = False

    def feed(self, data: bytes) -> tuple[bool, bool]:
        """Process audio and report state transitions.

        Args:
            data: Mono PCM16 audio at `sample_rate`.

        Returns:
            tuple[bool, bool]: (speech started, speech stopped) during this chunk.
        """
        self._buf.extend(data)
        started = stopped = False
        while len(self._buf) >= self._frame_bytes:
            frame = bytes(self._buf[: self._frame_bytes])
            del self._buf[: self._frame_bytes]
            if self._vad.is_speech(frame, self.sample_rate):
                self._speech_run += 1
                self._silence_run = 0
                if not self.speaking and self._speech_run >= self._speech_frames_needed:
                    self.speaking = True
                    started = True
            else:
                self._silence_run += 1
                self._speech_run = 0
                if self.speaking and self._silence_run >= self._silence_frames_needed:
                    self.speaking = False
                    stopped = True
        return started, stopped
