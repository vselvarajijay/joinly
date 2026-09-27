import asyncio
import math
import time
from collections import deque

import numpy as np

_INT16_MAX = 32767
_INT16_MIN = -32768


class Pcm16Carry:
    """Keeps PCM16 streams aligned to whole samples across arbitrary chunks."""

    def __init__(self) -> None:
        """Initialize with an empty carry-over byte."""
        self._pending = b""

    def push(self, data: bytes) -> bytes:
        """Return the longest prefix of complete 16-bit samples.

        Args:
            data: The next bytes of the stream.

        Returns:
            bytes: Complete samples; a trailing odd byte is kept for the next call.
        """
        data = self._pending + data
        complete = len(data) - len(data) % 2
        self._pending = data[complete:]
        return data[:complete]


class StreamResampler:
    """Stateful polyphase resampler for mono PCM16 streams.

    Uses a Kaiser-windowed sinc low-pass filter. Filter state is kept across calls,
    so resampling a stream in chunks produces the same output as resampling it in
    one call. The output is aligned to the input (group delay compensated), with
    the last few output samples of a stream held back until more input arrives.
    """

    def __init__(
        self,
        src_rate: int,
        dst_rate: int,
        *,
        half_taps: int = 16,
        beta: float = 8.0,
        rolloff: float = 0.9,
    ) -> None:
        """Initialize the resampler.

        Args:
            src_rate: Input sample rate in Hz.
            dst_rate: Output sample rate in Hz.
            half_taps: Filter half-length in taps per phase (quality vs. cost).
            beta: Kaiser window beta (stopband attenuation).
            rolloff: Cutoff as a fraction of the lower Nyquist frequency.
        """
        if src_rate <= 0 or dst_rate <= 0:
            msg = "Sample rates must be positive."
            raise ValueError(msg)
        g = math.gcd(src_rate, dst_rate)
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self._up = dst_rate // g
        self._down = src_rate // g
        self._carry = Pcm16Carry()
        self._passthrough = self._up == self._down == 1
        if self._passthrough:
            return

        factor = max(self._up, self._down)
        n_taps = 2 * half_taps * factor + 1
        cutoff = rolloff * 0.5 / factor  # cycles per upsampled sample
        n = np.arange(n_taps) - (n_taps - 1) / 2
        h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.kaiser(n_taps, beta)
        h *= self._up / h.sum()  # unity DC gain after zero-stuffing

        self._taps_per_phase = math.ceil(n_taps / self._up)
        padded = np.zeros(self._taps_per_phase * self._up)
        padded[:n_taps] = h
        # row p holds the taps h[p], h[p + up], h[p + 2 * up], ...
        self._phases = padded.reshape(self._taps_per_phase, self._up).T.copy()
        self._delay = (n_taps - 1) // 2
        history = self._taps_per_phase - 1
        self._buf = np.zeros(history, dtype=np.float64)
        self._buf_start = -history  # absolute input index of _buf[0]
        self._next_out = 0  # absolute output index

    def process(self, data: bytes) -> bytes:
        """Resample the next chunk of PCM16 audio.

        Args:
            data: Mono signed 16-bit little-endian PCM at the source rate.

        Returns:
            bytes: Mono PCM16 at the destination rate (possibly empty).
        """
        data = self._carry.push(data)
        if self._passthrough or not data:
            return data

        x = np.frombuffer(data, dtype="<i2").astype(np.float64)
        self._buf = np.concatenate([self._buf, x])
        last = self._buf_start + len(self._buf) - 1

        # output k uses upsampled index n = k*down + delay, newest input n // up
        k_max = ((last + 1) * self._up - 1 - self._delay) // self._down
        if k_max < self._next_out:
            return b""
        k = np.arange(self._next_out, k_max + 1)
        n = k * self._down + self._delay
        newest = n // self._up
        phase = n - newest * self._up
        idx = (newest - self._buf_start)[:, None] - np.arange(self._taps_per_phase)
        y = np.einsum("km,km->k", self._phases[phase], self._buf[idx])
        self._next_out = k_max + 1

        oldest_needed = (
            (self._next_out * self._down + self._delay) // self._up
            - self._taps_per_phase
            + 1
        )
        drop = oldest_needed - self._buf_start
        if drop > 0:
            self._buf = self._buf[drop:]
            self._buf_start += drop

        return np.clip(np.rint(y), _INT16_MIN, _INT16_MAX).astype("<i2").tobytes()


class PlaybackBuffer:
    """Unbounded, flushable FIFO of PCM bytes between the network and the mic."""

    def __init__(self) -> None:
        """Initialize an empty buffer."""
        self._chunks: deque[bytes] = deque()
        self._size = 0
        self._nonempty = asyncio.Event()
        self.last_pop: float = 0.0

    def __len__(self) -> int:
        """Return the number of buffered bytes."""
        return self._size

    def push(self, data: bytes) -> None:
        """Append audio to the buffer."""
        if not data:
            return
        self._chunks.append(data)
        self._size += len(data)
        self._nonempty.set()

    def pop(self, max_bytes: int) -> bytes:
        """Remove and return up to `max_bytes` from the front of the buffer."""
        out = bytearray()
        while self._chunks and len(out) < max_bytes:
            head = self._chunks.popleft()
            take = max_bytes - len(out)
            out.extend(head[:take])
            if len(head) > take:
                self._chunks.appendleft(head[take:])
        self._size -= len(out)
        if not self._chunks:
            self._nonempty.clear()
        if out:
            self.last_pop = time.monotonic()
        return bytes(out)

    def clear(self) -> int:
        """Drop all buffered audio.

        Returns:
            int: The number of bytes dropped.
        """
        dropped = self._size
        self._chunks.clear()
        self._size = 0
        self._nonempty.clear()
        return dropped

    async def wait(self) -> None:
        """Wait until the buffer holds audio."""
        await self._nonempty.wait()
