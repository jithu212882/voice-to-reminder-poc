import io
import wave
from typing import List, Optional, Tuple
import numpy as np
import torch
from silero_vad import load_silero_vad

_shared_vad_model = None

def get_vad_model():
    global _shared_vad_model
    if _shared_vad_model is None:
        _shared_vad_model = load_silero_vad()
    return _shared_vad_model

def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1, sampwidth: int = 2) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buffer.getvalue()

class SileroVADBuffer:
    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.50,
        pause_duration_ms: int = 1000,
        min_speech_duration_ms: int = 500,
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.window_size_samples = 512
        self.window_size_bytes = self.window_size_samples * 2

        self.pause_frames_limit = int((pause_duration_ms / 1000.0) * (sample_rate / self.window_size_samples))
        self.min_speech_frames = int((min_speech_duration_ms / 1000.0) * (sample_rate / self.window_size_samples))

        self.model = get_vad_model()
        self.raw_stream_buffer = bytearray()
        self.speech_buffer = bytearray()
        self.is_speech_active = False
        self.consecutive_silence_frames = 0
        self.speech_frames_count = 0

    def process_chunk(self, chunk: bytes) -> List[Tuple[bytes, bool]]:
        """Processes 16kHz 16-bit PCM chunk.

        Returns list of (wav_bytes, is_final):
        - (wav_bytes, True): final completed speech segment after 1000ms silence pause.
        """
        emitted_chunks: List[Tuple[bytes, bool]] = []
        if not chunk:
            return emitted_chunks

        self.raw_stream_buffer.extend(chunk)

        while len(self.raw_stream_buffer) >= self.window_size_bytes:
            window_bytes = bytes(self.raw_stream_buffer[: self.window_size_bytes])
            del self.raw_stream_buffer[: self.window_size_bytes]

            pcm_int16 = np.frombuffer(window_bytes, dtype=np.int16)
            pcm_float = pcm_int16.astype(np.float32) / 32768.0
            tensor = torch.from_numpy(pcm_float)

            with torch.no_grad():
                prob = self.model(tensor, self.sample_rate).item()

            if prob >= self.threshold:
                self.is_speech_active = True
                self.consecutive_silence_frames = 0
                self.speech_frames_count += 1
                self.speech_buffer.extend(window_bytes)
            else:
                if self.is_speech_active:
                    self.speech_buffer.extend(window_bytes)
                    self.consecutive_silence_frames += 1

                    if self.consecutive_silence_frames >= self.pause_frames_limit:
                        # 1000ms silence pause reached -> emit final completed segment
                        if self.speech_frames_count >= self.min_speech_frames:
                            final_wav = pcm_to_wav(bytes(self.speech_buffer), self.sample_rate)
                            emitted_chunks.append((final_wav, True))

                        self.speech_buffer.clear()
                        self.is_speech_active = False
                        self.consecutive_silence_frames = 0
                        self.speech_frames_count = 0

        return emitted_chunks

    def flush(self) -> Optional[bytes]:
        """Flushes buffered speech on demand (e.g. stop recording)."""
        if self.speech_frames_count >= self.min_speech_frames and len(self.speech_buffer) > 0:
            wav_bytes = pcm_to_wav(bytes(self.speech_buffer), self.sample_rate)
            self.reset()
            return wav_bytes
        self.reset()
        return None

    def reset(self) -> None:
        self.raw_stream_buffer.clear()
        self.speech_buffer.clear()
        self.is_speech_active = False
        self.consecutive_silence_frames = 0
        self.speech_frames_count = 0
