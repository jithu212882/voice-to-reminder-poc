import asyncio
import io
import logging
import wave
import numpy as np
from app.config import settings
from app.stt.base import BaseSTTEngine

logger = logging.getLogger("app.stt.local_whisper")

WHISPER_ARTIFACTS = {
    "thank you",
    "thank you.",
    "thank you!",
    "thanks for watching",
    "thanks for watching.",
    "thanks for watching!",
}

class LocalWhisperSTTEngine(BaseSTTEngine):
    provider_name = "local_whisper"

    def __init__(self, model_size: str = None):
        self.model_size = model_size or settings.LOCAL_WHISPER_SIZE
        self._model = None

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            logger.info(f"Loading local faster-whisper model ({self.model_size})...")
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self._model

    def _sync_transcribe(self, audio_wav_bytes: bytes) -> str:
        model = self._get_model()
        with wave.open(io.BytesIO(audio_wav_bytes), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            pcm16 = np.frombuffer(frames, dtype=np.int16)
            audio_f32 = pcm16.astype(np.float32) / 32768.0

        if len(audio_f32) == 0:
            return ""

        segments, _ = model.transcribe(
            audio_f32,
            beam_size=5,
            language="en",
            temperature=0.0,
            vad_filter=True,
        )
        text = " ".join([segment.text.strip() for segment in segments if segment.text])
        cleaned = text.strip()

        # Simple exact-match artifact filter
        normalized = cleaned.lower().strip(".,!?\"' ")
        if normalized in WHISPER_ARTIFACTS or normalized == "thank you" or normalized == "thanks for watching":
            return ""

        return cleaned

    async def transcribe(self, audio_wav_bytes: bytes) -> str:
        try:
            return await asyncio.to_thread(self._sync_transcribe, audio_wav_bytes)
        except Exception as exc:
            logger.error(f"Local Whisper transcription failed: {exc}")
            raise
