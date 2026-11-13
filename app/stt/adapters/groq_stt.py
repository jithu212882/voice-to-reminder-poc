import logging
from groq import AsyncGroq, APIStatusError, RateLimitError
from app.config import settings
from app.stt.base import BaseSTTEngine

logger = logging.getLogger("app.stt.groq")

WHISPER_ARTIFACT_KEYWORDS = {
    "thank you",
    "thank you.",
    "thank you!",
    "thanks for watching",
    "thanks for watching.",
    "thanks for watching!",
}

def is_hallucination_or_noise(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return True

    normalized = cleaned.lower().strip(".,!?\"' ")
    if normalized in WHISPER_ARTIFACT_KEYWORDS or normalized == "thank you" or normalized == "thanks for watching":
        return True

    return False

class GroqSTTEngine(BaseSTTEngine):
    provider_name = "groq"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        if not self.api_key:
            logger.warning("[WARNING] GROQ_API_KEY is not set. Groq STT calls will fail.")
        self.client = AsyncGroq(api_key=self.api_key) if self.api_key else None

    async def transcribe(self, audio_wav_bytes: bytes) -> str:
        if not self.client:
            raise ValueError("Groq API key not configured")

        try:
            response = await self.client.audio.transcriptions.create(
                file=("audio.wav", audio_wav_bytes),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                language="en",
                temperature=0.0,
            )

            # Extract valid segments, discarding high no_speech_prob (> 0.6)
            segments = getattr(response, "segments", None)
            if segments is not None and isinstance(segments, list) and len(segments) > 0:
                valid_segments_text = []
                for seg in segments:
                    no_speech_prob = getattr(seg, "no_speech_prob", None)
                    if no_speech_prob is None and isinstance(seg, dict):
                        no_speech_prob = seg.get("no_speech_prob")

                    seg_text = getattr(seg, "text", "")
                    if not seg_text and isinstance(seg, dict):
                        seg_text = seg.get("text", "")

                    if no_speech_prob is not None and float(no_speech_prob) > 0.6:
                        logger.info(f"Discarding segment with high no_speech_prob ({float(no_speech_prob):.2f}): '{seg_text}'")
                        continue

                    if seg_text and seg_text.strip():
                        valid_segments_text.append(seg_text.strip())

                raw_text = " ".join(valid_segments_text).strip()
            else:
                raw_text = getattr(response, "text", "").strip()

            # Simple exact-match hallucination filtering
            if is_hallucination_or_noise(raw_text):
                logger.info(f"Filtered Whisper hallucination/noise: '{raw_text}'")
                return ""

            return raw_text
        except RateLimitError as exc:
            logger.warning(f"[WARNING] Groq STT Rate Limit (HTTP 429): {exc}")
            raise
        except APIStatusError as exc:
            if exc.status_code >= 500:
                logger.warning(f"[WARNING] Groq STT Server Error (HTTP {exc.status_code}): {exc}")
            raise
