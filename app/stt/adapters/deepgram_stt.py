import logging
import httpx
from app.config import settings
from app.stt.base import BaseSTTEngine

logger = logging.getLogger("app.stt.deepgram")

class DeepgramSTTEngine(BaseSTTEngine):
    provider_name = "deepgram"
    API_URL = "https://api.deepgram.com/v1/listen"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.DEEPGRAM_API_KEY
        if not self.api_key:
            logger.warning("[WARNING] DEEPGRAM_API_KEY is not set. Deepgram STT calls will fail.")

    async def transcribe(self, audio_wav_bytes: bytes) -> str:
        if not self.api_key:
            raise ValueError("Deepgram API key not configured")

        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/wav",
        }
        params = {
            "model": "nova-3",
            "language": "en-IN",
            "smart_format": "true",
            "punctuate": "true",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.API_URL,
                params=params,
                headers=headers,
                content=audio_wav_bytes,
            )

            if response.status_code == 429:
                logger.warning(f"[WARNING] Deepgram STT Rate Limit (HTTP 429): {response.text}")
                response.raise_for_status()

            if response.status_code >= 500:
                logger.warning(f"[WARNING] Deepgram STT Server Error (HTTP {response.status_code}): {response.text}")
                response.raise_for_status()

            response.raise_for_status()
            res_json = response.json()

            results = res_json.get("results", {})
            channels = results.get("channels", [])
            if not channels:
                return ""
            alternatives = channels[0].get("alternatives", [])
            if not alternatives:
                return ""

            best_alt = alternatives[0]
            transcript = best_alt.get("transcript", "").strip()
            confidence = best_alt.get("confidence")

            if confidence is not None and float(confidence) < 0.35:
                logger.info(
                    f"Deepgram transcript discarded due to low confidence ({float(confidence):.2f}): '{transcript}'"
                )
                return ""

            return transcript

