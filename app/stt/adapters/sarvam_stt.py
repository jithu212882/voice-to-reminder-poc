import logging
import re
import httpx
from app.config import settings
from app.stt.base import BaseSTTEngine

logger = logging.getLogger("app.stt.sarvam")

def clean_sarvam_output(raw_text: str) -> str:
    if not raw_text:
        return ""

    # Strip repeated punctuation and artifacts (e.g. "...", "???", ",,,", "---")
    cleaned = re.sub(r'([.,!?;:\-_~`@#$%^&*()+=|\\/])\1+', r'\1', raw_text).strip()

    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Ignore empty or <3 alphanumeric character noise outputs
    alphanumeric_content = re.sub(r'[\W_]+', '', cleaned)
    if len(alphanumeric_content) < 3:
        return ""

    # Single-word output under 3 characters (e.g., "ah", "um", "ok")
    words = cleaned.split()
    if len(words) == 1 and len(alphanumeric_content) < 3:
        return ""

    return cleaned

class SarvamSTTEngine(BaseSTTEngine):
    provider_name = "sarvam"
    API_URL = "https://api.sarvam.ai/speech-to-text"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.SARVAM_API_KEY
        if not self.api_key:
            logger.warning("[WARNING] SARVAM_API_KEY is not set. Sarvam STT calls will fail.")

    async def transcribe(self, audio_wav_bytes: bytes) -> str:
        if not self.api_key:
            raise ValueError("Sarvam API key not configured")

        headers = {
            "api-subscription-key": self.api_key,
        }
        files = {
            "file": ("audio.wav", audio_wav_bytes, "audio/wav")
        }
        data = {
            "model": "saaras:v4",
            "mode": "codemix",
            "language_code": "en-IN",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.API_URL, headers=headers, files=files, data=data)

            if response.status_code == 429:
                logger.warning(f"[WARNING] Sarvam STT Rate Limit (HTTP 429): {response.text}")
                response.raise_for_status()

            if response.status_code >= 500:
                logger.warning(f"[WARNING] Sarvam STT Server Error (HTTP {response.status_code}): {response.text}")
                response.raise_for_status()

            response.raise_for_status()
            res_json = response.json()
            raw_transcript = res_json.get("transcript", "").strip()
            return clean_sarvam_output(raw_transcript)
