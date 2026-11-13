import json
import logging
from typing import Dict, List, Optional
from app.config import settings
from app.stt.base import BaseSTTEngine
from app.stt.adapters.groq_stt import GroqSTTEngine
from app.stt.adapters.sarvam_stt import SarvamSTTEngine
from app.stt.adapters.local_whisper import LocalWhisperSTTEngine
from app.stt.adapters.deepgram_stt import DeepgramSTTEngine

logger = logging.getLogger("app.stt.factory")

class STTFactory:
    _engines: Dict[str, BaseSTTEngine] = {}

    @classmethod
    def get_engine(cls, provider: Optional[str] = None) -> BaseSTTEngine:
        provider_name = (provider or settings.STT_PROVIDER).lower()

        if provider_name not in cls._engines:
            if provider_name == "groq":
                cls._engines[provider_name] = GroqSTTEngine()
            elif provider_name == "sarvam":
                cls._engines[provider_name] = SarvamSTTEngine()
            elif provider_name == "local_whisper":
                cls._engines[provider_name] = LocalWhisperSTTEngine()
            elif provider_name == "deepgram":
                cls._engines[provider_name] = DeepgramSTTEngine()
            else:
                raise ValueError(f"Unknown STT provider: {provider_name}")

        return cls._engines[provider_name]

    @classmethod
    def get_failover_candidates(cls, primary: str) -> List[str]:
        candidates: List[str] = []
        if primary == "deepgram":
            if settings.GROQ_API_KEY:
                candidates.append("groq")
            if settings.SARVAM_API_KEY:
                candidates.append("sarvam")
            candidates.append("local_whisper")
        elif primary == "groq":
            if settings.DEEPGRAM_API_KEY:
                candidates.append("deepgram")
            if settings.SARVAM_API_KEY:
                candidates.append("sarvam")
            candidates.append("local_whisper")
        elif primary == "sarvam":
            if settings.DEEPGRAM_API_KEY:
                candidates.append("deepgram")
            if settings.GROQ_API_KEY:
                candidates.append("groq")
            candidates.append("local_whisper")
        elif primary == "local_whisper":
            if settings.DEEPGRAM_API_KEY:
                candidates.append("deepgram")
            if settings.GROQ_API_KEY:
                candidates.append("groq")
            if settings.SARVAM_API_KEY:
                candidates.append("sarvam")
        else:
            candidates = ["local_whisper"]
        return candidates

    @classmethod
    async def transcribe_with_failover(cls, audio_wav_bytes: bytes) -> str:
        primary_provider = settings.STT_PROVIDER
        providers_to_try = [primary_provider]

        if settings.ENABLE_FAILOVER:
            providers_to_try.extend(cls.get_failover_candidates(primary_provider))

        last_error = None
        for idx, provider_name in enumerate(providers_to_try):
            try:
                engine = cls.get_engine(provider_name)
                text = await engine.transcribe(audio_wav_bytes)
                if idx > 0:
                    logger.info(
                        json.dumps({
                            "event": "stt_failover_success",
                            "primary_provider": primary_provider,
                            "active_provider": provider_name,
                        })
                    )
                return text
            except Exception as exc:
                last_error = exc
                if idx < len(providers_to_try) - 1 and settings.ENABLE_FAILOVER:
                    next_provider = providers_to_try[idx + 1]
                    logger.warning(
                        f"[WARNING] STT provider '{provider_name}' failed with: {exc}. "
                        f"Failing over to '{next_provider}'."
                    )
                else:
                    logger.error(
                        f"All STT providers failed or failover disabled. Last error: {exc}"
                    )

        raise last_error or RuntimeError("Transcription failed on all providers")

