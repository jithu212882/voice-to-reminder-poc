from abc import ABC, abstractmethod

class BaseSTTEngine(ABC):
    provider_name: str

    @abstractmethod
    async def transcribe(self, audio_wav_bytes: bytes) -> str:
        """Transcribe in-memory WAV audio bytes to text string."""
        pass

