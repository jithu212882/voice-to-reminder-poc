import json
import logging
from datetime import datetime
from google import genai
from google.genai import types
from google.genai.errors import APIError
from app.config import settings
from app.parser.base import BaseReminderParser, ReminderSchema

logger = logging.getLogger("app.parser.gemini")

SYSTEM_INSTRUCTION = """You are an expert structured data extractor for voice reminders.
Analyze the voice transcript and reference date to extract:
- title: Clean, concise reminder title (strip dictation commands).
- category: one of ["Birthday", "Anniversary", "To-Do", "Meeting", "Custom"].
- date: accurate target date YYYY-MM-DD resolved against the provided reference datetime.
- time: 24-hour target time HH:MM (default 09:00 if not stated).
- repeat: one of ["None", "Yearly", "Monthly", "Weekly"]. If category is Birthday or Anniversary, repeat should be Yearly.
- advance_alert: advance alert text (e.g., "Morning of event", "1 day before", "1 hour before").

The transcript may contain live spoken corrections where the user fixes an earlier STT misrecognition or changes details mid-session (e.g., 'not RMS, it's Ram's', 'change time to 6 PM', 'make it next Friday instead'). Always apply later corrections to overwrite earlier values in the extracted ReminderSchema fields.
"""

class GeminiReminderParser(BaseReminderParser):
    provider_name = "gemini"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        if not self.api_key:
            logger.warning("[WARNING] GEMINI_API_KEY is not set. Gemini parser calls will fail.")
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

    async def parse(self, transcript: str, current_datetime: datetime) -> ReminderSchema:
        if not self.client:
            raise ValueError("Gemini API key not configured")

        prompt = (
            f"Reference Current DateTime: {current_datetime.strftime('%Y-%m-%d %H:%M:%S (%A)')}\n"
            f"Voice Transcript: \"{transcript}\"\n\n"
            f"Extract the reminder details following the schema:"
        )

        models_to_try = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
        last_err = None

        for model_name in models_to_try:
            try:
                response = await self.client.aio.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        response_mime_type="application/json",
                        response_schema=ReminderSchema,
                        temperature=0.1,
                    ),
                )
                raw_text = response.text or "{}"
                parsed_data = json.loads(raw_text)

                if parsed_data.get("category") in ["Birthday", "Anniversary"] and parsed_data.get("repeat") in [None, "None"]:
                    parsed_data["repeat"] = "Yearly"

                return ReminderSchema(**parsed_data)
            except APIError as exc:
                last_err = exc
                if exc.code == 429:
                    logger.warning(f"[WARNING] Gemini Rate Limit (HTTP 429) on model {model_name}: {exc}")
                    raise
                elif exc.code and exc.code >= 500:
                    logger.warning(f"[WARNING] Gemini Server Error (HTTP {exc.code}) on model {model_name}: {exc}")
                    continue
                logger.info(f"Retrying Gemini with fallback model from {model_name}...")
            except Exception as exc:
                last_err = exc
                logger.info(f"Retrying Gemini model after error: {exc}")

        raise last_err or RuntimeError("Gemini parsing failed on all model candidates")

