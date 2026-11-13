import json
import logging
from datetime import datetime
from groq import AsyncGroq, APIStatusError, RateLimitError
from app.config import settings
from app.parser.base import BaseReminderParser, ReminderSchema

logger = logging.getLogger("app.parser.groq")

SYSTEM_PROMPT = """You are an expert structured data extractor for voice reminders.
Analyze the user's transcript and extract the reminder information into a strict JSON object with these keys:
- title: concise, human-readable reminder title (remove command phrases like "remind me to", "set a reminder for").
- category: one of ["Birthday", "Anniversary", "To-Do", "Meeting", "Custom"].
- date: target date in YYYY-MM-DD format, accurately resolved relative to the reference date provided.
- time: target time in HH:MM (24-hour) format. If a time of day is mentioned (e.g., 'morning' -> 09:00, 'afternoon' -> 14:00, 'evening' -> 18:00, 'night' -> 20:00), resolve it accordingly. Default to "09:00" if no specific time was mentioned.
- repeat: one of ["None", "Yearly", "Monthly", "Weekly"]. If category is "Birthday" or "Anniversary", repeat MUST default to "Yearly" unless explicitly stated otherwise.
- advance_alert: advance alert string (e.g., "Morning of event", "1 day before", "1 hour before", "1 week before").

The transcript may contain live spoken corrections where the user fixes an earlier STT misrecognition or changes details mid-session (e.g., 'not RMS, it's Ram's', 'change time to 6 PM', 'make it next Friday instead'). Always apply later corrections to overwrite earlier values in the extracted ReminderSchema fields.

Respond ONLY with a valid JSON object matching the schema. Do not include markdown codeblocks or extra text.
"""

class GroqReminderParser(BaseReminderParser):
    provider_name = "groq"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        if not self.api_key:
            logger.warning("[WARNING] GROQ_API_KEY is not set. Groq parser calls will fail.")
        self.client = AsyncGroq(api_key=self.api_key) if self.api_key else None

    async def parse(self, transcript: str, current_datetime: datetime) -> ReminderSchema:
        if not self.client:
            raise ValueError("Groq API key not configured")

        user_prompt = (
            f"Reference Current DateTime: {current_datetime.strftime('%Y-%m-%d %H:%M:%S (%A)')}\n"
            f"Voice Transcript: \"{transcript}\"\n\n"
            f"Extract the reminder details as a valid JSON object:"
        )

        models_to_try = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
            "qwen/qwen3.8-27b",
        ]
        last_exc = None

        for model_name in models_to_try:
            try:
                response = await self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                raw_text = response.choices[0].message.content or "{}"
                parsed_data = json.loads(raw_text)

                # Enforce Birthday/Anniversary Yearly repeat default
                category = parsed_data.get("category", "To-Do")
                if category in ["Birthday", "Anniversary"] and parsed_data.get("repeat") in [None, "None"]:
                    parsed_data["repeat"] = "Yearly"

                return ReminderSchema(**parsed_data)
            except RateLimitError as exc:
                logger.warning(f"[WARNING] Groq Parser Rate Limit (HTTP 429) on model {model_name}: {exc}")
                last_exc = exc
                raise
            except APIStatusError as exc:
                last_exc = exc
                if exc.status_code == 404:
                    logger.info(f"Groq model {model_name} not found, trying fallback...")
                    continue
                if exc.status_code >= 500:
                    logger.warning(f"[WARNING] Groq Parser Server Error (HTTP {exc.status_code}) on model {model_name}: {exc}")
                    continue
                raise
            except Exception as exc:
                last_exc = exc
                logger.info(f"Retrying Groq parser after error on {model_name}: {exc}")
                continue

        raise last_exc or RuntimeError("All Groq model candidates failed")

