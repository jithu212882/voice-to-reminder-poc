import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from app.config import settings
from app.parser.base import BaseReminderParser, ReminderSchema
from app.parser.adapters.groq_parser import GroqReminderParser
from app.parser.adapters.gemini_parser import GeminiReminderParser
from app.parser.adapters.local_rules_parser import LocalRulesReminderParser

logger = logging.getLogger("app.parser.factory")

PARSER_SYSTEM_PROMPT = """You are an expert structured data extractor for voice reminders.
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

class ParserFactory:
    _parsers: Dict[str, BaseReminderParser] = {}

    @classmethod
    def get_parser(cls, provider: Optional[str] = None) -> BaseReminderParser:
        provider_name = (provider or settings.PARSER_PROVIDER).lower()

        if provider_name not in cls._parsers:
            if provider_name == "groq":
                cls._parsers[provider_name] = GroqReminderParser()
            elif provider_name == "gemini":
                cls._parsers[provider_name] = GeminiReminderParser()
            elif provider_name == "local_rules":
                cls._parsers[provider_name] = LocalRulesReminderParser()
            else:
                raise ValueError(f"Unknown parser provider: {provider_name}")

        return cls._parsers[provider_name]

    @classmethod
    def get_failover_candidates(cls, primary: str) -> List[str]:
        candidates: List[str] = []
        if primary == "groq":
            if settings.GEMINI_API_KEY:
                candidates.append("gemini")
            candidates.append("local_rules")
        elif primary == "gemini":
            if settings.GROQ_API_KEY:
                candidates.append("groq")
            candidates.append("local_rules")
        elif primary == "local_rules":
            if settings.GROQ_API_KEY:
                candidates.append("groq")
            if settings.GEMINI_API_KEY:
                candidates.append("gemini")
        else:
            candidates = ["local_rules"]
        return candidates

    @classmethod
    async def parse_with_failover(cls, transcript: str, current_datetime: Optional[datetime] = None) -> ReminderSchema:
        ref_dt = current_datetime or datetime.now()
        primary_provider = settings.PARSER_PROVIDER
        providers_to_try = [primary_provider]

        if settings.ENABLE_FAILOVER:
            providers_to_try.extend(cls.get_failover_candidates(primary_provider))

        last_error = None
        for idx, provider_name in enumerate(providers_to_try):
            try:
                parser = cls.get_parser(provider_name)
                result = await parser.parse(transcript, ref_dt)

                # Guarantee Birthday/Anniversary defaults to Yearly repeat
                if result.category in ["Birthday", "Anniversary"] and result.repeat in [None, "None"]:
                    result.repeat = "Yearly"

                if idx > 0:
                    logger.info(
                        json.dumps({
                            "event": "parser_failover_success",
                            "primary_provider": primary_provider,
                            "active_provider": provider_name,
                        })
                    )
                return result
            except Exception as exc:
                last_error = exc
                if idx < len(providers_to_try) - 1 and settings.ENABLE_FAILOVER:
                    next_provider = providers_to_try[idx + 1]
                    logger.warning(
                        f"[WARNING] Parser provider '{provider_name}' failed with: {exc}. "
                        f"Failing over to '{next_provider}'."
                    )
                else:
                    logger.error(
                        f"All parsers failed or failover disabled. Last error: {exc}"
                    )

        raise last_error or RuntimeError("Parsing failed on all providers")

