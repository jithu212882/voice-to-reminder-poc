import re
from datetime import datetime, timedelta
import dateparser
from app.parser.base import BaseReminderParser, ReminderSchema, CategoryType, RepeatType

class LocalRulesReminderParser(BaseReminderParser):
    provider_name = "local_rules"

    async def parse(self, transcript: str, current_datetime: datetime) -> ReminderSchema:
        raw = transcript.strip()
        lower = raw.lower()

        # 1. Category extraction
        category: CategoryType = "To-Do"
        if re.search(r"\b(birthday|bday)\b", lower):
            category = "Birthday"
        elif re.search(r"\banniversary\b", lower):
            category = "Anniversary"
        elif re.search(r"\b(meeting|sync|call|interview|standup|appointment|conference|catch up)\b", lower):
            category = "Meeting"
        elif re.search(r"\b(buy|submit|pay|clean|finish|grocery|groceries|pickup|pick up|deliver|fix)\b", lower):
            category = "To-Do"

        # 2. Repeat extraction
        repeat: RepeatType = "None"
        if re.search(r"\b(every year|yearly|annually)\b", lower):
            repeat = "Yearly"
        elif re.search(r"\b(every month|monthly)\b", lower):
            repeat = "Monthly"
        elif re.search(r"\b(every week|weekly)\b", lower):
            repeat = "Weekly"
        elif category in ["Birthday", "Anniversary"]:
            repeat = "Yearly"

        # 3. Advance alert extraction
        advance_alert = "Morning of event"
        if re.search(r"\b(1 week before|a week before|one week before)\b", lower):
            advance_alert = "1 week before"
        elif re.search(r"\b(2 days before|two days before)\b", lower):
            advance_alert = "2 days before"
        elif re.search(r"\b(1 day before|a day before|one day before|the day before)\b", lower):
            advance_alert = "1 day before"
        elif re.search(r"\b(1 hour before|an hour before|one hour before)\b", lower):
            advance_alert = "1 hour before"
        elif re.search(r"\b(morning of|morning of event)\b", lower):
            advance_alert = "Morning of event"

        # 4. Time extraction (prioritize latest spoken time or "change time to" correction)
        time_str = "09:00"
        all_time_matches = list(re.finditer(r"\b(?:(?:change\s+time\s+to|make\s+it|at)\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", lower))
        if all_time_matches:
            last_match = all_time_matches[-1]
            hour = int(last_match.group(1))
            minute = int(last_match.group(2) or 0)
            ampm = last_match.group(3)
            if ampm == "pm" and hour < 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0
            time_str = f"{hour:02d}:{minute:02d}"
        elif re.search(r"\b(?:at\s+)?(\d{1,2}):(\d{2})\b", lower):
            t_match = re.search(r"\b(?:at\s+)?(\d{1,2}):(\d{2})\b", lower)
            h = int(t_match.group(1))
            m = int(t_match.group(2))
            if 0 <= h < 24 and 0 <= m < 60:
                time_str = f"{h:02d}:{m:02d}"
        elif re.search(r"\b(in the evening|evening)\b", lower):
            time_str = "18:00"
        elif re.search(r"\b(in the afternoon|afternoon)\b", lower):
            time_str = "14:00"
        elif re.search(r"\b(in the morning|morning)\b", lower):
            time_str = "09:00"
        elif re.search(r"\b(noon|midday)\b", lower):
            time_str = "12:00"
        elif re.search(r"\bmidnight\b", lower):
            time_str = "00:00"

        # 5. Date extraction
        target_date = current_datetime.date()
        date_extracted = False

        # Check explicit relative phrases
        next_month_match = re.search(r"\bnext month\s+(\d{1,2})\b", lower)
        if next_month_match:
            day = int(next_month_match.group(1))
            year = current_datetime.year
            month = current_datetime.month + 1
            if month > 12:
                month = 1
                year += 1
            try:
                target_date = datetime(year, month, day).date()
                date_extracted = True
            except ValueError:
                pass

        if not date_extracted:
            if "day after tomorrow" in lower:
                target_date = (current_datetime + timedelta(days=2)).date()
                date_extracted = True
            elif "tomorrow" in lower:
                target_date = (current_datetime + timedelta(days=1)).date()
                date_extracted = True

        if not date_extracted:
            date_phrases = re.findall(
                r"\b(?:on\s+)?(?:next\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
                r"(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?|"
                r"\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december))\b",
                lower
            )
            for phrase in date_phrases:
                parsed_dt = dateparser.parse(
                    phrase,
                    settings={"RELATIVE_BASE": current_datetime, "PREFER_DATES_FROM": "future"}
                )
                if parsed_dt:
                    target_date = parsed_dt.date()
                    date_extracted = True
                    break

        date_str = target_date.strftime("%Y-%m-%d")

        # 6. Title extraction
        title = raw

        # Conversational self-correction: e.g. "not RMS, it's Ram's"
        corr_m = re.search(r"\bnot\s+([^,]+),\s*(?:it's|it is)\s+([^,.]+)", raw, re.IGNORECASE)
        if corr_m:
            wrong_term = corr_m.group(1).strip()
            correct_term = corr_m.group(2).strip()
            title = re.sub(r"\bnot\s+[^,]+,\s*(?:it's|it is)\s+[^,.]+[.,]?", "", title, flags=re.IGNORECASE).strip()
            title = re.sub(r"\b" + re.escape(wrong_term) + r"\b", correct_term, title, flags=re.IGNORECASE).strip()
            if not title:
                title = correct_term
        title = re.sub(
            r"^(please\s+)?(set a reminder (to|for)|remind me (to|about|for|of)|create a reminder (to|for)|add a reminder (to|for)|reminder (to|for))\s+",
            "",
            title,
            flags=re.IGNORECASE,
        ).strip()

        # Remove trailing date/time fragments from title
        patterns_to_remove = [
            r"\b(tomorrow|day after tomorrow|today)\b",
            r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|month(?:\s+\d{1,2})?|week)\b",
            r"\b(every year|yearly|every month|monthly|every week|weekly)\b",
            r"\b(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b",
            r"\b(in the morning|in the evening|in the afternoon|at night)\b",
            r"\b(1 day before|1 hour before|1 week before|morning of event)\b",
            r"\bon\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        ]
        for pat in patterns_to_remove:
            title = re.sub(pat, "", title, flags=re.IGNORECASE).strip()

        title = re.sub(r"\s{2,}", " ", title).strip(" ,.-")
        if not title:
            title = f"{category} Reminder"
        else:
            title = title[0].upper() + title[1:]

        return ReminderSchema(
            title=title,
            category=category,
            date=date_str,
            time=time_str,
            repeat=repeat,
            advance_alert=advance_alert,
        )

