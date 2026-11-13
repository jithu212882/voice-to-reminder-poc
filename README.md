# Voice-to-Reminder Application

Production-ready, modular Voice-to-Reminder application built with FastAPI and an ultra-clean, minimal HTML/CSS/JS frontend.

---

## 1. Key Features

- **Strict Minimal Light-Mode UI**:
  - System/Inter sans-serif typography.
  - Zero backend jargon or telemetry (no latency ms, provider names, model names, or confidence scores shown on the UI).
  - Zero AI sparkles or decorative badges.
  - Interactive top microphone button with subtle pulse animation while recording.
  - Real-time live transcript box and editable reminder card.
  - Persistent SQLite-backed reminder history.
- **In-Memory Silero VAD (`silero-vad`)**:
  - Processes 16kHz 16-bit mono PCM audio frames directly from WebSocket streams.
  - Emits complete in-memory WAV bytes (`io.BytesIO`) upon a 1200ms speech pause with zero disk I/O.
- **Pluggable STT Adapters with Automatic Failover**:
  - `deepgram`: Nova-3 (`nova-3`, language: en-IN, smart_format: true, punctuate: true, confidence filtering)
  - `groq`: `whisper-large-v3-turbo` (language: en, temperature: 0.0, hallucination filtering)
  - `sarvam`: `saaras:v4` (codemix mode, language_code: en-IN)
  - `local_whisper`: `faster-whisper`
  - Automatic fallback on HTTP 429 / 5xx / timeout when `ENABLE_FAILOVER=true`.
- **Pluggable Reminder Extraction Parsers**:
  - `groq`: `llama-3.3-70b-versatile` JSON mode.
  - `gemini`: `gemini-3.5-flash-lite` structured schema via `google-genai`.
  - `local_rules`: `dateparser` + regex + keyword rules (runs 100% locally with zero external API keys).
  - Injects current timestamp (`datetime.now()`) into prompts so relative dates (e.g., "next month 30") resolve accurately.
  - Automatically defaults Birthday and Anniversary categories to `Yearly` recurrence.
- **DevOps Monitoring**:
  - Structured JSON logs formatted for log aggregators (Datadog, CloudWatch, Loki).
  - Explicit `[WARNING]` logs on rate limits, HTTP status codes, and failover triggers.
  - `GET /health` endpoint reporting active engines and failover status.

---

## 2. Directory Structure

```
├── .env.example
├── requirements.txt
├── README.md
├── main.py
├── static/
│   ├── index.html
│   ├── style.css
│   └── app.js
└── app/
    ├── __init__.py
    ├── config.py
    ├── database.py
    ├── vad.py
    ├── stt/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── factory.py
    │   └── adapters/
    │       ├── groq_stt.py
    │       ├── sarvam_stt.py
    │       ├── local_whisper.py
    │       └── deepgram_stt.py
    └── parser/
        ├── __init__.py
        ├── base.py
        ├── factory.py
        └── adapters/
            ├── groq_parser.py
            ├── gemini_parser.py
            └── local_rules_parser.py
```

---

## 3. Configuration & Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `STT_PROVIDER` | `groq` | Active STT engine: `groq`, `sarvam`, or `local_whisper` |
| `PARSER_PROVIDER` | `groq` | Active parser: `groq`, `gemini`, or `local_rules` |
| `ENABLE_FAILOVER` | `true` | Automatically failover to secondary providers on errors |
| `GROQ_API_KEY` | `""` | API key for Groq Whisper & LLaMA 3.3 |
| `SARVAM_API_KEY` | `""` | API key for Sarvam Saarika |
| `GEMINI_API_KEY` | `""` | API key for Google Gemini |
| `LOCAL_WHISPER_SIZE`| `base` | Local faster-whisper model (`tiny`, `base`, `small`) |
| `DATABASE_PATH` | `reminders.db` | SQLite database file location |
| `HOST` | `0.0.0.0` | Server bind host |
| `PORT` | `8000` | Server bind port |

---

## 4. Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Running the Service
```bash
python main.py
```
Or via Uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in any modern web browser to access the application.

---

## 5. API Reference

### Health Check
- **`GET /health`**
  ```json
  {
    "status": "healthy",
    "timestamp": "2026-11-13T11:00:00",
    "active_stt_provider": "groq",
    "active_parser_provider": "groq",
    "failover_enabled": true,
    "providers": {
      "stt": ["groq", "sarvam", "local_whisper"],
      "parser": ["groq", "gemini", "local_rules"]
    }
  }
  ```

### Reminders REST API
- **`GET /api/reminders`**: Retrieve all saved reminders ordered chronologically.
- **`POST /api/reminders`**: Create a new reminder.
  ```json
  {
    "title": "Team Planning Sync",
    "category": "Meeting",
    "date": "2026-11-20",
    "time": "14:00",
    "repeat": "Weekly",
    "advance_alert": "1 hour before"
  }
  ```
- **`DELETE /api/reminders/{id}`**: Delete a reminder by ID.

### Voice WebSocket Protocol
- **Endpoint**: `/ws/voice`
- **Incoming**:
  - Binary frames: 16kHz 16-bit mono PCM chunks.
  - JSON text: `{"type": "stop"}` to finalize dictation.
- **Outgoing**:
  - `{"type": "transcript", "text": "...", "accumulated": "..."}`
  - `{"type": "reminder", "data": {...}, "transcript": "..."}`

