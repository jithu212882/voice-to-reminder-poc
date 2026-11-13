import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import create_reminder, delete_reminder, get_reminders, init_db
from app.parser.base import ReminderSchema
from app.parser.factory import ParserFactory
from app.stt.factory import STTFactory
from app.vad import SileroVADBuffer

# Structured logging configuration for DevOps
class DevOpsJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_payload = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_payload)

log_handler = logging.StreamHandler()
log_handler.setFormatter(DevOpsJsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[log_handler])
logger = logging.getLogger("voice_reminder_service")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info(
        json.dumps({
            "event": "startup",
            "stt_provider": settings.STT_PROVIDER,
            "parser_provider": settings.PARSER_PROVIDER,
            "failover_enabled": settings.ENABLE_FAILOVER,
        })
    )
    yield
    logger.info(json.dumps({"event": "shutdown"}))

app = FastAPI(
    title="Voice-to-Reminder Service",
    description="Production-ready modular voice reminder processing with pluggable STT and Parser engines.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST Endpoints
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_stt_provider": settings.STT_PROVIDER,
        "active_parser_provider": settings.PARSER_PROVIDER,
        "failover_enabled": settings.ENABLE_FAILOVER,
        "providers": {
            "stt": ["groq", "sarvam", "local_whisper"],
            "parser": ["groq", "gemini", "local_rules"],
        },
    }

@app.get("/api/reminders")
async def list_reminders() -> List[Dict[str, Any]]:
    return get_reminders()

@app.post("/api/reminders")
async def save_reminder(reminder: ReminderSchema) -> Dict[str, Any]:
    saved = create_reminder(reminder.model_dump())
    logger.info(f"Reminder created with ID: {saved.get('id')}")
    return saved

@app.delete("/api/reminders/{reminder_id}")
async def remove_reminder(reminder_id: int) -> Dict[str, Any]:
    success = delete_reminder(reminder_id)
    if not success:
        raise HTTPException(status_code=404, detail="Reminder not found")
    logger.info(f"Reminder deleted with ID: {reminder_id}")
    return {"status": "deleted", "id": reminder_id}

# WebSocket Voice Stream
@app.websocket("/ws/voice")
async def voice_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket voice client connected")

    vad_buffer = SileroVADBuffer(
        pause_duration_ms=1000,
        min_speech_duration_ms=500,
        threshold=0.50,
    )
    session_transcript_chunks: List[str] = []

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                pcm_chunk = message["bytes"]
                emitted_chunks = vad_buffer.process_chunk(pcm_chunk)

                for wav_bytes, is_final in emitted_chunks:
                    try:
                        text_chunk = await STTFactory.transcribe_with_failover(wav_bytes)
                        if text_chunk and text_chunk.strip():
                            session_transcript_chunks.append(text_chunk.strip())
                            full_text = " ".join(session_transcript_chunks).strip()
                            await websocket.send_json({
                                "type": "transcript",
                                "text": text_chunk.strip(),
                                "is_final": True,
                                "accumulated": full_text,
                            })
                            # Parse full accumulated session transcript with conversational self-correction
                            try:
                                parsed_reminder = await ParserFactory.parse_with_failover(
                                    full_text, datetime.now()
                                )
                                await websocket.send_json({
                                    "type": "reminder",
                                    "data": parsed_reminder.model_dump(),
                                    "transcript": full_text,
                                })
                            except Exception as parse_err:
                                logger.error(f"Parser failed on speech pause interval: {parse_err}")
                    except Exception as exc:
                        logger.error(f"STT transcription failed on VAD chunk: {exc}")

            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                except Exception:
                    payload = {"type": message["text"]}

                msg_type = payload.get("type")
                if msg_type in ["stop", "finish", "flush"]:
                    flushed_wav = vad_buffer.flush()
                    if flushed_wav:
                        try:
                            text_chunk = await STTFactory.transcribe_with_failover(flushed_wav)
                            if text_chunk:
                                session_transcript_chunks.append(text_chunk)
                        except Exception as exc:
                            logger.error(f"STT transcription failed on flush: {exc}")

                    full_text = " ".join(session_transcript_chunks).strip()

                    if full_text:
                        await websocket.send_json({
                            "type": "transcript",
                            "text": "",
                            "is_final": True,
                            "accumulated": full_text,
                        })
                        try:
                            parsed_reminder = await ParserFactory.parse_with_failover(
                                full_text, datetime.now()
                            )
                            await websocket.send_json({
                                "type": "reminder",
                                "data": parsed_reminder.model_dump(),
                                "transcript": full_text,
                            })
                        except Exception as exc:
                            logger.error(f"Parser failed on transcript: {exc}")
                            await websocket.send_json({
                                "type": "error",
                                "message": "Could not parse reminder details from transcript.",
                            })
                    else:
                        await websocket.send_json({
                            "type": "empty",
                            "message": "No speech detected.",
                        })

                    vad_buffer.reset()
                    session_transcript_chunks.clear()

    except WebSocketDisconnect:
        logger.info("WebSocket voice client disconnected")
    except Exception as exc:
        logger.error(f"WebSocket session error: {exc}")
    finally:
        vad_buffer.reset()

# Static Frontend Files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)

