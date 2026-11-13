import asyncio
from datetime import datetime
import numpy as np
from fastapi.testclient import TestClient

from main import app
from app.config import settings
from app.database import init_db
from app.parser.factory import ParserFactory
from app.stt.factory import STTFactory
from app.vad import SileroVADBuffer, pcm_to_wav

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "active_stt_provider" in data
    assert "active_parser_provider" in data
    assert data["failover_enabled"] is True

def test_database_crud():
    # 1. Create reminder
    payload = {
        "title": "Dentist Appointment",
        "category": "Meeting",
        "date": "2026-12-01",
        "time": "14:30",
        "repeat": "None",
        "advance_alert": "1 day before",
    }
    create_resp = client.post("/api/reminders", json=payload)
    assert create_resp.status_code == 200
    created = create_resp.json()
    assert created["title"] == "Dentist Appointment"
    reminder_id = created["id"]

    # 2. List reminders
    list_resp = client.get("/api/reminders")
    assert list_resp.status_code == 200
    reminders = list_resp.json()
    assert any(r["id"] == reminder_id for r in reminders)

    # 3. Delete reminder
    del_resp = client.delete(f"/api/reminders/{reminder_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"

    # 4. Verify deleted
    list_resp_after = client.get("/api/reminders")
    reminders_after = list_resp_after.json()
    assert not any(r["id"] == reminder_id for r in reminders_after)

def test_vad_in_memory_buffer():
    vad = SileroVADBuffer(
        sample_rate=16000,
        threshold=0.50,
        pause_duration_ms=1000,
        min_speech_duration_ms=500,
    )
    assert vad.threshold == 0.50
    assert vad.pause_frames_limit == int((1000 / 1000.0) * (16000 / 512))
    assert vad.min_speech_frames == int((500 / 1000.0) * (16000 / 512))

    # Test pcm_to_wav zero disk write conversion of raw unamplified PCM
    raw_samples = (np.sin(np.linspace(0, 100, 16000)) * 2000).astype(np.int16).tobytes()
    wav_bytes = pcm_to_wav(raw_samples)
    assert wav_bytes.startswith(b"RIFF")
    assert b"WAVE" in wav_bytes

def test_groq_hallucination_filtering():
    from app.stt.adapters.groq_stt import is_hallucination_or_noise
    # Hallucinations that must return True (filtered)
    assert is_hallucination_or_noise("thank you") is True
    assert is_hallucination_or_noise("Thank you.") is True
    assert is_hallucination_or_noise("Thanks for watching!") is True
    assert is_hallucination_or_noise("thanks for watching") is True

    # Valid voice reminder sentences that must NOT be filtered
    assert is_hallucination_or_noise("Remind me to buy groceries") is False
    assert is_hallucination_or_noise("Team meeting tomorrow at 10 am") is False
    assert is_hallucination_or_noise("Sarah's birthday party") is False

def test_sarvam_cleaning_and_configuration():
    from app.stt.adapters.sarvam_stt import SarvamSTTEngine, clean_sarvam_output
    engine = SarvamSTTEngine(api_key="test_key")
    assert engine.provider_name == "sarvam"
    assert engine.API_URL == "https://api.sarvam.ai/speech-to-text"

    # Repeated punctuation stripping
    assert clean_sarvam_output("Call doctor tomorrow... Please??") == "Call doctor tomorrow. Please?"
    # Rejection of <3 char noise
    assert clean_sarvam_output("..") == ""
    assert clean_sarvam_output("ok") == ""
    assert clean_sarvam_output("  ") == ""
    # Valid output preserved
    assert clean_sarvam_output("Submit quarterly report on Monday") == "Submit quarterly report on Monday"

def test_deepgram_adapter():
    from unittest.mock import patch, AsyncMock
    from app.stt.adapters.deepgram_stt import DeepgramSTTEngine
    import httpx

    engine = DeepgramSTTEngine(api_key="test_deepgram_key")
    assert engine.provider_name == "deepgram"
    assert engine.API_URL == "https://api.deepgram.com/v1/listen"

    wav = pcm_to_wav(np.zeros(16000, dtype=np.int16).tobytes())

    # 1. Successful transcription with nova-3 and en-IN
    mock_success_response = httpx.Response(
        200,
        json={
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": "Remind me to call John tomorrow at 5 PM",
                                "confidence": 0.94,
                            }
                        ]
                    }
                ]
            }
        },
        request=httpx.Request("POST", "https://api.deepgram.com/v1/listen"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_success_response
        transcript = asyncio.run(engine.transcribe(wav))
        assert transcript == "Remind me to call John tomorrow at 5 PM"

        # Verify arguments passed to Deepgram API
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["params"]["model"] == "nova-3"
        assert kwargs["params"]["language"] == "en-IN"
        assert kwargs["params"]["smart_format"] == "true"
        assert kwargs["params"]["punctuate"] == "true"
        assert kwargs["headers"]["Authorization"] == "Token test_deepgram_key"
        assert kwargs["headers"]["Content-Type"] == "audio/wav"
        assert kwargs["content"] == wav

    # 2. Low confidence (< 0.35) returns empty string
    mock_low_conf_response = httpx.Response(
        200,
        json={
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": "Unintelligible muffled noise",
                                "confidence": 0.24,
                            }
                        ]
                    }
                ]
            }
        },
        request=httpx.Request("POST", "https://api.deepgram.com/v1/listen"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_low_conf_response
        transcript = asyncio.run(engine.transcribe(wav))
        assert transcript == ""

    # 3. HTTP 429 raises error for failover
    mock_error_response = httpx.Response(
        429,
        text="Rate limited",
        request=httpx.Request("POST", "https://api.deepgram.com/v1/listen"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_error_response
        try:
            asyncio.run(engine.transcribe(wav))
            assert False, "Should have raised HTTPStatusError on 429"
        except httpx.HTTPStatusError as e:
            assert e.response.status_code == 429

def test_parser_local_rules():
    from app.parser.adapters.local_rules_parser import LocalRulesReminderParser
    parser = LocalRulesReminderParser()
    ref_dt = datetime(2026, 11, 13, 10, 0, 0)

    # Test 1: Birthday with relative date and yearly repeat default
    res1 = asyncio.run(parser.parse(
        "Remind me about Sarah's birthday next month 30 at 5pm",
        current_datetime=ref_dt
    ))
    assert res1.category == "Birthday"
    assert res1.repeat == "Yearly"
    assert res1.date == "2026-12-30"
    assert res1.time == "17:00"

    # Test 2: Meeting with tomorrow and time
    res2 = asyncio.run(parser.parse(
        "Quarterly budget meeting tomorrow at 10:30 am",
        current_datetime=ref_dt
    ))
    assert res2.category == "Meeting"
    assert res2.date == "2026-11-14"
    assert res2.time == "10:30"

    # Test 3: Anniversary with 1 week before alert
    res3 = asyncio.run(parser.parse(
        "Wedding anniversary on December 25th 1 week before",
        current_datetime=ref_dt
    ))
    assert res3.category == "Anniversary"
    assert res3.repeat == "Yearly"
    assert res3.advance_alert == "1 week before"

def test_conversational_self_correction():
    from app.parser.adapters.local_rules_parser import LocalRulesReminderParser
    parser = LocalRulesReminderParser()
    ref_dt = datetime(2026, 11, 13, 10, 0, 0)

    # Conversational correction on title ("not RMS, it's Ram's") and time ("change time to 6 PM")
    transcript = "Remind me about RMS birthday at 4 PM, not RMS, it's Ram's, change time to 6 PM"
    res = asyncio.run(parser.parse(transcript, current_datetime=ref_dt))
    assert "ram's" in res.title.lower()
    assert "rms" not in res.title.lower()
    assert res.time == "18:00"
    assert res.category == "Birthday"
    assert res.repeat == "Yearly"

def test_parser_failover():
    saved_provider = settings.PARSER_PROVIDER
    saved_failover = settings.ENABLE_FAILOVER
    try:
        settings.PARSER_PROVIDER = "groq"
        settings.ENABLE_FAILOVER = True
        ref_dt = datetime(2026, 11, 13, 10, 0, 0)
        res = asyncio.run(ParserFactory.parse_with_failover(
            "Buy groceries tomorrow evening",
            current_datetime=ref_dt
        ))
        assert res.category == "To-Do"
        assert res.date == "2026-11-14"
        assert res.time == "18:00"
    finally:
        settings.PARSER_PROVIDER = saved_provider
        settings.ENABLE_FAILOVER = saved_failover

def test_stt_failover():
    saved_provider = settings.STT_PROVIDER
    saved_failover = settings.ENABLE_FAILOVER
    try:
        settings.STT_PROVIDER = "groq"
        settings.ENABLE_FAILOVER = True
        wav = pcm_to_wav(np.zeros(16000, dtype=np.int16).tobytes())
        res = asyncio.run(STTFactory.transcribe_with_failover(wav))
        assert isinstance(res, str)
    finally:
        settings.STT_PROVIDER = saved_provider
        settings.ENABLE_FAILOVER = saved_failover

if __name__ == "__main__":
    init_db()
    print("Running test_health_check...")
    test_health_check()
    print("Running test_database_crud...")
    test_database_crud()
    print("Running test_vad_in_memory_buffer...")
    test_vad_in_memory_buffer()
    print("Running test_groq_hallucination_filtering...")
    test_groq_hallucination_filtering()
    print("Running test_sarvam_cleaning_and_configuration...")
    test_sarvam_cleaning_and_configuration()
    print("Running test_deepgram_adapter...")
    test_deepgram_adapter()
    print("Running test_parser_local_rules...")
    test_parser_local_rules()
    print("Running test_conversational_self_correction...")
    test_conversational_self_correction()
    print("Running test_parser_failover...")
    test_parser_failover()
    print("Running test_stt_failover...")
    test_stt_failover()
    print("ALL TESTS PASSED SUCCESSFULLY!")
