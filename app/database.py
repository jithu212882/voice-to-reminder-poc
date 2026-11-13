import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional
from app.config import settings

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                repeat TEXT NOT NULL DEFAULT 'None',
                advance_alert TEXT NOT NULL DEFAULT 'Morning of event',
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()

def get_reminders() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.execute("SELECT * FROM reminders ORDER BY date ASC, time ASC, id DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def create_reminder(reminder_data: Dict[str, Any]) -> Dict[str, Any]:
    with get_connection() as conn:
        created_at = datetime.now().isoformat()
        cursor = conn.execute(
            """
            INSERT INTO reminders (title, category, date, time, repeat, advance_alert, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                reminder_data.get("title", "Untitled Reminder"),
                reminder_data.get("category", "To-Do"),
                reminder_data.get("date", datetime.now().strftime("%Y-%m-%d")),
                reminder_data.get("time", "09:00"),
                reminder_data.get("repeat", "None"),
                reminder_data.get("advance_alert", "Morning of event"),
                created_at,
            ),
        )
        conn.commit()
        reminder_id = cursor.lastrowid
        cursor = conn.execute("SELECT * FROM reminders WHERE id = ?", (reminder_id,))
        row = cursor.fetchone()
        return dict(row) if row else {"id": reminder_id, **reminder_data, "created_at": created_at}

def delete_reminder(reminder_id: int) -> bool:
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
        conn.commit()
        return cursor.rowcount > 0

