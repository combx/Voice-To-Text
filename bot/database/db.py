"""SQLite database layer using aiosqlite."""

import aiosqlite
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent.parent / "data" / "bot.db"

# Singleton instance
_db_instance: Optional["Database"] = None


def get_db() -> "Database":
    """Get the database singleton instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance


class Database:
    """Async SQLite database wrapper."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    async def init(self) -> None:
        """Initialize database — create tables if not exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    full_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS transcriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    file_name TEXT,
                    file_type TEXT,
                    duration_seconds REAL,
                    language TEXT,
                    speakers_count INTEGER,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            await db.commit()

    async def add_user(
        self,
        user_id: int,
        username: Optional[str],
        full_name: str,
        status: str = "pending",
    ) -> None:
        """Add a new user to the database."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR IGNORE INTO users (user_id, username, full_name, status) "
                "VALUES (?, ?, ?, ?)",
                (user_id, username, full_name, status),
            )
            await db.commit()

    async def get_user(self, user_id: int) -> Optional[dict]:
        """Get user by Telegram ID."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def get_all_users(self) -> list[dict]:
        """Get all users."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM users ORDER BY created_at DESC"
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def update_user_status(self, user_id: int, status: str) -> None:
        """Update user authorization status."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE users SET status = ? WHERE user_id = ?",
                (status, user_id),
            )
            await db.commit()

    async def add_transcription(
        self,
        user_id: int,
        file_name: str,
        file_type: str,
        duration_seconds: float = 0,
    ) -> int:
        """Record a new transcription task. Returns the transcription ID."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "INSERT INTO transcriptions (user_id, file_name, file_type, duration_seconds, status) "
                "VALUES (?, ?, ?, ?, 'processing')",
                (user_id, file_name, file_type, duration_seconds),
            )
            await db.commit()
            return cursor.lastrowid

    async def complete_transcription(
        self,
        transcription_id: int,
        language: str = None,
        speakers_count: int = None,
    ) -> None:
        """Mark transcription as completed."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE transcriptions SET status = 'completed', "
                "language = ?, speakers_count = ?, "
                "completed_at = CURRENT_TIMESTAMP "
                "WHERE id = ?",
                (language, speakers_count, transcription_id),
            )
            await db.commit()

    async def fail_transcription(self, transcription_id: int) -> None:
        """Mark transcription as failed."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE transcriptions SET status = 'failed', "
                "completed_at = CURRENT_TIMESTAMP WHERE id = ?",
                (transcription_id,),
            )
            await db.commit()

    async def get_stats(self) -> dict:
        """Get bot usage statistics."""
        async with aiosqlite.connect(self.db_path) as db:
            # User counts
            async with db.execute("SELECT COUNT(*) FROM users") as c:
                total_users = (await c.fetchone())[0]
            async with db.execute(
                "SELECT COUNT(*) FROM users WHERE status = 'approved'"
            ) as c:
                approved_users = (await c.fetchone())[0]
            async with db.execute(
                "SELECT COUNT(*) FROM users WHERE status = 'pending'"
            ) as c:
                pending_users = (await c.fetchone())[0]
            async with db.execute(
                "SELECT COUNT(*) FROM users WHERE status = 'rejected'"
            ) as c:
                rejected_users = (await c.fetchone())[0]

            # Transcription stats
            async with db.execute(
                "SELECT COUNT(*) FROM transcriptions WHERE status = 'completed'"
            ) as c:
                total_transcriptions = (await c.fetchone())[0]
            async with db.execute(
                "SELECT COALESCE(SUM(duration_seconds), 0) FROM transcriptions "
                "WHERE status = 'completed'"
            ) as c:
                total_duration = (await c.fetchone())[0]

            # Format duration
            hours = int(total_duration // 3600)
            minutes = int((total_duration % 3600) // 60)
            if hours > 0:
                duration_formatted = f"{hours}ч {minutes}мин"
            else:
                duration_formatted = f"{minutes}мин"

            return {
                "total_users": total_users,
                "approved_users": approved_users,
                "pending_users": pending_users,
                "rejected_users": rejected_users,
                "total_transcriptions": total_transcriptions,
                "total_duration": total_duration,
                "total_duration_formatted": duration_formatted,
            }
