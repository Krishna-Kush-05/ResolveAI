"""SQLite database for storing analysed grievances.
Uses the standard library sqlite3 — no ORM, no extra dependencies.
One table: grievances."""
import json, sqlite3, os
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = os.environ.get("DB_PATH", str(Path(__file__).resolve().parent / "grievance.db"))

def _conn() -> sqlite3.Connection:
    """Get a connection with row_factory so we can use dict-like access."""
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    """Create the grievances table if it doesn't exist."""
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS grievances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            english_text TEXT NOT NULL,
            language TEXT NOT NULL,
            category TEXT NOT NULL,
            priority TEXT NOT NULL,
            confidence REAL NOT NULL,
            sentiment TEXT NOT NULL,
            sentiment_score REAL NOT NULL,
            entities TEXT NOT NULL DEFAULT '[]',
            reply TEXT,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            resolved_at TEXT
        )""")
        columns = {row["name"] for row in c.execute("PRAGMA table_info(grievances)")}
        if "status" not in columns:
            c.execute("ALTER TABLE grievances ADD COLUMN status TEXT NOT NULL DEFAULT 'pending'")
        if "resolved_at" not in columns:
            c.execute("ALTER TABLE grievances ADD COLUMN resolved_at TEXT")

def save(result: dict) -> int:
    """Insert one analysed grievance and return its id."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO grievances
               (text, english_text, language, category, priority, confidence,
                sentiment, sentiment_score, entities, reply, created_at, status, resolved_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (result["text"], result["english_text"], result["language"],
             result["category"], result["priority"], result["confidence"],
             result["sentiment"], result["sentiment_score"],
             json.dumps(result.get("entities", [])),
             json.dumps(result.get("reply")),
             now, result.get("status", "pending"), result.get("resolved_at")))
        return cur.lastrowid

def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a database row to a dict matching the API response format."""
    d = dict(row)
    d["entities"] = json.loads(d["entities"]) if d["entities"] else []
    d["reply"] = json.loads(d["reply"]) if d["reply"] else None
    return d

def get_history(limit: int = 100) -> list[dict]:
    """Return the most recent grievances, newest first."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM grievances ORDER BY id DESC LIMIT ?",
            (min(max(1, limit), 200),)).fetchall()
    return [_row_to_dict(r) for r in rows]

def delete_one(gid: int) -> bool:
    """Delete a single grievance. Returns True if it existed."""
    with _conn() as c:
        return c.execute("DELETE FROM grievances WHERE id = ?", (gid,)).rowcount > 0

def delete_all() -> int:
    """Delete all grievances. Returns the count deleted."""
    with _conn() as c:
        return c.execute("DELETE FROM grievances").rowcount

def update_status(gid: int, status: str) -> bool:
    resolved_at = datetime.now(timezone.utc).isoformat() if status == "resolved" else None
    with _conn() as c:
        return c.execute("UPDATE grievances SET status = ?, resolved_at = ? WHERE id = ?",
                         (status, resolved_at, gid)).rowcount > 0

def update_status_many(ids: list[int], status: str) -> int:
    resolved_at = datetime.now(timezone.utc).isoformat() if status == "resolved" else None
    with _conn() as c:
        cur = c.executemany("UPDATE grievances SET status = ?, resolved_at = ? WHERE id = ?",
                            [(status, resolved_at, gid) for gid in ids])
        return cur.rowcount
