"""SQLite database for storing analysed grievances.
Uses the standard library sqlite3 — no ORM, no extra dependencies.
One table: grievances."""
import json, sqlite3, os
from pathlib import Path
from datetime import datetime, timezone, timedelta

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

def save_many(results: list[dict]) -> list[int]:
    """Insert a batch in one SQLite transaction and return its row IDs."""
    if not results:
        return []
    now = datetime.now(timezone.utc).isoformat()
    values = [(r["text"], r["english_text"], r["language"], r["category"], r["priority"],
               r["confidence"], r["sentiment"], r["sentiment_score"], json.dumps(r.get("entities", [])),
               json.dumps(r.get("reply")), now, r.get("status", "pending"), r.get("resolved_at"))
              for r in results]
    with _conn() as c:
        c.executemany(
            """INSERT INTO grievances
               (text, english_text, language, category, priority, confidence,
                sentiment, sentiment_score, entities, reply, created_at, status, resolved_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", values)
        last_id = c.execute("SELECT last_insert_rowid()").fetchone()[0]
    return list(range(last_id - len(results) + 1, last_id + 1))

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

def get_by_id(gid: int) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM grievances WHERE id = ?", (gid,)).fetchone()
    return _row_to_dict(row) if row else None

def get_recent_for_similarity(limit: int = 500) -> list[dict]:
    with _conn() as c:
        rows = c.execute("""SELECT id, english_text, category, status
                             FROM grievances ORDER BY id DESC LIMIT ?""", (min(max(1, limit), 500),)).fetchall()
    return [dict(row) for row in rows]

def get_stats() -> dict:
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=13)
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM grievances").fetchone()[0]
        status_counts = {row[0]: row[1] for row in c.execute("SELECT status, COUNT(*) FROM grievances GROUP BY status")}
        categories = {row[0]: row[1] for row in c.execute("SELECT category, COUNT(*) FROM grievances GROUP BY category")}
        priorities = {row[0]: row[1] for row in c.execute("SELECT priority, COUNT(*) FROM grievances GROUP BY priority")}
        languages = {row[0]: row[1] for row in c.execute("SELECT language, COUNT(*) FROM grievances GROUP BY language")}
        average = c.execute("SELECT AVG(sentiment_score) FROM grievances").fetchone()[0]
        daily_rows = {row[0]: row[1] for row in c.execute(
            "SELECT date(created_at), COUNT(*) FROM grievances WHERE date(created_at) >= ? GROUP BY date(created_at)",
            (start.isoformat(),))}
    days = [(start + timedelta(days=i)).isoformat() for i in range(14)]
    return {
        "total": total, "pending": status_counts.get("pending", 0), "resolved": status_counts.get("resolved", 0),
        "by_category": categories, "by_priority": priorities, "average_sentiment_score": round(average or 0, 3),
        "by_language": languages, "per_day": [{"date": day, "count": daily_rows.get(day, 0)} for day in days],
    }

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
