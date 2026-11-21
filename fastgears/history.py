"""Bounded asynchronous SQLite journal. A batch is committed atomically."""
import json
import queue
import sqlite3
import threading
from pathlib import Path

TABLES = ("sessions", "events", "patterns", "pattern_occurrences", "alerts")


class History:
    def __init__(self, path):
        self.path = Path(path)
        self.pending = queue.Queue(256)
        self.dropped = 0
        self.error = None
        self.ready = threading.Event()
        self.worker = threading.Thread(target=self._write, daemon=True)
        self.worker.start()
        self.ready.wait()

    def submit(self, batch):
        if not batch:
            return
        try:
            self.pending.put_nowait(batch)
        except queue.Full:
            self.dropped += 1

    def _write(self):
        db = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(self.path)
            db.execute("PRAGMA journal_mode=WAL")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > 1:
                raise ValueError("Unsupported database version")
            with db:
                for table in TABLES:
                    db.execute(f"CREATE TABLE IF NOT EXISTS {table} "
                               "(id TEXT PRIMARY KEY, session_id TEXT, timestamp TEXT, record TEXT NOT NULL)")
                    db.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_time ON {table}(timestamp)")
                db.execute("PRAGMA user_version=1")
        except (OSError, ValueError, sqlite3.Error) as exc:
            self.error = type(exc).__name__
            if db:
                db.close()
            db = None
        self.ready.set()
        while True:
            batch = self.pending.get()
            try:
                if batch is None:
                    break
                if db is None:
                    self.dropped += 1
                    continue
                with db:
                    for table, record in batch:
                        assert table in TABLES
                        db.execute(f"INSERT INTO {table} VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                                   "timestamp=excluded.timestamp, record=excluded.record",
                                   (record["id"], record.get("session_id"), record["timestamp"], json.dumps(record)))
            except sqlite3.Error as exc:
                self.error = type(exc).__name__
                self.dropped += 1
            finally:
                self.pending.task_done()
        if db:
            db.close()

    def recent(self, table, limit=50):
        if table not in TABLES or not self.path.is_file():
            return []
        try:
            db = sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True)
            try:
                rows = db.execute(f"SELECT record FROM {table} ORDER BY timestamp DESC, rowid DESC LIMIT ?",
                                  (max(1, min(limit, 512)),))
                return [json.loads(row[0]) for row in rows]
            finally:
                db.close()
        except (sqlite3.Error, ValueError) as exc:
            self.error = type(exc).__name__
            return []

    def close(self):
        self.pending.put(None)
        self.worker.join()
