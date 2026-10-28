"""
SQLite cache backend for LongCache.

Default backend with zero external dependencies (beyond Python stdlib).
Stores vectors as BLOBs, responses as JSON, and uses WAL mode
for thread-safe concurrent reads.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Optional

import numpy as np

from longcache.backends.base import AbstractCacheBackend
from longcache.core.entry import CacheEntry


class SQLiteCacheBackend(AbstractCacheBackend):
    """
    SQLite-based cache backend with WAL mode for concurrent access.

    Vectors are stored as binary BLOBs (numpy float32 bytes).
    Responses are JSON-serialized. All operations are thread-safe
    via a reentrant lock.

    Usage:
        backend = SQLiteCacheBackend(db_path=".longcache/cache.db")
        entry_id = backend.store(entry)
        result = backend.retrieve(entry_id)
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS cache_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        namespace TEXT NOT NULL,
        query TEXT NOT NULL,
        vector BLOB NOT NULL,
        response TEXT NOT NULL,
        created_at REAL NOT NULL,
        expires_at REAL,
        hit_count INTEGER DEFAULT 0,
        model TEXT DEFAULT '',
        system_prompt TEXT DEFAULT '',
        temperature REAL DEFAULT -1.0,
        metadata TEXT NOT NULL DEFAULT '{}'
    );
    CREATE INDEX IF NOT EXISTS idx_namespace ON cache_entries(namespace);
    CREATE INDEX IF NOT EXISTS idx_expires ON cache_entries(expires_at);
    CREATE INDEX IF NOT EXISTS idx_namespace_created ON cache_entries(namespace, created_at);

    CREATE TABLE IF NOT EXISTS cache_tags (
        entry_id INTEGER NOT NULL REFERENCES cache_entries(id) ON DELETE CASCADE,
        tag TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_tags_tag ON cache_tags(tag);
    CREATE INDEX IF NOT EXISTS idx_tags_entry_id ON cache_tags(entry_id);

    CREATE TABLE IF NOT EXISTS cache_stats (
        key   TEXT PRIMARY KEY,
        value INTEGER NOT NULL DEFAULT 0
    );
    INSERT OR IGNORE INTO cache_stats(key, value) VALUES ('hits', 0);
    INSERT OR IGNORE INTO cache_stats(key, value) VALUES ('misses', 0);
    INSERT OR IGNORE INTO cache_stats(key, value) VALUES ('total_saved_ms', 0);
    """

    def __init__(self, db_path: str = ".longcache/cache.db", max_entries: int = 10_000):
        """
        Initialize the SQLite backend.

        Creates the database file and schema if they don't exist.

        Args:
            db_path: Path to the SQLite database file.
                Parent directories are created automatically.
            max_entries: Maximum entries per namespace (used by evict_oldest).
        """
        self._db_path = db_path
        self._max_entries = max_entries
        self._lock = threading.RLock()
        self._local = threading.local()

        # Ensure directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        conn = self._get_connection()
        try:
            with conn:
                columns = {r["name"] for r in conn.execute("PRAGMA table_info(cache_entries)")}
                if columns and "metadata" not in columns:
                    raise ValueError(
                        "Legacy cache format: use a new v0.2 database path; old data was not modified"
                    )
                required = {
                    "id",
                    "namespace",
                    "query",
                    "vector",
                    "response",
                    "created_at",
                    "expires_at",
                    "hit_count",
                    "model",
                    "system_prompt",
                    "temperature",
                    "metadata",
                }
                if columns and not required.issubset(columns):
                    raise ValueError("Incompatible cache schema; use a new v0.2 database path")
                if columns and conn.execute("SELECT 1 FROM cache_entries LIMIT 1").fetchone():
                    manifest_table = conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='cache_manifest'"
                    ).fetchone()
                    if (
                        not manifest_table
                        or not conn.execute("SELECT 1 FROM cache_manifest WHERE id=1").fetchone()
                    ):
                        raise ValueError(
                            "Missing storage manifest in nonempty cache; use a new store"
                        )
                # Begin inside executescript, which otherwise commits an existing transaction.
                conn.executescript("BEGIN IMMEDIATE;\n" + self.SCHEMA)
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS cache_manifest (id INTEGER PRIMARY KEY CHECK(id=1), data TEXT NOT NULL)"
                )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")
        except BaseException:
            conn.rollback()
            self.close()
            raise

    def read_manifest(self):
        row = (
            self._get_connection().execute("SELECT data FROM cache_manifest WHERE id=1").fetchone()
        )
        return json.loads(row[0]) if row else None

    def ensure_manifest(self, manifest: dict) -> None:
        from longcache.core.identity import canonical

        encoded = canonical(manifest)
        with self._lock, self._get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO cache_manifest (id, data) VALUES (1, ?)", (encoded,)
            )
            if conn.execute("SELECT data FROM cache_manifest WHERE id=1").fetchone()[0] != encoded:
                raise ValueError("Incompatible embedding or storage generation; use a new store")

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a thread-local database connection.

        Each thread gets its own connection to avoid SQLite threading issues.
        Connections are cached per-thread for performance.

        Returns:
            sqlite3.Connection instance.
        """
        if not hasattr(self._local, "connection") or self._local.connection is None:
            conn = sqlite3.connect(self._db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            self._local.connection = conn
        return self._local.connection

    def close(self) -> None:
        """Close the connection owned by the calling thread."""
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

    def _entry_to_row(self, entry: CacheEntry) -> dict:
        """Convert a CacheEntry to a row dict for database insertion."""
        vector_bytes = entry.vector.tobytes() if entry.vector is not None else b""
        response_json = json.dumps(entry.response, allow_nan=False)

        return {
            "namespace": entry.namespace,
            "query": entry.query,
            "vector": vector_bytes,
            "response": response_json,
            "created_at": entry.created_at,
            "expires_at": entry.expires_at,
            "hit_count": entry.hit_count,
            "model": entry.model,
            "system_prompt": entry.system_prompt,
            "temperature": entry.temperature,
            "metadata": json.dumps(entry.metadata, sort_keys=True, allow_nan=False),
        }

    def _row_to_entry(self, row) -> CacheEntry:
        """Convert a database row to a CacheEntry."""
        vector = None
        if row["vector"]:
            vector = np.frombuffer(row["vector"], dtype=np.float32).copy()

        response = row["response"]
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                pass  # Keep as string if not valid JSON

        model = row["model"] if "model" in row.keys() else ""
        system_prompt = row["system_prompt"] if "system_prompt" in row.keys() else ""
        temperature = (
            float(row["temperature"])
            if "temperature" in row.keys() and row["temperature"] is not None
            else -1.0
        )

        tags = []
        try:
            conn = self._get_connection()
            tag_rows = conn.execute(
                "SELECT tag FROM cache_tags WHERE entry_id = ?", (row["id"],)
            ).fetchall()
            tags = [t["tag"] for t in tag_rows]
        except Exception:
            pass

        return CacheEntry(
            id=row["id"],
            namespace=row["namespace"],
            query=row["query"],
            vector=vector,
            response=response,
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            hit_count=row["hit_count"],
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            tags=tags,
            metadata=json.loads(row["metadata"]),
        )

    def store(self, entry: CacheEntry) -> int:
        """
        Store a cache entry in the SQLite database.

        Args:
            entry: The CacheEntry to store.

        Returns:
            The auto-generated integer ID of the stored entry.
        """
        row = self._entry_to_row(entry)

        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                """
                INSERT INTO cache_entries (namespace, query, vector, response, created_at, expires_at, hit_count, model, system_prompt, temperature, metadata)
                VALUES (:namespace, :query, :vector, :response, :created_at, :expires_at, :hit_count, :model, :system_prompt, :temperature, :metadata)
                """,
                row,
            )
            entry_id = cursor.lastrowid
            if entry.tags:
                tag_rows = [(entry_id, tag) for tag in entry.tags]
                conn.executemany(
                    "INSERT INTO cache_tags (entry_id, tag) VALUES (?, ?)",
                    tag_rows,
                )
            conn.commit()
            return entry_id

    def retrieve(self, entry_id: int) -> Optional[CacheEntry]:
        """
        Retrieve a cache entry by its ID.

        Args:
            entry_id: The unique ID of the entry.

        Returns:
            CacheEntry if found and not expired, None otherwise.
        """
        with self._lock:
            conn = self._get_connection()
            row = conn.execute("SELECT * FROM cache_entries WHERE id = ?", (entry_id,)).fetchone()

            if row is None:
                return None

            entry = self._row_to_entry(row)
            if entry.is_expired():
                # Auto-delete expired entries on retrieval
                self.delete(entry_id)
                return None

            return entry

    def delete(self, entry_id: int) -> bool:
        """
        Delete a cache entry by its ID.

        Args:
            entry_id: The ID of the entry to delete.

        Returns:
            True if the entry was deleted, False if not found.
        """
        with self._lock:
            conn = self._get_connection()
            conn.execute("DELETE FROM cache_tags WHERE entry_id = ?", (entry_id,))
            cursor = conn.execute("DELETE FROM cache_entries WHERE id = ?", (entry_id,))
            conn.commit()
            return cursor.rowcount > 0

    def delete_namespace(self, namespace: str) -> int:
        """
        Delete all entries in a namespace.

        Args:
            namespace: The namespace to clear.

        Returns:
            The number of entries deleted.
        """
        with self._lock:
            conn = self._get_connection()
            rows = conn.execute(
                "SELECT id FROM cache_entries WHERE namespace = ?", (namespace,)
            ).fetchall()
            if not rows:
                return 0
            entry_ids = [r["id"] for r in rows]
            placeholders = ",".join("?" * len(entry_ids))
            conn.execute(
                f"DELETE FROM cache_tags WHERE entry_id IN ({placeholders})",
                entry_ids,
            )
            cursor = conn.execute("DELETE FROM cache_entries WHERE namespace = ?", (namespace,))
            conn.commit()
            return cursor.rowcount

    def delete_by_tag(self, tag: str, namespace: str) -> int:
        """
        Delete all entries in a namespace that carry the given tag.

        Args:
            tag: The tag string to match.
            namespace: Scope the deletion to this namespace.

        Returns:
            The number of entries deleted.
        """
        with self._lock:
            conn = self._get_connection()
            rows = conn.execute(
                """
                SELECT DISTINCT t.entry_id FROM cache_tags t
                JOIN cache_entries e ON e.id = t.entry_id
                WHERE t.tag = ? AND e.namespace = ?
                """,
                (tag, namespace),
            ).fetchall()
            if not rows:
                return 0
            entry_ids = [r["entry_id"] for r in rows]
            placeholders = ",".join("?" * len(entry_ids))
            conn.execute(
                f"DELETE FROM cache_tags WHERE entry_id IN ({placeholders})",
                entry_ids,
            )
            conn.execute(
                f"DELETE FROM cache_entries WHERE id IN ({placeholders})",
                entry_ids,
            )
            conn.commit()
            return len(entry_ids)

    def clear(self) -> int:
        """
        Clear all entries across all namespaces.

        Returns:
            The total number of entries deleted.
        """
        with self._lock:
            conn = self._get_connection()
            conn.execute("DELETE FROM cache_tags")
            count = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
            conn.execute("DELETE FROM cache_entries")
            conn.commit()
            return count

    def count(self, namespace: str) -> int:
        """
        Count entries in a namespace.

        Only counts non-expired entries.

        Args:
            namespace: The namespace to count.

        Returns:
            The number of non-expired entries.
        """
        with self._lock:
            conn = self._get_connection()
            now = time.time()
            row = conn.execute(
                """
                SELECT COUNT(*) FROM cache_entries
                WHERE namespace = ? AND (expires_at IS NULL OR expires_at > ?)
                """,
                (namespace, now),
            ).fetchone()
            return row[0]

    def list_entries(self, namespace: str) -> list[CacheEntry]:
        """
        List all non-expired entries in a namespace.

        Args:
            namespace: The namespace to list.

        Returns:
            List of CacheEntry objects with vectors populated.
        """
        with self._lock:
            conn = self._get_connection()
            now = time.time()
            rows = conn.execute(
                """
                SELECT * FROM cache_entries
                WHERE namespace = ? AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY created_at DESC
                """,
                (namespace, now),
            ).fetchall()
            return [self._row_to_entry(row) for row in rows]

    def list_all_entries(self) -> list[CacheEntry]:
        """List live entries across every namespace for index reconstruction."""
        with self._lock:
            rows = (
                self._get_connection()
                .execute(
                    "SELECT * FROM cache_entries WHERE expires_at IS NULL OR expires_at > ?",
                    (time.time(),),
                )
                .fetchall()
            )
            return [self._row_to_entry(row) for row in rows]

    def update_hit_count(self, entry_id: int) -> None:
        """
        Increment the hit count for a cache entry.

        Args:
            entry_id: The ID of the entry to update.
        """
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                "UPDATE cache_entries SET hit_count = hit_count + 1 WHERE id = ?",
                (entry_id,),
            )
            conn.commit()

    def evict_oldest(self, namespace: str, count: int) -> int:
        """
        Evict the oldest entries in a namespace.

        Deletes entries with the oldest created_at timestamps.

        Args:
            namespace: The namespace to evict from.
            count: The number of entries to evict.

        Returns:
            The number of entries actually evicted.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                """
                DELETE FROM cache_entries
                WHERE id IN (
                    SELECT id FROM cache_entries
                    WHERE namespace = ?
                    ORDER BY created_at ASC
                    LIMIT ?
                )
                """,
                (namespace, count),
            )
            conn.commit()
            return cursor.rowcount

    def get_expired(self, namespace: str) -> list[int]:
        """
        Get IDs of all expired entries in a namespace.

        Args:
            namespace: The namespace to check.

        Returns:
            List of expired entry IDs.
        """
        with self._lock:
            conn = self._get_connection()
            now = time.time()
            rows = conn.execute(
                """
                SELECT id FROM cache_entries
                WHERE namespace = ? AND expires_at IS NOT NULL AND expires_at <= ?
                """,
                (namespace, now),
            ).fetchall()
            return [row[0] for row in rows]

    def export_all(self) -> list[dict]:
        """
        Export all cache entries as dictionaries.

        Used by the CLI export command.

        Returns:
            List of dictionaries with entry data.
        """
        with self._lock:
            conn = self._get_connection()
            rows = conn.execute("SELECT * FROM cache_entries ORDER BY created_at DESC").fetchall()
            results = []
            for row in rows:
                entry = self._row_to_entry(row)
                results.append(
                    {
                        "id": entry.id,
                        "namespace": entry.namespace,
                        "query": entry.query,
                        "response": entry.response,
                        "created_at": entry.created_at,
                        "expires_at": entry.expires_at,
                        "hit_count": entry.hit_count,
                    }
                )
            return results

    def sweep_expired(self) -> int:
        """
        Delete all expired entries across all namespaces.
        Returns the number of entries deleted.
        """
        with self._lock:
            conn = self._get_connection()
            now = time.time()
            cursor = conn.execute(
                "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,)
            )
            conn.commit()
            return cursor.rowcount

    def increment_hits(self) -> None:
        with self._lock:
            self._get_connection().execute(
                "UPDATE cache_stats SET value = value + 1 WHERE key = 'hits'"
            )
            self._get_connection().commit()

    def increment_misses(self, saved_ms: int = 0) -> None:
        with self._lock:
            conn = self._get_connection()
            conn.execute("UPDATE cache_stats SET value=value+1 WHERE key='misses'")
            conn.execute(
                "UPDATE cache_stats SET value=value+? WHERE key='total_saved_ms'", (int(saved_ms),)
            )
            conn.commit()

    def load_stats(self) -> dict:
        with self._lock:
            rows = self._get_connection().execute("SELECT key, value FROM cache_stats").fetchall()
            return {row["key"]: row["value"] for row in rows}

    def __repr__(self) -> str:
        return f"SQLiteCacheBackend(db_path={self._db_path!r})"
