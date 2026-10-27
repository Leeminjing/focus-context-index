"""Immutable winner stores. No transaction is held during model work."""

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

from .api import IndexArtifact
from .segment_projection import SegmentProjectionRecord


class MemoryIndexStore:
    def __init__(self):
        self._segments: dict[tuple[str, str, str], str] = {}
        self._artifacts: dict[str, str] = {}

    async def get_segment(self, namespace, context_id, cache_key):
        raw = self._segments.get((namespace, context_id, cache_key))
        return SegmentProjectionRecord.model_validate_json(raw) if raw is not None else None

    async def put_segment(self, namespace, record):
        record = SegmentProjectionRecord.model_validate_json(record.model_dump_json())
        raw = self._segments.setdefault(
            (namespace, record.context_id, record.cache_key), record.model_dump_json()
        )
        return SegmentProjectionRecord.model_validate_json(raw)

    async def get_artifact(self, cache_key):
        raw = self._artifacts.get(cache_key)
        return IndexArtifact.model_validate_json(raw) if raw is not None else None

    async def put_artifact(self, artifact):
        artifact = IndexArtifact.model_validate_json(artifact.model_dump_json())
        raw = self._artifacts.setdefault(artifact.cache_key, artifact.model_dump_json())
        return IndexArtifact.model_validate_json(raw)


class SQLiteIndexStore:
    """One short connection/transaction per operation, on a worker thread.

    Suitable for restart recovery and cooperating local processes. Model-call leases
    and cross-process budget ownership remain responsibilities of the host.
    """

    def __init__(self, path: str | Path):
        if str(path) == ":memory:":
            raise ValueError("use MemoryIndexStore for an in-memory store")
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS index_cache ("
                "kind TEXT NOT NULL, scope TEXT NOT NULL, key TEXT NOT NULL, "
                "payload TEXT NOT NULL, PRIMARY KEY (kind, scope, key))"
            )

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def _operation(self, kind, scope, key, payload=None):
        with closing(self._connect()) as connection, connection:
            if payload is not None:
                connection.execute(
                    "INSERT OR IGNORE INTO index_cache VALUES (?, ?, ?, ?)",
                    (kind, scope, key, payload),
                )
            row = connection.execute(
                "SELECT payload FROM index_cache WHERE kind=? AND scope=? AND key=?",
                (kind, scope, key),
            ).fetchone()
            return row[0] if row is not None else None

    @staticmethod
    def _scope(namespace, context_id):
        from .contracts import stable_expansion_hash

        return stable_expansion_hash("segment-store-scope-v1", namespace, context_id)

    async def get_segment(self, namespace, context_id, cache_key):
        raw = await asyncio.to_thread(
            self._operation, "segment", self._scope(namespace, context_id), cache_key
        )
        return SegmentProjectionRecord.model_validate_json(raw) if raw is not None else None

    async def put_segment(self, namespace, record):
        record = SegmentProjectionRecord.model_validate_json(record.model_dump_json())
        raw = await asyncio.to_thread(
            self._operation,
            "segment",
            self._scope(namespace, record.context_id),
            record.cache_key,
            record.model_dump_json(),
        )
        return SegmentProjectionRecord.model_validate_json(raw)

    async def get_artifact(self, cache_key):
        raw = await asyncio.to_thread(self._operation, "artifact", "", cache_key)
        return IndexArtifact.model_validate_json(raw) if raw is not None else None

    async def put_artifact(self, artifact):
        artifact = IndexArtifact.model_validate_json(artifact.model_dump_json())
        raw = await asyncio.to_thread(
            self._operation, "artifact", "", artifact.cache_key, artifact.model_dump_json()
        )
        return IndexArtifact.model_validate_json(raw)
