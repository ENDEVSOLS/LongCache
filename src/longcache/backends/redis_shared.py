"""Redis-authoritative storage for opt-in multi-process cache use.

The local FAISS index is a read accelerator. Redis serializes mutations and
provides a revision that every reader checks before consulting that index.
"""

from __future__ import annotations

import hashlib
import json
import time
from uuid import uuid4

from longcache.backends.redis import RedisCacheBackend
from longcache.core.entry import CacheEntry
from longcache.core.identity import canonical


_STORE = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return redis.error_reply('incompatible manifest') end
if redis.call('EXISTS', KEYS[11]) == 1 then return redis.error_reply('shared generation is uncertain') end
if redis.call('GET', KEYS[2]) ~= ARGV[2] then return 0 end
if ARGV[10] ~= '' and redis.call('GET', ARGV[10]) ~= ARGV[11] then return 0 end
if redis.call('SCARD', KEYS[4]) >= tonumber(ARGV[8]) then return -1 end
if redis.call('SCARD', KEYS[6]) >= tonumber(ARGV[9]) then return -1 end
redis.call('SET', KEYS[11], '1')
local id = redis.call('INCR', KEYS[3])
local key = ARGV[3] .. ':' .. id
local fields = cjson.decode(ARGV[4])
for name, value in pairs(fields) do redis.call('HSET', key, name, value) end
redis.call('HSET', key, 'id', id)
redis.call('SADD', KEYS[4], id)
redis.call('SADD', KEYS[5], ARGV[5])
redis.call('SADD', KEYS[6], id)
redis.call('HSET', KEYS[9], id, ARGV[5])
redis.call('HSET', KEYS[10], id, ARGV[6])
local tags = cjson.decode(ARGV[6])
for _, tag_key in ipairs(tags) do redis.call('SADD', tag_key, id) end
local ttl = tonumber(ARGV[7])
if ttl > 0 then redis.call('PEXPIRE', key, ttl) end
local revision = redis.call('INCR', KEYS[7])
redis.call('HSET', key, 'commit_seq', revision)
redis.call('XADD', KEYS[8], revision .. '-0', 'type', 'store', 'id', id)
redis.call('XTRIM', KEYS[8], 'MAXLEN', '~', 100000)
redis.call('DEL', KEYS[11])
return id
"""

_REMOVE = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return redis.error_reply('incompatible manifest') end
if redis.call('EXISTS', KEYS[8]) == 1 then return redis.error_reply('shared generation is uncertain') end
local ids = cjson.decode(ARGV[2])
redis.call('SET', KEYS[8], '1')
local count = 0
for _, id in ipairs(ids) do
    local key = ARGV[3] .. ':' .. id
    local namespace = redis.call('HGET', KEYS[6], id)
    if namespace then
        local tags = redis.call('HGET', KEYS[7], id)
        if tags then
            for _, tag_key in ipairs(cjson.decode(tags)) do
                redis.call('SREM', tag_key, id)
            end
        end
        redis.call('SREM', ARGV[5] .. ':' .. namespace, id)
        local existed = redis.call('EXISTS', key)
        redis.call('DEL', key)
        redis.call('SREM', KEYS[2], id)
        redis.call('HDEL', KEYS[6], id)
        redis.call('HDEL', KEYS[7], id)
        count = count + existed
    else
        redis.call('SREM', KEYS[2], id)
    end
end
if count > 0 or ARGV[6] == 'invalidate' then
    local revision = redis.call('INCR', KEYS[3])
    redis.call('XADD', KEYS[4], revision .. '-0', 'type', ARGV[6])
    redis.call('XTRIM', KEYS[4], 'MAXLEN', '~', 100000)
    if ARGV[6] == 'invalidate' then redis.call('SET', ARGV[7], revision) end
end
if ARGV[6] == 'invalidate' then redis.call('INCR', KEYS[5]) end
redis.call('DEL', KEYS[8])
return count
"""

_VALIDATE = """
if redis.call('EXISTS', KEYS[4]) == 1 then return 0 end
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
if redis.call('GET', KEYS[2]) ~= ARGV[2] then return 0 end
if redis.call('HGET', KEYS[3], 'namespace') ~= ARGV[3] then return 0 end
if redis.call('HGET', KEYS[3], 'response') ~= ARGV[4] then return 0 end
if redis.call('HGET', KEYS[3], 'metadata') ~= ARGV[5] then return 0 end
local birth = tonumber(redis.call('HGET', KEYS[3], 'commit_seq'))
if not birth then return 0 end
if birth <= tonumber(redis.call('GET', ARGV[6]) or '0') then return 0 end
if birth <= tonumber(redis.call('GET', ARGV[7]) or '0') then return 0 end
local tags = redis.call('HGET', KEYS[3], 'tags')
if tags then
    for _, tag in ipairs(cjson.decode(tags)) do
        local key = ARGV[8] .. ':' .. ARGV[3] .. ':' .. redis.sha1hex(tag)
        if birth <= tonumber(redis.call('GET', key) or '0') then return 0 end
    end
end
local ttl = redis.call('PTTL', KEYS[3])
if ttl == -2 or ttl == 0 then return 0 end
return 1
"""


class SharedRedisCacheBackend(RedisCacheBackend):
    """A distinct generation whose mutations are visible to all participants."""

    def __init__(
        self,
        *args,
        shared_max_entries: int = 50_000,
        distributed_lease_seconds: float = 15.0,
        distributed_renew_interval_seconds: float = 5.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        cache_id, generation = kwargs["cache_id"], kwargs["generation"]
        self.ROOT = f"longcache:shared-v1:{cache_id}:{generation}"
        self.KEY_PREFIX = self.ROOT + ":entry"
        self.NAMESPACE_SET = self.ROOT + ":namespaces"
        self.ID_COUNTER = self.ROOT + ":id_counter"
        self.REVISION = self.ROOT + ":revision"
        self.EPOCH = self.ROOT + ":epoch"
        self.JOURNAL = self.ROOT + ":journal"
        self.ALL_IDS = self.ROOT + ":ids"
        self.DIRTY = self.ROOT + ":dirty"
        self._shared_max_entries = shared_max_entries
        self._lease_seconds = distributed_lease_seconds
        self._renew_interval = distributed_renew_interval_seconds
        self._manifest = None

    def _namespace_key(self, namespace: str) -> str:
        return f"{self.ROOT}:ns:{namespace}"

    def _tag_key(self, namespace: str, tag: str) -> str:
        digest = hashlib.sha1(tag.encode("utf-8")).hexdigest()
        return f"{self.ROOT}:tag:{namespace}:{digest}"

    def _namespace_watermark(self, namespace: str) -> str:
        return f"{self.ROOT}:wm:ns:{namespace}"

    def _tag_watermark(self, namespace: str, tag: str) -> str:
        return f"{self.ROOT}:wm:tag:{namespace}:{hashlib.sha1(tag.encode('utf-8')).hexdigest()}"

    def ensure_manifest(self, manifest: dict) -> None:
        expected = canonical(
            {
                **manifest,
                "coordination": "shared-v1",
                "max_entries": self._max_entries,
                "shared_max_entries": self._shared_max_entries,
                "lease_seconds": self._lease_seconds,
                "renew_interval": self._renew_interval,
            }
        )
        key = self.ROOT + ":manifest"
        existing = self._client.get(key)
        if existing is None and self._client.exists(self.ID_COUNTER, self.ALL_IDS, self.JOURNAL):
            raise ValueError("Missing shared manifest in nonempty generation")
        self._client.set(key, expected, nx=True)
        if self._client.get(key) != expected:
            raise ValueError("Incompatible shared cache generation")
        self._client.set(self.EPOCH, 0, nx=True)
        self._manifest = expected

    def read_manifest(self):
        raw = self._client.get(self.ROOT + ":manifest")
        return json.loads(raw) if raw else None

    def revision(self) -> int:
        self._healthy()
        return int(self._client.get(self.REVISION) or 0)

    def _healthy(self) -> None:
        if self._client.exists(self.DIRTY):
            raise ValueError("Shared generation has an uncertain mutation")

    def events(self, start: int, end: int, limit: int = 256):
        """Return a bounded, ordered slice of authoritative mutations."""
        self._healthy()
        return [
            (int(event_id.split("-", 1)[0]), fields)
            for event_id, fields in self._client.xrange(
                self.JOURNAL, min=f"{start}-0", max=f"{end}-0", count=limit
            )
        ]

    def epoch(self) -> int:
        self._healthy()
        return int(self._client.get(self.EPOCH) or 0)

    def store(
        self,
        entry: CacheEntry,
        *,
        expected_epoch: int | None = None,
        lease_key: str = "",
        lease_token: str = "",
    ) -> int:
        if self._manifest is None:
            raise ValueError("Shared manifest was not initialized")
        if expected_epoch is None:
            expected_epoch = self.epoch()
        encoded_tags = json.dumps([self._tag_key(entry.namespace, tag) for tag in entry.tags or []])
        data = {
            "namespace": entry.namespace,
            "query": entry.query,
            "vector": self._vector_to_str(entry.vector) if entry.vector is not None else "",
            "response": json.dumps(entry.response, allow_nan=False),
            "created_at": str(entry.created_at),
            "expires_at": str(entry.expires_at) if entry.expires_at is not None else "",
            "hit_count": str(entry.hit_count),
            "model": entry.model,
            "system_prompt": entry.system_prompt,
            "temperature": str(entry.temperature),
            "tags": json.dumps(list(entry.tags or [])),
            "metadata": json.dumps(entry.metadata, sort_keys=True, allow_nan=False),
        }
        ttl_ms = max(1, int((entry.expires_at - time.time()) * 1000)) if entry.expires_at else 0
        identifier = int(
            self._client.eval(
                _STORE,
                11,
                self.ROOT + ":manifest",
                self.EPOCH,
                self.ID_COUNTER,
                self._namespace_key(entry.namespace),
                self.NAMESPACE_SET,
                self.ALL_IDS,
                self.REVISION,
                self.JOURNAL,
                self.ROOT + ":entry_partition",
                self.ROOT + ":entry_tags",
                self.DIRTY,
                self._manifest,
                str(expected_epoch),
                self.KEY_PREFIX,
                json.dumps(data, allow_nan=False),
                entry.namespace,
                encoded_tags,
                ttl_ms,
                self._max_entries,
                self._shared_max_entries,
                lease_key,
                lease_token,
            )
        )
        return identifier

    def acquire_flight(self, exact_key: str, ttl_ms: int) -> tuple[str, str | None]:
        self._healthy()
        key = f"{self.ROOT}:flight:{exact_key}"
        token = uuid4().hex
        acquired = self._client.eval(
            "if redis.call('SET', KEYS[1], ARGV[1], 'NX', 'PX', ARGV[2]) then "
            "redis.call('DEL', KEYS[2]); return 1 end return 0",
            2,
            key,
            key + ":terminal",
            token,
            ttl_ms,
        )
        return key, token if acquired else None

    def finish_flight(self, key: str, token: str, outcome: str) -> bool:
        if outcome not in ("error", "cancelled", "non_shareable"):
            raise ValueError("Invalid terminal flight outcome")
        return bool(
            self._client.eval(
                "if redis.call('GET', KEYS[1]) == ARGV[1] then "
                "redis.call('SET', KEYS[2], ARGV[2], 'PX', 30000); "
                "redis.call('DEL', KEYS[1]); return 1 end return 0",
                2,
                key,
                key + ":terminal",
                token,
                outcome,
            )
        )

    def flight_outcome(self, key: str) -> str | None:
        return self._client.get(key + ":terminal")

    def renew_flight(self, key: str, token: str, ttl_ms: int) -> bool:
        return bool(
            self._client.eval(
                "if redis.call('GET', KEYS[1]) == ARGV[1] then "
                "return redis.call('PEXPIRE', KEYS[1], ARGV[2]) end return 0",
                1,
                key,
                token,
                ttl_ms,
            )
        )

    def release_flight(self, key: str, token: str) -> None:
        self._client.eval(
            "if redis.call('GET', KEYS[1]) == ARGV[1] then "
            "return redis.call('DEL', KEYS[1]) end return 0",
            1,
            key,
            token,
        )

    def flight_active(self, key: str) -> bool:
        return bool(self._client.exists(key))

    def retrieve(self, entry_id: int) -> CacheEntry | None:
        self._healthy()
        pipe = self._client.pipeline(transaction=True)
        key = self._entry_key(entry_id)
        pipe.hgetall(key)
        pipe.pttl(key)
        raw, ttl_ms = pipe.execute()
        if not raw or ttl_ms == -2 or ttl_ms == 0:
            return None
        entry = self._data_to_entry(raw)
        entry.expires_at = time.time() + ttl_ms / 1000 if ttl_ms > 0 else None
        birth = int(raw.get("commit_seq", 0))
        if birth <= int(self._client.get(self.ROOT + ":wm:all") or 0):
            return None
        if birth <= int(self._client.get(self._namespace_watermark(entry.namespace)) or 0):
            return None
        for tag in entry.tags or []:
            if birth <= int(self._client.get(self._tag_watermark(entry.namespace, tag)) or 0):
                return None
        return entry

    def validate_hit(self, entry: CacheEntry, *, expected_epoch: int) -> bool:
        return bool(
            self._client.eval(
                _VALIDATE,
                4,
                self.ROOT + ":manifest",
                self.EPOCH,
                self._entry_key(entry.id),
                self.DIRTY,
                self._manifest,
                str(expected_epoch),
                entry.namespace,
                json.dumps(entry.response, allow_nan=False),
                json.dumps(entry.metadata, sort_keys=True, allow_nan=False),
                self.ROOT + ":wm:all",
                self._namespace_watermark(entry.namespace),
                self.ROOT + ":wm:tag",
            )
        )

    def _remove(self, ids, *, invalidate=False, watermark=None):
        if not ids and not invalidate:
            return 0
        return int(
            self._client.eval(
                _REMOVE,
                8,
                self.ROOT + ":manifest",
                self.ALL_IDS,
                self.REVISION,
                self.JOURNAL,
                self.EPOCH,
                self.ROOT + ":entry_partition",
                self.ROOT + ":entry_tags",
                self.DIRTY,
                self._manifest,
                json.dumps([str(identifier) for identifier in ids]),
                self.KEY_PREFIX,
                self.ROOT + ":tag",
                self.ROOT + ":ns",
                "invalidate" if invalidate else "delete",
                watermark or self.ROOT + ":wm:all",
            )
        )

    def delete(self, entry_id: int) -> bool:
        return bool(self._remove([entry_id]))

    def delete_namespace(self, namespace: str) -> int:
        return self._remove(
            self._client.smembers(self._namespace_key(namespace)),
            invalidate=True,
            watermark=self._namespace_watermark(namespace),
        )

    def delete_by_tag(self, tag: str, namespace: str) -> int:
        return self._remove(
            self._client.smembers(self._tag_key(namespace, tag)),
            invalidate=True,
            watermark=self._tag_watermark(namespace, tag),
        )

    def clear(self) -> int:
        if self._manifest is None:
            return 0
        return self._remove(self._client.smembers(self.ALL_IDS), invalidate=True)

    def update_hit_count(self, entry_id: int) -> None:
        self._client.eval(
            "if redis.call('EXISTS', KEYS[1]) == 1 then "
            "return redis.call('HINCRBY', KEYS[1], 'hit_count', 1) end return 0",
            1,
            self._entry_key(entry_id),
        )

    def count(self, namespace: str) -> int:
        return sum(
            self.retrieve(int(identifier)) is not None
            for identifier in self._client.smembers(self._namespace_key(namespace))
        )

    def evict_oldest(self, namespace: str, count: int) -> int:
        entries = sorted(self.list_entries(namespace), key=lambda item: item.created_at)
        return sum(self.delete(entry.id) for entry in entries[:count])

    def get_expired(self, namespace: str) -> list[int]:
        return []

    def sweep_expired(self) -> int:
        """Bound cleanup of expired and logically invalidated shared entries."""
        cursor, ids = self._client.sscan(
            self.ALL_IDS, cursor=getattr(self, "_scan_cursor", 0), count=256
        )
        self._scan_cursor = cursor
        invalid = [identifier for identifier in ids if self.retrieve(int(identifier)) is None]
        self._remove(invalid)
        return len(invalid)
