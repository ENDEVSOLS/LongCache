"""Redis-authoritative storage for opt-in multi-process cache use.

The local FAISS index is a read accelerator. Redis serializes mutations and
provides a revision that every reader checks before consulting that index.
"""

from __future__ import annotations

import hashlib
from itertools import islice
import json
import time
from typing import Any, Iterable
from uuid import uuid4

from longcache.backends.redis import RedisCacheBackend
from longcache.core.entry import CacheEntry
from longcache.core.errors import CacheOperationError, SingleflightCapacityError
from longcache.core.identity import canonical


_STORE = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return redis.error_reply('incompatible manifest') end
if redis.call('EXISTS', KEYS[11]) == 1 then return redis.error_reply('shared generation is uncertain') end
local prior = redis.call('GET', KEYS[12])
if prior then
    local receipt = cjson.decode(prior)
    if receipt.digest ~= ARGV[12] then return redis.error_reply('operation receipt mismatch') end
    return {receipt.id, receipt.revision}
end
if redis.call('GET', KEYS[2]) ~= ARGV[2] then return 0 end
if ARGV[10] ~= '' and redis.call('GET', ARGV[10]) ~= ARGV[11] then return 0 end
local clock = redis.call('TIME')
local now = tonumber(clock[1]) * 1000 + math.floor(tonumber(clock[2]) / 1000)
local expired = redis.call('ZRANGEBYSCORE', KEYS[13], '-inf', now - 1, 'LIMIT', 0, 256)
for _, key in ipairs(expired) do
    redis.call('DEL', key)
    redis.call('ZREM', KEYS[13], key)
end
if redis.call('ZCARD', KEYS[13]) >= tonumber(ARGV[13]) then return -2 end
if redis.call('SISMEMBER', KEYS[5], ARGV[5]) == 0 and redis.call('SCARD', KEYS[5]) >= tonumber(ARGV[15]) then return -1 end
local tags = cjson.decode(ARGV[6])
local new_tags = 0
for _, tag_key in ipairs(tags) do
    if redis.call('SISMEMBER', KEYS[14], tag_key) == 0 then new_tags = new_tags + 1 end
end
if redis.call('SCARD', KEYS[14]) + new_tags > tonumber(ARGV[16]) then return -1 end
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
for _, tag_key in ipairs(tags) do
    redis.call('SADD', KEYS[14], tag_key)
    redis.call('SADD', tag_key, id)
end
local ttl = tonumber(ARGV[7])
if ttl > 0 then redis.call('PEXPIRE', key, ttl) end
local revision = redis.call('INCR', KEYS[7])
redis.call('HSET', key, 'commit_seq', revision)
redis.call('XADD', KEYS[8], revision .. '-0', 'type', 'store', 'id', id)
redis.call('XTRIM', KEYS[8], 'MAXLEN', '~', 100000)
redis.call('SET', KEYS[12], cjson.encode({id=id, revision=revision, digest=ARGV[12]}), 'PX', ARGV[14])
redis.call('ZADD', KEYS[13], now + tonumber(ARGV[14]), KEYS[12])
redis.call('DEL', KEYS[11])
return {id, revision}
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

_INVALIDATE = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return redis.error_reply('incompatible manifest') end
if redis.call('EXISTS', KEYS[5]) == 1 then return redis.error_reply('shared generation is uncertain') end
local prior = redis.call('GET', KEYS[6])
if prior then
    local receipt = cjson.decode(prior)
    if receipt.digest ~= ARGV[5] then return redis.error_reply('operation receipt mismatch') end
    return receipt.count
end
if tonumber(redis.call('GET', KEYS[2]) or '0') ~= tonumber(ARGV[2]) then return -1 end
local clock = redis.call('TIME')
local now = tonumber(clock[1]) * 1000 + math.floor(tonumber(clock[2]) / 1000)
local expired = redis.call('ZRANGEBYSCORE', KEYS[7], '-inf', now - 1, 'LIMIT', 0, 256)
for _, key in ipairs(expired) do
    redis.call('DEL', key)
    redis.call('ZREM', KEYS[7], key)
end
if redis.call('ZCARD', KEYS[7]) >= tonumber(ARGV[6]) then return -2 end
if ARGV[10] ~= '' and redis.call('SISMEMBER', KEYS[9], ARGV[10]) == 0 and redis.call('SCARD', KEYS[9]) >= tonumber(ARGV[11]) then return -3 end
if ARGV[8] ~= '' and redis.call('SISMEMBER', KEYS[8], ARGV[8]) == 0 and redis.call('SCARD', KEYS[8]) >= tonumber(ARGV[9]) then return -3 end
redis.call('SET', KEYS[5], '1')
local revision = redis.call('INCR', KEYS[2])
if ARGV[10] ~= '' then redis.call('SADD', KEYS[9], ARGV[10]) end
if ARGV[8] ~= '' then redis.call('SADD', KEYS[8], ARGV[8]) end
redis.call('SET', ARGV[3], revision)
redis.call('INCR', KEYS[4])
redis.call('XADD', KEYS[3], revision .. '-0', 'type', 'invalidate')
redis.call('XTRIM', KEYS[3], 'MAXLEN', '~', 100000)
redis.call('SET', KEYS[6], cjson.encode({count=tonumber(ARGV[4]), digest=ARGV[5]}), 'PX', ARGV[7])
redis.call('ZADD', KEYS[7], now + tonumber(ARGV[7]), KEYS[6])
redis.call('DEL', KEYS[5])
return tonumber(ARGV[4])
"""

_ACQUIRE_FLIGHT = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return redis.error_reply('incompatible manifest') end
if redis.call('EXISTS', KEYS[5]) == 1 then return redis.error_reply('shared generation is uncertain') end
local clock = redis.call('TIME')
local now = tonumber(clock[1]) * 1000 + math.floor(tonumber(clock[2]) / 1000)
redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', now - 1)
if redis.call('EXISTS', KEYS[2]) == 1 then return {0, 0} end
if redis.call('ZCARD', KEYS[3]) >= tonumber(ARGV[3]) then return {-1, 0} end
local token = ARGV[2] .. ':' .. redis.call('INCR', KEYS[4])
redis.call('SET', KEYS[2], token, 'PX', ARGV[4])
redis.call('ZADD', KEYS[3], now + tonumber(ARGV[4]), KEYS[2])
redis.call('DEL', KEYS[6])
return {1, token}
"""

_RENEW_FLIGHT = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
local clock = redis.call('TIME')
local now = tonumber(clock[1]) * 1000 + math.floor(tonumber(clock[2]) / 1000)
redis.call('PEXPIRE', KEYS[1], ARGV[2])
redis.call('ZADD', KEYS[2], now + tonumber(ARGV[2]), KEYS[1])
return 1
"""

_RELEASE_FLIGHT = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('DEL', KEYS[1])
redis.call('ZREM', KEYS[2], KEYS[1])
return 1
"""

_FINISH_FLIGHT = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('SET', KEYS[3], ARGV[2], 'PX', 30000)
redis.call('DEL', KEYS[1])
redis.call('ZREM', KEYS[2], KEYS[1])
return 1
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
        *args: Any,
        shared_max_entries: int = 50_000,
        shared_max_leases: int = 1024,
        shared_max_receipts: int = 100_000,
        shared_max_partitions: int = 50_000,
        shared_max_distinct_tags: int = 100_000,
        shared_max_tags_per_entry: int = 64,
        shared_max_tag_chars: int = 256,
        distributed_lease_seconds: float = 15.0,
        distributed_renew_interval_seconds: float = 5.0,
        **kwargs: Any,
    ) -> None:
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
        self._shared_max_leases = shared_max_leases
        self._shared_max_receipts = shared_max_receipts
        self._shared_max_partitions = shared_max_partitions
        self._shared_max_distinct_tags = shared_max_distinct_tags
        self._shared_max_tags_per_entry = shared_max_tags_per_entry
        self._shared_max_tag_chars = shared_max_tag_chars
        self.RECEIPTS = self.ROOT + ":receipts"
        self.TAGS = self.ROOT + ":tags"
        self.LEASES = self.ROOT + ":leases"
        self.LEASE_COUNTER = self.ROOT + ":lease_counter"
        self._lease_seconds = distributed_lease_seconds
        self._renew_interval = distributed_renew_interval_seconds
        self._manifest: str | None = None
        self._scan_cursor: int = 0

    def _namespace_key(self, namespace: str) -> str:
        return f"{self.ROOT}:ns:{namespace}"

    def _tag_key(self, namespace: str, tag: str) -> str:
        digest = hashlib.sha1(tag.encode("utf-8")).hexdigest()
        return f"{self.ROOT}:tag:{namespace}:{digest}"

    def _namespace_watermark(self, namespace: str) -> str:
        return f"{self.ROOT}:wm:ns:{namespace}"

    def _tag_watermark(self, namespace: str, tag: str) -> str:
        return f"{self.ROOT}:wm:tag:{namespace}:{hashlib.sha1(tag.encode('utf-8')).hexdigest()}"

    def ensure_manifest(self, manifest: dict[str, Any]) -> None:
        definition = {
            **manifest,
            "coordination": "shared-v1",
            "max_entries": self._max_entries,
            "shared_max_entries": self._shared_max_entries,
            "shared_max_leases": self._shared_max_leases,
            "shared_max_receipts": self._shared_max_receipts,
            "shared_max_partitions": self._shared_max_partitions,
            "shared_max_distinct_tags": self._shared_max_distinct_tags,
            "shared_max_tags_per_entry": self._shared_max_tags_per_entry,
            "shared_max_tag_chars": self._shared_max_tag_chars,
            "lease_seconds": self._lease_seconds,
            "renew_interval": self._renew_interval,
        }
        key = self.ROOT + ":manifest"
        existing = self._client.get(key)
        if existing is None and self._client.exists(
            self.ID_COUNTER,
            self.ALL_IDS,
            self.JOURNAL,
            self.EPOCH,
            self.REVISION,
            self.NAMESPACE_SET,
            self.TAGS,
            self.LEASES,
            self.RECEIPTS,
            self.DIRTY,
        ):
            raise ValueError("Missing shared manifest in nonempty generation")
        if existing is None:
            self._client.set(key, canonical({**definition, "incarnation": uuid4().hex}), nx=True)
        actual = self._client.get(key)
        if actual is None:
            raise ValueError("Shared generation lost its manifest during initialization")
        parsed = json.loads(actual)
        incarnation = parsed.pop("incarnation", None)
        if (
            not isinstance(incarnation, str)
            or len(incarnation) != 32
            or canonical(parsed) != canonical(definition)
        ):
            raise ValueError("Incompatible shared cache generation")
        self._client.set(self.EPOCH, 0, nx=True)
        self._manifest = actual

    def read_manifest(self) -> dict[str, Any] | None:
        raw = self._client.get(self.ROOT + ":manifest")
        if not raw:
            return None
        parsed: Any = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None

    def revision(self) -> int:
        self._healthy()
        return int(self._client.get(self.REVISION) or 0)

    def _healthy(self) -> None:
        if self._client.exists(self.DIRTY):
            raise ValueError("Shared generation has an uncertain mutation")
        if (
            self._manifest is not None
            and self._client.get(self.ROOT + ":manifest") != self._manifest
        ):
            raise ValueError("Shared generation changed or lost its manifest")

    def events(self, start: int, end: int, limit: int = 256) -> list[tuple[int, dict[str, Any]]]:
        """Return a bounded, ordered slice of authoritative mutations."""
        self._healthy()
        return [
            (int(str(event_id).split("-", 1)[0]), dict(fields))
            for event_id, fields in self._client.xrange(
                self.JOURNAL, min=f"{start}-0", max=f"{end}-0", count=limit
            )
        ]

    def epoch(self) -> int:
        self._healthy()
        return int(self._client.get(self.EPOCH) or 0)

    def store_shared(
        self,
        entry: CacheEntry,
        *,
        expected_epoch: int | None = None,
        lease_key: str = "",
        lease_token: str = "",
    ) -> tuple[int, int | None]:
        if self._manifest is None:
            raise ValueError("Shared manifest was not initialized")
        if expected_epoch is None:
            expected_epoch = self.epoch()
        encoded_tags = json.dumps(
            list(dict.fromkeys(self._tag_key(entry.namespace, tag) for tag in entry.tags or []))
        )
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
        payload = json.dumps(data, sort_keys=True, allow_nan=False)
        receipt_key = self.ROOT + ":receipt:" + uuid4().hex
        payload_digest = hashlib.sha256(payload.encode()).hexdigest()

        def commit() -> Any:
            return self._client.eval(
                _STORE,
                14,
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
                receipt_key,
                self.RECEIPTS,
                self.TAGS,
                self._manifest,
                str(expected_epoch),
                self.KEY_PREFIX,
                payload,
                entry.namespace,
                encoded_tags,
                ttl_ms,
                self._max_entries,
                self._shared_max_entries,
                lease_key,
                lease_token,
                payload_digest,
                self._shared_max_receipts,
                60_000,
                self._shared_max_partitions,
                self._shared_max_distinct_tags,
            )

        import redis

        try:
            result: Any = commit()
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
            try:
                result = commit()
            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as exc:
                raise CacheOperationError("store_indeterminate") from exc
        if isinstance(result, (list, tuple)):
            return int(result[0]), int(result[1])
        if int(result) == -2:
            return -1, None
        return int(result), None

    def store(self, entry: CacheEntry) -> int:
        identifier, _ = self.store_shared(entry)
        return identifier

    def acquire_flight(self, exact_key: str, ttl_ms: int) -> tuple[str, str | None]:
        self._healthy()
        key = f"{self.ROOT}:flight:{exact_key}"
        result: Any = self._client.eval(
            _ACQUIRE_FLIGHT,
            6,
            self.ROOT + ":manifest",
            key,
            self.LEASES,
            self.LEASE_COUNTER,
            self.DIRTY,
            key + ":terminal",
            self._manifest,
            uuid4().hex,
            self._shared_max_leases,
            ttl_ms,
        )
        if int(result[0]) == -1:
            raise SingleflightCapacityError("Shared lease capacity reached")
        return key, str(result[1]) if int(result[0]) == 1 else None

    def finish_flight(self, key: str, token: str, outcome: str) -> bool:
        if outcome not in ("error", "cancelled", "non_shareable"):
            raise ValueError("Invalid terminal flight outcome")
        return bool(
            self._client.eval(
                _FINISH_FLIGHT,
                3,
                key,
                self.LEASES,
                key + ":terminal",
                token,
                outcome,
            )
        )

    def flight_outcome(self, key: str) -> str | None:
        val = self._client.get(key + ":terminal")
        return str(val) if val is not None else None

    def renew_flight(self, key: str, token: str, ttl_ms: int) -> bool:
        return bool(
            self._client.eval(
                _RENEW_FLIGHT,
                2,
                key,
                self.LEASES,
                token,
                ttl_ms,
            )
        )

    def release_flight(self, key: str, token: str) -> None:
        self._client.eval(
            _RELEASE_FLIGHT,
            2,
            key,
            self.LEASES,
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
        if entry.id is None:
            return False
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

    def _remove(self, ids: Iterable[int | str]) -> int:
        total = 0
        iterator = iter(ids)
        while batch := list(islice(iterator, 256)):
            total += int(
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
                    json.dumps([str(identifier) for identifier in batch]),
                    self.KEY_PREFIX,
                    self.ROOT + ":tag",
                    self.ROOT + ":ns",
                    "delete",
                    self.ROOT + ":wm:all",
                )
            )
        return total

    def _count_live(self, key: str) -> int:
        cursor, count = 0, 0
        while True:
            cursor, identifiers = self._client.sscan(key, cursor=cursor, count=256)
            count += sum(self.retrieve(int(identifier)) is not None for identifier in identifiers)
            if cursor == 0:
                return count

    def _invalidate(
        self, membership: str, watermark: str, *, partition: str = "", tag_key: str = ""
    ) -> int:
        import redis

        for _ in range(5):
            before = self.revision()
            count = self._count_live(membership)
            receipt_key = self.ROOT + ":receipt:" + uuid4().hex
            payload_digest = hashlib.sha256(
                f"{membership}:{watermark}:{before}:{count}".encode()
            ).hexdigest()

            def commit() -> Any:
                return self._client.eval(
                    _INVALIDATE,
                    9,
                    self.ROOT + ":manifest",
                    self.REVISION,
                    self.JOURNAL,
                    self.EPOCH,
                    self.DIRTY,
                    receipt_key,
                    self.RECEIPTS,
                    self.TAGS,
                    self.NAMESPACE_SET,
                    self._manifest,
                    before,
                    watermark,
                    count,
                    payload_digest,
                    self._shared_max_receipts,
                    60_000,
                    tag_key,
                    self._shared_max_distinct_tags,
                    partition,
                    self._shared_max_partitions,
                )

            try:
                result = int(commit())
            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
                try:
                    result = int(commit())
                except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as exc:
                    raise CacheOperationError("invalidation_indeterminate") from exc
            if result == -2:
                raise CacheOperationError("receipt_capacity")
            if result == -3:
                raise CacheOperationError("metadata_capacity")
            if result >= 0:
                return result
        raise CacheOperationError("invalidation_conflict")

    def delete(self, entry_id: int) -> bool:
        return bool(self._remove([entry_id]))

    def delete_namespace(self, namespace: str) -> int:
        return self._invalidate(
            self._namespace_key(namespace),
            self._namespace_watermark(namespace),
            partition=namespace,
        )

    def delete_by_tag(self, tag: str, namespace: str) -> int:
        tag_key = self._tag_key(namespace, tag)
        return self._invalidate(
            tag_key, self._tag_watermark(namespace, tag), partition=namespace, tag_key=tag_key
        )

    def clear(self) -> int:
        if self._manifest is None:
            return 0
        return self._invalidate(self.ALL_IDS, self.ROOT + ":wm:all")

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
            for identifier in list(self._client.smembers(self._namespace_key(namespace)))
        )

    def evict_oldest(self, namespace: str, count: int) -> int:
        entries = sorted(self.list_entries(namespace), key=lambda item: item.created_at)
        return sum(self.delete(entry.id) for entry in entries[:count] if entry.id is not None)

    def get_expired(self, namespace: str) -> list[int]:
        return []

    def sweep_expired(self) -> int:
        """Bound cleanup of expired and logically invalidated shared entries."""
        cursor, ids = self._client.sscan(
            self.ALL_IDS, cursor=self._scan_cursor, count=256
        )
        self._scan_cursor = int(cursor)
        invalid = [identifier for identifier in ids if self.retrieve(int(identifier)) is None]
        self._remove(invalid)
        return len(invalid)
