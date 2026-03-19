"""Tests for fasteval.cache.memory."""

import pytest
from pydantic import BaseModel

import fasteval.cache.memory as cache_module
from fasteval.cache.memory import (
    CacheStats,
    MemoryCache,
    clear_cache,
    get_cache,
)


class SampleModel(BaseModel):
    name: str
    value: int


class TestCacheStats:
    def test_hit_rate_with_data(self):
        stats = CacheStats(hits=7, misses=3)
        assert stats.hit_rate == 0.7

    def test_hit_rate_zero_total(self):
        stats = CacheStats(hits=0, misses=0)
        assert stats.hit_rate == 0.0


class TestMemoryCache:
    def test_basic_get_set(self):
        cache = MemoryCache(max_size=10)
        cache.set("k1", "v1")
        assert cache.get("k1") == "v1"

    def test_get_miss(self):
        cache = MemoryCache(max_size=10)
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        cache = MemoryCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # evicts "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_lru_access_order(self):
        cache = MemoryCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # moves "a" to most recent
        cache.set("c", 3)  # evicts "b" (least recent)
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_update_existing_key(self):
        cache = MemoryCache(max_size=10)
        cache.set("k1", "old")
        cache.set("k1", "new")
        assert cache.get("k1") == "new"
        assert len(cache) == 1

    def test_get_or_set_callable(self):
        cache = MemoryCache(max_size=10)
        result = cache.get_or_set("k1", lambda: 42)
        assert result == 42
        # Second call returns cached value
        result = cache.get_or_set("k1", lambda: 99)
        assert result == 42

    def test_get_or_set_non_callable(self):
        cache = MemoryCache(max_size=10)
        result = cache.get_or_set("k1", "static_value")
        assert result == "static_value"

    def test_delete_existing(self):
        cache = MemoryCache(max_size=10)
        cache.set("k1", "v1")
        assert cache.delete("k1") is True
        assert cache.get("k1") is None

    def test_delete_missing(self):
        cache = MemoryCache(max_size=10)
        assert cache.delete("nonexistent") is False

    def test_clear(self):
        cache = MemoryCache(max_size=10)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert len(cache) == 0
        assert cache.get("a") is None

    def test_stats_property(self):
        cache = MemoryCache(max_size=100)
        cache.set("k1", "v1")
        cache.get("k1")  # hit
        cache.get("k2")  # miss
        stats = cache.stats
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.size == 1
        assert stats.max_size == 100

    def test_eviction_stats(self):
        cache = MemoryCache(max_size=1)
        cache.set("a", 1)
        cache.set("b", 2)  # evicts "a"
        stats = cache.stats
        assert stats.evictions == 1

    def test_len(self):
        cache = MemoryCache(max_size=10)
        assert len(cache) == 0
        cache.set("a", 1)
        assert len(cache) == 1

    def test_contains(self):
        cache = MemoryCache(max_size=10)
        cache.set("a", 1)
        assert "a" in cache
        assert "b" not in cache

    def test_make_key_basic(self):
        cache = MemoryCache()
        key1 = cache._make_key("arg1", key="val")
        key2 = cache._make_key("arg1", key="val")
        key3 = cache._make_key("arg2", key="val")
        assert key1 == key2
        assert key1 != key3

    def test_make_key_with_pydantic_model(self):
        cache = MemoryCache()
        model = SampleModel(name="test", value=42)
        key = cache._make_key(model)
        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 hex digest

    def test_make_key_with_dict_and_list(self):
        cache = MemoryCache()
        key = cache._make_key({"a": [1, 2, 3]})
        assert isinstance(key, str)

    def test_unlimited_cache(self):
        cache = MemoryCache(max_size=0)
        for i in range(100):
            cache.set(str(i), i)
        assert len(cache) == 100


class TestGlobalCache:
    def setup_method(self):
        cache_module._global_cache = None

    def test_get_cache_singleton(self):
        c1 = get_cache()
        c2 = get_cache()
        assert c1 is c2

    def test_clear_cache(self):
        cache = get_cache()
        cache.set("k1", "v1")
        clear_cache()
        assert cache.get("k1") is None

    def test_clear_cache_when_none(self):
        cache_module._global_cache = None
        clear_cache()  # Should not raise
