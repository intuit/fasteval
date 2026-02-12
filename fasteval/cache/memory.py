"""In-memory LRU cache for evaluation results."""

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from typing import Any, Dict, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheEntry(BaseModel, Generic[T]):
    """A single cache entry with metadata."""

    key: str
    value: Any  # Generic T doesn't work well with Pydantic
    hit_count: int = 0


class CacheStats(BaseModel):
    """Cache statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class MemoryCache:
    """
    Thread-safe in-memory LRU cache.

    Uses OrderedDict for O(1) LRU operations.

    Example:
        cache = MemoryCache(max_size=1000)

        # Store a value
        cache.set("key1", {"score": 0.95})

        # Retrieve (moves to front of LRU)
        value = cache.get("key1")

        # Check stats
        print(cache.stats)
    """

    def __init__(self, max_size: int = 10000) -> None:
        """
        Initialize cache.

        Args:
            max_size: Maximum number of entries (0 = unlimited)
        """
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.RLock()
        self._stats = CacheStats(max_size=max_size)

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.size = len(self._cache)
            return self._stats.model_copy()

    def _make_key(self, *args: Any, **kwargs: Any) -> str:
        """
        Create a deterministic cache key from arguments.

        Handles nested dicts, lists, and Pydantic models.
        """

        def serialize(obj: Any) -> Any:
            if isinstance(obj, BaseModel):
                return obj.model_dump()
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in sorted(obj.items())}
            elif isinstance(obj, (list, tuple)):
                return [serialize(v) for v in obj]
            return obj

        key_data = {"args": serialize(args), "kwargs": serialize(kwargs)}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache, marking as recently used.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._stats.hits += 1
                return self._cache[key]
            else:
                self._stats.misses += 1
                return None

    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.

        If cache is full, evicts least recently used entry.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            if key in self._cache:
                # Update existing, move to end
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                # Check if eviction needed
                if self._max_size > 0 and len(self._cache) >= self._max_size:
                    # Remove oldest (first) item
                    evicted_key, _ = self._cache.popitem(last=False)
                    self._stats.evictions += 1
                    logger.debug(f"Cache evicted: {evicted_key[:16]}...")

                self._cache[key] = value

    def get_or_set(self, key: str, factory: Any) -> Any:
        """
        Get value or compute and cache it.

        Args:
            key: Cache key
            factory: Callable or value to use if key not found

        Returns:
            Cached or computed value
        """
        with self._lock:
            value = self.get(key)
            if value is not None:
                return value

            # Compute new value
            if callable(factory):
                value = factory()
            else:
                value = factory

            self.set(key, value)
            return value

    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if entry was deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats(max_size=self._max_size)

    def __len__(self) -> int:
        """Return number of cached entries."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if key is in cache (doesn't update LRU)."""
        with self._lock:
            return key in self._cache


# Global cache instance
_global_cache: Optional[MemoryCache] = None


def get_cache(max_size: int = 10000) -> MemoryCache:
    """
    Get or create the global cache instance.

    Args:
        max_size: Maximum cache size (only used on first call)

    Returns:
        Global MemoryCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = MemoryCache(max_size=max_size)
    return _global_cache


def clear_cache() -> None:
    """Clear the global cache."""
    global _global_cache
    if _global_cache:
        _global_cache.clear()
