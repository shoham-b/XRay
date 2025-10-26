from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from diskcache import Cache


class DataFrameCache:
    """A simple cache wrapper for Pandas DataFrames backed by diskcache.

    This class encapsulates the cache directory and exposes small helpers for
    making deterministic keys and storing/loading DataFrames.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self._ensure_dir()

    # -------------------
    # Key helpers
    # -------------------
    @staticmethod
    def _json_dumps_canonical(obj: Any) -> str:
        """Dump JSON with sorted keys and stable formatting for hashing.

        Non-serializable objects should be converted by the caller to basic types.
        """
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @staticmethod
    def make_key(config: dict) -> str:
        """Make a stable cache key from a configuration dictionary.

        The dictionary should contain only JSON-serializable primitives
        (str, int, float, bool, None, list, dict).
        """
        canon = DataFrameCache._json_dumps_canonical(config)
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    # -------------------
    # IO helpers
    # -------------------
    def _ensure_dir(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_df(self, key: str) -> pd.DataFrame | None:
        """Load a cached Pandas DataFrame by key; return None on miss or error."""
        try:
            with Cache(self.cache_dir.as_posix()) as cache:
                obj = cache.get(key, default=None)
                if isinstance(obj, pd.DataFrame):
                    return obj
                return None
        except Exception:
            return None

    def save_df(self, df: pd.DataFrame, key: str) -> Path:
        """Save a Pandas DataFrame and return a logical cache path (cache_dir/key).

        Note: diskcache manages the underlying storage; the returned Path is for
        logging only.
        """
        with Cache(self.cache_dir.as_posix()) as cache:
            cache.set(key, df)
        return self.cache_dir / key


# ---------------------------------------------------------------------------
# Backward-compatible functional wrappers (deprecated)
# ---------------------------------------------------------------------------


def make_key(config: dict) -> str:
    """Deprecated functional wrapper around DataFrameCache.make_key."""
    return DataFrameCache.make_key(config)


def load_df(cache_dir: Path, key: str) -> pd.DataFrame | None:
    """Deprecated functional wrapper; prefer DataFrameCache(cache_dir).load_df(key)."""
    return DataFrameCache(cache_dir).load_df(key)


def save_df(df: pd.DataFrame, cache_dir: Path, key: str) -> Path:
    """Deprecated functional wrapper; prefer DataFrameCache(cache_dir).save_df(df, key)."""
    return DataFrameCache(cache_dir).save_df(df, key)
