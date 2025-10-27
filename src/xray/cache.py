from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd
from diskcache import Cache

# Define the cache directory in the project root
CACHE_DIR = Path(__file__).parent.parent.parent / ".xray_cache"

# Initialize the cache
cache = Cache(CACHE_DIR)


def get_df_hash(df: pd.DataFrame) -> str:
    """Creates a stable hash of a DataFrame's contents."""
    return str(pd.util.hash_pandas_object(df).sum())


def cached(func):
    """A decorator to cache function results based on their arguments."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a stable key based on function name and arguments
        key_parts = [func.__name__]
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                key_parts.append(get_df_hash(arg))
            else:
                key_parts.append(str(arg))
        for k, v in kwargs.items():
            if isinstance(v, pd.DataFrame):
                key_parts.append(f"{k}={get_df_hash(v)}")
            else:
                key_parts.append(f"{k}={v}")

        key = "|".join(key_parts)

        # Check if the result is in the cache
        if key in cache:
            print(f"[Cache] HIT: Loading result for {func.__name__}")
            return cache[key]

        # If not, run the function and cache the result
        print(f"[Cache] MISS: Running {func.__name__} and caching result.")
        result = func(*args, **kwargs)
        cache[key] = result
        return result

    return wrapper
