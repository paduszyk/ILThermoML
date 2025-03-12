import os

from joblib import Memory

ilt_memory = Memory(
    location=".ilt_cache",
    verbose=int(os.getenv("JOBLIB_CACHE_VERBOSITY", "0")),
)
"""Memory object for caching ILThermo data."""
