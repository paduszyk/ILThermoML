from joblib import Memory

from . import settings

ilt_memory = Memory(location=".ilt_cache", verbose=settings.JOBLIB_CACHE_VERBOSITY)
"""Memory object for caching ILThermo data."""
