from joblib import Memory

ilt_memory = Memory(location=".ilt_cache", verbose=0)
"""Memory object for caching ILThermo data."""
