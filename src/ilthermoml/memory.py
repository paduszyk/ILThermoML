import os

from dotenv import load_dotenv
from joblib import Memory

load_dotenv()
cache_verbosity = os.getenv("JOBLIB_CACHE_VERBOSITY", "0")

ilt_memory = Memory(location=".ilt_cache", verbose=int(cache_verbosity))
"""Memory object for caching ILThermo data."""
