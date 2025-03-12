from environs import env

env.read_env()


# Joblib

JOBLIB_CACHE_VERBOSITY = env.int("JOBLIB_CACHE_VERBOSITY", default=0)
