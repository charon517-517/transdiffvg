from time import perf_counter
from functools import wraps
def timed(fn):
 
    @wraps(fn)
    def inner(*args, **kwargs):
        start = perf_counter()
        result = fn(*args, **kwargs)
        elapsed = perf_counter() - start
        print(f'{fn.__name__} took {elapsed:.6f}s to execute')
        return result
 
    return inner