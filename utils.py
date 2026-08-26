import logging
import time

logger = logging.getLogger('RESNet.utils')

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        rv = func(*args, **kwargs)
        total = time.time() - start
        logger.info('%s finished in %.2f seconds', func.__name__, total)
        return rv
    return wrapper
