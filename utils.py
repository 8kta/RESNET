import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        rv = func(*args, **kwargs)
        total = time.time() - start
        print(f'Total: {total}')
        return rv
    return wrapper
