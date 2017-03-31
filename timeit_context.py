from contextlib import contextmanager
import time

@contextmanager
def timeit_context(name):
    """
    Use it to time a specific code snippet
    Usage: 'with timeit_context('Testcase1'):'
    :param name: Name of the context
    """
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    print('[{}] finished in {} ms'.format(name, int(elapsed_time * 1000)))