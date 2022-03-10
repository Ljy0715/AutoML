from contextlib import contextmanager
from time import time
import numpy as np


@contextmanager
def timer():
    s = time()
    yield
    e = time() - s
    print("{0}: {1} ms".format('Elapsed time', e))

