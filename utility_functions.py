from functools import wraps
import numpy as np

def coroutine(func):
    @wraps(func)
    def inner(*args, **kwargs):
        gen = func(*args, **kwargs)
        next(gen)
        return gen

    return inner


@coroutine
def averager():
    total = 0
    count = 0
    average = None
    cont = True
    while cont:
        val = yield average
        if val is None:
            cont = False
            continue
        else:
            total += val
            count += 1.
            average = total / count
    return average


def extract_averager_value(averager):
    try:
        averager.send(None)
    except StopIteration as e:
        return e.value


def relu(x):
    return np.where(x > 0, x, 0)


def relu_prime(x):
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))