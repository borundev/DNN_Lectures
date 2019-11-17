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


def relu(x, der=False):
    if not der:
        return np.where(x > 0, x, 0)
    else:
        return np.where(x > 0, 1, 0)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def selu(x,der=False,l=1.0507009873554804934193349852946,a=1.6732632423543772848170429916717):
    if not der:
        return np.where(x>0,l*x,l*a*(np.exp(x)-1))
    else:
        return np.where(x>0,l,l*a*np.exp(x))
