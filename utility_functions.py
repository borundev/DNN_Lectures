from functools import wraps
import numpy as np
from scipy.special import erf

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


def sigmoid(x, der=False):
    if not der:
        return 1. / (1. + np.exp(-x))
    else:
        xs=sigmoid(x)
        return xs*(1-xs)

def selu(x,der=False,l=1.0507009873554804934193349852946,a=1.6732632423543772848170429916717):
    if not der:
        return np.where(x>0,l*x,l*a*(np.exp(x)-1))
    else:
        return np.where(x>0,l,l*a*np.exp(x))

def tanh(x,der=False):
    if not der:
        return np.tanh(x)
    else:
        return 1-np.tanh(x)**2

def ghelu(z,der=False,x=-1.13,l1=1.1,l2=0.3,m=0,s=1):
    def remove_mean(x, l1, l2, m, s):
        a = (x - m) / s
        y = s * a * (l1 + l2) / 2. - (l1 - l2) * s * (
                    a * erf(a / np.sqrt(2)) / 2. + np.exp(-a ** 2 / 2) / np.sqrt(2 * np.pi))
        return y
    y=remove_mean(x,l1,l2,m,s)
    if not der:
        return np.where(z>x,l1*(z-x),l2*(z-x))+y
    else:
        return np.where(z>x,l1,l2)


from functools import wraps

def random_state_contolled(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        state = np.random.get_state()
        np.random.seed(42)
        value=func(*args,**kwargs)
        np.random.set_state(state)
        return value
    return wrapper

@random_state_contolled
def np_random_normal(*args,**kwargs):
    return np.random.normal(*args,**kwargs)

