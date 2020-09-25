import random
import math
import numpy as np


def rosenbrock(x):
    return (1 + x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def gradient_rosenbrock(x):
    g1 = -400 * x[0] * x[1] + 400 * x[0] ** 3 + 2 * x[0] - 2
    g2 = 200 * x[1] - 200 * x[0] ** 2
    return np.array([g1, g2])


def pseudo_regression_quartic(x):
    # min bei x=5/2
    # b = [1, 2, 3, 4]
    # xs = [(x - b[i]) ** 4 for i in range(0, 3)]
    # return 1 / 4 * sum(xs)
    return 177 / 2 + 20 * (-5 * x + x ** 2) + (-5 * x + (x ** 2)) ** 2


def gradient_pseudo_regression_quartic(x):
    return -100 + 90 * x - 30 * (x ** 2) + 4 * (x ** 3)


def gradient_stochastic_pseudo_regression_quartic(x):
    b = [1, 2, 3, 4]
    r = random.randrange(0, 4)
    return 4 * (x - b[r]) ** 3


def f_flat_region(x):
    a = 5
    if x > 0:
        return math.exp(x)
    if x < -a:
        return -20 * (x + a) + 1 - a / 10
    return 1 + x / 10


def gradient_f_flat_region(x):
    a = 5
    if x > 0:
        return math.exp(x)
    if x < -a:
        return -20 * a
    return 1 / 10
