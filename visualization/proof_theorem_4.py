import math
import numpy as np
import matplotlib.pyplot as plt

from algorithms.gradient_descent import gd


def case1(x):
    L_1 = 2
    if x < -1 / L_1:
        return math.exp(-L_1 * x) / (L_1 * math.e)
    if -1 / L_1 <= x <= 1 / L_1:
        return L_1 * (x ** 2) / 2 + 1 / (2 * L_1)
    return math.exp(L_1 * x) / (L_1 * math.e)


def case1_gradient(x):
    L_1 = 2
    if x < -1 / L_1:
        return - math.exp(-L_1 * x) / math.e
    if -1 / L_1 <= x <= 1 / L_1:
        return L_1 * x
    return math.exp(L_1 * x) / math.e


def case2(x):
    eps = 0.01
    if x < -1:
        return (-2 * eps * (x + 1)) + (5 * eps / 4)
    if -1 <= x <= 1:
        return eps / 4 * (6 * (x ** 2) - (x ** 4))
    return (2 * eps * (x - 1)) + (5 * eps / 4)


def case2_gradient(x):
    eps = 0.01
    if x < -1:
        return -2 * eps * x
    if -1 <= x <= 1:
        return eps / 4 * (12 * x - (4 * (x ** 3)))
    return 2 * eps * x


def plot_basic_gradient_descent(function, gradient, start_point, learning_rate, max_iterations, x_min=-5, x_max=5):
    x_val = gd(gradient, start_point, learning_rate=learning_rate,
               max_iterations=max_iterations)
    y_val = [function(x) for x in x_val]
    z_val = np.arange(len(x_val))
    xs = np.linspace(x_min, x_max, 100)
    ys = [function(x) for x in xs]
    plt.plot(xs, ys)
    plt.plot(x_val, y_val)
    cm = plt.cm.get_cmap('RdYlBu')
    plt.scatter(x_val, y_val, c=z_val, vmin=0, vmax=len(x_val), cmap=cm)
    plt.show()


def plot_case_1():
    L_1 = 2
    M = 2
    x_0 = (math.log1p(M) + 1) / L_1
    print("x_0: ", x_0)
    h = 3 * x_0 / M
    plot_basic_gradient_descent(case1, case1_gradient, x_0, h, 1, -4, 4)


def plot_case_2():
    L_1 = 2
    M = 2
    eps = 0.01
    delta = 0.01
    x_0 = 1 + (delta / eps)
    h = (2 * math.log1p(M) + 2) / (M * L_1)
    plot_basic_gradient_descent(case2, case2_gradient, x_0, h, 10, -4, 4)


plot_case_1()
plot_case_2()
