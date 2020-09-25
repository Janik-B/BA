import matplotlib.pyplot as plt

from algorithms.gradient_descent import *


def plot_iterates_on_function(ax, function, x_val, x_min, x_max, title):
    y_val = [function(x) for x in x_val]
    z_val = np.arange(len(x_val))
    xs = np.linspace(x_min, x_max, 100)
    ys = [function(x) for x in xs]
    ax.plot(xs, ys)
    ax.plot(x_val, y_val)
    cm = plt.cm.get_cmap('RdYlBu')
    ax.scatter(x_val, y_val, c=z_val, vmin=0, vmax=len(x_val), cmap=cm)
    ax.set_title(title)


def plot_iterates_on_level_sets(ax, function, x_val, x_min, x_max, y_min, y_max, title):
    x = np.linspace(x_min, x_max, 250)
    y = np.linspace(y_min, y_max, 250)
    X, Y = np.meshgrid(x, y)
    Z = function([X, Y])
    number_of_level_sets = 10
    ax.contour(X, Y, Z, number_of_level_sets, cmap='jet')
    angles_x = x_val[1:, 0] - x_val[:-1, 0]
    angles_y = x_val[1:, 1] - x_val[:-1, 1]
    ax.scatter(x_val[:, 0], x_val[:, 1], color='r', marker='*', s=1)
    ax.quiver(x_val[:-1, 0], x_val[:-1, 1], angles_x, angles_y, scale_units='xy', angles='xy', scale=1, color='r',
              alpha=.3)
    ax.set_xlabel("{} Iterationen".format(len(x_val[1:])))
    ax.set_title(title)


def plot_error_convergence(ax, x_vals, learning_rates, function, minimum, title, param_name):
    f_min = function(minimum)
    for i in range(len(x_vals)):
        y_val = [abs(function(x) - f_min) for x in x_vals[i]]
        ax.plot(range(len(x_vals[i])), y_val,
                label=param_name + format(learning_rates[i], '.6f'))
    ax.legend()
    ax.set_yscale('log')
    ax.set_ylabel("Fehler")
    ax.set_xlabel("Iteration")
    ax.set_title(title)
