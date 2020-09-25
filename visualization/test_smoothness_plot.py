import matplotlib.pyplot as plt
import numpy as np

from algorithms.gradient_descent import gd, cgd
from functions.test_functions import gradient_pseudo_regression_quartic


def smoothness_estimator(gradient, old_point, new_point, num_pts=1):
    # 1 dimensional smoothness estimator
    old_grad = gradient(old_point)
    alphas = np.arange(1, num_pts + 1, dtype=np.float32) / (num_pts + 1)
    old_norm = np.linalg.norm(old_grad)
    update_size = np.abs(old_point - new_point)
    max_smooth = -1.0
    for alpha in alphas:
        between_point = alpha * old_point + (1 - alpha) * new_point
        between_grad = gradient(between_point)
        smooth = np.linalg.norm(old_grad - between_grad) / (update_size * (1 - alpha))
        max_smooth = max(smooth, max_smooth)
    return old_norm, max_smooth


def visualize_smoothness(gradient, iterates):
    values = [smoothness_estimator(gradient, iterates[i], iterates[i + 1]) for i in range(len(iterates) - 1)]
    xs = [np.log(value[0]) for value in values]
    ys = [np.log(value[1]) for value in values]
    fig, ax = plt.subplots()
    i = np.arange(len(xs))
    cax = ax.scatter(xs, ys, c=i, edgecolors='none',
                     cmap='viridis')
    fig.colorbar(cax)
    ax.grid(True)
    ax.set(xlabel='log(gradient norm)', ylabel='log(smoothness)')
    plt.show()


def plot_estimates_smoothness(clip_norms, learning_rates, gradient, iterates):
    print(iterates[0])
    fig, axs = plt.subplots(1, 2, gridspec_kw={'hspace': 0.5})
    clip_norm_index = 0
    for clip_norm in clip_norms:
        values = [smoothness_estimator(gradient, iterates[clip_norm_index][j],
                                       iterates[clip_norm_index][j + 1]) for
                  j
                  in range(len(iterates[clip_norm_index]) - 1)]
        xs = [np.log(value[0]) for value in values]
        ys = [np.log(value[1]) for value in values]
        i = np.arange(len(xs))
        cax = axs[clip_norm_index].scatter(xs, ys, c=i, edgecolors='none',
                                           cmap='viridis')
        fig.colorbar(cax, ax=axs[clip_norm_index])
        axs[clip_norm_index].grid(True)
        axs[clip_norm_index].set(xlabel='log(Norm des Gradienten)',
                                 ylabel='log(Schätzer der Glattheit)')
        step_size_str = format(learning_rates[clip_norm_index], '.5f')
        if clip_norm is None:
            axs[clip_norm_index].set_title("GD h {}".format(step_size_str))
        else:
            axs[clip_norm_index].set_title("CGD $\\eta$ {}".format(step_size_str))
        clip_norm_index = clip_norm_index + 1
    plt.show()


start_point_pseudo_regression = 30
learning_rate = 0.9 / 1520
eta = 2 / 150
gamma = 15
precision = 0.00000001
max_iter = 1000
GRADIENT = gradient_pseudo_regression_quartic
x_vals_gd = [gd(gradient_pseudo_regression_quartic, start_point_pseudo_regression,
                learning_rate=learning_rate / (1 + i + (i - 1) * i * (1 / 2)),
                precision=precision,
                max_iterations=max_iter) for i in range(3)]
x_vals_cgd = [cgd(gradient_pseudo_regression_quartic, start_point_pseudo_regression,
                  gamma=gamma, eta=eta / (1 + i + (i - 1) * i * (1 / 2)),
                  precision=precision,
                  max_iterations=max_iter) for i in range(3)]

clip_norms = [None, 15]
learning_rates = [[learning_rate / (1 + i + (i - 1) * i * (1 / 2)) for i in range(3)],
                  [eta / (1 + i + (i - 1) * i * (1 / 2)) for i in range(3)]]

plot_estimates_smoothness(clip_norms, [learning_rates[0][2], learning_rates[1][2]], GRADIENT,
                          [x_vals_gd[2], x_vals_cgd[2]])
