from functions.test_functions import *
from visualization.convergence_plot_utils import *
from visualization.convergence_plot_utils import plot_error_convergence


def plot_compare_iterates(gradient, function, start_point, learning_rate, eta, gamma, x_min, x_max):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    x_val_gd = gd(gradient, start_point,
                  learning_rate=learning_rate,
                  max_iterations=102)
    plot_iterates_on_function(ax1, function, x_val_gd, x_min=x_min, x_max=x_max, title='Gradientenverfahren')
    ax1.set_ylabel('f(x)')
    x_val_clipped = cgd(gradient, start_point, eta=eta,
                        gamma=gamma,
                        max_iterations=12)
    plot_iterates_on_function(ax2, function, x_val_clipped, x_min=x_min, x_max=x_max,
                              title='clipped \n Gradientenverfahren')
    plt.legend()
    plt.show()


plot_compare_iterates(gradient=gradient_f_flat_region, function=f_flat_region,
                      start_point=-4, learning_rate=0.1, eta=1, gamma=0.1, x_min=-7, x_max=5.1)


def plot_compare_error_convergence(gradient, function, minimum, start_point, learning_rates, etas, gamma,
                                   max_iter,
                                   precision):
    fig3, (ax9, ax10) = plt.subplots(1, 2)
    x_vals_gd = [gd(gradient, start_point,
                    learning_rate=lr,
                    precision=precision,
                    max_iterations=max_iter) for lr in learning_rates]
    plot_error_convergence(ax9, x_vals_gd, learning_rates, function, minimum=minimum,
                           title="GD", param_name='h ')
    x_vals_cgd = [cgd(gradient, start_point,
                      gamma=gamma, eta=eta,
                      precision=precision,
                      max_iterations=max_iter) for eta in etas]

    plot_error_convergence(ax10, x_vals_cgd, etas, function, minimum=minimum,
                           title="CGD", param_name="$\\eta$ ")
    plt.legend(loc='upper right')
    plt.subplots_adjust(wspace=0.5)
    plt.show()


MINIMUM_PSEUDO_REGRESSION = 5 / 2
START_POINT_PSEUDO_REGRESSION = 30
LR_PR = 0.9 / 1520
LEARNING_RATES_PSEUDO_REGRESSION = [LR_PR / 2, LR_PR, LR_PR * 2]
ETAS_PSEUDO_REGRESSION = [(2 / 150) / 2, 2 / 150, (2 / 150) * 2]
plot_compare_error_convergence(gradient=gradient_pseudo_regression_quartic,
                               function=pseudo_regression_quartic,
                               minimum=MINIMUM_PSEUDO_REGRESSION,
                               start_point=START_POINT_PSEUDO_REGRESSION,
                               learning_rates=LEARNING_RATES_PSEUDO_REGRESSION,
                               etas=ETAS_PSEUDO_REGRESSION,
                               gamma=15,
                               max_iter=1000,
                               precision=0.00000001)

LR_SPR = 3.03 / 9090
LEARNING_RATES_STOCHASTIC_PSEUDO_REGRESSION = [LR_SPR / 2, LR_SPR, LR_SPR * 2]
ETAS_STOCHASTIC_PSEUDO_REGRESSION = [(15 / 150) / 2, 15 / 150, (15 / 150) * 2]
plot_compare_error_convergence(gradient=gradient_stochastic_pseudo_regression_quartic,
                               function=pseudo_regression_quartic,
                               minimum=MINIMUM_PSEUDO_REGRESSION,
                               start_point=START_POINT_PSEUDO_REGRESSION,
                               learning_rates=LEARNING_RATES_STOCHASTIC_PSEUDO_REGRESSION,
                               etas=ETAS_STOCHASTIC_PSEUDO_REGRESSION,
                               gamma=16,
                               max_iter=100,
                               precision=0.000001)


def plot_compare_iterates_level_sets(gradient, function, start_point, learning_rate,
                                     gamma, x_min, x_max, y_min, y_max):
    fig2, (ax7, ax8) = plt.subplots(1, 2)
    x_val_gd = gd(gradient, start_point, learning_rate=learning_rate, max_iterations=20000)
    plot_iterates_on_level_sets(ax7, function, x_val_gd, x_min, x_max, y_min, y_max, title='ohne Momentum')

    x_val_momentum = gradient_descent_momentum(gradient, start_point, learning_rate=learning_rate,
                                               gamma=gamma,
                                               max_iterations=5000)
    plot_iterates_on_level_sets(ax8, function, x_val_momentum, x_min, x_max, y_min, y_max, title='mit Momentum')
    plt.legend()
    plt.show()


plot_compare_iterates_level_sets(gradient=gradient_rosenbrock, function=rosenbrock, start_point=[-1.5, 2],
                                 learning_rate=0.00125, gamma=0.95, x_min=-2, x_max=2, y_min=-1, y_max=3)
