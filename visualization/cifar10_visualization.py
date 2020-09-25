from visualization.visualization_utils import plot_training_loss_histories

DIR_NAME_LOSS_CIFAR = '../neural_network/data/Cifar10_losses'
clip_norms = [None, 0.5]
learning_rates = [0.05, 0.1, 0.25]
N_EPOCHS = 64
LEGEND_LOCATION = 'upper right'
MOMENTUM_FILENAME = "../neural_network/data/Momentum/Cifar10.csv"
plot_training_loss_histories(
    dir_name=DIR_NAME_LOSS_CIFAR,
    clip_norms=clip_norms,
    learning_rates=learning_rates,
    epochs=N_EPOCHS,
    loc=LEGEND_LOCATION, file_name_momentum=MOMENTUM_FILENAME)
CLIP_NORMS_SMOOTH = [None, 0.5]
LEARNING_RATES_SMOOTH = [0.05, 0.1, 0.25]
DIR_NAME_SMOOTHNESS_CIFAR = '../neural_network/data/Cifar10_smoothness'
# plot_estimates_smoothness(DIR_NAME_SMOOTHNESS_CIFAR, CLIP_NORMS_SMOOTH, LEARNING_RATES_SMOOTH)
