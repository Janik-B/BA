from visualization.visualization_utils import plot_training_loss_histories

DIR_NAME_LOSS_CIFAR = '../neural_network/data/Movie_losses'
clip_norms = [None, 0.5]
learning_rates = [0.2, 0.4, 0.6, 0.8]
N_EPOCHS = 32
LEGEND_LOCATION = 'lower left'
MOMENTUM_FILE_NAME = "../neural_network/data/Momentum/Movie.csv"

plot_training_loss_histories(dir_name=DIR_NAME_LOSS_CIFAR, clip_norms=clip_norms, learning_rates=learning_rates,
                             epochs=N_EPOCHS, loc=LEGEND_LOCATION, file_name_momentum=MOMENTUM_FILE_NAME, lr_momentum=0.5)
CLIP_NORMS_SMOOTH = [None, 0.5]
LEARNING_RATES_SMOOTH = [0.4, 0.6, 0.8]
DIR_NAME_SMOOTHNESS_CIFAR = '../neural_network/data/Movie_smoothnesses'
# plot_estimates_smoothness(DIR_NAME_SMOOTHNESS_CIFAR, CLIP_NORMS_SMOOTH, LEARNING_RATES_SMOOTH)
# visualize_smoothness("./Movie_smoothnesses/smoothness_clip_norm0.5_lr0.8.csv")
