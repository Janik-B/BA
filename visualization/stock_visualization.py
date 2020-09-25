import matplotlib.pyplot as plt
import os
import csv


DIR_NAME_LOSS_STOCK = '../neural_network/data/Stock_losses'
CLIP_NORMS = [None, 0.1]
LEARNING_RATES = [[0.01, 0.05, 0.1], [0.01, 0.5, 0.1, 0.25, 0.5]]
N_EPOCHS = 32
LEGEND_LOCATION = 'lower left'
MOMENTUM_FILE_NAME = "../neural_network/data/Momentum/Stock.csv"
lr_momentum = 0.01


def plot_training_loss_histories(dir_name, clip_norms, learning_rates, epochs=10, loc='lower left'):
    labels = []
    norm_index = 0
    for item in learning_rates:
        for index, lr in enumerate(item):
            norm = clip_norms[norm_index]
            if norm is None:
                labels.append('GD, h {}'.format(lr))
            else:
                labels.append('CGD, h {}'.format(lr))
            loss = []
            csv_path = os.path.join(dir_name, 'clip{}_loss_lr{}.csv'.format(norm, lr))
            with open(csv_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    loss.append(float(row['clip{}_loss_lr{}.csv'.format(norm, lr)]))
            if norm is None:
                colormap_none = plt.get_cmap('autumn')
                plt.plot(loss[:min(epochs, len(loss))], color=colormap_none(100*index))
            else:
                colormap_clip = plt.get_cmap('winter')
                plt.plot(loss[:min(epochs, len(loss))], color=colormap_clip(75*index))
        norm_index = norm_index + 1
    with open(MOMENTUM_FILE_NAME, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        loss = []
        for row in reader:
            loss.append(float(row['clip{}_loss_lr{}.csv'.format(None, lr_momentum)]))
        labels.append('Momentum')
        plt.plot(loss[:min(epochs, len(loss))], color='purple')
    plt.ylabel('Training Loss')
    plt.xlabel('Epoche')
    plt.legend(labels, loc=loc)
    plt.grid(True)
    plt.show()


plot_training_loss_histories(DIR_NAME_LOSS_STOCK, CLIP_NORMS, LEARNING_RATES, N_EPOCHS, loc='lower left')
CLIP_NORMS_SMOOTH = [None, 0.1]
LEARNING_RATES_SMOOTH = [0.01, 0.05, 0.1]
DIR_NAME_SMOOTHNESS_STOCK = '../neural_network/data/Stock_smoothness'
# plot_estimates_smoothness(DIR_NAME_SMOOTHNESS_STOCK, CLIP_NORMS_SMOOTH, LEARNING_RATES_SMOOTH)
