import matplotlib.pyplot as plt
import csv
import numpy as np
import os


def get_gradient_and_smoothness(file):
    xs = []
    ys = []
    with open(file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            g = float(row['grad_norm'])
            s = float(row['smoothness'])
            if g != 0 and s != 0:
                xs.append(np.math.log(g))
                ys.append(np.math.log(s))
    return xs, ys


def visualize_smoothness(file):
    fig, ax = plt.subplots()
    xs, ys = get_gradient_and_smoothness(file)
    i = np.arange(len(xs))
    cax = ax.scatter(xs, ys, c=i, edgecolors='none',
                     cmap='viridis')
    fig.colorbar(cax)
    ax.grid(True)
    ax.set(xlabel='log(Norm des Gradienten)', ylabel='log(Schätzer der Glattheit)')
    plt.show()


def plot_training_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def plot_training_loss(history):
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def plot_training_loss_histories(dir_name, clip_norms, learning_rates, epochs=10, loc='lower left',
                                 file_name_momentum="",
                                 lr_momentum=0.01):
    labels = []
    colormap_none = plt.get_cmap('autumn')
    colormap_clip = plt.get_cmap('winter')
    for norm in clip_norms:
        for index, lr in enumerate(learning_rates):
            labels.append(get_label(clip_norms, lr, norm))
            csv_path = os.path.join(dir_name, 'clip{}_loss_lr{}.csv'.format(norm, lr))
            loss = get_loss(csv_path, lr, norm)
            if norm is None:
                color = colormap_none(100 * index)
            else:
                color = colormap_clip(100 * index)
            plt.plot(loss[:min(epochs, len(loss))], color=color)

    momentum_loss = get_loss(file_name_momentum, lr_momentum, None)
    labels.append('Momentum')
    plt.plot(momentum_loss[:min(epochs, len(momentum_loss))], color='purple')

    plt.ylabel('Training Loss')
    plt.xlabel('Epoche')
    plt.legend(labels, loc=loc)
    plt.grid(True)
    plt.show()


def get_loss(csv_path, lr, norm):
    loss = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            loss.append(float(row['clip{}_loss_lr{}.csv'.format(norm, lr)]))
    return loss


def get_label(clip_norms, lr, norm):
    if norm is None:
        return 'GD h {}'.format(lr)
    elif len(clip_norms) > 2:
        return 'CGD {}, h {}'.format(norm, lr)
    else:
        return 'CGD h {}'.format(lr)


def plot_estimates_smoothness(dir_name, clip_norms, learning_rates):
    fig, axs = plt.subplots(len(clip_norms), len(learning_rates), figsize=(15, 10), gridspec_kw={'hspace': 0.5})
    for clip_index, clip_norm in enumerate(clip_norms):
        for lr_index, lr in enumerate(learning_rates):
            csv_path = os.path.join(dir_name, 'smoothness_clip_norm{}_lr{}.csv'.format(clip_norm, lr))
            xs, ys = get_gradient_and_smoothness(csv_path)
            i = np.arange(len(xs))
            cax = axs[clip_index, lr_index].scatter(xs, ys, c=i, edgecolors='none',
                                                    cmap='viridis')
            fig.colorbar(cax, ax=axs[clip_index, lr_index])
            axs[clip_index, lr_index].grid(True)
            axs[clip_index, lr_index].set(xlabel='log(Norm des Gradienten)',
                                          ylabel='log(Schätzer der Glattheit)')
            if clip_norm is None:
                title = "GD h {}".format(lr)
            else:
                title = "CGD h {}".format(lr)
            axs[clip_index, lr_index].set_title(title)
    plt.show()
