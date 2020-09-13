import matplotlib.pyplot as plt
import csv
import numpy as np
import os


def visualize_smoothness(file):
    with open(file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        fig, ax = plt.subplots()
        xs = []
        ys = []

        for row in reader:
            xs.append(np.math.log(float(row['grad_norm'])))
            ys.append(np.math.log(float(row['smoothness'])))
        i = np.arange(len(xs))
        cax = ax.scatter(xs, ys, c=i, edgecolors='none',
                         cmap='viridis')
        fig.colorbar(cax)
        ax.grid(True)
        ax.set(xlabel='log(gradient norm)', ylabel='log(smoothness)')
        plt.show()


def plot_training_accuracy(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def plot_training_loss(history):
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def plot_training_loss_histories(dir_name, clip_norms, learning_rates, epochs=10, loc='lower left'):
    labels = []
    for norm in clip_norms:
        for lr in learning_rates:
            if norm is None:
                labels.append('festes h {}'.format(lr))
            else:
                labels.append('clip Norm {}, h {}'.format(norm, lr))
            loss = []
            csv_path = os.path.join(dir_name, 'clip{}_loss_lr{}.csv'.format(norm, lr))
            with open(csv_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    loss.append(float(row['clip{}_loss_lr{}.csv'.format(norm, lr)]))
            plt.plot(loss[:min(epochs, len(loss))])

    plt.title('Training loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(labels, loc=loc)
    plt.show()


def plot_training_val_loss_histories(dir_name, clip_norms, learning_rates, epochs=10, loc='lower left'):
    labels = []
    for norm in clip_norms:
        for lr in learning_rates:
            if norm is None:
                labels.append('festes h {}'.format(lr))
            else:
                labels.append('clip Norm {}, h {}'.format(norm, lr))

            loss = []
            csv_path = os.path.join(dir_name, 'clip{}_val_loss_lr{}.csv'.format(norm, lr))
            with open(csv_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    loss.append(float(row['clip{}_val_loss_lr{}.csv'.format(norm, lr)]))
            plt.plot(loss[:min(epochs, len(loss))])

    plt.title('Validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(labels, loc=loc)
    plt.show()


def plot_estimates_smoothness(dir_name, clip_norms, learning_rates):
    fig, axs = plt.subplots(len(learning_rates), len(clip_norms), figsize=(15, 15), gridspec_kw={'hspace': 0.5})
    lr_index = 0
    for lr in learning_rates:
        clip_norm_index = 0
        for clip_norm in clip_norms:
            csv_path = os.path.join(dir_name, 'smoothness_clip_norm{}_lr{}.csv'.format(clip_norm, lr))
            with open(csv_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                xs = []
                ys = []
                for row in reader:
                    xs.append(np.math.log(float(row['grad_norm'])))
                    ys.append(np.math.log(float(row['smoothness'])))
                i = np.arange(len(xs))
                cax = axs[lr_index, clip_norm_index].scatter(xs, ys, c=i, edgecolors='none',
                                                             cmap='viridis')
                fig.colorbar(cax, ax=axs[lr_index, clip_norm_index])
                axs[lr_index, clip_norm_index].grid(True)
                axs[lr_index, clip_norm_index].set(xlabel='log(Norm des Gradienten)',
                                                   ylabel='log(Schätzer der Glattheit)')
                if clip_norm is None:
                    axs[lr_index, clip_norm_index].set_title("festes h {}".format(lr))
                else:
                    axs[lr_index, clip_norm_index].set_title("clip Norm {}, h {}".format(clip_norm, lr))
            clip_norm_index = clip_norm_index + 1
        lr_index = lr_index + 1
    plt.show()
