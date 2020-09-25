import os
import csv
import numpy as np
from keras.models import load_model
from keras.callbacks import LambdaCallback
import tensorflow as tf
import random


# TODO correctly cited?
# Orientierung des Codes an utils.py der Autoren des Papers
# "why gradient clipping accelerates training of neural networks"
@tf.function
def l2_norm(grads):
    return tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.square(g)) for g in grads]))


def l2_norm_diff(v1, v2):
    return tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.square(tf.math.subtract(g1, g2))) for g1, g2 in zip(v1, v2)]))


def get_gradients(model, data, labels, loss):
    with tf.GradientTape() as tape:
        predictions = model(data, training=True)
        loss_value = loss(labels, predictions)
    return tape.gradient(loss_value, model.trainable_weights)


def write_and_read_gradients(logger, model, file, data, labels, loss_object):
    logger.write_row(eval_smoothness(file, model, data, labels, loss_object))
    model.save(file)


def convex_comb_weights(alpha, weights1, weights2):
    weights = []
    for w1, w2 in zip(weights1, weights2):
        weights.append(alpha * w1 + (1 - alpha) * w2)
    return weights


def get_subset_of_data_and_labels(data, labels):
    # Evaluierung des Gradienten nur auf x Prozent des Datensatzes
    # Wie in dem Paper vorgeschlagen sind 10% repräsentativ
    percentage = 10
    k = len(data) * percentage // 100
    indicies = random.sample(range(len(data)), k)
    data_subset = data[indicies]
    label_subset = labels[indicies]
    return data_subset, label_subset


# Schätzer der Glattheit wie im Paper vorgschlagen
def eval_smoothness(file, new_model, data, labels, loss_object, num_pts=1):
    old_model = load_model(file)
    data_subset, label_subset = get_subset_of_data_and_labels(data, labels)
    old_grad = get_gradients(old_model, data_subset, label_subset, loss_object)
    gnorm = l2_norm(old_grad)
    alphas = np.arange(1, num_pts + 1, dtype=np.float32) / (num_pts + 1)
    update_size = l2_norm_diff(new_model.trainable_weights, old_model.trainable_weights)
    max_smooth = tf.constant(-1.0, dtype=tf.float32)
    for alpha in alphas:
        between_model = load_model(file)
        between_model.set_weights(
            convex_comb_weights(alpha, between_model.trainable_weights, new_model.trainable_weights))
        between_grad = get_gradients(between_model, data_subset, label_subset, loss_object)
        smooth = l2_norm_diff(old_grad, between_grad) / (update_size * (1 - alpha))
        max_smooth = tf.math.maximum(smooth, max_smooth)
    max_smooth = max_smooth.numpy()
    gnorm = gnorm.numpy()
    return max_smooth, gnorm


def write_data_to_csv_file(dir_name, filename, data):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    csv_path = os.path.join(dir_name, filename)
    with open(csv_path, 'a') as logfile:
        log_writer = csv.writer(logfile, delimiter=',')
        log_writer.writerow([filename])
        for element in data:
            log_writer.writerow([element])


def train_model_and_save_losses_to_file(build_model_function, data, labels, learning_rates, clip_norms,
                                        n_epochs, batch_size, dir_name, momentum):
    for clip_norm in clip_norms:
        print("start clip_norm: ", clip_norm)
        for lr in learning_rates:
            print("start lr: ", lr)
            tf.keras.backend.clear_session()
            model = build_model_function(lr, clip_norm, momentum)
            model.fit(data, labels, epochs=n_epochs, batch_size=batch_size, verbose=1)
            write_data_to_csv_file(dir_name, 'clip{}_loss_lr{}.csv'.format(clip_norm, lr),
                                   model.history.history['loss'])


def estimate_smoothness_for_different_hyperparmeters_and_save_results_to_file(build_model_function, data, labels,
                                                                              learning_rates, clip_norms, n_epochs,
                                                                              batch_size, dir_name, loss_object):
    csv_logger_keys = ['smoothness', 'grad_norm']
    for clip_norm in clip_norms:
        print("start clip norm: ", clip_norm)
        for lr in learning_rates:
            print("start lr: ", lr)
            tf.keras.backend.clear_session()
            model = build_model_function(lr, clip_norm)
            csv_path = os.path.join(dir_name, 'smoothness_clip_norm{}_lr{}.csv'.format(clip_norm, lr))
            iterationlogger = CSVLogger(csv_path, csv_logger_keys)
            estimate_smoothness = LambdaCallback(
                on_batch_end=lambda batch, logs: write_and_read_gradients(iterationlogger, model, "model.h5", data,
                                                                          labels, loss_object))
            model.save("model.h5")
            model.fit(data, labels, epochs=n_epochs, batch_size=batch_size, callbacks=[estimate_smoothness], verbose=1)


class CSVLogger(object):
    def __init__(self, filename, keys):
        self.filename = filename
        self.keys = keys
        self.values = {k: [] for k in keys}
        self.init_file()

    def init_file(self):
        # This will overwrite previous file
        if os.path.exists(self.filename):
            return

        directory = os.path.dirname(self.filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(self.filename, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(self.keys)

    def write_row(self, values):
        assert len(values) == len(self.keys)
        if not os.path.exists(self.filename):
            self.init_file()
        with open(self.filename, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(values)
