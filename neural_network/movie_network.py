from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import SGD

from neural_network.utils import *
from visualization.visualization_utils import *


def get_data(vocab_size):
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)
    word_index = imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3
    train_data = pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=256)
    test_data = pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=256)
    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1))
    test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))
    return (train_data, train_labels), (test_data, test_labels)


def build_simple_lstm_model(learning_rate, clip_norm, momentum=0.0):
    model = Sequential([
        Embedding(VOCAB_SIZE, 16),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
        tf.keras.layers.Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(loss=BinaryCrossentropy(from_logits=True),
                  optimizer=SGD(learning_rate=learning_rate, clipnorm=clip_norm, momentum=momentum),
                  metrics=['accuracy'])
    return model


VOCAB_SIZE = 10000
(x_train, y_train), (x_test, y_test) = get_data(VOCAB_SIZE)

clip_norms = [None, 0.5]
learning_rates = [0.2, 0.4, 0.6, 0.8]
N_EPOCHS = 35
DIR_NAME = "Movie_losses"
BATCH_SIZE = 64
train_model_and_save_losses_to_file(
    build_model_function=build_simple_lstm_model,
    data=x_train,
    labels=y_train,
    learning_rates=learning_rates,
    clip_norms=clip_norms,
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    dir_name=DIR_NAME,
    momentum=0.0)

CLIP_NORM_MOMENTUM = [None]
LEARNING_RATE_MOMENTUM = [0.01]
MOMENTUM = 0.5
DIR_NAME_MOMENTUM = "Movie_loss_momentum"
train_model_and_save_losses_to_file(
    build_model_function=build_simple_lstm_model,
    data=x_train,
    labels=y_train,
    learning_rates=LEARNING_RATE_MOMENTUM,
    clip_norms=CLIP_NORM_MOMENTUM,
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    dir_name=DIR_NAME_MOMENTUM,
    momentum=0.5)

plot_training_loss_histories(DIR_NAME, clip_norms, learning_rates, N_EPOCHS,
                             file_name_momentum="Movie_loss_momentum/clip{}_loss_lr{}.csv".format(None, 0.01),
                             lr_momentum=0.01)

DIR_NAME_SMOOTHNESS = "smoothness_LSTM_movie_review"
# Die Berechnung des Schätzers ist zeitaufwendig und verbraucht viel RAM.
# Daher kann es bei einem Rechner mit wenig RAM sein, dass man die verschiedenen Modelle einzeln trainieren muss.
estimate_smoothness_for_different_hyperparmeters_and_save_results_to_file(
    build_model_function=build_simple_lstm_model,
    data=x_train,
    labels=y_train,
    learning_rates=learning_rates,
    clip_norms=clip_norms,
    n_epochs=1,
    batch_size=BATCH_SIZE,
    dir_name=DIR_NAME_SMOOTHNESS,
    loss_object=BinaryCrossentropy())

plot_estimates_smoothness(DIR_NAME_SMOOTHNESS, clip_norms, learning_rates)

# Quellen
# 1.   https://www.tensorflow.org/guide/keras/rnn
# 2.   https://www.tensorflow.org/tutorials/keras/text_classification
# 3.   https://www.tensorflow.org/tutorials/text/text_classification_rnn
