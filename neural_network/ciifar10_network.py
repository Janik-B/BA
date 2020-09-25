from tensorflow.keras.datasets import cifar10
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from neural_network.utils import *
from visualization.visualization_utils import *


def get_preprocessed_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    return (x_train, y_train), (x_test, y_test)


def build_model(learning_rate=0.2, clip_norm=0.2, momentum=0.0):
    model = keras.Sequential()
    model.add(Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(filters=16, activation='relu', kernel_size=3))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(Conv2D(32, activation='relu', kernel_size=3))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=SGD(learning_rate=learning_rate, clipnorm=clip_norm, momentum=momentum
                                ), loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


(x_train, y_train), (x_test, y_test) = get_preprocessed_data()
learning_rates = [0.05, 0.1, 0.25]
clip_norms = [None, 0.5]
N_EPOCHS = 1
BATCH_SIZE = 512
DIR_NAME = "Cifar10_smoothness"
# Die Berechnung des Schätzers ist zeitaufwendig und verbraucht viel RAM.
# Daher kann es bei einem Rechner mit wenig RAM sein, dass man die verschiedenen Modelle einzeln trainieren muss.
estimate_smoothness_for_different_hyperparmeters_and_save_results_to_file(
    build_model_function=build_model,
    data=x_train,
    labels=y_train,
    learning_rates=learning_rates,
    clip_norms=clip_norms,
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    dir_name=DIR_NAME,
    loss_object=SparseCategoricalCrossentropy())

plot_estimates_smoothness(DIR_NAME, clip_norms, learning_rates)

learning_rates = [0.1, 0.05, 0.25]
clip_norms = [None, 0.25, 0.5]
BATCH_SIZE = 128
N_EPOCHS = 64
DIR_NAME = "Cifar10_losses"
train_model_and_save_losses_to_file(
    build_model_function=build_model,
    data=x_train,
    labels=y_train,
    learning_rates=learning_rates,
    clip_norms=clip_norms,
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    dir_name=DIR_NAME,
    momentum=0.0)

learning_rate_momentum = [0.01]
clip_norm_momentum = [None]
BATCH_SIZE = 128
N_EPOCHS = 64
DIR_NAME_MOMENTUM = "Cifar10_loss_momentum"
train_model_and_save_losses_to_file(
    build_model_function=build_model,
    data=x_train,
    labels=y_train,
    learning_rates=learning_rate_momentum,
    clip_norms=clip_norm_momentum,
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    dir_name=DIR_NAME_MOMENTUM,
    momentum=0.9)

plot_training_loss_histories(DIR_NAME, clip_norms, learning_rates, N_EPOCHS,
                             file_name_momentum="Cifar10_loss_momentum/clip{}_loss_lr{}.csv".format(None, 0.01),
                             lr_momentum=0.01)

# Quelle https://www.tensorflow.org/xla/tutorials/autoclustering_xla
