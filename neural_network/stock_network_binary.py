import pandas_datareader as web
from keras.layers import Dense, LSTM
from keras.losses import BinaryCrossentropy
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dropout

from neural_network.utils import *
from visualization.visualization_utils import *


def generate_data_binary_in_30_days(stock_ticker_list, start_date, end_date, scaler):
    labels = []
    data = []
    observed_days = 60
    for stock_ticker in stock_ticker_list:
        df = web.DataReader(stock_ticker, data_source='yahoo', start=start_date, end=end_date)
        dataset = df.filter(['Close']).values
        scaled_data = scaler.fit_transform(dataset)
        for i in range(0, len(dataset) - observed_days):
            # compute difference on unscaled data
            labels.append((dataset[i + observed_days] - dataset[i])[0] > 0)
            # training data is scaled
            data.append(scaled_data[i:i + observed_days, 0])
    data = np.array(data)
    labels = np.array(labels)
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    return data, labels


def build_lstm_model_binary(learning_rate=0.2, clip_norm=0.2, momentum=0.0):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer=SGD(learning_rate=learning_rate, clipnorm=clip_norm, momentum=momentum),
                  loss='binary_crossentropy')
    return model


def plot_is_price_increasing(real_results, predicted_results, time_steps_to_show):
    how_long_act = min(len(real_results), time_steps_to_show)
    fig, axs = plt.subplots(1, 1, figsize=(21, 6))
    axs.step(range(how_long_act), real_results[:how_long_act])
    axs.step(range(how_long_act), predicted_results[:how_long_act])
    axs.set_title("Real vs. Vorhersage")
    axs.set_yticks(np.arange(2))
    axs.set_yticklabels(['Falsch', 'Richtig'])
    axs.legend(('Real', 'Vorhersage'), loc='upper right', shadow=True)

    plt.show()


STOCK_TICKER_LIST = ['AAPL', 'LHA.DE']
# STOCK_TICKER_LIST = ['DAI.DE', 'WFC']
START_DATE = '2018-01-01'
END_DATE = '2020-09-09'
SCALER = MinMaxScaler(feature_range=(0, 1))
BATCH_SIZE = 16
x_train, y_train = generate_data_binary_in_30_days(STOCK_TICKER_LIST, START_DATE, END_DATE, SCALER)
x_test, y_test = generate_data_binary_in_30_days(['BMW.DE'], START_DATE, END_DATE, SCALER)
# x_test, y_test = generate_data_binary_in_30_days(['JPM'], START_DATE, END_DATE, SCALER)

clip_norms = [None, 0.1]
learning_rates = [0.01, 0.05, 0.1]
# Die Berechnung des Schätzers ist zeitaufwendig und verbraucht viel RAM.
# Daher kann es bei einem Rechner mit wenig RAM sein, dass man die verschiedenen Modelle einzeln trainieren muss.
estimate_smoothness_for_different_hyperparmeters_and_save_results_to_file(
    build_model_function=build_lstm_model_binary,
    data=x_train,
    labels=y_train,
    learning_rates=learning_rates,
    clip_norms=clip_norms,
    n_epochs=1,
    batch_size=BATCH_SIZE,
    dir_name="Stock_smoothness",
    loss_object=BinaryCrossentropy())

plot_estimates_smoothness('Stock_smoothness', [None, 0.1], [0.01, 0.05, 0.1])

clip_norms = [None, 0.1]
learning_rates = [0.01, 0.05, 0.1]
N_EPOCHS = 32
DIR_NAME = "Stock_losses"
train_model_and_save_losses_to_file(
    build_model_function=build_lstm_model_binary,
    data=x_train,
    labels=y_train,
    learning_rates=learning_rates,
    clip_norms=clip_norms,
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    dir_name=DIR_NAME,
    momentum=0.0)

clip_norms = [None]
learning_rates = [0.01]
N_EPOCHS = 32
DIR_NAME_MOMENTUM = "Stock_losses_momentum"
train_model_and_save_losses_to_file(
    build_model_function=build_lstm_model_binary,
    data=x_train,
    labels=y_train,
    learning_rates=learning_rates,
    clip_norms=clip_norms,
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    dir_name=DIR_NAME,
    momentum=0.5)

plot_training_loss_histories(DIR_NAME, [None, 0.1], [0.01, 0.05, 0.1], N_EPOCHS,
                             DIR_NAME_MOMENTUM + '/clip{}_loss_lr{}.csv'.format(None, 0.01))

lstm_model = build_lstm_model_binary(learning_rate=0.25, clip_norm=0.1, momentum=0.0)
lstm_model.fit(x_train, y_train, batch_size=16, epochs=80)

# Plot der Vorhersage und der Realität zur Veranschaulichung wie gut das Modell vorhersagen trifft.
predictions = lstm_model.predict(x_test)
plot_predictions = [i[0] >= 0.5 for i in predictions]
TIME_STEPS_TO_SHOW = 1000
plot_is_price_increasing(y_test, plot_predictions, TIME_STEPS_TO_SHOW)

# Vergleich mit verschiedenen naiven Schätzern.
# 1. naiver Schätzer der immer vorhersagt, dass der Preis steigt.
estimator_up = [1] * len(y_test)
estimator_up_is_correct = [a == b for a, b in zip(y_test, estimator_up)]
print("positiver naiver Schätzer: ", sum(estimator_up_is_correct), " von ", len(y_test))

# 2. naiver Schätzer der immer vorhersagt, dass der Preis fällt.
estimator_down = [0] * len(y_test)
estimator_down_is_correct = [a == b for a, b in zip(y_test, estimator_down)]
print("negativer naiver Schätzer: ", sum(estimator_down_is_correct), " von ", len(y_test))

# 3. Schätzer der zufällige Vorhersagen trifft
random_estimator = np.random.randint(low=0, high=2, size=len(y_test))
random_estimator_is_correct = [a == b for a, b in zip(y_test, random_estimator)]
print("Zufalls Schätzer: ", sum(random_estimator_is_correct), " von ", len(y_test))

# das trainierte neuronale Netz
plot_predictions_is_correct = [a == b for a, b in zip(y_test, plot_predictions)]
print("Neuronales Netz: ", sum(plot_predictions_is_correct), " von ", len(y_test))

# Man sieht, das Modell ist sowohl besser als der zufällige Schätzer,
# als auch besser als der naive pessimistische und der naive optimistische Schätzer.

# Quelle https://medium.com/@randerson112358/stock-price-prediction-using-python-machine-learning-e82a039ac2bb
