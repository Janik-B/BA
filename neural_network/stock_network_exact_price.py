import pandas_datareader as web
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dropout

from visualization.visualization_utils import *


# pip install pandas-datareader
def generate_data_true_value_tomorrow(stock_ticker_list, start_date, end_date, scaler):
    labels = []
    data = []
    observed_days = 60
    for stock_ticker in stock_ticker_list:
        df = web.DataReader(stock_ticker, data_source='yahoo', start=start_date, end=end_date)
        dataset = df.filter(['Close']).values
        scaled_data = scaler.fit_transform(dataset)
        for i in range(observed_days, len(dataset)):
            labels.append(scaled_data[i, 0])
            data.append(scaled_data[i - observed_days:i, 0])
    data = np.array(data)
    labels = np.array(labels)
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    return data, labels


def build_lstm_model(learning_rate=0.2, clip_norm=0.2):
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
    model.add(Dense(units=1))
    model.compile(optimizer=SGD(learning_rate=learning_rate, clipnorm=clip_norm), loss='mean_squared_error')
    return model


def plot_error_history(real_labels, predicted_labels):
    # Plot der Abweichung des Schätzers vom Kurswert
    # der scaler funktioniert nur mit 2-dimensionalen Arrays
    valid = [[val] for val in real_labels]
    valid = SCALER.inverse_transform(valid)
    # der dummy Schätzer sagt einfach den heutigen Kurs für Morgen voraus.
    # ein gutes Modell sollte besser als der dummy Schätzer sein.
    dummy_estimator = valid[:-1][1:]

    dummy_error = get_error(dummy_estimator, valid)
    prediction_error = get_error(predicted_labels, valid)

    plt.figure(figsize=(16, 8))
    plt.xlabel('Day', fontsize=18)
    plt.ylabel('absolute deviation from true value in $', fontsize=18)
    plt.plot(range(len(prediction_error)), prediction_error)
    plt.plot(range(len(dummy_error)), dummy_error)
    plt.legend(['Predictionerror', 'dummy error'], loc='lower right')
    plt.show()


def get_error(estimator, valid):
    return [abs(a - b) for a, b in zip(valid, estimator)]


STOCK_TICKER_LIST = ['AAPL', 'LHA.DE']
START_DATE = '2018-01-01'
END_DATE = '2020-09-09'
SCALER = MinMaxScaler(feature_range=(0, 1))
x_train, y_train = generate_data_true_value_tomorrow(STOCK_TICKER_LIST, START_DATE, END_DATE, SCALER)
x_test, y_test = generate_data_true_value_tomorrow(['BMW.DE'], START_DATE, END_DATE, SCALER)
lstm_model = build_lstm_model()
lstm_model.fit(x_train, y_train, batch_size=16, epochs=32, validation_split=0.1)

predictions = lstm_model.predict(x_test)
predictions = SCALER.inverse_transform(predictions)

plot_error_history(y_test, predictions)

# Das trainierte Modell ist schlechter als der dummy Schätzer.
# Daher habe ich zunächst die Repräsentation der Daten verändert.
# Dies geschah basierend auf der Idee, dass im Vordergrund steht, ob der Aktienkurs fällt, oder steigt,
# was durch eine binäre Entscheidung modelliert wird.
# Zudem wurde der Zeitrahmen vergrößert.
# Es soll entschieden werden ob in 30 Tagen der Kurs gestiegen ist, oder gefallen.
# siehe stock_network_binary.py

# Quelle https://medium.com/@randerson112358/stock-price-prediction-using-python-machine-learning-e82a039ac2bb
