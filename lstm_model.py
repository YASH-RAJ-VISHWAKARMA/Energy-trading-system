import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class LSTMForecast:

    def __init__(self, dataframe):

        self.df = dataframe
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self):

        data = self.df['price'].values.reshape(-1, 1)

        scaled_data = self.scaler.fit_transform(data)

        X = []
        y = []

        sequence_length = 10

        for i in range(sequence_length, len(scaled_data)):

            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])

        X = np.array(X)
        y = np.array(y)

        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        return X, y

    def build_model(self):

        model = Sequential()

        model.add(LSTM(64, return_sequences=True,
                       input_shape=(10, 1)))

        model.add(LSTM(64))

        model.add(Dense(1))

        model.compile(
            optimizer='adam',
            loss='mean_squared_error'
        )

        return model

    def train(self):

        X, y = self.prepare_data()

        model = self.build_model()

        model.fit(
            X,
            y,
            epochs=10,
            batch_size=16
        )

        return model
    
    def predict_future(self, model):

        X, y = self.prepare_data()

        prediction = model.predict(X[-1].reshape(1, 10, 1))

        predicted_price = self.scaler.inverse_transform(
            prediction
        )

        return predicted_price[0][0]

class EnergyLSTM:

    def __init__(self, csv_path):

        self.df = pd.read_csv(csv_path)

        self.scaler = MinMaxScaler()

    def preprocess(self):

        self.df["price actual"] = pd.to_numeric(
            self.df["price actual"],
            errors="coerce"
        )

        self.df = self.df.dropna(
            subset=["price actual"]
        )

        data = self.df["price actual"].values.reshape(-1,1)

        scaled = self.scaler.fit_transform(data)

        return scaled

    def create_sequences(
        self,
        data,
        seq_length=24
    ):

        X = []
        y = []

        for i in range(
            seq_length,
            len(data)
        ):

            X.append(
                data[i-seq_length:i]
            )

            y.append(
                data[i]
            )

        X = np.array(X)
        y = np.array(y)

        return X,y

    def train(self):

        scaled = self.preprocess()

        X,y = self.create_sequences(
            scaled
        )

        model = Sequential()

        model.add(
            LSTM(
                64,
                return_sequences=True,
                input_shape=(
                    X.shape[1],
                    1
                )
            )
        )

        model.add(
            LSTM(64)
        )

        model.add(
            Dense(1)
        )

        model.compile(
            optimizer="adam",
            loss="mse"
        )

        model.fit(
            X,
            y,
            epochs=5,
            batch_size=32
        )

        self.model = model

        return model

    def predict_next_price(self):

        scaled = self.preprocess()

        last_sequence = scaled[-24:]

        X = np.array(
            [last_sequence]
        )

        prediction = self.model.predict(X)

        prediction = self.scaler.inverse_transform(
            prediction
        )

        return float(
            prediction[0][0]
        )