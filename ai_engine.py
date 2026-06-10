import pandas as pd
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


class AITradingEngine:

    def __init__(self, dataframe):
        self.df = dataframe
        self.xgb_model = None
        
    def train_xgboost(self):

        X = self.df[['demand']]
        y = self.df['price']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = XGBRegressor(n_estimators=100)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        self.xgb_model = model

        return rmse

    def predict_price(self, demand):
        return self.xgb_model.predict([[demand]])[0]

    def prophet_forecast(self):

        prophet_df = pd.DataFrame()
        prophet_df['ds'] = self.df['date']
        prophet_df['y'] = self.df['price']

        model = Prophet()
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        return forecast

    def decision_engine(self, demand, predicted_price):

        if demand > 90 and predicted_price > 80:
            return "SELL"

        elif predicted_price < 50:
            return "BUY"
        else:
            return "HOLD"