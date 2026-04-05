import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from prophet import Prophet
from pathlib import Path

class EnsembleForecaster:
    def __init__(self):
        self.ensemble = None
        self.scaler = None
        self.prophet = None
        self.is_trained = False

    def prepare_features(self, hist: pd.DataFrame) -> np.ndarray:
        closes = hist["Close"].tail(30).values
        if len(closes) < 14:
            raise ValueError("Not enough data")

        lag1 = closes[-1]
        lag7 = closes[-7]
        ma7 = np.mean(closes[-7:])
        vol7 = np.std(np.diff(closes[-7:]))

        return np.array([[lag1, lag7, ma7, vol7]])

    def predict(self, hist: pd.DataFrame) -> float:
        if self.ensemble is None or self.scaler is None or self.prophet is None:
            raise ValueError("Model not loaded")

        X = self.prepare_features(hist) 
        X_scaled = self.scaler.transform(X)
        tabular_pred = self.ensemble.predict(X_scaled)[0]

        close_series = hist["Close"].copy()
        prophet_df = close_series.reset_index()
        prophet_df.columns = ["ds", "y"]
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        prophet_df["y"] = pd.to_numeric(prophet_df["y"], errors="coerce")
        prophet_df.dropna(inplace=True)

        future = self.prophet.make_future_dataframe(periods=7)
        prophet_pred = self.prophet.predict(future)["yhat"].iloc[-1]

        return float(0.5 * tabular_pred + 0.5 * prophet_pred)

    def train_sample(self):
        import yfinance as yf

        tickers = ["AAPL", "MSFT"]
        all_data = []
        prophet_data = []

        for t in tickers:
            df = yf.download(t, period="2y", progress=False)
            close_series = df["Close"]
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.squeeze()

            df_tab = close_series.reset_index()
            df_tab.columns = ["Date", "close"]
            df_tab["return"] = df_tab["close"].pct_change()
            df_tab["lag1"] = df_tab["close"].shift(1)
            df_tab["lag7"] = df_tab["close"].shift(7)
            df_tab["ma7"] = df_tab["close"].rolling(7).mean()
            df_tab["vol7"] = df_tab["return"].rolling(7).std()
            df_tab["target"] = df_tab["close"].shift(-7)
            df_tab.dropna(inplace=True)
            all_data.append(df_tab)

            df_prophet = close_series.reset_index()
            df_prophet.columns = ["ds", "y"]
            df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])
            df_prophet["y"] = pd.to_numeric(df_prophet["y"], errors="coerce")
            df_prophet.dropna(inplace=True)
            prophet_data.append(df_prophet)

        df_all = pd.concat(all_data, ignore_index=True)
        X = df_all[["lag1", "lag7", "ma7", "vol7"]]
        y = df_all["target"]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        xgb = XGBRegressor(n_estimators=100, random_state=42)

        self.ensemble = VotingRegressor([("rf", rf), ("xgb", xgb)])
        self.ensemble.fit(X_scaled, y)

        prophet_df = pd.concat(prophet_data, ignore_index=True)
        self.prophet = Prophet()
        self.prophet.fit(prophet_df)

        self.is_trained = True
        Path("models").mkdir(exist_ok=True)
        joblib.dump(self, "models/ensemble.pkl")
        print("✅ Model trained & saved!")


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "ensemble.pkl"

def load_forecaster(path=MODEL_PATH):
    return joblib.load(path)