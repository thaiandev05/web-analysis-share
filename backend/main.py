from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np
from vnstock import Vnstock
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from typing import List, Dict
from datetime import datetime, timedelta
from dotenv import load_dotenv
import redis.asyncio as aioredis
import os
import json
import pickle
import hashlib
import joblib

# Try to import talib, fallback to custom implementations if not available
try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    print("⚠️ TA-Lib not available - Advanced features may be limited")

    # Custom implementations for basic indicators
    class talib:
        @staticmethod
        def RSI(prices, timeperiod=14):
            """Custom RSI implementation"""
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        @staticmethod
        def MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9):
            """Custom MACD implementation"""
            ema_fast = prices.ewm(span=fastperiod).mean()
            ema_slow = prices.ewm(span=slowperiod).mean()
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=signalperiod).mean()
            histogram = macd - signal
            return macd, signal, histogram

        @staticmethod
        def BBANDS(prices, timeperiod=20, nbdevup=2, nbdevdn=2):
            """Custom Bollinger Bands implementation"""
            middle = prices.rolling(window=timeperiod).mean()
            std = prices.rolling(window=timeperiod).std()
            upper = middle + (std * nbdevup)
            lower = middle - (std * nbdevdn)
            return upper, middle, lower

        @staticmethod
        def ATR(high, low, close, timeperiod=14):
            """Custom ATR implementation"""
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            tr = np.maximum(high_low, np.maximum(high_close, low_close))
            return tr.rolling(window=timeperiod).mean()

        @staticmethod
        def STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3):
            """Custom Stochastic implementation"""
            lowest_low = low.rolling(window=fastk_period).min()
            highest_high = high.rolling(window=fastk_period).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            k_percent = k_percent.rolling(window=slowk_period).mean()
            d_percent = k_percent.rolling(window=slowd_period).mean()
            return k_percent, d_percent

        @staticmethod
        def WILLR(high, low, close, timeperiod=14):
            """Custom Williams %R implementation"""
            highest_high = high.rolling(window=timeperiod).max()
            lowest_low = low.rolling(window=timeperiod).min()
            return -100 * ((highest_high - close) / (highest_high - lowest_low))

        @staticmethod
        def CCI(high, low, close, timeperiod=14):
            """Custom CCI implementation"""
            tp = (high + low + close) / 3
            ma = tp.rolling(window=timeperiod).mean()
            md = tp.rolling(window=timeperiod).apply(
                lambda x: np.abs(x - x.mean()).mean()
            )
            return (tp - ma) / (0.015 * md)

        @staticmethod
        def ADX(high, low, close, timeperiod=14):
            """Simplified ADX implementation"""
            tr = talib.ATR(high, low, close, timeperiod)
            return tr.rolling(window=timeperiod).mean()

        @staticmethod
        def OBV(close, volume):
            """On Balance Volume implementation"""
            obv = pd.Series(index=close.index, dtype=float)
            obv.iloc[0] = volume.iloc[0]
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i - 1]:
                    obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i - 1]:
                    obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i - 1]
            return obv

        @staticmethod
        def VPT(close, volume):
            """Volume Price Trend implementation"""
            return (volume * close.pct_change()).cumsum()

        @staticmethod
        def AD(high, low, close, volume):
            """Accumulation/Distribution implementation"""
            clv = ((close - low) - (high - close)) / (high - low)
            clv = clv.fillna(0)  # Handle division by zero
            ad = (clv * volume).cumsum()
            return ad

        @staticmethod
        def TRANGE(high, low, close):
            """True Range implementation"""
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            return np.maximum(high_low, np.maximum(high_close, low_close))


load_dotenv()

app = FastAPI(title="Stock Prediction API")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
redis = None


@app.on_event("startup")
async def startup():
    global redis
    redis = await aioredis.from_url(
        f"redis://{REDIS_HOST}:{REDIS_PORT}", encoding="utf-8", decode_responses=True
    )


symbols = ["ACB", "FPT", "VNM"]
data = {}


# ================== Cache Helper Functions ==================
def get_cache_key(prefix: str, **kwargs) -> str:
    """Tạo cache key duy nhất từ parameters"""
    key_data = f"{prefix}:" + ":".join([f"{k}={v}" for k, v in sorted(kwargs.items())])
    return hashlib.md5(key_data.encode()).hexdigest()


async def get_from_cache(key: str):
    """Lấy dữ liệu từ cache"""
    try:
        data = await redis.get(key)
        if data:
            return pickle.loads(data.encode("latin1"))
    except Exception as e:
        print(f"Cache get error: {e}")
    return None


async def set_to_cache(key: str, data, expire_seconds: int = 3600):
    """Lưu dữ liệu vào cache"""
    try:
        serialized = pickle.dumps(data).decode("latin1")
        await redis.set(key, serialized, ex=expire_seconds)
    except Exception as e:
        print(f"Cache set error: {e}")


async def get_json_from_cache(key: str):
    """Lấy JSON data từ cache"""
    try:
        data = await redis.get(key)
        if data:
            return json.loads(data)
    except Exception as e:
        print(f"Cache get JSON error: {e}")
    return None


async def set_json_to_cache(key: str, data, expire_seconds: int = 3600):
    """Lưu JSON data vào cache"""
    try:
        await redis.set(key, json.dumps(data, default=str), ex=expire_seconds)
    except Exception as e:
        print(f"Cache set JSON error: {e}")


# ================== Model Loading Helper ==================
def load_pretrained_models(symbol: str, model_type: str = "auto"):
    """Load pre-trained models for a specific symbol

    Args:
        symbol: Stock symbol (ACB, FPT, VNM)
        model_type: "auto", "advanced", "basic"
    """
    try:
        if model_type == "auto" or model_type == "advanced":
            # Try advanced models first
            advanced_path = f"models_advanced/{symbol}_advanced_models.pkl"
            if os.path.exists(advanced_path):
                print(f"🚀 Loading ADVANCED models for {symbol}")
                model_data = joblib.load(advanced_path)
                return (
                    model_data["classifier_1d"],
                    model_data["price_model_1d"],
                    "advanced",
                    model_data.get("selected_features", []),  # Return selected features
                )

        if model_type == "auto" or model_type == "basic":
            # Fallback to basic models
            clf_path = f"models/{symbol}_classifier.pkl"
            price_path = f"models/{symbol}_price_model.pkl"

            if os.path.exists(clf_path) and os.path.exists(price_path):
                print(f"📊 Loading BASIC models for {symbol}")
                clf_model = joblib.load(clf_path)
                price_model = joblib.load(price_path)
                # Basic features list
                basic_features = [
                    "MA5",
                    "MA20",
                    "Daily Return",
                    "Volume",
                    "Volume Change",
                ]
                return clf_model, price_model, "basic", basic_features

        print(f"⚠️ No pre-trained models found for {symbol}")
        return None, None, None, None

    except Exception as e:
        print(f"❌ Error loading models for {symbol}: {e}")
        return None, None, None, None


# ================== 1. Data Loader with Cache ==================
async def load_stock(sym: str, start: str, end: str):
    # Cache key cho dữ liệu stock
    cache_key = get_cache_key("stock_data", symbol=sym, start=start, end=end)

    # Kiểm tra cache trước
    cached_data = await get_from_cache(cache_key)
    if cached_data is not None:
        return cached_data

    # Nếu không có cache, lấy dữ liệu mới
    stock = Vnstock().stock(symbol=sym, source="TCBS")
    df = stock.quote.history(start=start, end=end, interval="1D")
    df = df.rename(
        columns={
            "time": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    # Cache dữ liệu trong 1 giờ (3600 giây)
    await set_to_cache(cache_key, df, expire_seconds=3600)
    return df


# ================== 2. Features with Cache ==================
async def make_features(df: pd.DataFrame, symbol: str):
    # Cache key cho features
    cache_key = get_cache_key(
        "features", symbol=symbol, shape=str(df.shape), last_date=str(df.index[-1])
    )

    # Kiểm tra cache
    cached_features = await get_from_cache(cache_key)
    if cached_features is not None:
        return cached_features

    df = df.copy()

    # Basic features (backward compatibility)
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["Daily Return"] = df["Close"].pct_change()
    df["Cumulative Return"] = (1 + df["Daily Return"]).cumprod()
    df["Volume Change"] = df["Volume"].pct_change()

    # Enhanced features for better accuracy
    try:
        import talib

        # Additional moving averages
        df["MA10"] = df["Close"].rolling(10).mean()
        df["MA50"] = df["Close"].rolling(50).mean()
        df["EMA12"] = df["Close"].ewm(span=12).mean()
        df["EMA26"] = df["Close"].ewm(span=26).mean()

        # Technical indicators
        df["RSI"] = talib.RSI(df["Close"], timeperiod=14)
        df["MACD"], df["MACD_signal"], df["MACD_hist"] = talib.MACD(df["Close"])
        df["BB_upper"], df["BB_middle"], df["BB_lower"] = talib.BBANDS(
            df["Close"], timeperiod=20
        )
        df["ATR"] = talib.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)

        # Price position in Bollinger Bands
        df["BB_position"] = (df["Close"] - df["BB_lower"]) / (
            df["BB_upper"] - df["BB_lower"]
        )

        # Price vs moving averages
        df["Price_MA5_ratio"] = df["Close"] / df["MA5"] - 1
        df["Price_MA20_ratio"] = df["Close"] / df["MA20"] - 1

        # Volatility
        df["Volatility"] = df["Daily Return"].rolling(20).std() * np.sqrt(252)

        # Volume indicators
        df["Volume_MA"] = df["Volume"].rolling(20).mean()
        df["Volume_ratio"] = df["Volume"] / df["Volume_MA"] - 1

        # Enhanced feature list
        features = [
            "MA5",
            "MA20",
            "MA10",
            "MA50",
            "EMA12",
            "EMA26",
            "Daily Return",
            "Volume",
            "Volume Change",
            "Volume_ratio",
            "RSI",
            "MACD",
            "MACD_signal",
            "MACD_hist",
            "BB_position",
            "ATR",
            "Price_MA5_ratio",
            "Price_MA20_ratio",
            "Volatility",
        ]

    except ImportError:
        print("⚠️ TA-Lib not available, using basic features")
        # Fallback to basic features
        features = ["MA5", "MA20", "Daily Return", "Volume", "Volume Change"]

    # Labels
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    df["Price_Target"] = df["Close"].shift(-1)

    # Remove NaN values
    df = df.dropna()

    # Filter features that exist in dataframe
    available_features = [
        f for f in features if f in df.columns and not df[f].isna().all()
    ]

    result = (df[available_features], df["Target"], df["Price_Target"], df)

    # Cache features trong 30 phút
    await set_to_cache(cache_key, result, expire_seconds=1800)
    return result


# ================== 2b. Advanced Features with Cache ==================
async def make_advanced_features(df: pd.DataFrame, symbol: str):
    """Tạo features nâng cao với 50+ technical indicators - tương thích với advanced models"""
    # Cache key cho advanced features
    cache_key = get_cache_key(
        "advanced_features",
        symbol=symbol,
        shape=str(df.shape),
        last_date=str(df.index[-1]),
    )

    # Kiểm tra cache
    cached_features = await get_from_cache(cache_key)
    if cached_features is not None:
        return cached_features

    df = df.copy()

    # 1. ========== Price-based features ==========
    # Moving averages (nhiều kỳ hạn)
    for period in [3, 5, 10, 20, 50, 100]:
        df[f"MA{period}"] = df["Close"].rolling(period).mean()
        df[f"MA{period}_ratio"] = df["Close"] / df[f"MA{period}"] - 1

    # Exponential Moving Averages
    for period in [12, 26, 50]:
        df[f"EMA{period}"] = df["Close"].ewm(span=period).mean()
        df[f"EMA{period}_ratio"] = df["Close"] / df[f"EMA{period}"] - 1

    # Price channels và Bollinger Bands
    df["BB_upper"], df["BB_middle"], df["BB_lower"] = talib.BBANDS(
        df["Close"], timeperiod=20
    )
    df["BB_position"] = (df["Close"] - df["BB_lower"]) / (
        df["BB_upper"] - df["BB_lower"]
    )
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]

    # 2. ========== Momentum indicators ==========
    # RSI với nhiều kỳ hạn
    for period in [7, 14, 21]:
        df[f"RSI{period}"] = talib.RSI(df["Close"], timeperiod=period)

    # MACD
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = talib.MACD(df["Close"])

    # Stochastic
    df["STOCH_k"], df["STOCH_d"] = talib.STOCH(df["High"], df["Low"], df["Close"])

    # Williams %R
    df["WILLR"] = talib.WILLR(df["High"], df["Low"], df["Close"])

    # Commodity Channel Index
    df["CCI"] = talib.CCI(df["High"], df["Low"], df["Close"])

    # Average Directional Index
    df["ADX"] = talib.ADX(df["High"], df["Low"], df["Close"])

    # 3. ========== Volume indicators ==========
    # Volume Moving Averages
    df["Volume_MA5"] = df["Volume"].rolling(5).mean()
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_ratio"] = df["Volume"] / df["Volume_MA20"] - 1

    # On Balance Volume
    df["OBV"] = talib.OBV(df["Close"], df["Volume"])
    df["OBV_MA"] = df["OBV"].rolling(10).mean()
    df["OBV_signal"] = df["OBV"] / df["OBV_MA"] - 1

    # Volume Price Trend (custom, TA-Lib does NOT have VPT)
    df["VPT"] = (df["Volume"] * df["Close"].pct_change()).cumsum()

    # Accumulation/Distribution Line (TA-Lib has AD)
    df["AD"] = talib.AD(df["High"], df["Low"], df["Close"], df["Volume"])

    # 4. ========== Volatility indicators ==========
    # True Range và Average True Range
    df["TR"] = talib.TRANGE(df["High"], df["Low"], df["Close"])
    df["ATR"] = talib.ATR(df["High"], df["Low"], df["Close"])
    df["ATR_pct"] = df["ATR"] / df["Close"] * 100

    # Price volatility
    for period in [5, 10, 20]:
        df[f"Volatility{period}"] = df["Close"].pct_change().rolling(
            period
        ).std() * np.sqrt(252)

    # 5. ========== Return features ==========
    # Returns với nhiều kỳ hạn
    for period in [1, 2, 3, 5, 10]:
        df[f"Return{period}"] = df["Close"].pct_change(period)
        df[f"Return{period}_MA"] = df[f"Return{period}"].rolling(10).mean()

    # Log returns
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Cumulative returns
    df["Cumulative_Return"] = (1 + df["Return1"]).cumprod()

    # 6. ========== Market microstructure ==========
    # High-Low spread
    df["HL_pct"] = (df["High"] - df["Low"]) / df["Close"] * 100

    # Open-Close relationship
    df["OC_pct"] = (df["Close"] - df["Open"]) / df["Open"] * 100

    # Gap analysis
    df["Gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1) * 100

    # Intraday return
    df["Intraday_Return"] = (df["Close"] - df["Open"]) / df["Open"]

    # 7. ========== Pattern recognition ==========
    # Support và Resistance levels
    df["Support"] = df["Low"].rolling(20).min()
    df["Resistance"] = df["High"].rolling(20).max()
    df["Support_distance"] = (df["Close"] - df["Support"]) / df["Close"] * 100
    df["Resistance_distance"] = (df["Resistance"] - df["Close"]) / df["Close"] * 100

    # Price position trong range
    df["Price_position"] = (df["Close"] - df["Support"]) / (
        df["Resistance"] - df["Support"]
    )

    # 8. ========== Time-based features ==========
    df["DayOfWeek"] = df.index.dayofweek
    df["DayOfMonth"] = df.index.day
    df["Month"] = df.index.month
    df["Quarter"] = df.index.quarter

    # Week/Month effects
    df["Monday"] = (df["DayOfWeek"] == 0).astype(int)
    df["Friday"] = (df["DayOfWeek"] == 4).astype(int)
    df["MonthEnd"] = (df.index.day > 25).astype(int)

    # 9. ========== Regime detection ==========
    # Trend detection
    df["Trend_5"] = np.where(df["MA5"] > df["MA20"], 1, -1)
    df["Trend_20"] = np.where(df["MA20"] > df["MA50"], 1, -1)

    # Market regime (Bull/Bear/Sideways)
    df["Market_Regime"] = 0  # Sideways
    df.loc[df["MA20"] > df["MA50"] * 1.02, "Market_Regime"] = 1  # Bull
    df.loc[df["MA20"] < df["MA50"] * 0.98, "Market_Regime"] = -1  # Bear

    # 10. ========== Statistical features ==========
    # Rolling statistics
    for period in [10, 20]:
        df[f"Skew{period}"] = df["Return1"].rolling(period).skew()
        df[f"Kurt{period}"] = df["Return1"].rolling(period).kurt()
        df[f"Mean{period}"] = df["Return1"].rolling(period).mean()
        df[f"Std{period}"] = df["Return1"].rolling(period).std()

    # Z-score
    df["ZScore_20"] = (df["Close"] - df["MA20"]) / df["Close"].rolling(20).std()

    # Labels - Multiple prediction horizons
    df["Target_1d"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    df["Target_3d"] = np.where(df["Close"].shift(-3) > df["Close"], 1, 0)
    df["Target_5d"] = np.where(df["Close"].shift(-5) > df["Close"], 1, 0)

    # Price targets
    df["Price_Target_1d"] = df["Close"].shift(-1)
    df["Price_Target_3d"] = df["Close"].shift(-3)
    df["Price_Target_5d"] = df["Close"].shift(-5)

    # Return targets
    df["Return_Target_1d"] = df["Close"].shift(-1) / df["Close"] - 1
    df["Return_Target_3d"] = df["Close"].shift(-3) / df["Close"] - 1
    df["Return_Target_5d"] = df["Close"].shift(-5) / df["Close"] - 1

    # Remove rows with NaN
    df = df.dropna()

    # Select features (exclude price columns, targets, and index columns)
    exclude_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Date",
        "Target_1d",
        "Target_3d",
        "Target_5d",
        "Price_Target_1d",
        "Price_Target_3d",
        "Price_Target_5d",
        "Return_Target_1d",
        "Return_Target_3d",
        "Return_Target_5d",
        "Support",
        "Resistance",
        "BB_upper",
        "BB_middle",
        "BB_lower",
    ]

    feature_columns = [col for col in df.columns if col not in exclude_columns]

    # Fixed feature selection để match với training
    fixed_features = [
        # Basic price features
        "MA5",
        "MA20",
        "Daily Return",
        "Volume",
        "Volume Change",
        # Moving averages ratios
        "MA5_ratio",
        "MA10_ratio",
        "MA20_ratio",
        # EMA features
        "EMA12",
        "EMA26",
        "EMA12_ratio",
        "EMA26_ratio",
        # RSI features
        "RSI14",
        "RSI7",
        "RSI21",
        # MACD features
        "MACD",
        "MACD_signal",
        "MACD_hist",
        # Bollinger Bands
        "BB_position",
        "BB_width",
        # Volume features
        "Volume_ratio",
        "OBV_signal",
        "VPT",
        "AD",
        # Volatility
        "ATR",
        "ATR_pct",
        "Volatility10",
        "Volatility20",
        # Returns
        "Return1",
        "Return2",
        "Return3",
        "Return5",
        # Market structure
        "HL_pct",
        "OC_pct",
        "Gap",
        "Intraday_Return",
        # Time features
        "DayOfWeek",
        "DayOfMonth",
        "Month",
    ]

    # Add basic compatibility
    if "Daily Return" not in df.columns:
        df["Daily Return"] = df["Return1"]
    if "Volume Change" not in df.columns:
        df["Volume Change"] = df["Volume"].pct_change()

    # Chỉ lấy những features có trong dataset
    available_features = [f for f in fixed_features if f in df.columns]

    # Return multiple targets for different prediction horizons
    result = (
        df[available_features],
        df["Target_1d"],
        df["Target_3d"],
        df["Target_5d"],
        df["Price_Target_1d"],
        df["Price_Target_3d"],
        df["Price_Target_5d"],
        df["Return_Target_1d"],
        df["Return_Target_3d"],
        df["Return_Target_5d"],
        df,
    )

    # Cache features trong 30 phút
    await set_to_cache(cache_key, result, expire_seconds=1800)
    return result


# ================== 3. Model Training with Cache ==================
async def train_model(X, y, symbol: str):
    # Cache key cho model
    cache_key = get_cache_key(
        "classification_model", symbol=symbol, data_shape=str(X.shape)
    )

    # Kiểm tra cache model
    cached_model = await get_from_cache(cache_key)
    if cached_model is not None:
        return cached_model

    tscv = TimeSeriesSplit(n_splits=5)
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        rf,
        param_distributions={
            "n_estimators": [200, 300],
            "max_depth": [6, 8, None],
            "min_samples_leaf": [10, 20],
            "max_features": ["sqrt", 0.5],
            "class_weight": [None, "balanced"],
        },
        n_iter=5,
        cv=tscv,
        scoring="balanced_accuracy",
        random_state=42,
    )
    search.fit(X, y)
    best_model = search.best_estimator_
    calibrated = CalibratedClassifierCV(best_model, method="isotonic", cv=3)
    calibrated.fit(X, y)

    # Cache model trong 4 giờ
    await set_to_cache(cache_key, calibrated, expire_seconds=14400)
    return calibrated


async def train_price_model(X, y_price, symbol: str):
    # Cache key cho price model
    cache_key = get_cache_key("price_model", symbol=symbol, data_shape=str(X.shape))

    # Kiểm tra cache model
    cached_model = await get_from_cache(cache_key)
    if cached_model is not None:
        return cached_model

    tscv = TimeSeriesSplit(n_splits=5)
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        rf,
        param_distributions={
            "n_estimators": [200, 300],
            "max_depth": [6, 8, None],
            "min_samples_leaf": [10, 20],
            "max_features": ["sqrt", 0.5],
        },
        n_iter=5,
        cv=tscv,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    search.fit(X, y_price)

    # Cache model trong 4 giờ
    await set_to_cache(cache_key, search.best_estimator_, expire_seconds=14400)
    return search.best_estimator_


# ================== 4. Backtest ==================
def backtest(df, model, X):
    df["Signal"] = model.predict(X)
    df["Strategy Return"] = df["Signal"].shift(1) * df["Daily Return"]
    df["Equity Curve"] = (1 + df["Strategy Return"].fillna(0)).cumprod()
    df["BuyHold"] = (1 + df["Daily Return"].fillna(0)).cumprod()
    return df


# ================== 5. Metrics ==================
def metrics(df):
    total_return = df["Equity Curve"].iloc[-1] - 1
    days = (df.index[-1] - df.index[0]).days
    years = days / 365
    cagr = (df["Equity Curve"].iloc[-1]) ** (1 / years) - 1
    roll_max = df["Equity Curve"].cummax()
    drawdown = df["Equity Curve"] / roll_max - 1
    max_drawdown = drawdown.min()
    sharpe = (
        np.mean(df["Strategy Return"].dropna())
        / np.std(df["Strategy Return"].dropna(), ddof=1)
        * np.sqrt(252)
    )
    win_rate = (df["Strategy Return"] > 0).sum() / df["Strategy Return"].count()
    gains = df["Strategy Return"][df["Strategy Return"] > 0].sum()
    losses = df["Strategy Return"][df["Strategy Return"] < 0].sum()
    profit_factor = gains / abs(losses) if losses != 0 else np.inf
    return {
        "Total Return": round(total_return, 3),
        "CAGR": round(cagr, 3),
        "Max Drawdown": round(max_drawdown, 3),
        "Sharpe": round(sharpe, 3),
        "Win Rate": round(win_rate, 3),
        "Profit Factor": round(profit_factor, 3),
    }


# ================== API with Cache ==================
@app.get("/prediction/{symbol}")
async def prediction(
    symbol: str, start: str = None, end: str = None, model_type: str = "auto"
):
    """
    Get stock prediction

    Args:
        symbol: Stock symbol (ACB, FPT, VNM)
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        model_type: "auto", "advanced", "basic"
    """
    if start is None:
        start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    # Cache key cho kết quả prediction
    cache_key = get_cache_key(
        "prediction", symbol=symbol, start=start, end=end, model_type=model_type
    )

    # Kiểm tra cache prediction (cache ngắn hạn - 15 phút)
    cached_result = await get_json_from_cache(cache_key)
    if cached_result is not None:
        return cached_result

    # Load dữ liệu
    df = await load_stock(symbol, start, end)

    # Thử load pre-trained models trước để biết sử dụng features nào
    clf_model, price_model, used_model_type, selected_features = load_pretrained_models(
        symbol, model_type
    )

    # Tạo features phù hợp với model type
    if used_model_type == "advanced":
        print(f"🚀 Using ADVANCED features for {symbol}")
        result = await make_advanced_features(df, symbol)
        (
            X,
            y_1d,
            y_3d,
            y_5d,
            price_1d,
            price_3d,
            price_5d,
            return_1d,
            return_3d,
            return_5d,
            df,
        ) = result
        # Chỉ sử dụng 1 day prediction cho compatibility
        y, y_price = y_1d, price_1d

        # Filter features theo selected_features đã train
        if selected_features:
            available_selected = [f for f in selected_features if f in X.columns]
            if len(available_selected) > 0:
                X = X[available_selected]
                print(f"✅ Using {len(available_selected)} pre-selected features")
            else:
                print(
                    f"⚠️ No pre-selected features found, using all {len(X.columns)} features"
                )
    else:
        print(f"📊 Using BASIC features for {symbol}")
        X, y, y_price, df = await make_features(df, symbol)

        # Filter basic features nếu có
        if selected_features:
            available_selected = [f for f in selected_features if f in X.columns]
            if len(available_selected) > 0:
                X = X[available_selected]

    if clf_model is not None and price_model is not None:
        print(f"✅ Using {used_model_type} models for {symbol}")
        model = clf_model
        price_model_final = price_model
    else:
        print(f"🔄 Training new models for {symbol}")
        model = await train_model(X, y, symbol)
        price_model_final = await train_price_model(X, y_price, symbol)
        used_model_type = "newly_trained"

    # Backtest với model
    df_bt = backtest(df, model, X)

    # 🔮 Dự đoán cho ngày mai
    last_features = X.iloc[[-1]]  # lấy row cuối cùng
    pred = model.predict(last_features)[0]
    prob = model.predict_proba(last_features)[0].tolist()

    # Dự đoán giá cụ thể
    predicted_price = price_model_final.predict(last_features)[0]
    current_price = df["Close"].iloc[-1]
    price_change = predicted_price - current_price
    percentage_change = (price_change / current_price) * 100

    result = {
        "symbol": symbol,
        "model_type": used_model_type,
        "metrics": metrics(df_bt),
        "last_equity": df_bt["Equity Curve"].iloc[-1],
        "prediction_next_day": {
            "signal": int(pred),  # 1 = tăng, 0 = giảm
            "probability": prob,
            "current_price": round(current_price, 2),
            "predicted_price": round(predicted_price, 2),
            "price_change": round(price_change, 2),
            "percentage_change": round(percentage_change, 2),
        },
        "history": df_bt[["Close", "Signal", "Equity Curve"]].tail(50).to_dict(),
    }

    # Cache kết quả trong 15 phút
    await set_json_to_cache(cache_key, result, expire_seconds=900)
    return result


@app.get("/portfolio")
async def portfolio(start: str = "2024-01-01", end: str = "2024-08-01"):
    # Cache key cho portfolio
    cache_key = get_cache_key("portfolio", start=start, end=end)

    # Kiểm tra cache portfolio (cache 30 phút)
    cached_result = await get_json_from_cache(cache_key)
    if cached_result is not None:
        return cached_result

    results = {}
    for sym in symbols:
        df = await load_stock(sym, start, end)
        X, y, y_price, df = await make_features(df, sym)
        model = await train_model(X, y, sym)
        df_bt = backtest(df, model, X)
        results[sym] = df_bt
    portfolio = pd.concat(
        [results[sym]["Equity Curve"].rename(sym) for sym in symbols], axis=1
    ).dropna()
    portfolio["Portfolio"] = portfolio.mean(axis=1)
    port_return = portfolio["Portfolio"].pct_change().fillna(0)
    portfolio_df = pd.DataFrame(
        {"Equity Curve": portfolio["Portfolio"], "Strategy Return": port_return}
    )

    result = {"metrics": metrics(portfolio_df)}

    # Cache portfolio trong 30 phút
    await set_json_to_cache(cache_key, result, expire_seconds=1800)
    return result


# ================== Model Comparison & Info ==================
@app.get("/models/compare/{symbol}")
async def compare_models(symbol: str, start: str = None, end: str = None):
    """So sánh hiệu suất giữa basic và advanced models"""
    if start is None:
        start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    # Cache key cho comparison
    cache_key = get_cache_key("model_comparison", symbol=symbol, start=start, end=end)

    cached_result = await get_json_from_cache(cache_key)
    if cached_result is not None:
        return cached_result

    # Load data
    df = await load_stock(symbol, start, end)

    results = {}

    # Test basic model
    clf_basic, price_basic, model_type = load_pretrained_models(symbol, "basic")
    if clf_basic is not None:
        # Use basic features for basic model
        X_basic, y_basic, y_price_basic, df_basic = await make_features(df, symbol)
        df_bt_basic = backtest(df, clf_basic, X_basic)
        results["basic"] = {
            "metrics": metrics(df_bt_basic),
            "model_type": "basic",
            "available": True,
            "features_count": len(X_basic.columns),
        }
    else:
        results["basic"] = {"available": False}

    # Test advanced model
    clf_advanced, price_advanced, model_type, selected_features_adv = (
        load_pretrained_models(symbol, "advanced")
    )
    if clf_advanced is not None:
        # Use advanced features for advanced model
        result_advanced = await make_advanced_features(df, symbol)
        (
            X_advanced,
            y_advanced,
            y_3d,
            y_5d,
            price_1d,
            price_3d,
            price_5d,
            return_1d,
            return_3d,
            return_5d,
            df_advanced,
        ) = result_advanced

        # Filter features theo selected_features
        if selected_features_adv:
            available_selected = [
                f for f in selected_features_adv if f in X_advanced.columns
            ]
            if len(available_selected) > 0:
                X_advanced = X_advanced[available_selected]

        df_bt_advanced = backtest(df, clf_advanced, X_advanced)
        results["advanced"] = {
            "metrics": metrics(df_bt_advanced),
            "model_type": "advanced",
            "available": True,
            "features_count": len(X_advanced.columns),
        }
    else:
        results["advanced"] = {"available": False}

    # Calculate improvement
    if results["basic"]["available"] and results["advanced"]["available"]:
        basic_sharpe = results["basic"]["metrics"]["Sharpe"]
        advanced_sharpe = results["advanced"]["metrics"]["Sharpe"]
        improvement = (
            ((advanced_sharpe - basic_sharpe) / abs(basic_sharpe)) * 100
            if basic_sharpe != 0
            else 0
        )

        results["comparison"] = {
            "sharpe_improvement_pct": round(improvement, 2),
            "recommendation": "advanced" if advanced_sharpe > basic_sharpe else "basic",
        }

    result = {
        "symbol": symbol,
        "comparison_period": f"{start} to {end}",
        "results": results,
    }

    # Cache kết quả trong 30 phút
    await set_json_to_cache(cache_key, result, expire_seconds=1800)
    return result


@app.get("/models/info")
async def models_info():
    """Thông tin về các models có sẵn"""
    info = {"basic_models": {}, "advanced_models": {}, "total_models": 0}

    for symbol in symbols:
        # Check basic models
        basic_clf = f"models/{symbol}_classifier.pkl"
        basic_price = f"models/{symbol}_price_model.pkl"
        info["basic_models"][symbol] = {
            "classifier": os.path.exists(basic_clf),
            "price_model": os.path.exists(basic_price),
            "complete": os.path.exists(basic_clf) and os.path.exists(basic_price),
        }

        # Check advanced models
        advanced_path = f"models_advanced/{symbol}_advanced_models.pkl"
        advanced_clf = f"models_advanced/{symbol}_classifier.pkl"
        advanced_price = f"models_advanced/{symbol}_price_model.pkl"

        info["advanced_models"][symbol] = {
            "complete_package": os.path.exists(advanced_path),
            "classifier": os.path.exists(advanced_clf),
            "price_model": os.path.exists(advanced_price),
            "complete": os.path.exists(advanced_path),
        }

        if info["basic_models"][symbol]["complete"]:
            info["total_models"] += 1
        if info["advanced_models"][symbol]["complete"]:
            info["total_models"] += 1

    return info


@app.get("/models/features/{symbol}")
async def model_features(symbol: str, model_type: str = "advanced"):
    """Xem features được sử dụng bởi model"""
    try:
        if model_type == "advanced":
            advanced_path = f"models_advanced/{symbol}_advanced_models.pkl"
            if os.path.exists(advanced_path):
                model_data = joblib.load(advanced_path)
                return {
                    "symbol": symbol,
                    "model_type": "advanced",
                    "feature_count": model_data["feature_count"],
                    "selected_features": model_data["selected_features"],
                }

        return {
            "symbol": symbol,
            "model_type": "basic",
            "feature_count": 5,  # Basic model uses 5 features
            "selected_features": [
                "MA5",
                "MA20",
                "Daily Return",
                "Volume",
                "Volume Change",
            ],
        }

    except Exception as e:
        return {"error": f"Cannot load features for {symbol}: {str(e)}"}


# ================== Cache Management ==================
@app.get("/cache/status")
async def cache_status():
    """Kiểm tra trạng thái cache"""
    try:
        info = await redis.info()
        return {
            "status": "connected",
            "used_memory": info.get("used_memory_human", "N/A"),
            "total_keys": await redis.dbsize(),
            "redis_version": info.get("redis_version", "N/A"),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.delete("/cache/clear")
async def clear_cache(pattern: str = "*"):
    """Xóa cache theo pattern"""
    try:
        keys = await redis.keys(pattern)
        if keys:
            await redis.delete(*keys)
            return {"message": f"Đã xóa {len(keys)} keys với pattern '{pattern}'"}
        return {"message": "Không có key nào để xóa"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/cache/keys")
async def list_cache_keys(pattern: str = "*", limit: int = 100):
    """Liệt kê các cache keys"""
    try:
        keys = await redis.keys(pattern)
        return {"total_keys": len(keys), "keys": keys[:limit], "pattern": pattern}
    except Exception as e:
        return {"status": "error", "message": str(e)}
