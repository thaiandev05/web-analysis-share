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


# ================== 2. Features with Cache ==================Model đã train: Cache các model đã train để không phải train lại
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
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["Daily Return"] = df["Close"].pct_change()
    df["Cumulative Return"] = (1 + df["Daily Return"]).cumprod()
    df["Volume Change"] = df["Volume"].pct_change()
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    df["Price_Target"] = df["Close"].shift(-1)
    df = df.dropna()
    features = ["MA5", "MA20", "Daily Return", "Volume", "Volume Change"]

    result = (df[features], df["Target"], df["Price_Target"], df)

    # Cache features trong 30 phút
    await set_to_cache(cache_key, result, expire_seconds=1800)
    return result


# ================== 3. Model with Cache ==================
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
async def prediction(symbol: str, start: str = None, end: str = None):
    if start is None:
        start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    # Cache key cho kết quả prediction
    cache_key = get_cache_key("prediction", symbol=symbol, start=start, end=end)

    # Kiểm tra cache prediction (cache ngắn hạn - 15 phút)
    cached_result = await get_json_from_cache(cache_key)
    if cached_result is not None:
        return cached_result

    df = await load_stock(symbol, start, end)
    X, y, y_price, df = await make_features(df, symbol)
    model = await train_model(X, y, symbol)
    price_model = await train_price_model(X, y_price, symbol)
    df_bt = backtest(df, model, X)

    # 🔮 Dự đoán cho ngày mai
    last_features = X.iloc[[-1]]  # lấy row cuối cùng
    pred = model.predict(last_features)[0]
    prob = model.predict_proba(last_features)[0].tolist()

    # Dự đoán giá cụ thể
    predicted_price = price_model.predict(last_features)[0]
    current_price = df["Close"].iloc[-1]
    price_change = predicted_price - current_price
    percentage_change = (price_change / current_price) * 100

    result = {
        "symbol": symbol,
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
