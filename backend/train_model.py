import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
import joblib
from vnstock import Vnstock
import os
import joblib

# ==================== Load dữ liệu ==================== #
def load_stock(sym: str, start: str, end: str):
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
    return df


# ==================== Tạo features ==================== #
def make_features(df: pd.DataFrame):
    df = df.copy()

    # Moving averages
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()

    # Returns
    df["Daily Return"] = df["Close"].pct_change()
    df["Cumulative Return"] = (1 + df["Daily Return"]).cumprod()

    # Volume features
    df["Volume Change"] = df["Volume"].pct_change()

    # Labels
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)   # tăng/giảm
    df["Price_Target"] = df["Close"].shift(-1)                            # giá ngày mai

    df = df.dropna()

    features = ["MA5", "MA20", "Daily Return", "Volume", "Volume Change"]
    X, y, y_price = df[features], df["Target"], df["Price_Target"]

    return X, y, y_price, df


# ==================== Train classifier (Up/Down) ==================== #
def train_model(X, y):
    tscv = TimeSeriesSplit(n_splits=5)

    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    search = RandomizedSearchCV(
        rf,
        param_distributions={
            "n_estimators": [200, 300, 400],
            "max_depth": [6, 8, 12, None],
            "min_samples_leaf": [2, 5, 10],
            "max_features": ["sqrt", "log2", 0.5],
        },
        n_iter=10,
        cv=tscv,
        scoring="balanced_accuracy",
        random_state=42,
        n_jobs=-1
    )

    search.fit(X, y)
    best_model = search.best_estimator_

    # Calibration cho xác suất tin cậy hơn
    calibrated = CalibratedClassifierCV(best_model, method="sigmoid", cv=3)
    calibrated.fit(X, y)

    return calibrated


# ==================== Train regressor (Price) ==================== #
def train_price_model(X, y_price):
    tscv = TimeSeriesSplit(n_splits=5)

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    search = RandomizedSearchCV(
        rf,
        param_distributions={
            "n_estimators": [200, 300, 400],
            "max_depth": [6, 8, 12, None],
            "min_samples_leaf": [2, 5, 10],
            "max_features": ["sqrt", "log2", 0.5],
        },
        n_iter=10,
        cv=tscv,
        scoring="neg_mean_squared_error",
        random_state=42,
        n_jobs=-1
    )

    search.fit(X, y_price)
    best_model = search.best_estimator_

    return best_model

# ==================== Run training ==================== #
if __name__ == "__main__":
	# Tạo thư mục lưu model nếu chưa có
	os.makedirs("models", exist_ok=True)
	
	# Danh sách các mã cổ phiếu cần train
	symbols = ["ACB", "FPT", "VNM"]
	
	for symbol in symbols:
		print(f"\n{'='*50}")
		print(f"🚀 Bắt đầu training cho mã {symbol}")
		print(f"{'='*50}")
		
		try:
			# Load dữ liệu
			print(f"📊 Đang tải dữ liệu cho {symbol}...")
			df = load_stock(symbol, "2020-01-01", "2025-01-01")
			X, y, y_price, df = make_features(df)
			
			print(f"� Dữ liệu {symbol}: {X.shape[0]} samples, {X.shape[1]} features")

			# Train classifier (Up/Down)
			print(f"�🔄 Training classifier (Up/Down) cho {symbol}...")
			clf = train_model(X, y)
			
			# Train regressor (Price)
			print(f"🔄 Training regressor (Price) cho {symbol}...")
			reg = train_price_model(X, y_price)

			# Lưu models riêng cho từng mã
			joblib.dump(clf, f"models/{symbol}_classifier.pkl")
			joblib.dump(reg, f"models/{symbol}_price_model.pkl")

			print(f"✅ Đã lưu models cho {symbol}:")
			print(f"   - models/{symbol}_classifier.pkl")
			print(f"   - models/{symbol}_price_model.pkl")
			
		except Exception as e:
			print(f"❌ Lỗi khi training {symbol}: {str(e)}")
			continue

	print(f"\n{'='*50}")
	print("🎉 Hoàn thành training cho tất cả các mã!")
	print(f"{'='*50}")
