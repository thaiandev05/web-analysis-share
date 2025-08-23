import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import StandardScaler
import joblib
from vnstock import Vnstock
import os
import warnings
from scipy import stats

warnings.filterwarnings("ignore")

# Try to import talib, fallback to custom implementations if not available
try:
    import talib

    HAS_TALIB = True
    print("✅ TA-Lib available - Using professional technical indicators")
except ImportError:
    HAS_TALIB = False
    print("⚠️ TA-Lib not available - Using basic technical indicators")
    print("💡 To install TA-Lib: pip install TA-Lib (may require system dependencies)")

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
            # This is a simplified version
            tr = talib.ATR(high, low, close, timeperiod)
            return tr.rolling(window=timeperiod).mean()  # Simplified

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


# ==================== Advanced Features Engineering ==================== #
def make_advanced_features(df: pd.DataFrame):
    """Tạo features nâng cao với 50+ technical indicators"""
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

    # Volume Price Trend - Custom implementation since TA-Lib doesn't have VPT
    df["VPT"] = (df["Volume"] * df["Close"].pct_change()).cumsum()

    # Accumulation/Distribution Line - Custom implementation since TA-Lib doesn't have AD
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (
        df["High"] - df["Low"]
    )
    clv = clv.fillna(0)  # Handle division by zero
    df["AD"] = (clv * df["Volume"]).cumsum()

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
        "Volume",
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

    # Return multiple targets for different prediction horizons
    return (
        df[feature_columns],
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


# ==================== Feature Selection ==================== #
def select_best_features(X, y, n_features=30, method="fixed"):
    """Chọn features tốt nhất với deterministic selection"""
    if method == "fixed":
        # Fixed feature list để đảm bảo consistency
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

        # Chỉ lấy những features có trong dataset
        available_features = [f for f in fixed_features if f in X.columns]

        # Nếu không đủ features, thêm từ importance
        if len(available_features) < n_features:
            # Dùng Random Forest để tìm thêm features quan trọng
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)

            # Lấy feature importance
            feature_importance = pd.DataFrame(
                {"feature": X.columns, "importance": rf.feature_importances_}
            ).sort_values("importance", ascending=False)

            # Thêm features chưa có trong danh sách fixed
            for feature in feature_importance["feature"]:
                if (
                    feature not in available_features
                    and len(available_features) < n_features
                ):
                    available_features.append(feature)

        # Chỉ lấy số lượng features theo yêu cầu
        selected_features = available_features[:n_features]

    elif method == "univariate":
        selector = SelectKBest(score_func=f_classif, k=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()

    elif method == "rfe":
        # Sử dụng Random Forest để feature selection
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        selector = RFE(rf, n_features_to_select=n_features, step=1)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()

    print(f"✅ Selected {len(selected_features)} features using {method}")
    print(f"🔥 Selected features: {selected_features}")

    return (
        pd.DataFrame(X[selected_features], columns=selected_features, index=X.index),
        selected_features,
    )


# ==================== Advanced Model Training ==================== #
def train_advanced_model(X, y, model_type="ensemble"):
    """Train advanced ML models"""
    tscv = TimeSeriesSplit(n_splits=5)

    if model_type == "ensemble":
        # Ensemble của nhiều models
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        gb = GradientBoostingClassifier(random_state=42)
        lr = LogisticRegression(random_state=42, max_iter=1000)

        # Hyperparameter tuning cho từng model
        rf_params = {
            "n_estimators": [300, 500],
            "max_depth": [8, 12, None],
            "min_samples_leaf": [1, 2, 5],
            "max_features": ["sqrt", "log2"],
            "class_weight": [None, "balanced"],
        }

        gb_params = {
            "n_estimators": [200, 300],
            "learning_rate": [0.05, 0.1, 0.15],
            "max_depth": [4, 6, 8],
            "subsample": [0.8, 0.9, 1.0],
        }

        # Grid search cho Random Forest
        rf_search = RandomizedSearchCV(
            rf,
            rf_params,
            n_iter=8,
            cv=tscv,
            scoring="balanced_accuracy",
            random_state=42,
            n_jobs=-1,
        )
        rf_search.fit(X, y)
        best_rf = rf_search.best_estimator_

        # Grid search cho Gradient Boosting
        gb_search = RandomizedSearchCV(
            gb,
            gb_params,
            n_iter=8,
            cv=tscv,
            scoring="balanced_accuracy",
            random_state=42,
            n_jobs=-1,
        )
        gb_search.fit(X, y)
        best_gb = gb_search.best_estimator_

        # Fit Logistic Regression với scaled data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lr.fit(X_scaled, y)

        # Tạo Voting Classifier
        voting_clf = VotingClassifier(
            estimators=[("rf", best_rf), ("gb", best_gb), ("lr", lr)],
            voting="soft",  # Sử dụng probabilities
        )

        # Cần fit lại với data gốc cho ensemble
        voting_clf.fit(X, y)

        # Calibration cho ensemble
        calibrated = CalibratedClassifierCV(voting_clf, method="isotonic", cv=3)
        calibrated.fit(X, y)

        print(f"✅ Best RF params: {rf_search.best_params_}")
        print(f"✅ Best GB params: {gb_search.best_params_}")
        print(f"✅ RF score: {rf_search.best_score_:.4f}")
        print(f"✅ GB score: {gb_search.best_score_:.4f}")

        return calibrated, scaler

    else:  # single model
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        rf.fit(X, y)
        calibrated = CalibratedClassifierCV(rf, method="isotonic", cv=3)
        calibrated.fit(X, y)

        return calibrated, None


# ==================== Advanced Price Model ==================== #
def train_advanced_price_model(X, y_price):
    """Train advanced regression model"""
    tscv = TimeSeriesSplit(n_splits=5)

    # Ensemble regressor
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(random_state=42)

    rf_params = {
        "n_estimators": [300, 500],
        "max_depth": [8, 12, None],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", 0.3, 0.5],
    }

    gb_params = {
        "n_estimators": [200, 300],
        "learning_rate": [0.05, 0.1],
        "max_depth": [4, 6, 8],
        "subsample": [0.8, 0.9],
    }

    # Hyperparameter tuning
    rf_search = RandomizedSearchCV(
        rf,
        rf_params,
        n_iter=8,
        cv=tscv,
        scoring="neg_mean_squared_error",
        random_state=42,
        n_jobs=-1,
    )
    rf_search.fit(X, y_price)

    gb_search = RandomizedSearchCV(
        gb,
        gb_params,
        n_iter=8,
        cv=tscv,
        scoring="neg_mean_squared_error",
        random_state=42,
        n_jobs=-1,
    )
    gb_search.fit(X, y_price)

    print(f"✅ Price model RF score: {-rf_search.best_score_:.6f}")
    print(f"✅ Price model GB score: {-gb_search.best_score_:.6f}")

    # Return best model
    if rf_search.best_score_ > gb_search.best_score_:
        return rf_search.best_estimator_
    else:
        return gb_search.best_estimator_


# ==================== Feature Importance Analysis ==================== #
def analyze_feature_importance(model, feature_names, top_n=20):
    """Phân tích tầm quan trọng của features"""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "estimators_"):  # For ensemble models
        # Average importance across all estimators
        importances = np.mean(
            [est.feature_importances_ for est in model.estimators_], axis=0
        )
    else:
        print("⚠️ Model không hỗ trợ feature importance")
        return None

    # Tạo DataFrame với feature importance
    feature_imp = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    print(f"\n🔥 Top {top_n} quan trọng nhất:")
    print(feature_imp.head(top_n).to_string(index=False))

    return feature_imp


# ==================== Main Training Pipeline ==================== #
if __name__ == "__main__":
    # Tạo thư mục lưu model nếu chưa có
    os.makedirs("models_advanced", exist_ok=True)

    # Danh sách các mã cổ phiếu cần train
    symbols = ["ACB", "FPT", "VNM"]

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"🚀 ADVANCED TRAINING cho mã {symbol}")
        print(f"{'='*60}")

        try:
            # Load dữ liệu với nhiều năm hơn để có đủ data cho advanced features
            print(f"📊 Đang tải dữ liệu cho {symbol}...")
            df = load_stock(symbol, "2018-01-01", "2025-01-01")

            # Tạo advanced features
            print(f"🔧 Tạo advanced features...")
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
                df_full,
            ) = make_advanced_features(df)

            print(f"📈 Dữ liệu {symbol}: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"🎯 Features gốc: {list(X.columns[:10])}...")

            # Feature Selection với method fixed để đảm bảo consistency
            print(f"🎯 Selecting best features...")
            X_selected, selected_features = select_best_features(
                X, y_1d, n_features=35, method="fixed"
            )

            # Train models cho prediction 1 ngày
            print(f"🤖 Training ENSEMBLE classifier cho {symbol} (1 day)...")
            clf_1d, scaler = train_advanced_model(
                X_selected, y_1d, model_type="ensemble"
            )

            print(f"💰 Training ADVANCED price model cho {symbol} (1 day)...")
            price_model_1d = train_advanced_price_model(X_selected, price_1d)

            # Train models cho prediction 3 ngày (optional)
            print(f"🤖 Training classifier cho {symbol} (3 days)...")
            clf_3d, _ = train_advanced_model(X_selected, y_3d, model_type="single")

            # Feature importance analysis
            print(f"📊 Analyzing feature importance...")
            try:
                # Analyze cho Random Forest trong ensemble
                if hasattr(clf_1d.base_estimator, "estimators_"):
                    rf_model = clf_1d.base_estimator.estimators_[
                        0
                    ]  # RF từ voting classifier
                    analyze_feature_importance(rf_model, selected_features, top_n=15)
            except:
                print("⚠️ Không thể phân tích feature importance")

            # Lưu models và metadata
            model_data = {
                "classifier_1d": clf_1d,
                "classifier_3d": clf_3d,
                "price_model_1d": price_model_1d,
                "scaler": scaler,
                "selected_features": selected_features,
                "feature_count": len(selected_features),
            }

            joblib.dump(model_data, f"models_advanced/{symbol}_advanced_models.pkl")

            # Lưu riêng từng model để tương thích với code cũ
            joblib.dump(clf_1d, f"models_advanced/{symbol}_classifier.pkl")
            joblib.dump(price_model_1d, f"models_advanced/{symbol}_price_model.pkl")

            print(f"✅ Đã lưu ADVANCED models cho {symbol}:")
            print(f"   📁 models_advanced/{symbol}_advanced_models.pkl")
            print(f"   📁 models_advanced/{symbol}_classifier.pkl")
            print(f"   📁 models_advanced/{symbol}_price_model.pkl")
            print(f"   🎯 Features: {len(selected_features)}/70+")

        except Exception as e:
            print(f"❌ Lỗi khi training {symbol}: {str(e)}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("🎉 HOÀN THÀNH ADVANCED TRAINING!")
    print("🚀 Models được lưu trong thư mục 'models_advanced/'")
    print(
        "💡 Để sử dụng: Sửa đường dẫn trong main.py từ 'models/' → 'models_advanced/'"
    )
    print(f"{'='*60}")
