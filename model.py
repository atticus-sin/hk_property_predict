"""
model.py — Feature engineering, model training, and price prediction
for the HK property transaction dataset.
"""

import numpy as np
import pandas as pd
import joblib
import os
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

MODEL_PATH = "data/model.joblib"
ENCODERS_PATH = "data/encoders.joblib"


def prepare_features(df: pd.DataFrame, remove_outliers: bool = False) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Engineer features from raw transaction DataFrame.

    Returns:
        X: feature DataFrame with spatial + temporal + size features
        y: target Series (price in HKD)
        encoders: dict containing LabelEncoders and reference info
    """
    df = df.copy()

    # Drop rows with missing critical fields
    df = df.dropna(subset=["price"])

    # Date features
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Extract temporal features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Calculate months since earliest transaction (continuous time feature)
    reference_date = df["date"].min()
    df["months_since_start"] = (df["date"] - reference_date).dt.days / 30.44

    # Cyclical encoding for month (captures seasonality)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Numeric floor: already parsed; fill missing with median
    df["floor"] = pd.to_numeric(df["floor"], errors="coerce")
    floor_median = df["floor"].median()
    df["floor"] = df["floor"].fillna(floor_median if not np.isnan(floor_median) else 10)

    # Categorical encoding
    encoders = {}

    for col in ["block", "flat"]:
        df[col] = df[col].fillna("Unknown").astype(str)
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le

    # Calculate trend adjustment for future extrapolation
    # Use most recent 20% of data to estimate if model underestimates recent prices
    df_sorted = df.sort_values("date")
    recent_cutoff = int(len(df_sorted) * 0.8)
    df_recent = df_sorted.iloc[recent_cutoff:]

    # Store reference info for future predictions
    encoders["reference_date"] = reference_date
    encoders["floor_median"] = float(floor_median if not np.isnan(floor_median) else 10)
    encoders["recent_date_threshold"] = df_recent["date"].min() if len(df_recent) > 0 else reference_date
    encoders["max_training_date"] = df["date"].max()

    # Feature set: spatial + temporal + interactions
    feature_cols = [
        "block_enc",
        "floor",
        "flat_enc",
        "year",
        "month_sin",
        "month_cos",
        "months_since_start",
    ]

    # Add polynomial time feature for non-linear trends
    df["months_squared"] = df["months_since_start"] ** 2
    feature_cols.append("months_squared")

    # Add year-month interaction features
    df["year_month_sin"] = df["year"] * df["month_sin"]
    df["year_month_cos"] = df["year"] * df["month_cos"]
    feature_cols.extend(["year_month_sin", "year_month_cos"])

    X = df[feature_cols].copy()
    y = df["price"].astype(float)

    n_removed = 0
    if remove_outliers:
        # Remove outliers using IQR method on price (more conservative: 3.0 instead of 1.5)
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3.0 * IQR
        upper_bound = Q3 + 3.0 * IQR

        price_mask = (y >= lower_bound) & (y <= upper_bound)

        n_before = len(X)
        X = X[price_mask]
        y = y[price_mask]
        n_after = len(X)
        n_removed = n_before - n_after

        print(f"Outlier removal: {n_removed} outliers removed ({n_removed/n_before*100:.1f}%)")
        print(f"Training with {n_after} samples")

        encoders["outlier_bounds"] = {"lower": float(lower_bound), "upper": float(upper_bound)}

    encoders["outliers_removed"] = n_removed

    return X, y, encoders


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[XGBRegressor, dict]:
    """
    Train an XGBoost regressor and return (model, metrics_dict).

    Metrics: R², MAE, RMSE averaged over TimeSeriesSplit folds, plus final
    model trained on the last fold's train split for calibration.
    """
    from sklearn.isotonic import IsotonicRegression

    # TimeSeriesSplit gives temporally honest evaluation for small datasets
    tscv = TimeSeriesSplit(n_splits=5)
    cv_r2, cv_mae, cv_rmse, cv_mape = [], [], [], []

    X_arr = X.values
    y_arr = y.values

    for train_idx, test_idx in tscv.split(X_arr):
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
        m = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.1,
            reg_alpha=1.0,
            reg_lambda=3.0,
            random_state=random_state,
            tree_method='hist',
            enable_categorical=False,
        )
        m.fit(X_tr, y_tr, verbose=False)
        preds = m.predict(X_te)
        cv_r2.append(r2_score(y_te, preds))
        cv_mae.append(mean_absolute_error(y_te, preds))
        cv_rmse.append(np.sqrt(mean_squared_error(y_te, preds)))
        cv_mape.append(float(np.mean(np.abs((y_te - preds) / np.where(y_te == 0, 1, y_te)) * 100)))

    # Final model on last fold's train split (most recent data in test)
    last_train_idx, last_test_idx = list(tscv.split(X_arr))[-1]
    X_train = X.iloc[last_train_idx]
    X_test = X.iloc[last_test_idx]
    y_train = y.iloc[last_train_idx]
    y_test = y.iloc[last_test_idx]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.1,
        reg_alpha=1.0,
        reg_lambda=3.0,
        random_state=random_state,
        tree_method='hist',
        enable_categorical=False,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(y_train_pred, y_train)

    trend_adjustment = np.mean(y_train / y_train_pred)

    metrics = {
        "r2": float(np.mean(cv_r2)),
        "mae": float(np.mean(cv_mae)),
        "rmse": float(np.mean(cv_rmse)),
        "mape": float(np.mean(cv_mape)),

        "cv_r2_scores": cv_r2,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "y_test": y_test.values,
        "y_pred": y_pred,
        "X_test": X_test,
        "feature_names": list(X.columns),
        "feature_importances": model.feature_importances_,
        "trend_adjustment": float(trend_adjustment),
        "calibration_model": iso_reg,
    }

    return model, metrics


def save_model(model: XGBRegressor, encoders: dict, metrics: dict = None) -> None:
    os.makedirs("data", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    # Store trend adjustment and calibration in encoders if available
    if metrics:
        if "trend_adjustment" in metrics:
            encoders["trend_adjustment"] = metrics["trend_adjustment"]
        if "calibration_model" in metrics:
            encoders["calibration_model"] = metrics["calibration_model"]
    joblib.dump(encoders, ENCODERS_PATH)


def load_model() -> tuple[XGBRegressor, dict] | tuple[None, None]:
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        return model, encoders
    return None, None


def predict_price(
    model: XGBRegressor,
    encoders: dict,
    inputs: dict,
) -> float:
    """
    Predict price given a dict with keys:
        block (str), floor (int), flat (str), year (int), month (int)
    Returns predicted price in HKD.
    """
    block_le = encoders["block"]
    flat_le = encoders["flat"]
    reference_date = encoders["reference_date"]
    floor_median = encoders.get("floor_median", 10)

    def safe_encode(le: LabelEncoder, val: str) -> int:
        val = str(val)
        if val in le.classes_:
            return int(le.transform([val])[0])
        return 0

    # Extract inputs with defaults
    floor = float(inputs.get("floor", floor_median))
    year = int(inputs.get("year", 2024))
    month = int(inputs.get("month", 6))

    # Calculate temporal features
    target_date = pd.Timestamp(year=year, month=month, day=15)
    months_since_start = (target_date - reference_date).days / 30.44
    months_squared = months_since_start ** 2

    # Cyclical month encoding
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # Year-month interaction features
    year_month_sin = year * month_sin
    year_month_cos = year * month_cos

    # Build feature vector
    features = pd.DataFrame([{
        "block_enc": safe_encode(block_le, inputs.get("block", "Unknown")),
        "floor": floor,
        "flat_enc": safe_encode(flat_le, inputs.get("flat", "Unknown")),
        "year": year,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "months_since_start": months_since_start,
        "months_squared": months_squared,
        "year_month_sin": year_month_sin,
        "year_month_cos": year_month_cos,
    }])

    base_prediction = model.predict(features)[0]

    # Apply isotonic calibration to correct systematic bias
    calibration_model = encoders.get("calibration_model")
    if calibration_model:
        calibrated_prediction = calibration_model.predict([base_prediction])[0]
    else:
        calibrated_prediction = base_prediction

    # Apply trend adjustment for future extrapolation
    trend_adjustment = encoders.get("trend_adjustment", 1.0)
    max_training_date = encoders.get("max_training_date")

    # Apply adjustment if predicting beyond training data
    if max_training_date and target_date > max_training_date:
        # Gradually increase adjustment for dates further in the future
        months_beyond = (target_date - max_training_date).days / 30.44
        # Cap adjustment at 6 months beyond training data
        adjustment_factor = min(months_beyond / 6.0, 1.0)
        adjusted_prediction = calibrated_prediction * (1 + (trend_adjustment - 1) * adjustment_factor)
        return float(adjusted_prediction)

    return float(calibrated_prediction)


def confidence_interval(
    X: pd.DataFrame,
    y: pd.Series,
    inputs: dict,
    encoders: dict,
    alpha: float = 0.9,
) -> tuple[float, float] | None:
    """
    Estimate a prediction interval using quantile XGBoost (lower and upper).
    Only feasible when the dataset has sufficient size (>= 50 rows).
    Returns (lower, upper) in HKD or None if insufficient data.
    """
    if len(X) < 50:
        return None

    block_le = encoders["block"]
    flat_le = encoders["flat"]
    reference_date = encoders["reference_date"]
    floor_median = encoders.get("floor_median", 10)

    def safe_encode(le, val):
        val = str(val)
        return int(le.transform([val])[0]) if val in le.classes_ else 0

    # Extract inputs
    floor = float(inputs.get("floor", floor_median))
    year = int(inputs.get("year", 2024))
    month = int(inputs.get("month", 6))

    # Calculate temporal features
    target_date = pd.Timestamp(year=year, month=month, day=15)
    months_since_start = (target_date - reference_date).days / 30.44
    months_squared = months_since_start ** 2
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    year_month_sin = year * month_sin
    year_month_cos = year * month_cos

    feature_row = pd.DataFrame([{
        "block_enc": safe_encode(block_le, inputs.get("block", "Unknown")),
        "floor": floor,
        "flat_enc": safe_encode(flat_le, inputs.get("flat", "Unknown")),
        "year": year,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "months_since_start": months_since_start,
        "months_squared": months_squared,
        "year_month_sin": year_month_sin,
        "year_month_cos": year_month_cos,
    }])

    quantiles = [(1 - alpha) / 2, 1 - (1 - alpha) / 2]
    results = []
    for q in quantiles:
        xgb_q = XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=q,
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.85,
            random_state=42,
            tree_method='hist',
        )
        xgb_q.fit(X, y, verbose=False)
        quantile_pred = float(xgb_q.predict(feature_row)[0])
        results.append(quantile_pred)

    return results[0], results[1]
