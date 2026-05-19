# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streamlit dashboard for Hong Kong property price prediction, specifically targeting 28hse.com estate transaction data. Uses XGBoost regression with temporal and spatial feature engineering.

## Running the Application

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Architecture

### Data Flow Pipeline

```
28hse.com → scraper.py → CSV cache → model.py (feature engineering) → XGBoost → predictions
```

### Core Components

**scraper.py** - Web scraping with intelligent caching
- Entry point: `load_or_scrape(cache_path, force=False, base_url)`
- URL normalization: `normalize_28hse_url()` converts estate/detail or transaction URLs to canonical form
- Caching: MD5-based cache paths per estate URL (see `cache_path_for_url()` in app.py)
- Deduplication: Registry source prioritized over market source for duplicate transactions
- Chinese floor parsing: `CHINESE_FLOOR_MAP` handles "低層", "中層", "高層", "十二樓", etc.
- Known limitation: 28hse.com may use JS rendering; fallback to manual CSV upload if scraping returns empty

**model.py** - Feature engineering and prediction
- Feature pipeline: `prepare_features(df, remove_outliers=False)` → `train_model(X, y)` → `save_model()`
- Features generated:
  - Temporal: year, month_sin/cos (cyclical), months_since_start, months_squared (polynomial trend)
  - Spatial: block_enc, floor, flat_enc (label encoded)
  - Interactions: year_month_sin, year_month_cos
- Model calibration: Isotonic regression applied to correct systematic bias
- Trend adjustment: Extrapolation factor for predictions beyond training date range
- Prediction intervals: `confidence_interval()` uses quantile XGBoost (requires ≥50 samples)
- Persistence: joblib saves model + encoders (including reference_date, floor_median, calibration_model)

**app.py** - Streamlit UI with 3 tabs
- Session state keys: `df`, `model`, `encoders`, `metrics`, `X`, `y`, `source_url`, `prediction_history`
- Tab 1 (數據): Load/scrape data, upload CSV fallback, visualize transaction history
- Tab 2 (模型): Train model with optional outlier removal, display metrics (R², MAE, RMSE), feature importance, error analysis
- Tab 3 (預測): Input unit details (block/floor/flat/year/month), predict price with confidence intervals

### Key Patterns

**URL-based caching**: Each estate gets a unique cache file via MD5 hash of normalized URL, allowing multi-estate support without conflicts.

**Encoders dict structure**: Contains LabelEncoders plus metadata:
- `reference_date`: Earliest transaction date (for temporal feature calculation)
- `floor_median`: Fallback for missing floor values
- `trend_adjustment`: Multiplicative factor for future extrapolation
- `calibration_model`: IsotonicRegression for bias correction
- `max_training_date`: Used to detect when prediction is beyond training range

**Safe encoding**: `safe_encode()` in `predict_price()` handles unseen categories by defaulting to 0 (first encoded value).

**Temporal feature engineering**: Uses cyclical encoding (sin/cos) for month to capture seasonality, plus polynomial time trend (months_squared) for non-linear price growth.

## Data Format

Expected CSV columns from scraper:
- `date` (YYYY-MM-DD)
- `block` (str: "A", "B", "C", etc.)
- `floor` (int or Chinese text)
- `flat` (str: "1", "A", etc.)
- `size_sqft` (float)
- `price` (float, HKD)
- `price_per_sqft` (float)
- `source` ("registry" or "market")

## Model Hyperparameters

XGBoost configuration (model.py:141-154):
- n_estimators=800, learning_rate=0.03, max_depth=6
- Regularization: reg_alpha=0.5, reg_lambda=2.0, gamma=0.05
- Sampling: subsample=0.85, colsample_bytree=0.85
- tree_method='hist' for efficiency

Quantile models for confidence intervals use reduced complexity (n_estimators=500, max_depth=5).

## Common Modifications

**Adding a new estate**: Just paste the 28hse.com URL in Tab 1. The app auto-generates a unique cache file.

**Adjusting feature engineering**: Modify `prepare_features()` in model.py. Remember to update `predict_price()` to match the feature set.

**Changing model**: Replace XGBRegressor in `train_model()`. Ensure calibration logic is compatible or remove it.

**Handling new floor formats**: Extend `CHINESE_FLOOR_MAP` in scraper.py or update `parse_address_cell()` regex patterns.
