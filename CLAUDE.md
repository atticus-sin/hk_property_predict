# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Streamlit dashboard for Hong Kong property price prediction using 28hse.com estate transaction data. XGBoost regression with temporal and spatial feature engineering.

## Running the Application

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Architecture

```
28hse.com → scraper.py → data/*.csv → model.py → XGBoost → predictions
```

### scraper.py
- Entry: `load_or_scrape(cache_path, force=False, base_url)`
- `normalize_28hse_url()` converts any estate/detail/transaction URL to canonical form
- Cache files are MD5-keyed per estate URL via `cache_path_for_url()` in app.py
- Registry source takes priority over market source for duplicate transactions
- `CHINESE_FLOOR_MAP` handles "低層", "中層", "高層", "十二樓", etc.
- Known limitation: JS-rendered pages may return empty — use CSV upload fallback

### model.py
- Pipeline: `prepare_features()` → `train_model()` → `save_model()`
- Features: year, month_sin/cos, months_since_start, months_squared, block_enc, floor, flat_enc, year_month_sin/cos
- Isotonic regression calibration corrects systematic bias
- `confidence_interval()` uses quantile XGBoost (requires ≥50 samples)
- Saved artifacts: model + encoders via joblib (includes reference_date, floor_median, calibration_model, max_training_date, trend_adjustment)

### app.py
- 3-tab Streamlit UI: 數據 / 模型 / 預測
- Session state keys: `df`, `model`, `encoders`, `metrics`, `X`, `y`, `source_url`, `prediction_history`
- `safe_encode()` handles unseen label categories by defaulting to 0

## Data Format

CSV columns: `date`, `block`, `floor`, `flat`, `size_sqft`, `price`, `price_per_sqft`, `source`

## Model Hyperparameters (model.py:141-154)

- n_estimators=800, learning_rate=0.03, max_depth=6
- reg_alpha=0.5, reg_lambda=2.0, gamma=0.05
- subsample=0.85, colsample_bytree=0.85, tree_method='hist'
- Quantile models: n_estimators=500, max_depth=5

## Common Modifications

- **New estate**: Paste URL in Tab 1 — cache file is auto-generated.
- **Feature engineering**: Edit `prepare_features()` in model.py and mirror changes in `predict_price()`.
- **New floor formats**: Extend `CHINESE_FLOOR_MAP` or update `parse_address_cell()` regex in scraper.py.
