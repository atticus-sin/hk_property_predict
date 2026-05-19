# HK Property Price Prediction

A Streamlit dashboard for predicting Hong Kong property transaction prices using historical data from 28hse.com. Built with XGBoost regression and temporal/spatial feature engineering.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

## Usage

1. **數據 tab** — Paste a `28hse.com` estate URL and click **載入數據** to load cached data or **重新抓取數據** to scrape fresh transactions. Alternatively, upload a CSV manually.
2. **模型 tab** — Click **訓練模型** to train the model. View R², MAE, RMSE, feature importance, and error analysis.
3. **預測 tab** — Select block, floor, flat, size, year, and month, then click **預測價格** to get a price estimate with confidence interval.

## Notes

- Scraped data is cached locally in `data/` by estate URL — re-scraping is only needed when new transactions are available.
- If scraping returns empty (JS rendering), use the CSV upload fallback.
- Confidence intervals require at least 50 training samples.
