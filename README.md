# 28hse 樓價預測 Dashboard

## Setup

```bash
pip install -r requirements.txt
```

## Launch

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

## Usage

1. **數據 tab** - Paste any valid `28hse.com` estate-detail or transaction link, then click **載入數據** to use that link's cache or **重新抓取數據** to scrape up to 25 transaction pages.
2. **模型 tab** - Click **訓練模型** to train the Gradient Boosting model and inspect R², MAE, RMSE, and feature importance charts.
3. **預測 tab** - Select block, floor, flat, size, year, and month, then click **預測價格** to estimate the current estate's transaction price in HKD.
