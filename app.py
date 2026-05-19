"""
app.py - Streamlit dashboard for HK Property Price Prediction.
"""

import hashlib
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model import (
    confidence_interval,
    load_model,
    predict_price,
    prepare_features,
    save_model,
    train_model,
)
from scraper import DEFAULT_BASE_URL, extract_estate_name, load_or_scrape, normalize_28hse_url


def cache_path_for_url(url: str) -> str:
    normalized = normalize_28hse_url(url)
    digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()[:12]
    return os.path.join("data", f"transactions_{digest}.csv")


st.set_page_config(
    page_title="28Hse 成交分析與預測",
    page_icon="🏠",
    layout="wide",
)

# Session state defaults
if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "encoders" not in st.session_state:
    st.session_state.encoders = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "X" not in st.session_state:
    st.session_state.X = None
if "y" not in st.session_state:
    st.session_state.y = None
if "source_url" not in st.session_state:
    st.session_state.source_url = DEFAULT_BASE_URL
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

current_url = st.session_state.source_url
current_estate_name = extract_estate_name(current_url)

# Header
st.title(f"🏠 {current_estate_name} 樓價分析與預測")
st.caption(f"資料來源：[28hse.com]({normalize_28hse_url(current_url)})")

tab_data, tab_model, tab_predict = st.tabs(["📋 數據", "🤖 模型", "🔮 預測"])


with tab_data:
    st.header("成交紀錄")

    url_input = st.text_input(
        "28hse 連結",
        value=current_url,
        help="可貼上屋苑詳情頁或成交頁，例如 https://www.28hse.com/estate/detail/... 或 .../transaction",
    )

    normalized_url = None
    cache_path = None
    try:
        normalized_url = normalize_28hse_url(url_input)
        cache_path = cache_path_for_url(normalized_url)
        detected_name = extract_estate_name(normalized_url)
        col_name, col_link = st.columns([3, 1])
        with col_name:
            st.caption(f"屋苑：**{detected_name}** ｜ 成交頁：`{normalized_url}`")
        with col_link:
            st.link_button("🔗 開啟連結", normalized_url, use_container_width=True)
    except ValueError as err:
        st.error(str(err))

    col_load, col_rescrape = st.columns([2, 1])
    with col_load:
        load_btn = st.button("📥 載入數據（使用快取）", use_container_width=True)
    with col_rescrape:
        rescrape_btn = st.button("🔄 重新抓取數據", use_container_width=True)

    st.markdown("**或上傳 CSV 檔案**（如抓取失敗，可手動上傳）")
    uploaded = st.file_uploader(
        "上傳 transactions.csv",
        type="csv",
        label_visibility="collapsed",
    )

    if uploaded is not None:
        if cache_path is None:
            st.error("請先輸入有效的 28hse 連結。")
        else:
            df = pd.read_csv(uploaded, parse_dates=["date"])
            st.session_state.df = df
            st.session_state.source_url = normalized_url
            os.makedirs("data", exist_ok=True)
            df.to_csv(cache_path, index=False, encoding="utf-8-sig")
            st.success(f"已上傳 {len(df)} 筆紀錄並儲存至目前連結的快取。")

    if load_btn:
        if normalized_url is None or cache_path is None:
            st.error("請輸入有效的 28hse 連結。")
        else:
            with st.spinner("載入中…"):
                df = load_or_scrape(cache_path=cache_path, force=False, base_url=normalized_url)
                st.session_state.df = df
                st.session_state.source_url = normalized_url
            if df.empty:
                st.warning(
                    "未能取得任何成交紀錄。網站可能使用 JavaScript 動態渲染，"
                    "請嘗試手動下載 CSV 並使用上方上傳功能。"
                )
            else:
                st.success(f"已載入 {len(df)} 筆紀錄。")

    if rescrape_btn:
        if normalized_url is None or cache_path is None:
            st.error("請輸入有效的 28hse 連結。")
        else:
            with st.spinner("正在抓取所有頁面…（可能需要數十秒）"):
                df = load_or_scrape(cache_path=cache_path, force=True, base_url=normalized_url)
                st.session_state.df = df
                st.session_state.source_url = normalized_url
            if df.empty:
                st.warning(
                    "抓取失敗：網站可能需要 JavaScript 渲染。\n\n"
                    "**替代方案**：\n"
                    "1. 安裝 `requests-html` 並修改 scraper.py 使用 `AsyncHTMLSession`\n"
                    "2. 或使用瀏覽器手動儲存頁面後轉為 CSV 上傳"
                )
            else:
                st.success(f"已抓取並儲存 {len(df)} 筆紀錄。")

    df = st.session_state.df
    if df is not None and not df.empty:
        st.divider()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("總筆數", f"{len(df):,}")
        with c2:
            date_min = pd.to_datetime(df["date"]).min()
            st.metric("最早日期", str(date_min)[:10] if pd.notna(date_min) else "N/A")
        with c3:
            date_max = pd.to_datetime(df["date"]).max()
            st.metric("最新日期", str(date_max)[:10] if pd.notna(date_max) else "N/A")
        with c4:
            avg_price = df["price"].mean() / 10000 if "price" in df.columns else 0
            st.metric("平均成交價", f"{avg_price:.0f} 萬")

        st.subheader("成交紀錄表")
        display_df = df.copy()
        if "price" in display_df.columns:
            display_df["price_萬"] = (display_df["price"] / 10000).round(1)
        st.dataframe(display_df, use_container_width=True, height=400)

        if "date" in df.columns and "price" in df.columns:
            st.subheader("成交價格走勢")
            chart_df = df.dropna(subset=["date", "price"]).copy()
            chart_df["date"] = pd.to_datetime(chart_df["date"])
            chart_df["price_萬"] = chart_df["price"] / 10000
            fig = px.scatter(
                chart_df,
                x="date",
                y="price_萬",
                color="block" if "block" in chart_df.columns else None,
                hover_data=["floor", "flat", "size_sqft"] if all(
                    c in chart_df.columns for c in ["floor", "flat", "size_sqft"]
                ) else None,
                labels={"price_萬": "成交價（萬）", "date": "成交日期"},
                title="歷史成交價格",
            )
            fig.update_traces(marker=dict(size=6, opacity=0.7))
            st.plotly_chart(fig, use_container_width=True)

        csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="⬇️ 下載 CSV",
            data=csv_bytes,
            file_name="transactions.csv",
            mime="text/csv",
        )
    else:
        st.info("請先輸入 28hse 連結，然後點擊「載入數據」或「重新抓取數據」。")


with tab_model:
    st.header("模型訓練")

    if st.session_state.model is None:
        model, encoders = load_model()
        if model is not None:
            st.session_state.model = model
            st.session_state.encoders = encoders
            st.info("已從磁碟載入之前訓練的模型。")

    remove_outliers_checkbox = st.checkbox(
        "移除異常值（極端價格）",
        value=False,
        help="使用 IQR 方法移除極端價格。可能提升或降低 R²，視數據而定。"
    )
    dedup_market_checkbox = st.checkbox(
        "移除重複的市場成交（保留登記版本）",
        value=True,
        help="同一單位若同時有「market」及「registry」紀錄，移除較早的「market」版本，避免重複計算。"
    )

    train_btn = st.button("🚀 訓練模型", use_container_width=False)

    df = st.session_state.df
    if train_btn:
        if df is None or df.empty:
            st.error("請先在「數據」分頁載入成交紀錄。")
        elif len(df) < 10:
            st.error(f"數據不足（僅 {len(df)} 筆），無法訓練模型，至少需要 10 筆。")
        else:
            with st.spinner("訓練中…"):
                train_df = df.copy()
                if dedup_market_checkbox and "source" in train_df.columns:
                    # For each (block, floor, flat, price) group, if a registry record exists,
                    # drop all market records in that group (they are preliminary duplicates).
                    has_registry = train_df.groupby(["block", "floor", "flat", "price"])["source"].transform(
                        lambda s: (s == "registry").any()
                    )
                    before = len(train_df)
                    train_df = train_df[~((train_df["source"] == "market") & has_registry)].reset_index(drop=True)
                    removed = before - len(train_df)
                    if removed:
                        st.info(f"已移除 {removed} 筆重複的市場成交紀錄。")
                X, y, encoders = prepare_features(train_df, remove_outliers=remove_outliers_checkbox)
                if len(X) < 10:
                    st.error("特徵工程後資料不足，請檢查數據品質。")
                else:
                    model, metrics = train_model(X, y)
                    save_model(model, encoders, metrics)
                    st.session_state.model = model
                    st.session_state.encoders = encoders
                    st.session_state.metrics = metrics
                    st.session_state.X = X
                    st.session_state.y = y
                    st.success("模型訓練完成並已儲存！")

    metrics = st.session_state.metrics
    if metrics:
        st.divider()
        st.subheader("模型表現")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            mape = metrics.get("mape")
            st.metric("MAPE", f"{mape:.1f}%" if mape is not None else "N/A", help="平均絕對百分比誤差 — 越低越好，<10% 為佳")
        with m2:
            st.metric("MAE", f"HK$ {metrics['mae']:,.0f}", help="平均絕對誤差（港元）")
        with m3:
            st.metric("RMSE", f"HK$ {metrics['rmse']:,.0f}", help="均方根誤差，對大誤差更敏感")
        with m4:
            st.metric("R²", f"{metrics['r2']:.3f}", help="決定係數，受數據噪音影響，僅供參考")

        trend_adj = metrics.get("trend_adjustment", 1.0)
        trend_pct = (trend_adj - 1.0) * 100
        outliers_removed = st.session_state.encoders.get("outliers_removed", 0) if st.session_state.encoders else 0
        st.caption(
            f"訓練集：{metrics['n_train']} 筆 ｜ 測試集：{metrics['n_test']} 筆 ｜ "
            f"趨勢調整：{trend_pct:+.2f}% ｜ 已移除異常值：{outliers_removed} 筆"
        )

        st.subheader("特徵重要性")
        feat_labels = {
            "block_enc": "座 (Block)",
            "floor": "樓層 (Floor)",
            "flat_enc": "室 (Flat)",
            "year": "年份",
            "month_sin": "月份 (sin)",
            "month_cos": "月份 (cos)",
            "months_since_start": "時間趨勢",
            "months_squared": "時間趨勢²",
            "year_month_sin": "年份×月份 (sin)",
            "year_month_cos": "年份×月份 (cos)",
        }
        names = [feat_labels.get(f, f) for f in metrics["feature_names"]]
        importances = metrics["feature_importances"]
        fi_df = pd.DataFrame({"Feature": names, "Importance": importances})
        fi_df = fi_df.sort_values("Importance", ascending=True)
        fig_fi = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Gradient Boosting 特徵重要性",
            labels={"Importance": "重要性", "Feature": "特徵"},
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        st.subheader("實際 vs 預測成交價")
        ap_df = pd.DataFrame({
            "實際價格（萬）": metrics["y_test"] / 10000,
            "預測價格（萬）": metrics["y_pred"] / 10000,
        })
        fig_ap = px.scatter(
            ap_df,
            x="實際價格（萬）",
            y="預測價格（萬）",
            opacity=0.7,
            title="實際 vs 預測（測試集）",
        )
        min_val = ap_df.min().min()
        max_val = ap_df.max().max()
        fig_ap.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="完美預測線",
            )
        )
        st.plotly_chart(fig_ap, use_container_width=True)

        st.subheader("預測價格時間序列")
        # Need to reconstruct dates from X_test features
        X_test_data = metrics.get("X_test")
        if X_test_data is not None and "months_since_start" in X_test_data.columns:
            # Reconstruct dates from months_since_start
            encoders_loaded = st.session_state.encoders
            if encoders_loaded and "reference_date" in encoders_loaded:
                reference_date = encoders_loaded["reference_date"]
                months_vals = X_test_data["months_since_start"].values
                dates = [reference_date + pd.Timedelta(days=m * 30.44) for m in months_vals]

                time_df = pd.DataFrame({
                    "日期": dates,
                    "實際價格（萬）": metrics["y_test"] / 10000,
                    "預測價格（萬）": metrics["y_pred"] / 10000,
                })
                time_df["誤差（萬）"] = time_df["實際價格（萬）"] - time_df["預測價格（萬）"]
                time_df["誤差百分比"] = (time_df["誤差（萬）"] / time_df["實際價格（萬）"]) * 100
                time_df = time_df.sort_values("日期")

                fig_time = go.Figure()
                fig_time.add_trace(go.Scatter(
                    x=time_df["日期"],
                    y=time_df["實際價格（萬）"],
                    mode="markers",
                    name="實際價格",
                    marker=dict(size=8, opacity=0.6, color="blue"),
                ))
                fig_time.add_trace(go.Scatter(
                    x=time_df["日期"],
                    y=time_df["預測價格（萬）"],
                    mode="markers",
                    name="預測價格",
                    marker=dict(size=8, opacity=0.6, color="red"),
                ))
                fig_time.update_layout(
                    title="實際 vs 預測價格（時間序列）",
                    xaxis_title="日期",
                    yaxis_title="價格（萬）",
                    hovermode="closest",
                )
                st.plotly_chart(fig_time, use_container_width=True)

                # Error over time chart
                st.subheader("預測誤差隨時間變化")

                # Calculate rolling average of error
                time_df_sorted = time_df.sort_values("日期").copy()
                window_size = max(5, len(time_df_sorted) // 20)  # Adaptive window
                time_df_sorted["誤差滾動平均"] = time_df_sorted["誤差（萬）"].rolling(
                    window=window_size, min_periods=1, center=True
                ).mean()

                fig_error = go.Figure()

                # Scatter points for individual errors
                fig_error.add_trace(go.Scatter(
                    x=time_df_sorted["日期"],
                    y=time_df_sorted["誤差（萬）"],
                    mode="markers",
                    name="個別誤差",
                    marker=dict(size=6, opacity=0.4, color="gray"),
                    hovertemplate="日期: %{x}<br>誤差: %{y:.1f} 萬<extra></extra>",
                ))

                # Line for rolling average
                fig_error.add_trace(go.Scatter(
                    x=time_df_sorted["日期"],
                    y=time_df_sorted["誤差滾動平均"],
                    mode="lines",
                    name=f"滾動平均 ({window_size}筆)",
                    line=dict(color="red", width=3),
                    hovertemplate="日期: %{x}<br>平均誤差: %{y:.1f} 萬<extra></extra>",
                ))

                # Zero line
                fig_error.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="black",
                    opacity=0.5,
                    annotation_text="零誤差線",
                )

                fig_error.update_layout(
                    title="預測誤差時間趨勢（正值=低估，負值=高估）",
                    xaxis_title="日期",
                    yaxis_title="誤差（萬）= 實際 - 預測",
                    hovermode="closest",
                )
                st.plotly_chart(fig_error, use_container_width=True)

                # Summary stats
                avg_error = time_df["誤差（萬）"].mean()
                avg_error_pct = time_df["誤差百分比"].mean()
                st.caption(
                    f"平均誤差：{avg_error:+.1f} 萬 ({avg_error_pct:+.1f}%) ｜ "
                    f"正值表示模型低估，負值表示模型高估"
                )
            else:
                st.caption("無法重建日期資訊")
        else:
            st.caption("測試集無時間特徵資料")
    else:
        st.info("點擊「訓練模型」以查看模型表現。")


with tab_predict:
    st.header("價格預測")

    model = st.session_state.model
    encoders = st.session_state.encoders
    df = st.session_state.df
    estate_name = extract_estate_name(st.session_state.source_url)

    if model is None:
        st.warning("請先在「模型」分頁訓練模型。")
    else:
        st.markdown("輸入單位資料以預測成交價格：")

        if df is not None and not df.empty:
            block_choices = sorted(df["block"].dropna().astype(str).unique().tolist())
            flat_choices = sorted(df["flat"].dropna().astype(str).unique().tolist())
            if not block_choices:
                block_choices = ["A", "B", "C", "D"]
            if not flat_choices:
                flat_choices = [chr(c) for c in range(ord("A"), ord("N"))]
        else:
            block_choices = ["A", "B", "C", "D"]
            flat_choices = [chr(c) for c in range(ord("A"), ord("N"))]

        if encoders and "block" in encoders:
            known_blocks = list(encoders["block"].classes_)
            all_blocks = sorted(set(block_choices + [b for b in known_blocks if b != "Unknown"]))
        else:
            all_blocks = block_choices

        if encoders and "flat" in encoders:
            known_flats = list(encoders["flat"].classes_)
            all_flats = sorted(set(flat_choices + [f for f in known_flats if f != "Unknown"]))
        else:
            all_flats = flat_choices

        col1, col2 = st.columns(2)
        with col1:
            sel_block = st.selectbox("座 (Block)", options=all_blocks or ["A"])
            sel_floor = st.number_input("樓層 (Floor)", min_value=1, max_value=60, value=10, step=1)
            sel_flat = st.selectbox("室 (Flat)", options=all_flats or ["A"])
        with col2:
            sel_year = st.number_input("成交年份", min_value=2000, max_value=2035, value=2026, step=1)
            sel_month = st.selectbox(
                "成交月份",
                options=list(range(1, 13)),
                format_func=lambda m: [
                    "一月", "二月", "三月", "四月", "五月", "六月",
                    "七月", "八月", "九月", "十月", "十一月", "十二月",
                ][m - 1],
                index=3,  # April
            )

        predict_btn = st.button("🔮 預測價格", type="primary", use_container_width=False)

        if predict_btn:
            inputs = {
                "block": sel_block,
                "floor": sel_floor,
                "flat": sel_flat,
                "year": sel_year,
                "month": sel_month,
            }
            predicted = predict_price(model, encoders, inputs)
            predicted_wan = predicted / 10000

            st.divider()
            st.subheader("預測結果")
            st.metric(
                label=f"{estate_name} - {sel_block}座 {sel_floor}樓 {sel_flat}室",
                value=f"HK$ {predicted:,.0f}",
                delta=f"約 {predicted_wan:.1f} 萬",
            )

            X_feat = st.session_state.X
            y_feat = st.session_state.y
            ci_lower, ci_upper = None, None
            if X_feat is not None and len(X_feat) >= 50:
                with st.spinner("計算預測區間中…"):
                    try:
                        ci = confidence_interval(X_feat, y_feat, inputs, encoders, alpha=0.9)
                        if ci:
                            ci_lower, ci_upper = ci
                            st.info(
                                f"**90% 預測區間**：HK$ {ci_lower:,.0f} 至 HK$ {ci_upper:,.0f} "
                                f"（即 {ci_lower/10000:.1f} 萬 至 {ci_upper/10000:.1f} 萬）"
                            )
                    except Exception as e:
                        st.caption(f"（預測區間計算失敗：{e}）")
            else:
                st.caption(
                    "注意：數據量不足（少於 50 筆），無法計算預測區間。"
                    "預測結果僅供參考，實際成交價受市場因素影響。"
                )

            # Store prediction in history
            from datetime import datetime
            prediction_record = {
                "時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "座": sel_block,
                "樓層": sel_floor,
                "室": sel_flat,
                "年份": sel_year,
                "月份": sel_month,
                "預測價格（萬）": round(predicted_wan, 1),
                "下限（萬）": round(ci_lower / 10000, 1) if ci_lower else None,
                "上限（萬）": round(ci_upper / 10000, 1) if ci_upper else None,
            }
            st.session_state.prediction_history.append(prediction_record)

        # Display prediction history
        if st.session_state.prediction_history:
            st.divider()
            st.subheader("預測記錄")

            col_clear, col_download = st.columns([1, 1])
            with col_clear:
                if st.button("🗑️ 清空記錄", use_container_width=True):
                    st.session_state.prediction_history = []
                    st.rerun()

            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(history_df, use_container_width=True, height=300)

            with col_download:
                csv_history = history_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    label="⬇️ 下載記錄",
                    data=csv_history,
                    file_name=f"prediction_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
