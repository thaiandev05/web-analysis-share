import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

API_URL = "http://localhost:8000"  # FastAPI backend

st.set_page_config(page_title="Stock Dashboard", layout="wide")
st.title("📊 Stock Prediction Dashboard")

symbols = ["ACB", "FPT", "VNM"]
choice = st.selectbox("Chọn mã cổ phiếu", symbols)

if st.button("Lấy dự đoán"):
    with st.spinner("Đang tải dữ liệu và dự đoán..."):
        res = requests.get(f"{API_URL}/prediction/{choice}")

    if res.status_code == 200:
        data = res.json()

        # Hiển thị thông tin cơ bản
        st.subheader(f"📈 Phân tích cho mã {data['symbol']}")

        # Chia layout thành 2 cột
        col1, col2 = st.columns(2)

        with col1:
            st.write("### 📊 Metrics hiệu suất")
            metrics_df = pd.DataFrame([data["metrics"]]).T
            metrics_df.columns = ["Giá trị"]
            st.dataframe(metrics_df, use_container_width=True)

            st.write(f"**Last Equity:** {data['last_equity']:.3f}")

        with col2:
            # 🔮 Hiển thị dự đoán ngày mai
            pred = data["prediction_next_day"]
            st.write("### 🔮 Dự đoán ngày mai")

            # Signal với màu sắc
            if pred["signal"] == 1:
                st.success("📈 **TĂNG GIÁ**")
            else:
                st.error("📉 **GIẢM GIÁ**")

            # Thông tin giá - FIX: Hiển thị đúng định dạng VND
            current_price = pred["current_price"] * 1000  # Convert to VND
            predicted_price = pred["predicted_price"] * 1000  # Convert to VND
            price_change = predicted_price - current_price
            percentage_change = (price_change / current_price) * 100

            st.write(f"**Giá hiện tại:** {current_price:,.0f} VNĐ")
            st.write(f"**Giá dự đoán:** {predicted_price:,.0f} VNĐ")
            st.write(
                f"**Thay đổi:** {price_change:+.0f} VNĐ ({percentage_change:+.2f}%)"
            )

            # Xác suất
            st.write("**Xác suất:**")
            prob_df = pd.DataFrame(
                {
                    "Hướng": ["Giảm (0)", "Tăng (1)"],
                    "Xác suất": [
                        f"{pred['probability'][0]:.1%}",
                        f"{pred['probability'][1]:.1%}",
                    ],
                }
            )
            st.dataframe(prob_df, hide_index=True)

        # Vẽ biểu đồ - FIX: Sửa lỗi indexing
        st.write("### 📈 Biểu đồ lịch sử")
        hist = data["history"]

        # Chuyển đổi dictionary thành DataFrame
        close_data = pd.Series(hist["Close"])
        signal_data = pd.Series(hist["Signal"])
        equity_data = pd.Series(hist["Equity Curve"])

        fig = go.Figure()

        # Đường giá Close
        fig.add_trace(
            go.Scatter(
                x=close_data.index,
                y=close_data.values,
                mode="lines",
                name="Giá đóng cửa",
                line=dict(color="blue", width=2),
            )
        )

        # Đường Equity Curve
        fig.add_trace(
            go.Scatter(
                x=equity_data.index,
                y=equity_data.values,
                mode="lines",
                name="Equity Curve",
                line=dict(color="green", width=2),
                yaxis="y2",
            )
        )

        # FIX: Tín hiệu mua/bán - Sửa lỗi indexing
        buy_signals_idx = signal_data[signal_data == 1].index
        sell_signals_idx = signal_data[signal_data == 0].index

        if len(buy_signals_idx) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals_idx,
                    y=[
                        close_data.loc[i] for i in buy_signals_idx
                    ],  # Sử dụng .loc thay vì .iloc
                    mode="markers",
                    name="Tín hiệu MUA",
                    marker=dict(color="green", size=8, symbol="triangle-up"),
                )
            )

        if len(sell_signals_idx) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals_idx,
                    y=[
                        close_data.loc[i] for i in sell_signals_idx
                    ],  # Sử dụng .loc thay vì .iloc
                    mode="markers",
                    name="Tín hiệu BÁN",
                    marker=dict(color="red", size=8, symbol="triangle-down"),
                )
            )

        # Cấu hình layout với dual y-axis
        fig.update_layout(
            title=f"Lịch sử giá và tín hiệu giao dịch - {choice}",
            xaxis_title="Thời gian",
            yaxis=dict(title="Giá (VNĐ)", side="left"),
            yaxis2=dict(title="Equity Curve", side="right", overlaying="y"),
            hovermode="x unified",
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error(f"Lỗi khi lấy dữ liệu: {res.status_code}")

if st.button("Danh mục (Portfolio)"):
    with st.spinner("Đang phân tích danh mục..."):
        res = requests.get(f"{API_URL}/portfolio")

    if res.status_code == 200:
        data = res.json()
        st.subheader("📊 Portfolio Metrics")

        # Hiển thị metrics dưới dạng bảng đẹp
        metrics_df = pd.DataFrame([data["metrics"]]).T
        metrics_df.columns = ["Giá trị"]

        # Tạo màu sắc cho các metrics
        styled_metrics = metrics_df.style.format(
            {
                "Giá trị": lambda x: (
                    f"{x:.1%}"
                    if "Rate" in str(x) or "Return" in str(x) or "CAGR" in str(x)
                    else f"{x:.3f}"
                )
            }
        )

        st.dataframe(styled_metrics, use_container_width=True)

        # Hiển thị một số thông tin nổi bật
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Return", f"{data['metrics']['Total Return']:.1%}", delta=None
            )

        with col2:
            st.metric("CAGR", f"{data['metrics']['CAGR']:.1%}", delta=None)

        with col3:
            st.metric("Sharpe Ratio", f"{data['metrics']['Sharpe']:.3f}", delta=None)
    else:
        st.error(f"Lỗi khi lấy dữ liệu portfolio: {res.status_code}")

# Thêm thông tin bổ sung
st.sidebar.markdown("### ℹ️ Thông tin")
st.sidebar.markdown(
    """
**Hướng dẫn sử dụng:**
1. Chọn mã cổ phiếu từ dropdown
2. Nhấn "Lấy dự đoán" để xem phân tích
3. Nhấn "Danh mục" để xem hiệu suất tổng thể

**Giải thích metrics:**
- **Total Return**: Tổng lợi nhuận
- **CAGR**: Tỷ suất tăng trưởng kép hàng năm
- **Max Drawdown**: Mức thua lỗ tối đa
- **Sharpe**: Tỷ lệ rủi ro/lợi nhuận
- **Win Rate**: Tỷ lệ giao dịch thắng
- **Profit Factor**: Hệ số lợi nhuận
"""
)
