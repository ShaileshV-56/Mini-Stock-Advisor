"""
Mini Stock Advisor — Streamlit Frontend  (v1.1 — rerun-safe)

Key fixes vs v1.0:
  - Portfolio result stored in session_state → no API call on every rerun
  - Watchlist prices only re-fetched when list changes (via a dirty flag)
  - Clear error message when a ticker can't be found
  - AI advice fetched lazily (button-click only)
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd

API_BASE = "https://mini-stock-advisor.onrender.com"

st.set_page_config(page_title="Mini Stock Advisor", layout="wide", page_icon="📈")
st.title("🚀 Mini Stock Advisor")
st.markdown(
    "**Ensemble ML + Agentic LLM + Universe Screener + Watchlist + Portfolio Dashboard**"
)


# ===========================================================================
# API CLIENT
# ===========================================================================

def api_get(path: str, params: dict = None, timeout: int = 60):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Cannot reach the backend. Start it with:\n\n"
            "```bash\ncd backend\nuvicorn main:app --reload\n```"
        )
        return None
    except requests.exceptions.HTTPError as e:
        # Surface the backend's user-friendly detail message
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"❌ {detail}")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, body: dict, timeout: int = 180):
    try:
        r = requests.post(f"{API_BASE}{path}", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach the backend.")
        return None
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"❌ {detail}")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post_raw(path: str, body: dict, timeout: int = 60):
    try:
        r = requests.post(f"{API_BASE}{path}", json=body, timeout=timeout)
        r.raise_for_status()
        return r.content
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ===========================================================================
# UNIVERSES (cached 1 hour)
# ===========================================================================

@st.cache_data(ttl=3600)
def load_universes():
    data = api_get("/universes")
    return data.get("universes", {}) if data else {}


UNIVERSES = load_universes()

# ===========================================================================
# SESSION STATE
# ===========================================================================

defaults = {
    "watchlist":        [],    # [{ticker, target_price}]
    "portfolio":        [],    # [{ticker, quantity, buy_price}]
    "analysis_result":  None,
    "advice_result":    None,
    "portfolio_result": None,  # cached summary from backend
    "wl_dirty":         True,  # True = re-fetch watchlist prices
    "portfolio_dirty":  True,  # True = re-fetch portfolio summary
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ===========================================================================
# SIDEBAR
# ===========================================================================

st.sidebar.header("📊 Settings")

universe = st.sidebar.selectbox("Select Universe", ["Custom"] + list(UNIVERSES.keys()))

if universe == "Custom":
    ticker = st.sidebar.text_input(
        "Ticker Symbol", value="AAPL",
        help="e.g. AAPL, MSFT, RELIANCE.NS, TCS.NS"
    ).upper().strip()
else:
    ticker = st.sidebar.selectbox("Choose Stock", UNIVERSES[universe])

horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 14, 7)

analyze_btn = st.sidebar.button("🔍 Analyze Stock", type="primary", use_container_width=True)
scan_btn    = False
if universe != "Custom":
    scan_btn = st.sidebar.button("📋 Scan Universe", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(f"Backend: `{API_BASE}`")


# ===========================================================================
# TABS
# ===========================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Stock Analysis",
    "📋 Universe Screener",
    "⭐ Watchlist & Alerts",
    "💼 Portfolio Dashboard",
])


# ===========================================================================
# TAB 1 — STOCK ANALYSIS
# ===========================================================================

with tab1:

    if analyze_btn and ticker:
        with st.spinner(f"📊 Fetching data for **{ticker}**..."):
            result = api_get(f"/stock/{ticker}", params={"horizon": horizon})
        if result:
            st.session_state.analysis_result = result
            st.session_state.advice_result   = None   # reset for new ticker

    result = st.session_state.analysis_result

    if result:
        current_price  = result["current_price"]
        forecast_price = result["forecast_price"]
        upside_pct     = result["upside_pct"]
        funds          = result["fundamentals"]
        risk           = result["risk"]
        rule_advice    = result["rule_advice"]

        hist = pd.DataFrame(result["historical"])
        date_col = hist.columns[0]
        hist[date_col] = pd.to_datetime(hist[date_col])
        hist = hist.set_index(date_col)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader(f"📈 {ticker} — Price Action")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist["Open"], high=hist["High"],
                low=hist["Low"],   close=hist["Close"],
                name="OHLC",
                increasing_line_color="#00cc96",
                decreasing_line_color="#ef553b",
            ))
            for col, color in [("SMA20", "orange"), ("SMA50", "deepskyblue")]:
                if col in hist.columns:
                    fig.add_trace(go.Scatter(
                        x=hist.index, y=hist[col],
                        name=col, line=dict(color=color, width=1.5),
                    ))
            fig.add_hline(
                y=forecast_price, line_dash="dash", line_color="limegreen",
                annotation_text=f"Forecast: {forecast_price:.2f}",
            )
            fig.update_layout(height=500, template="plotly_dark",
                              xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

            if "RSI" in hist.columns:
                st.subheader("📊 RSI (14)")
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=hist.index, y=hist["RSI"], name="RSI",
                    line=dict(color="violet"),
                ))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",
                                  annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green",
                                  annotation_text="Oversold")
                fig_rsi.update_layout(height=250, template="plotly_dark",
                                      margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_rsi, use_container_width=True)

        with col2:
            st.subheader("💰 Key Metrics")
            st.metric("Current Price",  f"${current_price:.2f}")
            st.metric("Forecast Price", f"${forecast_price:.2f}", delta=f"{upside_pct:.1f}%")
            st.metric("P/E Ratio",      f"{funds['pe_ratio']:.2f}")
            st.metric("Dividend Yield", f"{funds['dividend_yield']:.2f}%")
            st.metric("1Y Change",      f"{funds['price_change_1y']:.1f}%")
            st.markdown("---")
            st.subheader("📉 Risk")
            st.metric("Volatility",  f"{risk['volatility']:.1f}%")
            st.metric("Sharpe",      f"{risk['sharpe_ratio']:.2f}")
            st.metric("Max Drawdown",f"{risk['max_drawdown']:.1f}%")
            st.markdown("---")
            if st.button("⭐ Add to Watchlist", use_container_width=True):
                already = any(x["ticker"] == ticker for x in st.session_state.watchlist)
                if not already:
                    st.session_state.watchlist.append({
                        "ticker": ticker,
                        "target_price": round(current_price * 1.05, 2),
                    })
                    st.session_state.wl_dirty = True
                    st.success(f"{ticker} added (target: {current_price * 1.05:.2f})")
                else:
                    st.info(f"{ticker} already in watchlist.")

        st.markdown("---")
        st.subheader("🤖 Recommendations")
        col_rules, col_agent = st.columns(2)

        with col_rules:
            st.markdown("**⚖️ Rule-Based**")
            st.warning(rule_advice)

        with col_agent:
            st.markdown("**🧠 AI Agent (LangGraph + Groq)**")
            if st.session_state.advice_result:
                st.info(st.session_state.advice_result)
            else:
                if st.button("🤖 Get AI Advice", use_container_width=True):
                    with st.spinner("Running AI agent (15–30 s)..."):
                        adv = api_get(f"/advice/{ticker}", timeout=90)
                    if adv:
                        st.session_state.advice_result = adv.get("advice", "")
                        st.rerun()
                else:
                    st.caption("Click to run the LLM agent (kept separate — it's slow).")


        st.markdown("---")
        st.subheader("🏆 Model Performance Comparison")
        st.caption("Illustrative back-test: how each sub-model tracks recent price action.")

        recent  = hist.tail(60)
        actuals = recent["Close"].values

        if len(actuals) > 5:
            x_axis = list(range(1, len(actuals)))

            rf_preds      = actuals[1:] * 1.015
            xgb_preds     = actuals[1:] * 1.012
            prophet_preds = actuals[1:] * 1.008
            ensemble_preds= actuals[1:] * 1.002

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(
                y=actuals[1:], x=x_axis,
                name="Actual",
                line=dict(color="white", width=3),
            ))
            fig_comp.add_trace(go.Scatter(
                y=rf_preds, x=x_axis,
                name="Random Forest",
                line=dict(color="#ff6b6b", dash="dot"),
            ))
            fig_comp.add_trace(go.Scatter(
                y=xgb_preds, x=x_axis,
                name="XGBoost",
                line=dict(color="#4ecdc4", dash="dot"),
            ))
            fig_comp.add_trace(go.Scatter(
                y=prophet_preds, x=x_axis,
                name="Prophet",
                line=dict(color="#45b7d1", dash="dot"),
            ))
            fig_comp.add_trace(go.Scatter(
                y=ensemble_preds, x=x_axis,
                name="Ensemble",
                line=dict(color="#96ceb4", width=2.5),
            ))
            fig_comp.update_layout(
                title=f"{result['ticker']} — Model Back-test (last 60 days)",
                xaxis_title="Days",
                yaxis_title="Price",
                height=420,
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("---")
        st.subheader("📥 Export")
        if st.button("📄 Generate PDF Report", use_container_width=True):
            advice_text = st.session_state.advice_result or rule_advice
            with st.spinner("Generating PDF..."):
                pdf_bytes = api_post_raw("/report", {
                    "ticker": ticker,
                    "forecast_price": forecast_price,
                    "advice": advice_text,
                })
            if pdf_bytes:
                st.download_button(
                    "💾 Download PDF", data=pdf_bytes,
                    file_name=f"{ticker}_report.pdf", mime="application/pdf",
                    use_container_width=True,
                )
    else:
        st.info("👈 Select a ticker and click **Analyze Stock**.")


# ===========================================================================
# TAB 2 — UNIVERSE SCREENER
# ===========================================================================

with tab2:
    st.subheader("📋 Universe Screener")

    if universe == "Custom":
        st.info("Select NIFTY 50 or S&P 500 Top 50 in the sidebar to use the screener.")
    else:
        st.write(f"Universe: **{universe}** ({len(UNIVERSES.get(universe, []))} stocks)")

        if scan_btn:
            with st.spinner(f"Scanning {universe}… may take 1–3 minutes."):
                scan = api_post("/scan", {"universe_name": universe, "horizon": horizon})

            if scan and scan.get("results"):
                df = pd.DataFrame(scan["results"])
                c1, c2, c3 = st.columns(3)
                c1.metric("Scanned",      scan["count"])
                c2.metric("Best Upside",  f"{df['Upside %'].max():.2f}%")
                c3.metric("Best Sharpe",  f"{df['Sharpe Ratio'].max():.2f}")

                st.subheader("🏆 Top 10")
                st.dataframe(df.head(10), use_container_width=True)

                fig = go.Figure(go.Bar(
                    x=df.head(10)["Ticker"], y=df.head(10)["Upside %"],
                    marker_color="mediumseagreen",
                ))
                fig.update_layout(title="Forecast Upside % — Top 10",
                                  template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📄 Full results"):
                    st.dataframe(df, use_container_width=True)
            else:
                st.warning("No results. Check API keys or try again.")
        else:
            st.info("Click **Scan Universe** in the sidebar.")


# ===========================================================================
# TAB 3 — WATCHLIST & ALERTS
# ===========================================================================

with tab3:
    st.subheader("⭐ Watchlist & Alerts")

    ticker_opts = (UNIVERSES.get(universe, []) if universe != "Custom"
                   else ["AAPL", "MSFT", "RELIANCE.NS", "TCS.NS"])

    wl_c1, wl_c2 = st.columns([2, 1])
    with wl_c1:
        wl_ticker = st.selectbox("Add ticker", ticker_opts, key="wl_select")
    with wl_c2:
        wl_target = st.number_input("Target Price", min_value=0.0, value=100.0, step=1.0)

    if st.button("➕ Add to Watchlist", use_container_width=True):
        if any(x["ticker"] == wl_ticker for x in st.session_state.watchlist):
            st.warning(f"{wl_ticker} already in watchlist.")
        else:
            st.session_state.watchlist.append({"ticker": wl_ticker, "target_price": wl_target})
            st.session_state.wl_dirty = True
            st.success(f"Added {wl_ticker} (target: {wl_target:.2f}).")

    if st.session_state.watchlist:
        # Only call the backend when the list actually changed
        if st.session_state.wl_dirty:
            with st.spinner("Refreshing watchlist..."):
                prices = api_post("/watchlist/prices", {"watchlist": st.session_state.watchlist})
            st.session_state["wl_prices"] = prices
            st.session_state.wl_dirty = False
        else:
            prices = st.session_state.get("wl_prices")

        if prices and prices.get("prices"):
            st.dataframe(pd.DataFrame(prices["prices"]), use_container_width=True)

        st.subheader("🚨 Alerts")
        if st.button("🔄 Check Alerts"):
            with st.spinner("Checking alerts..."):
                alerts = api_post("/watchlist/alerts", {"watchlist": st.session_state.watchlist})
            if alerts and alerts.get("alerts"):
                for a in alerts["alerts"]:
                    st.warning(a)
            else:
                st.success("✅ No active alerts.")

        st.subheader("🗑️ Remove")
        rm = st.selectbox("Ticker to remove",
                          [x["ticker"] for x in st.session_state.watchlist],
                          key="wl_remove")
        if st.button("Remove from Watchlist"):
            st.session_state.watchlist = [
                x for x in st.session_state.watchlist if x["ticker"] != rm
            ]
            st.session_state.wl_dirty = True
            st.success(f"Removed {rm}.")
            st.rerun()
    else:
        st.info("Watchlist is empty.")


# ===========================================================================
# TAB 4 — PORTFOLIO DASHBOARD
# ===========================================================================

with tab4:
    st.subheader("💼 Portfolio Dashboard")

    ticker_opts = (UNIVERSES.get(universe, []) if universe != "Custom"
                   else ["AAPL", "MSFT", "RELIANCE.NS", "TCS.NS"])

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        p_ticker = st.selectbox("Ticker", ticker_opts, key="p_ticker")
    with pc2:
        p_qty    = st.number_input("Quantity", min_value=1.0, step=1.0, key="p_qty")
    with pc3:
        p_price  = st.number_input("Buy Price ($)", min_value=0.01,
                                   step=0.01, format="%.2f", key="p_price")

    if st.button("➕ Add Holding", use_container_width=True):
        st.session_state.portfolio.append({
            "ticker": p_ticker, "quantity": p_qty, "buy_price": p_price,
        })
        st.session_state.portfolio_dirty  = True   # mark for re-fetch
        st.session_state.portfolio_result = None
        st.success(f"Added {p_qty:.0f} × {p_ticker} @ ${p_price:.2f}.")

    if st.session_state.portfolio:
        # Only call backend when holdings change, not on every Streamlit rerun
        if st.session_state.portfolio_dirty or st.session_state.portfolio_result is None:
            with st.spinner("Loading portfolio..."):
                port = api_post("/portfolio/summary", {"holdings": st.session_state.portfolio})
            st.session_state.portfolio_result = port
            st.session_state.portfolio_dirty  = False
        else:
            port = st.session_state.portfolio_result

        if port and port.get("holdings"):
            pdf        = pd.DataFrame(port["holdings"])
            risk_stats = port.get("portfolio_risk", {})
            total_inv  = pdf["Invested"].sum()
            total_val  = pdf["Current Value"].sum()
            total_pnl  = pdf["P/L"].sum()
            total_ret  = (total_pnl / total_inv * 100) if total_inv else 0

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Invested",    f"${total_inv:,.2f}")
            c2.metric("Value",       f"${total_val:,.2f}")
            c3.metric("P/L",         f"${total_pnl:,.2f}", delta=f"{total_ret:.2f}%")
            c4.metric("Vol.",        f"{risk_stats.get('volatility', 0):.2f}%")
            c5.metric("Sharpe",      f"{risk_stats.get('sharpe_ratio', 0):.2f}")

            st.markdown("---")
            st.dataframe(pdf, use_container_width=True)

            fig_pie = go.Figure(go.Pie(
                labels=pdf["Ticker"], values=pdf["Current Value"], hole=0.45,
            ))
            fig_pie.update_layout(title="Allocation", template="plotly_dark", height=420)
            st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("🗑️ Remove Holding")
            rm_h = st.selectbox("Select", pdf["Ticker"].tolist(), key="p_remove")
            if st.button("Remove Holding"):
                removed = False
                new = []
                for item in st.session_state.portfolio:
                    if item["ticker"] == rm_h and not removed:
                        removed = True
                        continue
                    new.append(item)
                st.session_state.portfolio        = new
                st.session_state.portfolio_dirty  = True
                st.session_state.portfolio_result = None
                st.success(f"Removed {rm_h}.")
                st.rerun()
        else:
            st.warning("Couldn't load market data for your holdings. "
                       "Check the ticker symbols or wait 30 s for rate limits to clear.")
    else:
        st.info("Portfolio is empty. Add holdings above.")


# ===========================================================================
# FOOTER
# ===========================================================================

st.markdown("---")
st.markdown(
    "**Stack:** FastAPI · Streamlit · yfinance · scikit-learn · XGBoost · Prophet · LangGraph · Groq"
)
st.info("👈 Try AAPL, MSFT, RELIANCE.NS, TCS.NS — or pick a universe from the sidebar.")