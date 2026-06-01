"""
Minervini Trend Template Scanner - Streamlit web app.

Reuses the logic in minervini_scanner.py (same 8 conditions, same composite
ranking, same 5% "stop risk" cap on distance above the 150-day SMA). Runs the
live scan with free yfinance data. Deploy free on Streamlit Community Cloud.

Run locally:   streamlit run app.py
Educational technical screen - NOT investment advice.
"""
import io
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import minervini_scanner as ms

st.set_page_config(page_title="Minervini Trend Scanner", page_icon="📈", layout="wide")
st.title("Minervini Trend Template Scanner")
st.caption("Free, live technical screen (yfinance). Educational only - not investment advice.")

with st.sidebar:
    st.header("Scan settings")
    uni = st.selectbox("Universe", ["S&P 500", "Nasdaq-100", "Custom list"])
    custom = ""
    if uni == "Custom list":
        custom = st.text_area("Tickers (comma-separated)",
                              "NVDA,META,LLY,GS,VRT,MS,JPM,AVGO,LRCX,ANET,PANW,MMC")
    maxn = st.number_input("Max tickers (caps runtime)", 10, 600, 150, 10)
    period = st.selectbox("History window", ["2y", "1y"], 0)
    cap = st.slider("Max % above 150-day SMA  =  your stop risk", 1.0, 15.0, 5.0, 0.5,
                    help="If you set your stop at the 150-day line, this is your max loss. "
                         "Tighter = lower risk and fewer candidates.")
    emin = st.slider("Min % above 150-day SMA", 0.0, 5.0, 0.0, 0.5)
    ncharts = st.slider("Charts to show", 1, 20, 8)
    run = st.button("Run scan", type="primary")

st.info("Composite ranking: relative strength 35%, trend strength 20%, momentum 15%, "
        "non-overextension 30%. The 'stop risk' column is how far price sits above its "
        "150-day SMA - i.e. your loss if stopped out there.")


@st.cache_data(show_spinner=False)
def fetch(tickers, period):
    return ms.download_prices(list(tickers), period=period)


def make_fig(ticker, m):
    c = m["_close"].iloc[-300:]
    s50, s150, s200 = m["_sma50"].iloc[-300:], m["_sma150"].iloc[-300:], m["_sma200"].iloc[-300:]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(c.index, c.values, color="#111827", lw=1.5, label="Close")
    ax.plot(s50.index, s50.values, color="#2563eb", lw=1.0, label="50-day")
    ax.plot(s150.index, s150.values, color="#f59e0b", lw=1.0, label="150-day (stop)")
    ax.plot(s200.index, s200.values, color="#dc2626", lw=1.0, label="200-day")
    ax.axhline(m["high52"], color="#16a34a", ls="--", lw=0.8, alpha=0.7)
    ax.axhline(m["low52"], color="#9ca3af", ls="--", lw=0.8, alpha=0.7)
    ax.legend(loc="upper left", fontsize=8, ncol=4)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


if not run:
    st.write("Pick a universe and click **Run scan** in the sidebar. "
             "A full S&P 500 scan takes a couple of minutes the first time.")
    st.stop()

name = {"S&P 500": "sp500", "Nasdaq-100": "nasdaq100", "Custom list": custom}[uni]
tickers = ms.get_universe(name, custom if uni == "Custom list" else None)
tickers = tickers[:int(maxn)]
P = dict(ms.DEFAULTS)
P["ext_max"] = float(cap)
P["ext_min"] = float(emin)

with st.spinner(f"Downloading {len(tickers)} tickers from Yahoo Finance..."):
    data = fetch(tuple(tickers), period)

rows, ind = [], {}
for t, df in data.items():
    m = ms.compute_indicators(df, slope_window=P["slope_window"])
    if m is None:
        continue
    crit = ms.evaluate_trend_template(m, P)
    npass = sum(1 for k, v in crit.items() if k != "passed" and v)
    ind[t] = m
    rows.append(dict(ticker=t, price=m["price"], sma50=m["sma50"], sma150=m["sma150"],
                     sma200=m["sma200"], slope200=m["slope200"], pct_above_150=m["pct_above_150"],
                     pct_above_low=m["pct_above_low"], pct_from_high=m["pct_from_high"],
                     rs_raw=m["rs_raw"], passed=crit["passed"], n_pass=npass,
                     criteria={k: bool(v) for k, v in crit.items() if k != "passed"}))

if not rows:
    st.warning("Not enough price history was returned. Try a different universe or a longer window.")
    st.stop()

M = pd.DataFrame(rows).set_index("ticker")
M, Q = ms.add_scores(M)

c1, c2, c3 = st.columns(3)
c1.metric("Scanned", len(M))
c2.metric("Qualified", len(Q))
c3.metric("Top pick", Q.index[0] if len(Q) else "-")

if len(Q) == 0:
    st.info("No stocks passed all 8 conditions with these settings. Try loosening the stop-risk cap.")
    st.stop()

top = Q.iloc[0]
st.success(f"**Top pick: {top.name}** - composite {top['composite']:.1f}/100, RS {int(top['rs_rating'])}, "
           f"stop risk to the 150-day ~ {top['pct_above_150']:.1f}%. "
           f"Trades +{top['pct_above_low']:.0f}% off its 52-week low and {top['pct_from_high']:.0f}% from its high.")

show = Q[["rank", "price", "rs_rating", "composite", "pct_above_150",
          "slope200", "pct_above_low", "pct_from_high"]].rename(columns={
    "rs_rating": "RS", "composite": "Score", "pct_above_150": "StopRisk→150 %",
    "slope200": "200d slope %", "pct_above_low": "% > 52w low", "pct_from_high": "% from 52w high"})
st.dataframe(show.style.format({"price": "{:.2f}", "Score": "{:.1f}", "StopRisk→150 %": "{:.1f}",
             "200d slope %": "{:.1f}", "% > 52w low": "{:.0f}", "% from 52w high": "{:.0f}"}),
             use_container_width=True)

st.download_button("Download qualifiers (CSV)",
                   data=Q.drop(columns=["criteria"]).to_csv().encode(),
                   file_name="qualifiers.csv", mime="text/csv")

st.subheader("Price + moving-average charts")
for t in list(Q.index[:int(ncharts)]):
    st.markdown(f"**{t}** — stop risk to 150-day ≈ {Q.loc[t, 'pct_above_150']:.1f}%  ·  RS {int(Q.loc[t, 'rs_rating'])}")
    st.pyplot(make_fig(t, ind[t]))
    plt.close("all")
