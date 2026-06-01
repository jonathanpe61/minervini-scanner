"""Minervini Trend Template Scanner - Streamlit web app (Full template + Micho Method)."""
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
    screen = st.radio("Screen", ["Full trend template (8 conditions)", "Micho Method (simpler)"])
    micha_mode = screen.startswith("Micho")
    uni = st.selectbox("Universe", ["S&P 500", "Nasdaq-100", "Custom list"])
    custom = ""
    if uni == "Custom list":
        custom = st.text_area("Tickers (comma-separated)",
                              "NVDA,META,LLY,GS,VRT,MS,JPM,AVGO,LRCX,ANET,PANW,MMC,FAST")
    maxn = st.number_input("Max tickers (caps runtime)", 10, 600, 150, 10)
    period = st.selectbox("History window", ["2y", "1y"], 0,
                          help="Use 2y: it guarantees enough history for the 200-day SMA, its slope, "
                               "and the RS rating. 1y is borderline and makes RS unreliable.")
    if micha_mode:
        st.caption("Micho Method = 150-day SMA rising AND price inside this band vs the 150-day:")
        mlo = st.slider("Lower bound (% below 150-day allowed)", -10.0, 0.0, -2.0, 0.5)
        mhi = st.slider("Upper bound (% above 150-day allowed)", 0.0, 15.0, 5.0, 0.5)
        cap, emin = 5.0, 0.0
    else:
        cap = st.slider("Max % above 150-day SMA  =  your stop risk", 1.0, 15.0, 5.0, 0.5,
                        help="If your stop sits at the 150-day line, this is your max loss.")
        emin = st.slider("Min % above 150-day SMA", 0.0, 5.0, 0.0, 0.5)
        mlo, mhi = -2.0, 5.0
    ncharts = st.slider("Charts to show", 1, 20, 8)
    run = st.button("Run scan", type="primary")

if micha_mode:
    st.info("**Micho Method**: keeps stocks whose 150-day SMA is rising and whose price is within "
            f"{mlo:.0f}% to {mhi:.0f}% of that 150-day line. Ranked by the composite.")
else:
    st.info("**Full template**: all 8 trend-template conditions. 'Stop risk' = how far price is above "
            "its 150-day SMA (your loss if stopped out at the 150-day line).")


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
    st.write("Pick a screen and universe, then click **Run scan** in the sidebar.")
    st.stop()

name = {"S&P 500": "sp500", "Nasdaq-100": "nasdaq100", "Custom list": custom}[uni]
tickers = ms.get_universe(name, custom if uni == "Custom list" else None)[:int(maxn)]
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
    rows.append(dict(ticker=t, price=m["price"], sma50=m["sma50"], sma150=m["sma150"],
                     sma200=m["sma200"], slope200=m["slope200"], slope150=m["slope150"],
                     pct_above_150=m["pct_above_150"], pct_above_low=m["pct_above_low"],
                     pct_from_high=m["pct_from_high"], rs_raw=m["rs_raw"],
                     passed=crit["passed"], micha=ms.micha_test(m, mlo, mhi)))
    ind[t] = m

if not rows:
    st.warning("Not enough price history was returned. Try a different universe or window.")
    st.stop()

mask = "micha" if micha_mode else "passed"
M = pd.DataFrame(rows).set_index("ticker")
M, Q = ms.add_scores(M, mask_col=mask)

c1, c2, c3 = st.columns(3)
c1.metric("Scanned", len(M))
c2.metric("Qualified", len(Q))
c3.metric("Top pick", Q.index[0] if len(Q) else "-")

if len(Q) == 0:
    st.info("Nothing passed with these settings. Loosen the band (Micho Method) or the stop-risk cap.")
    st.stop()

top = Q.iloc[0]
if micha_mode:
    st.success(f"**Top Micho-Method pick: {top.name}** - 150-day SMA rising ({top['slope150']:.1f}%); "
               f"price {top['pct_above_150']:+.1f}% vs its 150-day. Stop at the 150-day ~{abs(top['pct_above_150']):.1f}% away. "
               f"RS {int(top['rs_rating'])}.")
else:
    st.success(f"**Top pick: {top.name}** - composite {top['composite']:.1f}/100, RS {int(top['rs_rating'])}, "
               f"stop risk to the 150-day ~ {top['pct_above_150']:.1f}%.")

show = Q[["rank", "price", "rs_rating", "composite", "pct_above_150", "slope150",
          "slope200", "pct_above_low", "pct_from_high"]].copy()
show["Micho"] = Q["micha"].map(lambda b: "✓" if b else "✗")
show = show.rename(columns={"rs_rating": "RS", "composite": "Score", "pct_above_150": "StopRisk→150 %",
                            "slope150": "150d slope %", "slope200": "200d slope %",
                            "pct_above_low": "% > 52w low", "pct_from_high": "% from 52w high"})
st.dataframe(show.style.format({"price": "{:.2f}", "Score": "{:.1f}", "StopRisk→150 %": "{:.1f}",
             "150d slope %": "{:.1f}", "200d slope %": "{:.1f}", "% > 52w low": "{:.0f}",
             "% from 52w high": "{:.0f}"}), use_container_width=True)

st.download_button("Download qualifiers (CSV)",
                   data=Q.drop(columns=["criteria"], errors="ignore").to_csv().encode(),
                   file_name="qualifiers.csv", mime="text/csv")

st.subheader("Price + moving-average charts")
for t in list(Q.index[:int(ncharts)]):
    st.markdown(f"**{t}** - stop risk to 150-day ~ {Q.loc[t, 'pct_above_150']:.1f}%  ·  "
                f"150d slope {Q.loc[t, 'slope150']:+.1f}%  ·  RS {int(Q.loc[t, 'rs_rating'])}  ·  "
                f"Micho {'✓' if Q.loc[t, 'micha'] else '✗'}")
    st.pyplot(make_fig(t, ind[t]))
    plt.close("all")
