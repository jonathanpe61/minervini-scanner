#!/usr/bin/env python3
"""
Minervini Trend Template Scanner (SEPA-style)
=============================================

A free, local stock scanner implementing Mark Minervini's "Trend Template" -- the
technical pre-screen behind his SEPA (Specific Entry Point Analysis) strategy.
It uses only free data (yfinance) and runs entirely on your machine. No API key,
no server, no paid services.

It keeps only stocks that pass ALL of these conditions:
  1. Price is above both the 150-day and 200-day SMAs
  2. The 150-day SMA is above the 200-day SMA
  3. The 200-day SMA is trending UP (positive slope over the last ~20-30 days)
  4. The 50-day SMA is above both the 150-day and 200-day SMAs
  5. Price is above the 50-day SMA
  6. Price is NOT overextended: within 5% above the 150-day SMA
     (this is your max risk if you set the stop at the 150-day line)
  7. Price is at least 30% above its 52-week low
  8. Price is within 25% of its 52-week high

Qualifiers are ranked by a transparent composite score (0-100) that blends -- as a
"mix of all" -- Relative Strength, trend strength, momentum, and non-overextension.
Weights are configurable and printed at runtime.

Outputs (written to --outdir, default ./scanner_output):
  - A ranked table in the console
  - qualifiers.csv         the table, for spreadsheets
  - results.json           full metrics + chart series; load this into the dashboard
  - charts/<TICKER>.png    price + 50/150/200-day SMA chart for the top picks

Install:
  pip install yfinance pandas numpy matplotlib lxml

Usage:
  python minervini_scanner.py                       # scan the S&P 500 (default)
  python minervini_scanner.py --universe nasdaq100
  python minervini_scanner.py --tickers NVDA,META,LLY,GS,VRT
  python minervini_scanner.py --file watchlist.txt --charts 15
  python minervini_scanner.py --ext-min 5           # strict 5-10% band on cond. 6
  python minervini_scanner.py --max 100             # cap universe size (faster)

DISCLAIMER: This is an educational technical screen, NOT investment advice. A
passing stock is a *candidate* for further research, not a recommendation. Past
performance does not guarantee future results.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Default trend-template parameters (all overridable from the CLI)
# ----------------------------------------------------------------------------
DEFAULTS = dict(
    slope_window=22,      # trading days used to measure the 200-day SMA slope (~1 month)
    ext_min=0.0,          # min % price may sit above the 150-day SMA
    ext_max=5.0,         # max % above 150-day SMA = max loss if you stop at the 150-day line
    low_min=30.0,         # min % above the 52-week low
    high_within=25.0,     # max % below the 52-week high
)

# Composite ranking weights (must sum to 1.0). "A mix of all", per request.
WEIGHTS = dict(rs=0.35, trend=0.20, momentum=0.15, nonext=0.30)

# A compact, sector-diversified fallback universe used only if the S&P 500 list
# cannot be fetched online. Not exhaustive -- the live Wikipedia fetch is preferred.
FALLBACK_UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","LLY",
    "JPM","V","XOM","UNH","MA","COST","HD","PG","JNJ","ABBV","NFLX","CRM",
    "BAC","ORCL","MRK","CVX","KO","AMD","PEP","WMT","ADBE","TMO","LIN","ACN",
    "MCD","CSCO","ABT","GE","DHR","WFC","TXN","QCOM","INTU","AMGN","NOW","CAT",
    "IBM","GS","MS","SPGI","ISRG","PFE","UBER","BKNG","BLK","C","AXP","NEE",
    "LRCX","VRT","PANW","SNPS","KLAC","MU","ANET","CDNS","REGN","PGR","HON",
    "ETN","BA","DE","LMT","ELV","SBUX","GILD","ADP","MDLZ","TJX","PLD","CB",
    "SO","DUK","MMC","BSX","SYK","CI","ZTS","MO","FI","WM","ITW","APH","CME",
    "ACHR","PLTR","COIN","SMCI","ARM","DELL","CRWD","SNOW","DDOG","NET",
]

NASDAQ100_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","COST",
    "NFLX","AMD","PEP","ADBE","CSCO","TMUS","INTU","QCOM","TXN","AMGN","CMCSA",
    "HON","BKNG","ISRG","VRTX","ADP","REGN","LRCX","PANW","SNPS","KLAC","MU",
    "ANET","CDNS","MELI","ASML","ORLY","CSX","MAR","FTNT","ADI","CRWD","ABNB",
    "PYPL","CTAS","NXPI","PCAR","MNST","ROP","WDAY","CHTR","ODFL","PAYX","KDP",
    "MRVL","DDOG","TEAM","CPRT","FAST","ROST","DXCM","IDXX","EA","BKR","CCEP",
    "XEL","KHC","TTD","CEG","ON","GEHC","CDW","BIIB","ANSS","DLTR","WBD","ZS",
    "ARM","SMCI","PLTR","LIN","MDB","ALGN","ENPH","SIRI","ILMN","WBA",
]


# ----------------------------------------------------------------------------
# Universe selection
# ----------------------------------------------------------------------------
def get_universe(name, custom=None, file=None):
    """Return a list of ticker symbols for the requested universe."""
    if custom:
        return [t.strip().upper() for t in custom.split(",") if t.strip()]
    if file:
        with open(file) as fh:
            return [ln.strip().upper() for ln in fh if ln.strip() and not ln.startswith("#")]

    name = (name or "sp500").lower()
    if name in ("sp500", "s&p500", "spx"):
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            syms = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
            syms = [s.strip().upper() for s in syms if s and s != "nan"]
            if len(syms) > 400:
                print(f"  Loaded {len(syms)} S&P 500 tickers from Wikipedia.")
                return syms
        except Exception as e:
            print(f"  Could not fetch S&P 500 list online ({e}); using fallback list.")
        return FALLBACK_UNIVERSE
    if name in ("nasdaq100", "ndx", "nasdaq"):
        return NASDAQ100_FALLBACK
    if name in ("fallback", "demo", "watchlist"):
        return FALLBACK_UNIVERSE
    # Treat anything else as a comma list passed directly to --universe
    return [t.strip().upper() for t in name.split(",") if t.strip()]


# ----------------------------------------------------------------------------
# Data download (yfinance is imported lazily so the pure logic can run offline)
# ----------------------------------------------------------------------------
def download_prices(tickers, period="2y", batch_size=40, pause=1.0, retries=2):
    """Download daily OHLCV for many tickers, batched to be gentle on Yahoo.

    Returns dict {ticker: DataFrame with columns Open/High/Low/Close/Volume}.
    """
    import yfinance as yf  # lazy import: only needed for the live scan

    out = {}
    total = len(tickers)
    for i in range(0, total, batch_size):
        batch = tickers[i:i + batch_size]
        raw = None
        for attempt in range(retries + 1):
            try:
                raw = yf.download(
                    batch, period=period, interval="1d",
                    auto_adjust=True, group_by="ticker",
                    threads=True, progress=False,
                )
                break
            except Exception as e:
                if attempt == retries:
                    print(f"  Batch {i//batch_size+1} failed: {e}")
                    raw = None
                else:
                    time.sleep(pause * (attempt + 1))
        if raw is not None and len(raw) > 0:
            if isinstance(raw.columns, pd.MultiIndex):
                level0 = set(raw.columns.get_level_values(0))
                for t in batch:
                    if t in level0:
                        df = raw[t].dropna(how="all")
                        if not df.empty and "Close" in df:
                            out[t] = df
            else:  # single ticker => flat columns
                df = raw.dropna(how="all")
                if not df.empty and "Close" in df:
                    out[batch[0]] = df
        done = min(i + batch_size, total)
        print(f"  Downloaded {done}/{total} tickers ({len(out)} usable so far)...")
        time.sleep(pause)
    return out


# ----------------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------------
def compute_indicators(df, slope_window=22):
    """Compute the metrics needed for the trend template from one price frame.

    Returns a dict of metrics, or None if there is not enough history.
    """
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    need = 200 + slope_window
    if len(close) < need:
        return None

    sma50 = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()

    price = float(close.iloc[-1])
    s50 = float(sma50.iloc[-1])
    s150 = float(sma150.iloc[-1])
    s200 = float(sma200.iloc[-1])
    s200_prev = float(sma200.iloc[-1 - slope_window])

    win = close.iloc[-252:] if len(close) >= 252 else close
    low52 = float(win.min())
    high52 = float(win.max())

    slope200 = (s200 - s200_prev) / s200_prev * 100.0 if s200_prev else 0.0
    pct_above_150 = (price / s150 - 1.0) * 100.0
    pct_above_50 = (price / s50 - 1.0) * 100.0
    pct_above_low = (price / low52 - 1.0) * 100.0
    pct_from_high = (price / high52 - 1.0) * 100.0   # <= 0; "-8" means 8% below high

    # IBD-style raw relative strength: recent quarter double-weighted.
    rs_raw = np.nan
    if len(close) >= 253:
        c = price
        def ret(n):
            return c / float(close.iloc[-1 - n]) - 1.0
        rs_raw = 0.40 * ret(63) + 0.20 * ret(126) + 0.20 * ret(189) + 0.20 * ret(252)

    return dict(
        price=price, sma50=s50, sma150=s150, sma200=s200,
        slope200=slope200, pct_above_150=pct_above_150, pct_above_50=pct_above_50,
        pct_above_low=pct_above_low, pct_from_high=pct_from_high,
        low52=low52, high52=high52, rs_raw=rs_raw, n_bars=int(len(close)),
        _close=close, _sma50=sma50, _sma150=sma150, _sma200=sma200,
    )


def evaluate_trend_template(m, p):
    """Return an ordered dict {criterion_label: bool} plus 'passed'."""
    c = {}
    c["1. Price > 150 & 200-day SMA"] = (m["price"] > m["sma150"]) and (m["price"] > m["sma200"])
    c["2. 150-day SMA > 200-day SMA"] = m["sma150"] > m["sma200"]
    c["3. 200-day SMA sloping up"] = m["slope200"] > 0.0
    c["4. 50-day SMA > 150 & 200"] = (m["sma50"] > m["sma150"]) and (m["sma50"] > m["sma200"])
    c["5. Price > 50-day SMA"] = m["price"] > m["sma50"]
    c["6. Not overextended vs 150"] = (m["pct_above_150"] <= p["ext_max"]) and (m["pct_above_150"] >= p["ext_min"])
    c["7. >= 30% above 52-wk low"] = m["pct_above_low"] >= p["low_min"]
    c["8. Within 25% of 52-wk high"] = m["pct_from_high"] >= -p["high_within"]
    c["passed"] = all(v for k, v in c.items())
    return c


# ----------------------------------------------------------------------------
# Composite ranking ("a mix of all")
# ----------------------------------------------------------------------------
def _pct_rank(series):
    """Percentile rank in [0,1]; NaNs -> 0.5 (neutral)."""
    r = series.rank(pct=True)
    return r.fillna(0.5)


def add_scores(M):
    """Add rs_rating (1-99 vs scanned universe) and composite (0-100 among qualifiers)."""
    M = M.copy()
    # RS rating is ranked across the WHOLE scanned universe (market-relative).
    M["rs_rating"] = (_pct_rank(M["rs_raw"]) * 98 + 1).round().astype(int)

    q = M[M["passed"]].copy()
    if len(q) == 0:
        M["composite"] = np.nan
        return M, q

    # Sub-scores ranked AMONG qualifiers (higher = better):
    pr_rs = _pct_rank(q["rs_raw"])
    pr_slope = _pct_rank(q["slope200"])
    pr_stack = _pct_rank((q["sma50"] / q["sma200"] - 1.0))      # how stacked the MAs are
    pr_mom = _pct_rank(q["pct_above_low"])                       # distance off the low
    pr_prox = _pct_rank(q["pct_from_high"])                      # closeness to the high (less negative = better)
    pr_nonext = _pct_rank(-q["pct_above_150"])                   # smaller premium over 150 = better

    trend = 0.5 * pr_slope + 0.5 * pr_stack
    momentum = 0.5 * pr_mom + 0.5 * pr_prox

    comp = (WEIGHTS["rs"] * pr_rs + WEIGHTS["trend"] * trend
            + WEIGHTS["momentum"] * momentum + WEIGHTS["nonext"] * pr_nonext) * 100.0
    q["composite"] = comp.round(1)
    M.loc[q.index, "composite"] = q["composite"]
    q = q.sort_values("composite", ascending=False)
    q.insert(0, "rank", range(1, len(q) + 1))
    return M, q


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def make_chart(ticker, ind, outdir, lookback=300):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    close = ind["_close"].iloc[-lookback:]
    s50 = ind["_sma50"].iloc[-lookback:]
    s150 = ind["_sma150"].iloc[-lookback:]
    s200 = ind["_sma200"].iloc[-lookback:]

    fig = plt.figure(figsize=(11, 6))
    gs = GridSpec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(close.index, close.values, color="#111827", lw=1.6, label="Close")
    ax.plot(s50.index, s50.values, color="#2563eb", lw=1.2, label="50-day SMA")
    ax.plot(s150.index, s150.values, color="#f59e0b", lw=1.2, label="150-day SMA")
    ax.plot(s200.index, s200.values, color="#dc2626", lw=1.2, label="200-day SMA")
    ax.axhline(ind["high52"], color="#16a34a", ls="--", lw=0.8, alpha=0.7)
    ax.axhline(ind["low52"], color="#9ca3af", ls="--", lw=0.8, alpha=0.7)
    ax.set_title(
        f"{ticker}  |  ${ind['price']:.2f}  |  RS {ind.get('rs_rating','-')}  |  "
        f"+{ind['pct_above_low']:.0f}% off low, {ind['pct_from_high']:.0f}% from high  |  "
        f"200d slope {ind['slope200']:+.1f}%",
        fontsize=10, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=8, ncol=4)
    ax.grid(alpha=0.25)
    ax.margins(x=0.01)
    fig.tight_layout()
    os.makedirs(os.path.join(outdir, "charts"), exist_ok=True)
    path = os.path.join(outdir, "charts", f"{ticker}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def build_params(args):
    p = dict(DEFAULTS)
    p["slope_window"] = args.slope_window
    p["ext_min"] = args.ext_min
    p["ext_max"] = args.ext_max
    p["low_min"] = args.low_min
    p["high_within"] = args.high_within
    return p


def main():
    ap = argparse.ArgumentParser(description="Minervini Trend Template stock scanner")
    ap.add_argument("--universe", default="sp500",
                    help="sp500 (default) | nasdaq100 | fallback | a comma list")
    ap.add_argument("--tickers", default=None, help="comma-separated custom tickers")
    ap.add_argument("--file", default=None, help="path to a watchlist file (one ticker per line)")
    ap.add_argument("--period", default="2y", help="history window for yfinance (default 2y)")
    ap.add_argument("--max", type=int, default=0, help="cap number of tickers (0 = no cap)")
    ap.add_argument("--charts", type=int, default=10, help="how many top picks to chart (default 10)")
    ap.add_argument("--outdir", default="scanner_output", help="output directory")
    ap.add_argument("--slope-window", dest="slope_window", type=int, default=DEFAULTS["slope_window"])
    ap.add_argument("--ext-min", dest="ext_min", type=float, default=DEFAULTS["ext_min"],
                    help="min %% above 150-day SMA (set 5 for the strict 5-10%% band)")
    ap.add_argument("--ext-max", dest="ext_max", type=float, default=DEFAULTS["ext_max"])
    ap.add_argument("--low-min", dest="low_min", type=float, default=DEFAULTS["low_min"])
    ap.add_argument("--high-within", dest="high_within", type=float, default=DEFAULTS["high_within"])
    args = ap.parse_args()

    p = build_params(args)
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 70)
    print("MINERVINI TREND TEMPLATE SCANNER")
    print("=" * 70)
    tickers = get_universe(args.universe, args.tickers, args.file)
    if args.max and len(tickers) > args.max:
        tickers = tickers[:args.max]
    print(f"Universe: {len(tickers)} tickers | period={args.period} | "
          f"ext band={p['ext_min']:.0f}-{p['ext_max']:.0f}% | weights={WEIGHTS}")
    print("Downloading data from Yahoo Finance (free, no key)...")

    data = download_prices(tickers, period=args.period)
    print(f"\nComputing indicators for {len(data)} tickers with usable history...\n")

    rows = []
    indicators = {}
    for t, df in data.items():
        m = compute_indicators(df, slope_window=p["slope_window"])
        if m is None:
            continue
        crit = evaluate_trend_template(m, p)
        n_pass = sum(1 for k, v in crit.items() if k != "passed" and v)
        indicators[t] = m
        rows.append(dict(
            ticker=t, price=m["price"], sma50=m["sma50"], sma150=m["sma150"],
            sma200=m["sma200"], slope200=m["slope200"], pct_above_150=m["pct_above_150"],
            pct_above_low=m["pct_above_low"], pct_from_high=m["pct_from_high"],
            rs_raw=m["rs_raw"], passed=crit["passed"], n_pass=n_pass,
            criteria={k: bool(v) for k, v in crit.items() if k != "passed"},
        ))

    if not rows:
        print("No tickers had enough history. Try a larger --period or different universe.")
        return

    M = pd.DataFrame(rows).set_index("ticker")
    M, Q = add_scores(M)

    # ---- Console table -----------------------------------------------------
    print("=" * 70)
    print(f"QUALIFIERS: {len(Q)} of {len(M)} stocks passed all 8 conditions")
    print("=" * 70)
    if len(Q):
        show = Q[["rank", "price", "rs_rating", "composite", "slope200",
                  "pct_above_150", "pct_above_low", "pct_from_high"]].copy()
        show.columns = ["#", "Price", "RS", "Score", "200dSlope%",
                        "Risk150%", "%>52wLow", "%fr52wHigh"]
        with pd.option_context("display.float_format", lambda x: f"{x:,.2f}"):
            print(show.to_string())
        best = Q.iloc[0]
        print("\n" + "-" * 70)
        print(f"TOP PICK: {best.name}  (composite {best['composite']:.1f}/100, "
              f"RS {int(best['rs_rating'])})")
        print(f"  ${best['price']:.2f} | {best['pct_above_low']:.0f}% above 52-wk low | "
              f"{best['pct_from_high']:.0f}% from 52-wk high | "
              f"200-day SMA slope {best['slope200']:+.1f}%")
        print("-" * 70)
    else:
        print("No stocks passed all conditions today. Near-misses are saved in results.json.")

    # ---- CSV ---------------------------------------------------------------
    csv_path = os.path.join(args.outdir, "qualifiers.csv")
    if len(Q):
        Q.drop(columns=["criteria"]).to_csv(csv_path)
        print(f"\nSaved table  -> {csv_path}")

    # ---- Charts for the top N ---------------------------------------------
    chart_tickers = list(Q.index[:args.charts]) if len(Q) else []
    for t in chart_tickers:
        indicators[t]["rs_rating"] = int(M.loc[t, "rs_rating"])
        try:
            make_chart(t, indicators[t], args.outdir)
        except Exception as e:
            print(f"  chart failed for {t}: {e}")
    if chart_tickers:
        print(f"Saved {len(chart_tickers)} charts -> {os.path.join(args.outdir, 'charts')}/")

    # ---- results.json (feeds the dashboard) --------------------------------
    def series_for(t, n=252):
        m = indicators[t]
        idx = m["_close"].index[-n:]
        def arr(s): return [None if pd.isna(v) else round(float(v), 4) for v in s.iloc[-n:]]
        return dict(
            dates=[d.strftime("%Y-%m-%d") for d in idx],
            close=arr(m["_close"]), sma50=arr(m["_sma50"]),
            sma150=arr(m["_sma150"]), sma200=arr(m["_sma200"]),
        )

    stocks = []
    for t in Q.index:
        r = Q.loc[t]
        stocks.append(dict(
            ticker=t, rank=int(r["rank"]), price=round(float(r["price"]), 2),
            sma50=round(float(r["sma50"]), 2), sma150=round(float(r["sma150"]), 2),
            sma200=round(float(r["sma200"]), 2), slope200=round(float(r["slope200"]), 2),
            pct_above_150=round(float(r["pct_above_150"]), 2),
            pct_above_low=round(float(r["pct_above_low"]), 2),
            pct_from_high=round(float(r["pct_from_high"]), 2),
            rs_rating=int(r["rs_rating"]), composite=round(float(r["composite"]), 1),
            criteria=r["criteria"], passed=True,
            series=series_for(t) if t in chart_tickers else None,
        ))
    # A few near-misses (failed only 1-2 conditions) are handy context.
    near = M[(~M["passed"]) & (M["n_pass"] >= 6)].sort_values("n_pass", ascending=False).head(20)
    near_misses = [dict(ticker=t, n_pass=int(near.loc[t, "n_pass"]),
                        rs_rating=int(near.loc[t, "rs_rating"]),
                        criteria=near.loc[t, "criteria"]) for t in near.index]

    payload = dict(
        generated_at=datetime.now(timezone.utc).isoformat(),
        params=p, weights=WEIGHTS, universe=args.universe,
        scanned=int(len(M)), qualified=int(len(Q)),
        stocks=stocks, near_misses=near_misses,
    )
    json_path = os.path.join(args.outdir, "results.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\nDone. Reminder: this is an educational screen, not investment advice.")


if __name__ == "__main__":
    main()
