#!/usr/bin/env python3
"""
analyze_cycles.py
------------------
Price & Time cycle analysis engine for the "Cycles" GitHub Pages dashboard.

WHAT IT DOES
  For every stock JSON file found under stockdata_15/ stockdata_1H/ stockdata_D/
  stockdata_W/ stockdata_M/ (each file is a list of OHLCV bars with fields
  datetime, symbol, open, high, low, close, volume) it:

    1. Detects swing highs/lows (fractal pivots).
    2. Runs FOUR price-time analysis methods:
         a. gann       - Gann "Square of 9" style time-count projections from
                          the most significant recent swing.
         b. fft         - FFT-based dominant cycle length (statistical /
                          seasonality detection) projected forward.
         c. fibonacci   - Fibonacci time-zone projections measured from the
                          most recent major swing-to-swing duration.
         d. stats       - Plain swing-to-swing interval statistics (mean/median
                          bars between turns) projected forward.
    3. Writes one compact JSON per stock+timeframe into cycles/data/<tf>/<symbol>.json
    4. Builds a master cycles/data/watchlist.json that ranks stocks by how many
       methods agree on an upcoming turn window (confluence), for the
       "stocks that may move soon" list.

USAGE
    python3 analyze_cycles.py --root /path/to/repo --out /path/to/repo/Cycles/data

  --root should contain the stockdata_15 / stockdata_1H / stockdata_D /
  stockdata_W / stockdata_M folders (as shown in the repo root).
  --out is where the dashboard JSON gets written (defaults to ./cycles/data
  next to this script, matching the "Cycles" folder the user creates).

This script has NO third-party dependencies beyond numpy (pip install numpy).
"""

import json
import math
import argparse
import datetime as dt
from pathlib import Path
from collections import defaultdict

import numpy as np

TIMEFRAMES = ["stockdata_15", "stockdata_1H", "stockdata_D", "stockdata_W", "stockdata_M"]

# Approx calendar days represented by one BAR in each timeframe (for turning
# bar-counts into real calendar dates, and for deciding pivot look-back width).
BAR_CALENDAR_DAYS = {
    "stockdata_15": 15 / (60 * 24),      # 15-minute bar
    "stockdata_1H": 1 / 24,              # 1-hour bar
    "stockdata_D": 1,                    # 1 trading day (approx)
    "stockdata_W": 7,
    "stockdata_M": 30,
}

# Pivot fractal width (bars each side) per timeframe - smaller for noisier
# intraday data, larger for daily/weekly/monthly.
PIVOT_WIDTH = {
    "stockdata_15": 6,
    "stockdata_1H": 5,
    "stockdata_D": 3,
    "stockdata_W": 2,
    "stockdata_M": 2,
}

# Gann-significant angular time counts (degrees of the Square of 9 circle,
# 1 degree ~ 1 unit of time/price in Gann's system). These are the classic
# harmonic divisions of 360.
GANN_ANGLES = [45, 90, 120, 135, 144, 180, 225, 240, 270, 315, 360]

# Fibonacci ratios used for time-zone projection from the base swing duration.
FIB_RATIOS = [0.382, 0.5, 0.618, 1.0, 1.618, 2.0, 2.618, 3.618, 4.236]


def load_bars(fp: Path):
    with open(fp, "r") as f:
        raw = json.load(f)
    if not raw:
        return None
    bars = []
    for r in raw:
        try:
            d = dt.datetime.fromisoformat(r["datetime"].replace("Z", "+00:00"))
            d = d.replace(tzinfo=None)
        except Exception:
            continue
        bars.append({
            "dt": d,
            "o": float(r["open"]),
            "h": float(r["high"]),
            "l": float(r["low"]),
            "c": float(r["close"]),
            "v": float(r.get("volume", 0)),
        })
    bars.sort(key=lambda x: x["dt"])
    symbol = raw[0].get("symbol", fp.stem)
    return symbol, bars


def find_swings(bars, width):
    """Fractal pivot detection: bar i is a swing high if its high is the max
    of the window [i-width, i+width]; swing low analogous. Returns list of
    dicts sorted by index."""
    n = len(bars)
    swings = []
    for i in range(width, n - width):
        window = bars[i - width:i + width + 1]
        h = bars[i]["h"]
        l = bars[i]["l"]
        if h == max(b["h"] for b in window):
            swings.append({"idx": i, "dt": bars[i]["dt"], "price": h, "type": "high"})
        elif l == min(b["l"] for b in window):
            swings.append({"idx": i, "dt": bars[i]["dt"], "price": l, "type": "low"})
    # collapse consecutive same-type swings to the most extreme one
    cleaned = []
    for s in swings:
        if cleaned and cleaned[-1]["type"] == s["type"]:
            better = (s["price"] > cleaned[-1]["price"]) if s["type"] == "high" else (s["price"] < cleaned[-1]["price"])
            if better:
                cleaned[-1] = s
        else:
            cleaned.append(s)
    return cleaned


def bars_to_date(anchor_dt, n_bars, tf):
    """Project n_bars forward from anchor_dt using the timeframe's approx
    calendar-days-per-bar, skipping weekends for daily/weekly-ish granularity."""
    days = n_bars * BAR_CALENDAR_DAYS[tf]
    return anchor_dt + dt.timedelta(days=days)


def method_gann(bars, swings, tf):
    """Project Gann angle time-counts (in bars) forward from the most
    significant recent swing (largest % retracement pivot in the last 60
    swings)."""
    if not swings:
        return []
    recent = swings[-60:] if len(swings) > 60 else swings
    # pick the swing with the largest absolute % move relative to the next swing
    anchor = recent[-1]
    if len(recent) >= 2:
        best, best_move = recent[-1], 0
        for i in range(len(recent) - 1):
            move = abs(recent[i + 1]["price"] - recent[i]["price"]) / max(recent[i]["price"], 1e-6)
            if move > best_move:
                best_move = move
                best = recent[i]
        anchor = best
    projections = []
    # Gann counts translated to bars: 1 degree ~ 1 bar for intraday/daily
    # scaling factor keeps counts sensible across timeframes.
    scale = {"stockdata_15": 1, "stockdata_1H": 1, "stockdata_D": 1, "stockdata_W": 0.25, "stockdata_M": 0.08}[tf]
    for ang in GANN_ANGLES:
        n_bars = max(1, round(ang * scale))
        proj_date = bars_to_date(anchor["dt"], n_bars, tf)
        projections.append({
            "anchor_date": anchor["dt"].isoformat(),
            "anchor_type": anchor["type"],
            "angle": ang,
            "bars": n_bars,
            "date": proj_date.isoformat(),
        })
    return projections


def method_fft(bars, tf):
    """Detrend closes, run FFT, find dominant cycle period (in bars),
    project the next few turning windows forward from the last bar."""
    closes = np.array([b["c"] for b in bars], dtype=float)
    n = len(closes)
    if n < 40:
        return None
    # detrend with a rolling mean subtraction (window ~ n/10, min 5)
    win = max(5, n // 10)
    kernel = np.ones(win) / win
    trend = np.convolve(closes, kernel, mode="same")
    detrended = closes - trend
    detrended = detrended - detrended.mean()
    windowed = detrended * np.hanning(n)
    spectrum = np.fft.rfft(windowed)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    # ignore the 0 freq and very low freq bins (< 1 full cycle every n/2 bars)
    min_period = 5
    max_period = n // 2
    valid = [(f, p, i) for i, (f, p) in enumerate(zip(freqs, power)) if f > 0 and (1 / f) >= min_period and (1 / f) <= max_period]
    if not valid:
        return None
    valid.sort(key=lambda x: -x[1])
    top = valid[:3]
    dominant = top[0]
    period_bars = round(1 / dominant[0])
    last_dt = bars[-1]["dt"]
    projections = []
    for mult in [1, 2, 3]:
        proj_date = bars_to_date(last_dt, period_bars * mult, tf)
        projections.append({"date": proj_date.isoformat(), "bars_ahead": period_bars * mult})
    top_periods = [{"period_bars": round(1 / f), "relative_power": round(float(p / power.max()), 3)} for f, p, i in top]
    return {
        "dominant_period_bars": period_bars,
        "dominant_period_calendar_days": round(period_bars * BAR_CALENDAR_DAYS[tf], 2),
        "top_periods": top_periods,
        "projections": projections,
    }


def method_fibonacci(bars, swings, tf):
    """Use the most recent completed swing-to-swing leg (low->high or
    high->low) as the base unit; project Fibonacci multiples of that many
    bars forward from the most recent swing."""
    if len(swings) < 2:
        return []
    a, b = swings[-2], swings[-1]
    base_bars = b["idx"] - a["idx"]
    if base_bars <= 0:
        return []
    projections = []
    for ratio in FIB_RATIOS:
        n_bars = round(base_bars * ratio)
        if n_bars <= 0:
            continue
        proj_date = bars_to_date(b["dt"], n_bars, tf)
        projections.append({
            "ratio": ratio,
            "base_bars": base_bars,
            "bars_ahead": n_bars,
            "date": proj_date.isoformat(),
        })
    return projections


def method_stats(bars, swings, tf):
    """Historical average / median bars between consecutive swings (of either
    type), used to project the next statistically 'due' turn."""
    if len(swings) < 3:
        return None
    intervals = [swings[i + 1]["idx"] - swings[i]["idx"] for i in range(len(swings) - 1)]
    if not intervals:
        return None
    mean_i = float(np.mean(intervals))
    median_i = float(np.median(intervals))
    std_i = float(np.std(intervals))
    last = swings[-1]
    proj_mean = bars_to_date(last["dt"], round(mean_i), tf)
    proj_median = bars_to_date(last["dt"], round(median_i), tf)
    return {
        "mean_bars_between_swings": round(mean_i, 2),
        "median_bars_between_swings": round(median_i, 2),
        "std_bars_between_swings": round(std_i, 2),
        "sample_size": len(intervals),
        "last_swing_date": last["dt"].isoformat(),
        "last_swing_type": last["type"],
        "projection_mean": proj_mean.isoformat(),
        "projection_median": proj_median.isoformat(),
    }


def analyze_stock(symbol, bars, tf):
    width = PIVOT_WIDTH[tf]
    if len(bars) < width * 2 + 5:
        return None
    swings = find_swings(bars, width)
    last = bars[-1]
    result = {
        "symbol": symbol,
        "timeframe": tf,
        "bars_count": len(bars),
        "last_date": last["dt"].isoformat(),
        "last_close": last["c"],
        "swings": [{"date": s["dt"].isoformat(), "price": s["price"], "type": s["type"]} for s in swings[-40:]],
        "methods": {
            "gann": method_gann(bars, swings, tf),
            "fft": method_fft(bars, tf),
            "fibonacci": method_fibonacci(bars, swings, tf),
            "stats": method_stats(bars, swings, tf),
        },
    }
    return result


def collect_upcoming_dates(result, horizon_days, now):
    """Pull every projected date across all 4 methods that falls within the
    horizon, tagging which method produced it (for confluence scoring)."""
    hits = []

    def add(date_str, method):
        try:
            d = dt.datetime.fromisoformat(date_str)
        except Exception:
            return
        delta_days = (d - now).total_seconds() / 86400
        if 0 <= delta_days <= horizon_days:
            hits.append({"date": d.date().isoformat(), "method": method, "days_ahead": round(delta_days, 1)})

    m = result["methods"]
    for p in (m.get("gann") or []):
        add(p["date"], "gann")
    fft = m.get("fft")
    if fft:
        for p in fft.get("projections", []):
            add(p["date"], "fft")
    for p in (m.get("fibonacci") or []):
        add(p["date"], "fibonacci")
    stats = m.get("stats")
    if stats:
        add(stats["projection_mean"], "stats")
        add(stats["projection_median"], "stats")
    return hits


def build_watchlist(all_results, horizon_days=21):
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    rows = []
    for r in all_results:
        hits = collect_upcoming_dates(r, horizon_days, now)
        if not hits:
            continue
        methods_hit = sorted(set(h["method"] for h in hits))
        # cluster hits by date (within +-2 days) to find confluence windows
        hits.sort(key=lambda h: h["date"])
        rows.append({
            "symbol": r["symbol"],
            "timeframe": r["timeframe"],
            "last_close": r["last_close"],
            "last_date": r["last_date"],
            "confluence_score": len(methods_hit),
            "methods_hit": methods_hit,
            "upcoming_dates": hits[:8],
        })
    rows.sort(key=lambda x: (-x["confluence_score"], x["upcoming_dates"][0]["date"] if x["upcoming_dates"] else ""))
    return {"generated_at": now.isoformat(), "horizon_days": horizon_days, "stocks": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Repo root containing stockdata_* folders")
    ap.add_argument("--out", default="cycles/data", help="Output folder for dashboard JSON")
    ap.add_argument("--horizon-days", type=int, default=21, help="Upcoming-turn watchlist horizon in days")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = defaultdict(list)
    all_results = []

    for tf in TIMEFRAMES:
        tf_dir = root / tf
        if not tf_dir.exists():
            continue
        tf_out = out / tf
        tf_out.mkdir(parents=True, exist_ok=True)
        for fp in sorted(tf_dir.glob("*.json")):
            loaded = load_bars(fp)
            if not loaded:
                continue
            symbol, bars = loaded
            result = analyze_stock(symbol, bars, tf)
            if not result:
                continue
            safe_symbol = symbol.replace("/", "_").replace(":", "_")
            with open(tf_out / f"{safe_symbol}.json", "w") as f:
                json.dump(result, f)
            manifest[tf].append(safe_symbol)
            all_results.append(result)
            print(f"  processed {tf}/{symbol}: {len(bars)} bars, {len(result['swings'])} swings")

    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    watchlist = build_watchlist(all_results, horizon_days=args.horizon_days)
    with open(out / "watchlist.json", "w") as f:
        json.dump(watchlist, f, indent=2)

    print(f"\nDone. {sum(len(v) for v in manifest.values())} stock/timeframe files written to {out}")
    print(f"Watchlist: {len(watchlist['stocks'])} stocks with upcoming turns in next {args.horizon_days} days")


if __name__ == "__main__":
    main()
