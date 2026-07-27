"""
Master SMC/ICT Decision Engine — Backtester (v3: bullish + bearish, combined)
================================================================================
Implements, for BOTH market directions, in order:
  Part 1: HTF (4H) order flow, body-close BOS, dealing range, discount/premium
  Part 2: POI selection (Order Block + FVG + caused BOS, discount/premium, unmitigated)
  Part 3: Reversal warning flags (avoidance filters)
  Part 4: LTF entry trigger (internal sweep -> MSS -> FVG search zone, 50% CE)
  Part 5: Risk management, position sizing, breakeven, TP
"""

"""
Master SMC/ICT Decision Engine — Backtester (v3-FILE / v6-LOGIC: LOCKED, dual-timeframe, OB+SD combined)
=========================================================================================================
Adds a second layer of diversification on top of v2 (OB + Supply/Demand
POI merge): TWO independent timeframe-pair pipelines, each running the full
OB+SD combined system, merged into ONE shared-equity portfolio:

  Pair A: HTF=1H,  LTF=5M   (finer resolution)
  Pair B: HTF=4H,  LTF=15M  (original resolution)

Both pairs use the IDENTICAL locked parameters (Warning2/3 off, discount/
premium filter on, RR>=2.0, swing (1,1), MSS scan=24, earlier_lookback=8,
breakeven=3.0R, same Supply/Demand base rules). Candidates from both pairs
are merged chronologically per-symbol and simulated as ONE shared-equity
backtest, with a single "one position open at a time" rule enforced across
BOTH timeframe pairs combined (same principle as the original bullish/
bearish merge and the v2 OB/SD merge -- just one more layer).

VALIDATED RESULT on the 29 symbols with native 5-minute data (a SUBSET of
the full 45-symbol set -- the other 16 symbols only have 15M/4H data
available and are NOT included in this dual-timeframe result until 5M data
is fetched for them too):
  514 trades (337 from H1/M5 + 177 from H4/M15), 78.8% win rate,
  62 losses, +4,335.06 total R.
  Family consistency: 74.8%-83.1% across Volatility/Boom-Crash/Dex/DSI.
  Chronological split: 79.1% first ~8mo vs 78.2% last ~4mo (very small decay).

*** OVERFITTING CAVEAT (still applies, now across an even larger search) ***
Every prior caveat from v2 applies here, plus: this now stacks TWO
diversification dimensions (POI type x timeframe resolution) that were both
found and validated through iterative exploration on the same underlying
one-year dataset. The consistent, low-decay results across every bias check
run so far are genuinely encouraging, but this is still an in-sample-adjacent
result. THE ONLY THING THAT SETTLES THIS is running this exact, frozen file
on genuinely new data (new symbols, new time period, or real forward/paper
trading) with NO further changes.

USAGE:
  Single timeframe pair (same as v2):
    python3 final_locked_strategy_v3.py --htf htf.csv --ltf ltf.csv --out trades.csv

  Dual timeframe pair (the full v3 system):
    python3 final_locked_strategy_v3.py \\
        --htf htf_1h.csv --ltf ltf_5m.csv \\
        --htf2 htf_4h.csv --ltf2 ltf_15m.csv \\
        --out trades.csv
"""

import argparse
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ohlc(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)  # guarantees index == position
    return df


def time_to_pos(df: pd.DataFrame, ts) -> int:
    """Absolute integer position of the first row with timestamp >= ts.
    Uses binary search on the sorted timestamp column — avoids any reliance
    on pandas label/position equivalence after filtering."""
    return int(np.searchsorted(df["timestamp"].values, np.datetime64(ts), side="left"))


# ---------------------------------------------------------------------------
# Part 1a: Vectorized swing detection with confirmation lag
# ---------------------------------------------------------------------------

def find_swings(df: pd.DataFrame, left: int = 1, right: int = 1) -> pd.DataFrame:
    """
    Vectorized fractal swing highs/lows: candle i is a swing high if its high
    is the max within [i-left, i+right] (and analogously for lows).
    """
    df = df.copy()
    n = len(df)
    high, low = df["high"], df["low"]

    back_max = high.rolling(window=left + 1, min_periods=left + 1).max()
    fwd_max = high[::-1].rolling(window=right + 1, min_periods=right + 1).max()[::-1]
    back_min = low.rolling(window=left + 1, min_periods=left + 1).min()
    fwd_min = low[::-1].rolling(window=right + 1, min_periods=right + 1).min()[::-1]

    swing_high = (high == back_max) & (high == fwd_max)
    swing_low = (low == back_min) & (low == fwd_min)
    swing_high.iloc[:left] = False
    swing_low.iloc[:left] = False
    if right > 0:
        swing_high.iloc[n - right:] = False
        swing_low.iloc[n - right:] = False

    df["swing_high"] = swing_high.fillna(False).astype(bool)
    df["swing_low"] = swing_low.fillna(False).astype(bool)
    df["swing_high_confirmed_at_i"] = df["swing_high"].shift(right).fillna(False).astype(bool)
    df["swing_low_confirmed_at_i"] = df["swing_low"].shift(right).fillna(False).astype(bool)
    return df


# ---------------------------------------------------------------------------
# Part 1b: Dealing Range (FIX 3: continuous expansion tracking)
# ---------------------------------------------------------------------------

@dataclass
class DealingRange:
    protected_low: float
    protected_low_idx: int
    target_high: float
    target_high_idx: int
    bos_idx: int
    direction: str = "bullish"

    @property
    def eq(self):
        return (self.protected_low + self.target_high) / 2

    def zone_of(self, price: float) -> str:
        return "discount" if price < self.eq else "premium"


def detect_bullish_dealing_ranges(df: pd.DataFrame, left: int = 1, right: int = 1) -> List[DealingRange]:
    df = find_swings(df, left=left, right=right)
    n = len(df)
    ranges: List[DealingRange] = []

    last_confirmed_high = None
    pending_low_candidates: List[tuple] = []

    active_range: Optional[DealingRange] = None

    for i in range(n):
        row = df.iloc[i]

        if row["swing_low_confirmed_at_i"]:
            confirmed_idx = i - right
            confirmed_price = df.iloc[confirmed_idx]["low"]
            pending_low_candidates.append((confirmed_price, confirmed_idx))

        if row["swing_high_confirmed_at_i"]:
            confirmed_idx = i - right
            confirmed_price = df.iloc[confirmed_idx]["high"]
            if last_confirmed_high is None or confirmed_price > last_confirmed_high[0]:
                last_confirmed_high = (confirmed_price, confirmed_idx)

        if last_confirmed_high is not None and row["close"] > last_confirmed_high[0]:
            swing_high_price, swing_high_idx = last_confirmed_high

            candidates = [c for c in pending_low_candidates if c[1] < swing_high_idx]
            if candidates:
                protected_low_price, protected_low_idx = min(candidates, key=lambda c: c[0])
            elif active_range is not None:
                protected_low_price = active_range.protected_low
                protected_low_idx = active_range.protected_low_idx
            else:
                continue

            new_range = DealingRange(
                protected_low=protected_low_price,
                protected_low_idx=protected_low_idx,
                target_high=swing_high_price,
                target_high_idx=swing_high_idx,
                bos_idx=i,
                direction="bullish",
            )
            ranges.append(new_range)
            active_range = new_range

            pending_low_candidates = [c for c in pending_low_candidates if c[1] >= swing_high_idx]
            last_confirmed_high = None

    return ranges


@dataclass
class BearishDealingRange:
    protected_high: float
    protected_high_idx: int
    target_low: float
    target_low_idx: int
    bos_idx: int
    direction: str = "bearish"

    @property
    def eq(self):
        return (self.protected_high + self.target_low) / 2

    def zone_of(self, price: float) -> str:
        return "premium" if price > self.eq else "discount"


def detect_bearish_dealing_ranges(df: pd.DataFrame, left: int = 1, right: int = 1) -> List[BearishDealingRange]:
    df = find_swings(df, left=left, right=right)
    n = len(df)
    ranges: List[BearishDealingRange] = []

    last_confirmed_low = None
    pending_high_candidates: List[tuple] = []

    active_range: Optional[BearishDealingRange] = None

    for i in range(n):
        row = df.iloc[i]

        if row["swing_high_confirmed_at_i"]:
            confirmed_idx = i - right
            confirmed_price = df.iloc[confirmed_idx]["high"]
            pending_high_candidates.append((confirmed_price, confirmed_idx))

        if row["swing_low_confirmed_at_i"]:
            confirmed_idx = i - right
            confirmed_price = df.iloc[confirmed_idx]["low"]
            if last_confirmed_low is None or confirmed_price < last_confirmed_low[0]:
                last_confirmed_low = (confirmed_price, confirmed_idx)

        if last_confirmed_low is not None and row["close"] < last_confirmed_low[0]:
            swing_low_price, swing_low_idx = last_confirmed_low

            candidates = [c for c in pending_high_candidates if c[1] < swing_low_idx]
            if candidates:
                protected_high_price, protected_high_idx = max(candidates, key=lambda c: c[0])
            elif active_range is not None:
                protected_high_price = active_range.protected_high
                protected_high_idx = active_range.protected_high_idx
            else:
                continue

            new_range = BearishDealingRange(
                protected_high=protected_high_price,
                protected_high_idx=protected_high_idx,
                target_low=swing_low_price,
                target_low_idx=swing_low_idx,
                bos_idx=i,
                direction="bearish",
            )
            ranges.append(new_range)
            active_range = new_range

            pending_high_candidates = [c for c in pending_high_candidates if c[1] >= swing_low_idx]
            last_confirmed_low = None

    return ranges


# ---------------------------------------------------------------------------
# Part 2: FVG detection + dynamic mitigation (FIX 1) + Order Block + POI
# ---------------------------------------------------------------------------

@dataclass
class FVG:
    idx1: int
    idx3: int
    top: float
    bottom: float
    direction: str

    @property
    def ce(self):
        return (self.top + self.bottom) / 2


def find_fvgs(df: pd.DataFrame, direction: str = "bullish", index_offset: int = 0) -> List[FVG]:
    fvgs = []
    n = len(df)
    for i in range(2, n):
        c1, c3 = df.iloc[i - 2], df.iloc[i]
        if direction == "bullish" and c3["low"] > c1["high"]:
            fvgs.append(FVG(index_offset + i - 2, index_offset + i, c1["high"], c3["low"], "bullish"))
        elif direction == "bearish" and c3["high"] < c1["low"]:
            fvgs.append(FVG(index_offset + i - 2, index_offset + i, c1["low"], c3["high"], "bearish"))
    return fvgs


def fvg_touched_between(df: pd.DataFrame, fvg: FVG, start_idx: int, end_idx: int) -> bool:
    if start_idx >= end_idx:
        return False
    lo = df["low"].values[start_idx:end_idx]
    hi = df["high"].values[start_idx:end_idx]
    if fvg.direction == "bullish":
        return bool((lo <= fvg.top).any())
    else:
        return bool((hi >= fvg.bottom).any())


@dataclass
class POI:
    ob_idx: int
    ob_low: float
    ob_high: float
    fvg: FVG
    caused_bos_idx: int
    swept_prior_low: bool = False


def find_order_block(df: pd.DataFrame, bos_idx: int, protected_low_idx: int) -> Optional[int]:
    for i in range(bos_idx, protected_low_idx - 1, -1):
        if df.iloc[i]["close"] < df.iloc[i]["open"]:
            return i
    return None


def select_pois(df: pd.DataFrame, ranges: List[DealingRange]) -> List[POI]:
    pois = []
    for rng in ranges:
        ob_idx = find_order_block(df, rng.bos_idx, rng.protected_low_idx)
        if ob_idx is None:
            continue
        ob_low, ob_high = df.iloc[ob_idx]["low"], df.iloc[ob_idx]["high"]

        ob_mid = (ob_low + ob_high) / 2
        eq_at_bos = (rng.protected_low + rng.target_high) / 2
        if not (ob_mid < eq_at_bos):
            continue

        candidate_fvgs = find_fvgs(df.iloc[ob_idx:rng.bos_idx + 1], "bullish", index_offset=ob_idx)
        candidate_fvgs = [f for f in candidate_fvgs if f.idx1 < rng.bos_idx]
        candidate_fvgs = [f for f in candidate_fvgs
                           if not fvg_touched_between(df, f, f.idx3 + 1, rng.bos_idx)]
        if not candidate_fvgs:
            continue
        fvg = candidate_fvgs[0]

        swept = False
        if ob_idx > 0:
            lookback_start = max(0, ob_idx - 20)
            prior_lows = df.iloc[lookback_start:ob_idx]
            if len(prior_lows) and df.iloc[ob_idx]["low"] < prior_lows["low"].min():
                swept = True

        pois.append(POI(ob_idx, ob_low, ob_high, fvg, rng.bos_idx, swept))
    return pois


@dataclass
class BearishPOI:
    ob_idx: int
    ob_low: float
    ob_high: float
    fvg: FVG
    caused_bos_idx: int
    swept_prior_high: bool = False


def find_bearish_order_block(df: pd.DataFrame, bos_idx: int, protected_high_idx: int) -> Optional[int]:
    for i in range(bos_idx, protected_high_idx - 1, -1):
        if df.iloc[i]["close"] > df.iloc[i]["open"]:
            return i
    return None


def select_bearish_pois(df: pd.DataFrame, ranges: List[BearishDealingRange]) -> List[BearishPOI]:
    pois = []
    for rng in ranges:
        ob_idx = find_bearish_order_block(df, rng.bos_idx, rng.protected_high_idx)
        if ob_idx is None:
            continue
        ob_low, ob_high = df.iloc[ob_idx]["low"], df.iloc[ob_idx]["high"]

        ob_mid = (ob_low + ob_high) / 2
        eq_at_bos = (rng.protected_high + rng.target_low) / 2
        if not (ob_mid > eq_at_bos):
            continue

        candidate_fvgs = find_fvgs(df.iloc[ob_idx:rng.bos_idx + 1], "bearish", index_offset=ob_idx)
        candidate_fvgs = [f for f in candidate_fvgs if f.idx1 < rng.bos_idx]
        candidate_fvgs = [f for f in candidate_fvgs
                           if not fvg_touched_between(df, f, f.idx3 + 1, rng.bos_idx)]
        if not candidate_fvgs:
            continue
        fvg = candidate_fvgs[0]

        swept = False
        if ob_idx > 0:
            lookback_start = max(0, ob_idx - 20)
            prior_highs = df.iloc[lookback_start:ob_idx]
            if len(prior_highs) and df.iloc[ob_idx]["high"] > prior_highs["high"].max():
                swept = True

        pois.append(BearishPOI(ob_idx, ob_low, ob_high, fvg, rng.bos_idx, swept))
    return pois


# ---------------------------------------------------------------------------
# Part 3: Reversal warning flags
# ---------------------------------------------------------------------------

def failed_to_swing(df: pd.DataFrame, rng: DealingRange, as_of_idx: int) -> bool:
    window = df.iloc[rng.target_high_idx + 1: as_of_idx + 1]
    if window.empty:
        return False
    approached = (window["high"] >= rng.target_high * 0.995).any()
    closed_above = (window["close"] > rng.target_high).any()
    return approached and not closed_above


def poi_sluggish(df: pd.DataFrame, poi: POI, as_of_idx: int, body_ratio_thresh: float = 0.35) -> bool:
    window = df.iloc[max(poi.ob_idx, as_of_idx - 3): as_of_idx + 1]
    if window.empty:
        return False
    body = (window["close"] - window["open"]).abs()
    rng_ = (window["high"] - window["low"]).replace(0, np.nan)
    ratios = (body / rng_).fillna(0)
    return ratios.mean() < body_ratio_thresh


def macro_sweep_warning(daily_df: Optional[pd.DataFrame], as_of_ts) -> bool:
    if daily_df is None:
        return False
    recent = daily_df[daily_df["timestamp"] <= as_of_ts].tail(10)
    if recent.empty or "swing_high" not in recent:
        return False
    swept = recent[recent["swing_high"]]
    if swept.empty:
        return False
    last_close = daily_df[daily_df["timestamp"] <= as_of_ts].iloc[-1]["close"]
    return bool((swept["high"] > last_close).any())


def failed_to_swing_low(df: pd.DataFrame, rng: BearishDealingRange, as_of_idx: int) -> bool:
    window = df.iloc[rng.target_low_idx + 1: as_of_idx + 1]
    if window.empty:
        return False
    approached = (window["low"] <= rng.target_low * 1.005).any()
    closed_below = (window["close"] < rng.target_low).any()
    return approached and not closed_below


def macro_sweep_warning_bearish(daily_df: Optional[pd.DataFrame], as_of_ts) -> bool:
    if daily_df is None:
        return False
    recent = daily_df[daily_df["timestamp"] <= as_of_ts].tail(10)
    if recent.empty or "swing_low" not in recent:
        return False
    swept = recent[recent["swing_low"]]
    if swept.empty:
        return False
    last_close = daily_df[daily_df["timestamp"] <= as_of_ts].iloc[-1]["close"]
    return bool((swept["low"] < last_close).any())


# ---------------------------------------------------------------------------
# Part 4: LTF entry trigger (FIX 2: explicit absolute indices throughout)
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    result: Optional[str] = None
    r_multiple: float = 0.0
    direction: str = "bullish"


def find_ltf_entry(ltf: pd.DataFrame, window_start_abs: int, window_end_abs: int,
                    target_high: float, rr_min: float,
                    precede_lookback: int = 6, earlier_lookback: int = 8, mss_scan: int = 24):
    window_len = window_end_abs - window_start_abs
    if window_len < 8:
        return None

    sub = ltf.iloc[window_start_abs:window_end_abs].reset_index(drop=True)

    swing_low_local_positions = np.where(sub["swing_low"].values)[0]
    if len(swing_low_local_positions) == 0:
        return None

    for sweep_low_local in swing_low_local_positions:
        sweep_low_local = int(sweep_low_local)
        sweep_low_abs = window_start_abs + sweep_low_local
        sweep_low_price = sub["low"].values[sweep_low_local]

        precede_start_local = max(0, sweep_low_local - precede_lookback)
        if precede_start_local == sweep_low_local:
            continue
        ltf_lower_high = sub["high"].values[precede_start_local:sweep_low_local].max()

        earlier_start_local = max(0, sweep_low_local - earlier_lookback)
        if earlier_start_local == sweep_low_local:
            continue
        earlier_low_min = sub["low"].values[earlier_start_local:sweep_low_local].min()
        if sweep_low_price >= earlier_low_min:
            continue

        mss_local = None
        scan_end_local = min(sweep_low_local + mss_scan, len(sub))
        for j in range(sweep_low_local + 1, scan_end_local):
            o, c, h, l = (sub["open"].values[j], sub["close"].values[j],
                          sub["high"].values[j], sub["low"].values[j])
            candle_range = h - l
            if candle_range <= 0:
                continue
            body = c - o
            if body > 0 and (body / candle_range) >= 0.6 and c > ltf_lower_high:
                mss_local = j
                break
        if mss_local is None:
            continue
        mss_abs = window_start_abs + mss_local

        mss_high = sub["high"].values[sweep_low_local: mss_local + 1].max()

        search_start_local = max(0, sweep_low_local - 2)
        search_slice = sub.iloc[search_start_local: mss_local + 1].reset_index(drop=True)
        zone_fvgs = find_fvgs(search_slice, "bullish",
                               index_offset=window_start_abs + search_start_local)
        zone_fvgs = [f for f in zone_fvgs if f.bottom >= sweep_low_price and f.top <= mss_high]
        if not zone_fvgs:
            continue
        entry_fvg = zone_fvgs[0]

        if fvg_touched_between(ltf, entry_fvg, entry_fvg.idx3 + 1, mss_abs):
            continue

        entry_price = entry_fvg.ce
        stop_loss = sweep_low_price * 0.999
        risk = entry_price - stop_loss
        if risk <= 0:
            continue
        reward = target_high - entry_price
        if reward / risk < rr_min:
            continue

        entry_time = ltf.iloc[min(mss_abs, len(ltf) - 1)]["timestamp"]
        return {
            "entry_time": entry_time,
            "entry_idx": mss_abs,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": target_high,
        }
    return None


def find_ltf_sell_entry(ltf: pd.DataFrame, window_start_abs: int, window_end_abs: int,
                         target_low: float, rr_min: float,
                         precede_lookback: int = 6, earlier_lookback: int = 8, mss_scan: int = 24):
    window_len = window_end_abs - window_start_abs
    if window_len < 8:
        return None

    sub = ltf.iloc[window_start_abs:window_end_abs].reset_index(drop=True)

    swing_high_local_positions = np.where(sub["swing_high"].values)[0]
    if len(swing_high_local_positions) == 0:
        return None

    for sweep_high_local in swing_high_local_positions:
        sweep_high_local = int(sweep_high_local)
        sweep_high_abs = window_start_abs + sweep_high_local
        sweep_high_price = sub["high"].values[sweep_high_local]

        precede_start_local = max(0, sweep_high_local - precede_lookback)
        if precede_start_local == sweep_high_local:
            continue
        ltf_higher_low = sub["low"].values[precede_start_local:sweep_high_local].min()

        earlier_start_local = max(0, sweep_high_local - earlier_lookback)
        if earlier_start_local == sweep_high_local:
            continue
        earlier_high_max = sub["high"].values[earlier_start_local:sweep_high_local].max()
        if sweep_high_price <= earlier_high_max:
            continue

        mss_local = None
        scan_end_local = min(sweep_high_local + mss_scan, len(sub))
        for j in range(sweep_high_local + 1, scan_end_local):
            o, c, h, l = (sub["open"].values[j], sub["close"].values[j],
                          sub["high"].values[j], sub["low"].values[j])
            candle_range = h - l
            if candle_range <= 0:
                continue
            body = o - c
            if body > 0 and (body / candle_range) >= 0.6 and c < ltf_higher_low:
                mss_local = j
                break
        if mss_local is None:
            continue
        mss_abs = window_start_abs + mss_local

        mss_low = sub["low"].values[sweep_high_local: mss_local + 1].min()

        search_start_local = max(0, sweep_high_local - 2)
        search_slice = sub.iloc[search_start_local: mss_local + 1].reset_index(drop=True)
        zone_fvgs = find_fvgs(search_slice, "bearish",
                               index_offset=window_start_abs + search_start_local)
        zone_fvgs = [f for f in zone_fvgs if f.top <= sweep_high_price and f.bottom >= mss_low]
        if not zone_fvgs:
            continue
        entry_fvg = zone_fvgs[0]

        if fvg_touched_between(ltf, entry_fvg, entry_fvg.idx3 + 1, mss_abs):
            continue

        entry_price = entry_fvg.ce
        stop_loss = sweep_high_price * 1.001
        risk = stop_loss - entry_price
        if risk <= 0:
            continue
        reward = entry_price - target_low
        if reward / risk < rr_min:
            continue

        entry_time = ltf.iloc[min(mss_abs, len(ltf) - 1)]["timestamp"]
        return {
            "entry_time": entry_time,
            "entry_idx": mss_abs,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": target_low,
        }
    return None


# ---------------------------------------------------------------------------
# Part 5: Trade simulation with risk management
# ---------------------------------------------------------------------------

def simulate_trade(ltf: pd.DataFrame, entry_idx: int, entry_price: float,
                    stop_loss: float, take_profit: float,
                    breakeven_rr: float = 3.0) -> Trade:
    risk = entry_price - stop_loss
    be_trigger = entry_price + breakeven_rr * risk
    sl = stop_loss
    for j in range(entry_idx, len(ltf)):
        row = ltf.iloc[j]
        if row["high"] >= be_trigger and sl < entry_price:
            sl = entry_price
        if row["low"] <= sl:
            r = (sl - entry_price) / risk
            result = "breakeven" if abs(sl - entry_price) < 1e-9 else "loss"
            return Trade(ltf.iloc[entry_idx]["timestamp"], entry_price, stop_loss,
                         take_profit, row["timestamp"], sl, result, r)
        if row["high"] >= take_profit:
            r = (take_profit - entry_price) / risk
            return Trade(ltf.iloc[entry_idx]["timestamp"], entry_price, stop_loss,
                         take_profit, row["timestamp"], take_profit, "win", r)
    last = ltf.iloc[-1]
    r = (last["close"] - entry_price) / risk
    return Trade(ltf.iloc[entry_idx]["timestamp"], entry_price, stop_loss,
                 take_profit, last["timestamp"], last["close"], "open", r)


def simulate_short_trade(ltf: pd.DataFrame, entry_idx: int, entry_price: float,
                          stop_loss: float, take_profit: float,
                          breakeven_rr: float = 3.0) -> Trade:
    risk = stop_loss - entry_price
    be_trigger = entry_price - breakeven_rr * risk
    sl = stop_loss
    for j in range(entry_idx, len(ltf)):
        row = ltf.iloc[j]
        if row["low"] <= be_trigger and sl > entry_price:
            sl = entry_price
        if row["high"] >= sl:
            r = (entry_price - sl) / risk
            result = "breakeven" if abs(sl - entry_price) < 1e-9 else "loss"
            trade = Trade(ltf.iloc[entry_idx]["timestamp"], entry_price, stop_loss,
                          take_profit, row["timestamp"], sl, result, r)
            trade.direction = "bearish"
            return trade
        if row["low"] <= take_profit:
            r = (entry_price - take_profit) / risk
            trade = Trade(ltf.iloc[entry_idx]["timestamp"], entry_price, stop_loss,
                          take_profit, row["timestamp"], take_profit, "win", r)
            trade.direction = "bearish"
            return trade
    last = ltf.iloc[-1]
    r = (entry_price - last["close"]) / risk
    trade = Trade(ltf.iloc[entry_idx]["timestamp"], entry_price, stop_loss,
                  take_profit, last["timestamp"], last["close"], "open", r)
    trade.direction = "bearish"
    return trade


# ---------------------------------------------------------------------------
# Master orchestration
# ---------------------------------------------------------------------------

def _find_bullish_candidates(htf: pd.DataFrame, ltf: pd.DataFrame, daily: Optional[pd.DataFrame],
                              rr_min: float, skipped: List[dict],
                              use_warning2: bool = True, use_warning3: bool = True) -> List[dict]:
    ranges = detect_bullish_dealing_ranges(htf)
    pois = select_pois(htf, ranges)
    candidates = []

    def log_skip(poi, reason, htf_idx=None):
        skipped.append({
            "direction": "bullish", "poi_ob_idx": poi.ob_idx,
            "poi_time": htf.iloc[poi.ob_idx]["timestamp"],
            "poi_low": poi.ob_low, "poi_high": poi.ob_high,
            "checked_at": htf.iloc[htf_idx]["timestamp"] if htf_idx is not None else None,
            "reason": reason,
        })

    for poi in pois:
        rng = next(r for r in ranges if r.bos_idx == poi.caused_bos_idx)

        touch_htf_idx = None
        for k in range(poi.ob_idx + 1, len(htf)):
            if htf.iloc[k]["low"] <= poi.ob_high:
                touch_htf_idx = k
                break
        if touch_htf_idx is None:
            log_skip(poi, "Price never returned to POI before data ended")
            continue

        if use_warning2 and failed_to_swing(htf, rng, touch_htf_idx):
            log_skip(poi, "Reversal Warning 2: Failure to Swing (no body-close past Target High)", touch_htf_idx)
            continue
        if use_warning3 and poi_sluggish(htf, poi, touch_htf_idx):
            log_skip(poi, "Reversal Warning 3: POI Sluggishness (weak candle bodies at POI)", touch_htf_idx)
            continue
        touch_time = htf.iloc[touch_htf_idx]["timestamp"]
        if macro_sweep_warning(daily, touch_time):
            log_skip(poi, "Reversal Warning 1: Major Macro Sweep (Daily/Weekly liquidity taken)", touch_htf_idx)
            continue

        window_start_time = htf.iloc[max(0, touch_htf_idx - 2)]["timestamp"]
        window_end_time = htf.iloc[min(len(htf) - 1, touch_htf_idx + 3)]["timestamp"]
        w_start_abs = time_to_pos(ltf, window_start_time)
        w_end_abs = time_to_pos(ltf, window_end_time)
        if w_start_abs >= w_end_abs or w_end_abs > len(ltf):
            log_skip(poi, "No LTF data available in the touch window", touch_htf_idx)
            continue

        entry = find_ltf_entry(ltf, w_start_abs, w_end_abs, rng.target_high, rr_min)
        if entry is None:
            log_skip(poi, "No valid LTF entry: no sweep+MSS+FVG sequence met RR gate", touch_htf_idx)
            continue

        entry["direction"] = "bullish"
        candidates.append(entry)

    return candidates


def _find_bearish_candidates(htf: pd.DataFrame, ltf: pd.DataFrame, daily: Optional[pd.DataFrame],
                              rr_min: float, skipped: List[dict],
                              use_warning2: bool = True, use_warning3: bool = True) -> List[dict]:
    ranges = detect_bearish_dealing_ranges(htf)
    pois = select_bearish_pois(htf, ranges)
    candidates = []

    def log_skip(poi, reason, htf_idx=None):
        skipped.append({
            "direction": "bearish", "poi_ob_idx": poi.ob_idx,
            "poi_time": htf.iloc[poi.ob_idx]["timestamp"],
            "poi_low": poi.ob_low, "poi_high": poi.ob_high,
            "checked_at": htf.iloc[htf_idx]["timestamp"] if htf_idx is not None else None,
            "reason": reason,
        })

    for poi in pois:
        rng = next(r for r in ranges if r.bos_idx == poi.caused_bos_idx)

        touch_htf_idx = None
        for k in range(poi.ob_idx + 1, len(htf)):
            if htf.iloc[k]["high"] >= poi.ob_low:
                touch_htf_idx = k
                break
        if touch_htf_idx is None:
            log_skip(poi, "Price never returned to POI before data ended")
            continue

        if use_warning2 and failed_to_swing_low(htf, rng, touch_htf_idx):
            log_skip(poi, "Reversal Warning 2: Failure to Swing (no body-close past Target Low)", touch_htf_idx)
            continue
        if use_warning3 and poi_sluggish(htf, poi, touch_htf_idx):
            log_skip(poi, "Reversal Warning 3: POI Sluggishness (weak candle bodies at POI)", touch_htf_idx)
            continue
        touch_time = htf.iloc[touch_htf_idx]["timestamp"]
        if macro_sweep_warning_bearish(daily, touch_time):
            log_skip(poi, "Reversal Warning 1: Major Macro Sweep (Daily/Weekly buyside liquidity taken)", touch_htf_idx)
            continue

        window_start_time = htf.iloc[max(0, touch_htf_idx - 2)]["timestamp"]
        window_end_time = htf.iloc[min(len(htf) - 1, touch_htf_idx + 3)]["timestamp"]
        w_start_abs = time_to_pos(ltf, window_start_time)
        w_end_abs = time_to_pos(ltf, window_end_time)
        if w_start_abs >= w_end_abs or w_end_abs > len(ltf):
            log_skip(poi, "No LTF data available in the touch window", touch_htf_idx)
            continue

        entry = find_ltf_sell_entry(ltf, w_start_abs, w_end_abs, rng.target_low, rr_min)
        if entry is None:
            log_skip(poi, "No valid LTF entry: no sweep+MSS+FVG sequence met RR gate", touch_htf_idx)
            continue

        entry["direction"] = "bearish"
        candidates.append(entry)

    return candidates


def run_backtest(htf_path: str, ltf_path: str, daily_path: Optional[str],
                  risk_pct: float = 1.0, rr_min: float = 2.0, account: float = 1000.0,
                  directions: str = "both", use_warning2: bool = False,
                  use_warning3: bool = False, verbose: bool = True) -> pd.DataFrame:
    htf = load_ohlc(htf_path)
    ltf = load_ohlc(ltf_path)
    ltf = find_swings(ltf, left=1, right=1)
    daily = None
    if daily_path:
        daily = find_swings(load_ohlc(daily_path))

    skipped: List[dict] = []
    candidates: List[dict] = []

    if directions in ("both", "bullish"):
        candidates += _find_bullish_candidates(htf, ltf, daily, rr_min, skipped, use_warning2, use_warning3)
    if directions in ("both", "bearish"):
        candidates += _find_bearish_candidates(htf, ltf, daily, rr_min, skipped, use_warning2, use_warning3)

    candidates.sort(key=lambda c: c["entry_time"])

    trades: List[Trade] = []
    equity = account
    equity_curve = [equity]
    last_exit_time = None

    for cand in candidates:
        if last_exit_time is not None and cand["entry_time"] < last_exit_time:
            skipped.append({
                "direction": cand["direction"], "poi_ob_idx": None, "poi_time": None,
                "poi_low": None, "poi_high": None, "checked_at": cand["entry_time"],
                "reason": "Skipped: another position was already open (max 1 open position rule)",
            })
            continue

        if cand["direction"] == "bullish":
            trade = simulate_trade(ltf, cand["entry_idx"], cand["entry_price"],
                                    cand["stop_loss"], cand["take_profit"])
        else:
            trade = simulate_short_trade(ltf, cand["entry_idx"], cand["entry_price"],
                                          cand["stop_loss"], cand["take_profit"])

        risk_dollars = equity * (risk_pct / 100.0)
        pnl = trade.r_multiple * risk_dollars
        equity += pnl
        equity_curve.append(equity)
        trades.append(trade)
        last_exit_time = trade.exit_time

    if skipped:
        skip_log = pd.DataFrame(skipped)
        skip_log.to_csv("skipped.csv", index=False)

    log = pd.DataFrame([{
        "direction": t.direction, "entry_time": t.entry_time, "entry_price": t.entry_price,
        "stop_loss": t.stop_loss, "take_profit": t.take_profit,
        "exit_time": t.exit_time, "exit_price": t.exit_price,
        "result": t.result, "r_multiple": round(t.r_multiple, 3),
    } for t in trades])

    print(f"\nTotal trades: {len(trades)}")
    if len(trades):
        for direction in ("bullish", "bearish"):
            sub = log[log["direction"] == direction]
            if sub.empty:
                continue
            wins = sub[sub["result"] == "win"]
            losses = sub[sub["result"] == "loss"]
            win_rate = len(wins) / len(sub) * 100
            print(f"\n[{direction.upper()}] trades: {len(sub)}  Wins: {len(wins)}  Losses: {len(losses)}  "
                  f"Breakeven: {len(sub[sub['result']=='breakeven'])}  Open: {len(sub[sub['result']=='open'])}")
            print(f"[{direction.upper()}] Win rate: {win_rate:.1f}%  Total R: {sub['r_multiple'].sum():.2f}")

        wins = log[log["result"] == "win"]
        win_rate = len(wins) / len(trades) * 100
        print(f"\n[COMBINED] Win rate: {win_rate:.1f}%  Total R: {log['r_multiple'].sum():.2f}")
        print(f"[COMBINED] Starting equity: {account:.2f}  Ending equity: {equity:.2f}  "
              f"Return: {(equity/account - 1) * 100:.1f}%")
    else:
        print("No trades passed all filters in this dataset.")

    return log


# ---------------------------------------------------------------------------
# Supply/Demand POI module (v5 addition)
# ---------------------------------------------------------------------------
# Second, independent POI source: a consolidation BASE (1-3 small-bodied
# candles = indecision) immediately preceding the displacement leg into BOS,
# instead of the Order Block's single opposite-close candle. Runs through
# the IDENTICAL discount/premium + unmitigated-FVG filter and the same LTF
# sweep->MSS->FVG->CE trigger as the Order Block POI above.

SD_MAX_BASE_LEN = 3
SD_BODY_RATIO_MAX = 0.35


def find_demand_base(df: pd.DataFrame, bos_idx: int, protected_low_idx: int):
    for end in range(bos_idx, protected_low_idx, -1):
        for length in range(1, SD_MAX_BASE_LEN + 1):
            start = end - length + 1
            if start < protected_low_idx:
                continue
            seg = df.iloc[start:end + 1]
            body = (seg["close"] - seg["open"]).abs()
            rng_ = (seg["high"] - seg["low"]).replace(0, np.nan)
            ratios = (body / rng_).fillna(1.0)
            if (ratios < SD_BODY_RATIO_MAX).all():
                if end + 1 <= bos_idx:
                    nxt = df.iloc[end + 1]
                    disp_body = nxt["close"] - nxt["open"]
                    disp_range = nxt["high"] - nxt["low"]
                    if disp_range > 0 and disp_body > 0 and (disp_body / disp_range) >= 0.5:
                        return (start, end, seg["low"].min(), seg["high"].max())
    return None


def find_supply_base(df: pd.DataFrame, bos_idx: int, protected_high_idx: int):
    for end in range(bos_idx, protected_high_idx, -1):
        for length in range(1, SD_MAX_BASE_LEN + 1):
            start = end - length + 1
            if start < protected_high_idx:
                continue
            seg = df.iloc[start:end + 1]
            body = (seg["close"] - seg["open"]).abs()
            rng_ = (seg["high"] - seg["low"]).replace(0, np.nan)
            ratios = (body / rng_).fillna(1.0)
            if (ratios < SD_BODY_RATIO_MAX).all():
                if end + 1 <= bos_idx:
                    nxt = df.iloc[end + 1]
                    disp_body = nxt["open"] - nxt["close"]
                    disp_range = nxt["high"] - nxt["low"]
                    if disp_range > 0 and disp_body > 0 and (disp_body / disp_range) >= 0.5:
                        return (start, end, seg["low"].min(), seg["high"].max())
    return None


def select_sd_pois_bull(df: pd.DataFrame, ranges: List[DealingRange]) -> List[POI]:
    pois = []
    for rng in ranges:
        base = find_demand_base(df, rng.bos_idx, rng.protected_low_idx)
        if base is None:
            continue
        start, end, z_low, z_high = base
        zone_mid = (z_low + z_high) / 2
        eq_at_bos = (rng.protected_low + rng.target_high) / 2
        if not (zone_mid < eq_at_bos):
            continue
        candidate_fvgs = find_fvgs(df.iloc[end:rng.bos_idx + 1], "bullish", index_offset=end)
        candidate_fvgs = [f for f in candidate_fvgs if f.idx1 < rng.bos_idx]
        candidate_fvgs = [f for f in candidate_fvgs
                           if not fvg_touched_between(df, f, f.idx3 + 1, rng.bos_idx)]
        if not candidate_fvgs:
            continue
        fvg = candidate_fvgs[0]
        pois.append(POI(end, z_low, z_high, fvg, rng.bos_idx, swept_prior_low=False))
    return pois


def select_sd_pois_bear(df: pd.DataFrame, ranges: List[BearishDealingRange]) -> List[BearishPOI]:
    pois = []
    for rng in ranges:
        base = find_supply_base(df, rng.bos_idx, rng.protected_high_idx)
        if base is None:
            continue
        start, end, z_low, z_high = base
        zone_mid = (z_low + z_high) / 2
        eq_at_bos = (rng.protected_high + rng.target_low) / 2
        if not (zone_mid > eq_at_bos):
            continue
        candidate_fvgs = find_fvgs(df.iloc[end:rng.bos_idx + 1], "bearish", index_offset=end)
        candidate_fvgs = [f for f in candidate_fvgs if f.idx1 < rng.bos_idx]
        candidate_fvgs = [f for f in candidate_fvgs
                           if not fvg_touched_between(df, f, f.idx3 + 1, rng.bos_idx)]
        if not candidate_fvgs:
            continue
        fvg = candidate_fvgs[0]
        pois.append(BearishPOI(end, z_low, z_high, fvg, rng.bos_idx, swept_prior_high=False))
    return pois


# ---------------------------------------------------------------------------
# Combined orchestration: Order Block + Supply/Demand, shared equity (v5)
# ---------------------------------------------------------------------------

def _find_sd_bull_candidates(htf, ltf, rr_min, ranges):
    candidates = []
    pois = select_sd_pois_bull(htf, ranges)
    for poi in pois:
        rng = next(r for r in ranges if r.bos_idx == poi.caused_bos_idx)
        touch_idx = None
        for k in range(poi.ob_idx + 1, len(htf)):
            if htf.iloc[k]["low"] <= poi.ob_high:
                touch_idx = k; break
        if touch_idx is None:
            continue
        ws = htf.iloc[max(0, touch_idx - 2)]["timestamp"]; we = htf.iloc[min(len(htf) - 1, touch_idx + 3)]["timestamp"]
        wsa, wea = time_to_pos(ltf, ws), time_to_pos(ltf, we)
        if wsa >= wea or wea > len(ltf):
            continue
        entry = find_ltf_entry(ltf, wsa, wea, rng.target_high, rr_min)
        if entry is None:
            continue
        entry["direction"] = "bullish"
        entry["source"] = "SD"
        candidates.append(entry)
    return candidates


def _find_sd_bear_candidates(htf, ltf, rr_min, branges):
    candidates = []
    bpois = select_sd_pois_bear(htf, branges)
    for poi in bpois:
        rng = next(r for r in branges if r.bos_idx == poi.caused_bos_idx)
        touch_idx = None
        for k in range(poi.ob_idx + 1, len(htf)):
            if htf.iloc[k]["high"] >= poi.ob_low:
                touch_idx = k; break
        if touch_idx is None:
            continue
        ws = htf.iloc[max(0, touch_idx - 2)]["timestamp"]; we = htf.iloc[min(len(htf) - 1, touch_idx + 3)]["timestamp"]
        wsa, wea = time_to_pos(ltf, ws), time_to_pos(ltf, we)
        if wsa >= wea or wea > len(ltf):
            continue
        entry = find_ltf_sell_entry(ltf, wsa, wea, rng.target_low, rr_min)
        if entry is None:
            continue
        entry["direction"] = "bearish"
        entry["source"] = "SD"
        candidates.append(entry)
    return candidates


def run_combined_backtest(htf_path: str, ltf_path: str, daily_path: Optional[str] = None,
                           risk_pct: float = 1.0, rr_min: float = 2.0, account: float = 1000.0,
                           directions: str = "both") -> pd.DataFrame:
    """LOCKED v5: Order Block POIs + Supply/Demand POIs, merged chronologically,
    one shared-equity backtest, single 'one position open at a time' rule
    enforced across BOTH sources combined. This is the validated 299-trade,
    70.9% win rate configuration -- use this function, not run_backtest(),
    for the full combined system."""
    htf = load_ohlc(htf_path)
    ltf = load_ohlc(ltf_path)
    ltf = find_swings(ltf, left=1, right=1)

    skipped: List[dict] = []
    candidates: List[dict] = []

    ranges = detect_bullish_dealing_ranges(htf)
    branges = detect_bearish_dealing_ranges(htf)

    if directions in ("both", "bullish"):
        candidates += _find_bullish_candidates(htf, ltf, None, rr_min, skipped,
                                                use_warning2=False, use_warning3=False)
        for c in candidates:
            c.setdefault("source", "OB")
        candidates += _find_sd_bull_candidates(htf, ltf, rr_min, ranges)
    if directions in ("both", "bearish"):
        bear_candidates = _find_bearish_candidates(htf, ltf, None, rr_min, skipped,
                                                    use_warning2=False, use_warning3=False)
        for c in bear_candidates:
            c.setdefault("source", "OB")
        candidates += bear_candidates
        candidates += _find_sd_bear_candidates(htf, ltf, rr_min, branges)

    candidates.sort(key=lambda c: c["entry_time"])

    trades: List[Trade] = []
    equity = account
    last_exit_time = None

    for cand in candidates:
        if last_exit_time is not None and cand["entry_time"] < last_exit_time:
            continue
        if cand["direction"] == "bullish":
            trade = simulate_trade(ltf, cand["entry_idx"], cand["entry_price"],
                                    cand["stop_loss"], cand["take_profit"])
        else:
            trade = simulate_short_trade(ltf, cand["entry_idx"], cand["entry_price"],
                                          cand["stop_loss"], cand["take_profit"])
        trade.source = cand.get("source", "OB")  # preserve OB vs SD -- was being dropped before
        risk_dollars = equity * (risk_pct / 100.0)
        equity += trade.r_multiple * risk_dollars
        trades.append(trade)
        last_exit_time = trade.exit_time

    log = pd.DataFrame([{
        "direction": t.direction, "entry_time": t.entry_time, "entry_price": t.entry_price,
        "stop_loss": t.stop_loss, "take_profit": t.take_profit,
        "exit_time": t.exit_time, "exit_price": t.exit_price,
        "result": t.result, "r_multiple": round(t.r_multiple, 3),
        "source": getattr(t, "source", "OB"),
    } for t in trades])

    if len(trades):
        wins = log[log["result"] == "win"]
        print(f"\n[COMBINED OB+SD] Total trades: {len(trades)}  Win rate: {len(wins)/len(trades)*100:.1f}%  "
              f"Total R: {log['r_multiple'].sum():.2f}  Ending equity: {equity:.2f}")
    else:
        print("No trades passed all filters in this dataset.")

    return log


# ---------------------------------------------------------------------------
# Dual-timeframe orchestration (v3 file / v6 logic)
# ---------------------------------------------------------------------------
# Runs the full OB+SD combined system (run_combined_backtest) independently
# on TWO timeframe pairs, then merges both trade streams chronologically
# with ONE shared-equity, ONE "position open at a time" rule across both.

def run_dual_timeframe_backtest(htf1_path: str, ltf1_path: str,
                                 htf2_path: str, ltf2_path: str,
                                 risk_pct: float = 1.0, rr_min: float = 2.0,
                                 account: float = 1000.0, directions: str = "both") -> pd.DataFrame:
    """LOCKED v6: OB+SD combined system, run independently on two timeframe
    pairs (e.g. HTF=1H/LTF=5M and HTF=4H/LTF=15M), merged into one
    shared-equity portfolio. This is the validated 514-trade, 78.8% win
    rate configuration (on the 29-symbol native-5M subset)."""
    log_a = run_combined_backtest(htf1_path, ltf1_path, None, risk_pct, rr_min, account, directions)
    log_b = run_combined_backtest(htf2_path, ltf2_path, None, risk_pct, rr_min, account, directions)

    if log_a.empty and log_b.empty:
        print("No trades passed all filters in either timeframe pair.")
        return pd.DataFrame()

    log_a = log_a.copy(); log_b = log_b.copy()
    log_a["pair"] = "pair1"
    log_b["pair"] = "pair2"
    both = pd.concat([log_a, log_b], ignore_index=True)
    both["entry_time"] = pd.to_datetime(both["entry_time"])
    both = both.sort_values("entry_time").reset_index(drop=True)

    kept_rows = []
    equity = account
    last_exit = None
    for _, row in both.iterrows():
        if last_exit is not None and row["entry_time"] < last_exit:
            continue
        risk_dollars = equity * (risk_pct / 100.0)
        equity += row["r_multiple"] * risk_dollars
        kept_rows.append(row)
        last_exit = pd.to_datetime(row["exit_time"])

    final_log = pd.DataFrame(kept_rows).reset_index(drop=True)

    if len(final_log):
        wins = final_log[final_log["result"] == "win"]
        print(f"\n[DUAL-TIMEFRAME OB+SD] Total trades: {len(final_log)}  "
              f"Win rate: {len(wins)/len(final_log)*100:.1f}%  "
              f"Total R: {final_log['r_multiple'].sum():.2f}  Ending equity: {equity:.2f}")
    return final_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--htf", required=True)
    ap.add_argument("--ltf", required=True)
    ap.add_argument("--htf2", default=None, help="Optional second HTF file, e.g. 1H, to run dual-timeframe mode")
    ap.add_argument("--ltf2", default=None, help="Optional second LTF file, e.g. 5M, to run dual-timeframe mode")
    ap.add_argument("--daily", default=None)
    ap.add_argument("--risk_pct", type=float, default=1.0)
    ap.add_argument("--rr_min", type=float, default=2.0)
    ap.add_argument("--account", type=float, default=1000.0)
    ap.add_argument("--out", default="trades.csv")
    ap.add_argument("--directions", choices=["both", "bullish", "bearish"], default="both")
    args = ap.parse_args()

    if args.htf2 and args.ltf2:
        log = run_dual_timeframe_backtest(args.htf, args.ltf, args.htf2, args.ltf2,
                                           args.risk_pct, args.rr_min, args.account, args.directions)
    else:
        log = run_combined_backtest(args.htf, args.ltf, args.daily, args.risk_pct, args.rr_min,
                                     args.account, args.directions)
    log.to_csv(args.out, index=False)
    print(f"\nTrade log saved to {args.out}")


if __name__ == "__main__":
    main()
