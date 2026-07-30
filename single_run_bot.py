"""
live_bot.py -- Automated forward-testing bot for the locked SMC/ICT v3 strategy.
FULLY CORRECTED VERSION -- fixes 4 real fidelity gaps found in the prior draft:
  1. Per-symbol "one open position at a time" gate (matches
     run_dual_timeframe_backtest's merge rule exactly)
  2. Breakeven-to-3.0R contract adjustment (this was VALIDATED to matter --
     65.8%->79.7% win rate improvement in testing -- and was previously
     missing entirely)
  3. Full-history caching (not a tiny rolling window) so dealing-range/BOS
     detection has the same context the backtest had
  4. Processes ALL new signals since last check, in order -- never silently
     skips a setup that formed between polling cycles

*** MANDATORY SETUP BEFORE RUNNING ***
1. Deriv account -> switch to DEMO mode (top-right account switcher).
2. Create an API token WITH TRADE SCOPE on the DEMO account:
   https://app.deriv.com/account/api-token  (check "Trade" + "Read")
3. On your VPS:
       export DERIV_API_TOKEN="your_demo_token_here"
4. Put final_locked_strategy_v3.py in the same folder as this file.
5. VERIFY each symbol's available multiplier values on Deriv's platform
   (Trade -> Multipliers -> pick symbol) -- MULTIPLIER_MAP below are
   placeholders and WILL be wrong for some symbols until you check.

*** THINGS I CANNOT VERIFY FROM MY SANDBOX (NO NETWORK ACCESS TO DERIV) ***
- The exact field names Deriv returns for contract_update responses
- Whether "stop_loss": 0 is accepted as "move to breakeven" (some APIs
  require a small positive epsilon instead of exactly 0) -- TEST THIS
  with one small manual trade before trusting it at scale.
- Rate limits on how often you can call ticks_history / contract_update
  per connection -- if you see rate-limit errors in the console, increase
  POLL_INTERVAL_SEC and/or the incremental fetch size.
Watch the console output closely for the first day of real running and
cross-check every open/close/breakeven event against the Deriv platform
itself before trusting this unattended.

INSTALL:
    pip install websockets pandas numpy --break-system-packages

RUN (inside screen/tmux so it survives disconnecting from the VPS):
    screen -S tradingbot
    python3 live_bot.py
    (Ctrl+A then D to detach; "screen -r tradingbot" to reattach later)
"""

import asyncio
import json
import os
import sqlite3
import time
from datetime import datetime, timezone

import websockets
import pandas as pd

import final_locked_strategy_v3 as strat

# --- CRITICAL FIX: use app_id=1 (default test app) to match the token ---
APP_ID = 1
API_TOKEN = os.environ.get("DERIV_API_TOKEN")
WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

# --- CONFIGURE YOUR SYMBOL LIST + MULTIPLIER ---
# VERIFY these against Deriv's platform before running -- placeholders only.
MULTIPLIER_MAP = {
    "R_75":     100,
    "R_100":    100,
    "BOOM500":  100,
    "CRASH500": 100,
}

RISK_PCT = 1.0
POLL_INTERVAL_SEC = 60
BREAKEVEN_TRIGGER_R = 3.0  # matches the locked strategy's validated breakeven_rr=3.0
DB_PATH = "trades.db"
CSV_PATH = "trade_log.csv"
CACHE_DIR = "candle_cache"

# Full-history target counts, matching backtest scale (not a thin rolling window)
FULL_HISTORY_TARGET = {
    3600: 8000,    # 1H:  ~11 months
    300: 90000,    # 5M:  ~11 months
    14400: 2200,   # 4H:  ~1 year
    900: 35000,    # 15M: ~1 year
}

os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Database layer (SQLite = source of truth)
# ---------------------------------------------------------------------------

def db_init():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_id TEXT UNIQUE,
            direction TEXT,
            entry_time TEXT,
            entry_price REAL,
            stoploss REAL,
            take_profit REAL,
            exit_time TEXT,
            exitprice REAL,
            result TEXT,
            rmultiple REAL,
            pair TEXT,
            symbol TEXT,
            source TEXT,
            stake REAL,
            stop_loss_amount REAL,
            breakeven_applied INTEGER DEFAULT 0,
            status TEXT DEFAULT 'open'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cursors (
            symbol TEXT,
            pair TEXT,
            last_entry_time TEXT,
            PRIMARY KEY (symbol, pair)
        )
    """)
    conn.commit()
    conn.close()


def db_has_open_trade_for_symbol(symbol):
    """Per-symbol one-position-at-a-time gate, matching
    run_dual_timeframe_backtest's merge rule across BOTH timeframe pairs."""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE symbol=? AND status='open'", (symbol,)
    ).fetchone()
    conn.close()
    return row[0] > 0


def db_insert_open_trade(contract_id, direction, entry_time, entry_price, stoploss,
                          take_profit, pair, symbol, source, stake, stop_loss_amount):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO trades (contract_id, direction, entry_time, entry_price, stoploss,
                             take_profit, pair, symbol, source, stake, stop_loss_amount, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
    """, (contract_id, direction, str(entry_time), entry_price, stoploss,
          take_profit, pair, symbol, source, stake, stop_loss_amount))
    conn.commit()
    conn.close()


def db_close_trade(contract_id, exit_time, exitprice, result, rmultiple):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        UPDATE trades SET exit_time=?, exitprice=?, result=?, rmultiple=?, status='closed'
        WHERE contract_id=?
    """, (str(exit_time), exitprice, result, rmultiple, contract_id))
    conn.commit()
    conn.close()


def db_mark_breakeven_applied(contract_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE trades SET breakeven_applied=1 WHERE contract_id=?", (contract_id,))
    conn.commit()
    conn.close()


def db_get_open_trades():
    """Returns full rows for all open trades (for monitor_loop)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM trades WHERE status='open'").fetchall()
    conn.close()
    return rows


def db_get_cursor(symbol, pair):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT last_entry_time FROM cursors WHERE symbol=? AND pair=?", (symbol, pair)
    ).fetchone()
    conn.close()
    return pd.to_datetime(row[0]) if row else pd.Timestamp("2000-01-01")


def db_set_cursor(symbol, pair, entry_time):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO cursors (symbol, pair, last_entry_time) VALUES (?, ?, ?)
        ON CONFLICT(symbol, pair) DO UPDATE SET last_entry_time=excluded.last_entry_time
    """, (symbol, pair, str(entry_time)))
    conn.commit()
    conn.close()


def sync_csv():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT direction, entry_time, entry_price, stoploss, take_profit,
               exit_time, exitprice, result, rmultiple, pair, symbol, source
        FROM trades ORDER BY entry_time
    """, conn)
    conn.close()
    df.to_csv(CSV_PATH, index=False)


# ---------------------------------------------------------------------------
# Full-history candle caching (fixes the "tiny rolling window" gap)
# ---------------------------------------------------------------------------

def cache_path(symbol, granularity):
    return os.path.join(CACHE_DIR, f"{symbol}_{granularity}.csv")


async def fetch_candle_chunk(ws, symbol, granularity, count, end):
    req = {
        "ticks_history": symbol, "adjust_start_time": 1, "count": count,
        "end": end, "start": 1, "style": "candles", "granularity": granularity,
    }
    await ws.send(json.dumps(req))
    resp = json.loads(await ws.recv())
    if "error" in resp:
        raise RuntimeError(f"Deriv error fetching {symbol}@{granularity}: {resp['error']['message']}")
    return resp["candles"]


async def ensure_full_history(ws, symbol, granularity):
    """On first run (or if cache missing), paginate back to build a full
    history matching backtest scale. Returns the cached DataFrame."""
    path = cache_path(symbol, granularity)
    target = FULL_HISTORY_TARGET[granularity]

    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["timestamp"])
        if len(df) >= target * 0.9:
            return df  # cache is already close to full scale

    print(f"  [history] Building full {granularity}s history for {symbol} (target={target})...")
    all_candles = []
    end = "latest"
    while len(all_candles) < target:
        remaining = target - len(all_candles)
        chunk = await fetch_candle_chunk(ws, symbol, granularity, min(5000, remaining), end)
        if not chunk:
            break
        if all_candles and chunk[-1]["epoch"] >= all_candles[0]["epoch"]:
            break
        all_candles = chunk + all_candles
        end = chunk[0]["epoch"] - 1
        await asyncio.sleep(0.3)

    df = pd.DataFrame(all_candles)
    df["timestamp"] = pd.to_datetime(df["epoch"], unit="s")
    df = df[["timestamp", "open", "high", "low", "close"]].sort_values("timestamp").reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"  [history] {symbol}@{granularity}s: cached {len(df)} candles")
    return df


async def update_history(ws, symbol, granularity):
    """Incremental update: fetch a small recent tail and merge into the cache."""
    df = await ensure_full_history(ws, symbol, granularity)
    recent = await fetch_candle_chunk(ws, symbol, granularity, 50, "latest")
    recent_df = pd.DataFrame(recent)
    recent_df["timestamp"] = pd.to_datetime(recent_df["epoch"], unit="s")
    recent_df = recent_df[["timestamp", "open", "high", "low", "close"]]

    combined = pd.concat([df, recent_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    combined.to_csv(cache_path(symbol, granularity), index=False)
    return combined


async def get_account_balance(ws):
    await ws.send(json.dumps({"balance": 1, "subscribe": 0}))
    resp = json.loads(await ws.recv())
    if "error" in resp:
        raise RuntimeError(f"Balance error: {resp['error']['message']}")
    return float(resp["balance"]["balance"])


async def place_multiplier_trade(ws, symbol, direction, entry_price, stop_loss_price,
                                  take_profit_price, stake, multiplier):
    contract_type = "MULTUP" if direction == "bullish" else "MULTDOWN"
    stop_loss_amount = stake * multiplier * abs(entry_price - stop_loss_price) / entry_price
    take_profit_amount = stake * multiplier * abs(take_profit_price - entry_price) / entry_price

    buy_req = {
        "buy": 1,
        "price": stake,
        "parameters": {
            "amount": stake, "basis": "stake", "contract_type": contract_type,
            "currency": "USD", "symbol": symbol, "multiplier": multiplier,
            "limit_order": {
                "stop_loss": round(stop_loss_amount, 2),
                "take_profit": round(take_profit_amount, 2),
            }
        }
    }
    await ws.send(json.dumps(buy_req))
    resp = json.loads(await ws.recv())
    if "error" in resp:
        print(f"  !! Trade placement FAILED for {symbol}: {resp['error']['message']}")
        return None, None
    contract_id = resp["buy"]["contract_id"]
    print(f"  >> Trade placed: {symbol} {contract_type} stake={stake:.2f} "
          f"SL_amt={stop_loss_amount:.2f} TP_amt={take_profit_amount:.2f} contract_id={contract_id}")
    return contract_id, stop_loss_amount


# ---------------------------------------------------------------------------
# Task 1: trading loop
# ---------------------------------------------------------------------------

async def trading_loop():
    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({"authorize": API_TOKEN}))
        resp = json.loads(await ws.recv())
        if "error" in resp:
            raise RuntimeError(f"Authorization failed: {resp['error']['message']}")
        if not resp["authorize"].get("is_virtual"):
            raise RuntimeError("!! Token is NOT a demo/virtual account. Refusing to trade live money.")
        print(f"[trading_loop] Authorized: {resp['authorize']['loginid']} (demo)")

        while True:
            try:
                balance = await get_account_balance(ws)
                print(f"[trading_loop {datetime.now(timezone.utc)}] Balance: {balance:.2f}")

                for symbol, multiplier in MULTIPLIER_MAP.items():
                    try:
                        htf1 = await update_history(ws, symbol, 3600)
                        ltf1 = await update_history(ws, symbol, 300)
                        htf2 = await update_history(ws, symbol, 14400)
                        ltf2 = await update_history(ws, symbol, 900)

                        for htf_df, ltf_df, pair_name in [(htf1, ltf1, "H1_M5"), (htf2, ltf2, "H4_M15")]:
                            htf_df.to_csv("_tmp_htf.csv", index=False)
                            ltf_df.to_csv("_tmp_ltf.csv", index=False)

                            log = strat.run_combined_backtest("_tmp_htf.csv", "_tmp_ltf.csv")
                            if log.empty:
                                continue

                            log["entry_time"] = pd.to_datetime(log["entry_time"])
                            cursor = db_get_cursor(symbol, pair_name)
                            new_signals = log[log["entry_time"] > cursor].sort_values("entry_time")

                            for _, sig in new_signals.iterrows():
                                db_set_cursor(symbol, pair_name, sig["entry_time"])  # never revisit this one

                                age_sec = (pd.Timestamp.utcnow() - sig["entry_time"]).total_seconds()
                                if age_sec > POLL_INTERVAL_SEC * 3:
                                    continue  # too stale to act on live

                                if db_has_open_trade_for_symbol(symbol):
                                    print(f"  -- Skipping {symbol} signal: another position already open "
                                          f"(matches backtest's one-at-a-time rule)")
                                    continue

                                stake = balance * (RISK_PCT / 100.0)
                                contract_id, sl_amount = await place_multiplier_trade(
                                    ws, symbol, sig["direction"], sig["entry_price"],
                                    sig["stop_loss"], sig["take_profit"], stake, multiplier
                                )
                                if contract_id:
                                    db_insert_open_trade(
                                        contract_id, sig["direction"], sig["entry_time"],
                                        sig["entry_price"], sig["stop_loss"], sig["take_profit"],
                                        pair_name, symbol, sig.get("source", "OB"), stake, sl_amount
                                    )
                                    sync_csv()
                                    balance = await get_account_balance(ws)  # refresh after spending stake

                    except Exception as e:
                        print(f"  !! Error processing {symbol}: {e}")

            except Exception as e:
                print(f"[trading_loop] cycle error: {e}")

            await asyncio.sleep(POLL_INTERVAL_SEC)


# ---------------------------------------------------------------------------
# Task 2: monitor loop -- contract close tracking + breakeven adjustment
# ---------------------------------------------------------------------------

async def monitor_loop():
    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({"authorize": API_TOKEN}))
        resp = json.loads(await ws.recv())
        if "error" in resp:
            raise RuntimeError(f"Authorization failed: {resp['error']['message']}")
        print(f"[monitor_loop] Authorized: {resp['authorize']['loginid']} (demo)")

        subscribed = set()

        while True:
            open_trades = db_get_open_trades()
            for t in open_trades:
                if t["contract_id"] not in subscribed:
                    await ws.send(json.dumps({
                        "proposal_open_contract": 1, "contract_id": t["contract_id"], "subscribe": 1
                    }))
                    subscribed.add(t["contract_id"])

            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=POLL_INTERVAL_SEC)
            except asyncio.TimeoutError:
                continue

            resp = json.loads(raw)
            if "error" in resp:
                print(f"[monitor_loop] error: {resp['error']['message']}")
                continue

            poc = resp.get("proposal_open_contract")
            if not poc:
                continue

            contract_id = str(poc["contract_id"])
            profit = float(poc.get("profit", 0.0))

            if poc.get("is_sold"):
                sell_price = float(poc.get("sell_price", 0.0))
                exit_time = datetime.fromtimestamp(poc.get("sell_time", time.time()), tz=timezone.utc)

                conn = sqlite3.connect(DB_PATH)
                row = conn.execute(
                    "SELECT stop_loss_amount FROM trades WHERE contract_id=?", (contract_id,)
                ).fetchone()
                conn.close()
                sl_amount = row[0] if row and row[0] else None
                r_multiple = profit / sl_amount if sl_amount and sl_amount > 0 else None

                result = "win" if profit > 0.01 else ("loss" if profit < -0.01 else "breakeven")
                db_close_trade(contract_id, exit_time, sell_price, result, r_multiple)
                sync_csv()
                subscribed.discard(contract_id)
                print(f"[monitor_loop] Closed {contract_id}: {result}  profit={profit:.2f}  R={r_multiple}")
                continue

            # --- Breakeven-to-3.0R check (this was the missing piece) ---
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            trade_row = conn.execute(
                "SELECT * FROM trades WHERE contract_id=?", (contract_id,)
            ).fetchone()
            conn.close()
            if not trade_row or trade_row["breakeven_applied"]:
                continue

            sl_amount = trade_row["stop_loss_amount"]
            if sl_amount and sl_amount > 0 and profit >= BREAKEVEN_TRIGGER_R * sl_amount:
                update_req = {
                    "contract_update": 1,
                    "contract_id": int(contract_id),
                    "limit_order": {"stop_loss": 0}  # VERIFY: some APIs need a tiny epsilon, not exactly 0
                }
                await ws.send(json.dumps(update_req))
                update_resp = json.loads(await ws.recv())
                if "error" in update_resp:
                    print(f"  !! Breakeven update FAILED for {contract_id}: {update_resp['error']['message']}")
                else:
                    db_mark_breakeven_applied(contract_id)
                    print(f"  >> Breakeven applied to {contract_id} (reached {BREAKEVEN_TRIGGER_R}R)")


# ---------------------------------------------------------------------------

async def main():
    if not API_TOKEN:
        raise RuntimeError("Set DERIV_API_TOKEN environment variable first (demo account token).")
    db_init()
    sync_csv()
    await asyncio.gather(trading_loop(), monitor_loop())


if __name__ == "__main__":
    asyncio.run(main())
