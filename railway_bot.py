"""
railway_bot.py -- Continuous-loop version of the forward-testing bot, for
running on Railway (or any always-on host) instead of GitHub Actions.

This is a thin wrapper: it imports run_trading_pass(), run_monitor_pass(),
db_init(), and sync_csv() directly from single_run_bot.py, so ALL the
tested logic (per-symbol one-position-at-a-time gate, breakeven-to-3.0R
adjustment, full-history caching, cursor-based signal capture, correct
position sizing, per-symbol fresh-connection-per-request to dodge Deriv's
short-lived OTP) is reused UNCHANGED, not rewritten.

The only thing this file adds: an infinite loop with a sleep between
cycles, so it behaves like a real continuously-running bot instead of
"run once and exit" -- which is what Railway needs, since it keeps one
process alive rather than re-triggering a fresh job every 15 minutes the
way GitHub Actions does.

Why this fixes the timing-reliability problem:
GitHub Actions' free scheduled triggers are best-effort and were
confirmed (via real run timestamps) to sometimes gap by hours instead of
the configured 15 minutes. A process that stays running and sleeps
internally isn't competing for a shared scheduler's prioritization at
all -- its timing is controlled entirely by this script's own clock.
"""

import asyncio
import os
import traceback
from datetime import datetime, timezone

# Reuse every tested function as-is -- no logic duplicated or rewritten.
from single_run_bot import (
    db_init,
    run_monitor_pass,
    run_trading_pass,
    sync_csv,
    API_TOKEN,
    APP_ID,
)

CYCLE_SECONDS = int(os.environ.get("CYCLE_SECONDS", "900"))  # 900s = 15 min, matches the original design


async def run_forever():
    if not API_TOKEN:
        raise RuntimeError("Set DERIV_API_TOKEN environment variable (demo account token).")
    if not APP_ID:
        raise RuntimeError(
            "Set DERIV_APP_ID environment variable (a REGISTERED app ID from "
            "developers.deriv.com's 'Create new app' page, not a plain number)."
        )

    print(f"[diagnostic] Token received, length={len(API_TOKEN)} chars "
          f"(never printing the value itself).")
    print(f"[startup] Continuous bot starting. Cycle interval: {CYCLE_SECONDS}s "
          f"({CYCLE_SECONDS / 60:.1f} min). This process stays alive and loops "
          f"itself -- it does not depend on any external scheduler.")

    db_init()
    cycle_count = 0

    while True:
        cycle_count += 1
        cycle_start = datetime.now(timezone.utc)
        print(f"\n{'=' * 70}")
        print(f"[cycle {cycle_count}] Starting at {cycle_start.isoformat()}")
        print(f"{'=' * 70}")

        try:
            await run_monitor_pass()   # check existing open trades first
            await run_trading_pass()   # then look for new setups
            sync_csv()
        except Exception as e:
            # A single bad cycle (e.g. a transient network blip) should
            # never kill the whole long-running process -- log it and
            # keep going, same "don't let one symbol's error stop
            # everything else" philosophy already used inside
            # run_trading_pass/run_monitor_pass, just applied one level up.
            print(f"!! Cycle {cycle_count} raised an unhandled error: {e}")
            traceback.print_exc()

        elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        sleep_for = max(5, CYCLE_SECONDS - elapsed)
        print(f"[cycle {cycle_count}] Finished in {elapsed:.1f}s. "
              f"Sleeping {sleep_for:.1f}s until next cycle.")
        await asyncio.sleep(sleep_for)


if __name__ == "__main__":
    asyncio.run(run_forever())
