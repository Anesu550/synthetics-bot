"""
railway_bot.py -- Continuous-loop version of the forward-testing bot, for
running on Railway (or any always-on host) instead of GitHub Actions.

This is a thin wrapper: it imports run_trading_pass(), run_monitor_pass(),
db_init(), and sync_csv() directly from single_run_bot.py, so ALL the
tested logic (per-symbol one-position-at-a-time gate, breakeven-to-3.0R
adjustment, full-history caching, cursor-based signal capture, correct
position sizing, per-symbol fresh-connection-per-request to dodge Deriv's
short-lived OTP) is reused UNCHANGED, not rewritten.

This file adds TWO things GitHub Actions used to do for you automatically
and Railway does NOT do on its own:
  1. An infinite loop with a sleep between cycles (continuous execution).
  2. A git commit+push of trades.db / trade_log.csv / candle_cache/ back
     to your GitHub repo after every cycle -- WITHOUT this, results only
     ever exist inside Railway's own container and are invisible to you
     (and would be lost on redeploy). GitHub Actions' workflow file had
     an explicit "git push" step; Railway has no equivalent built in, so
     we do it ourselves here using a GitHub Personal Access Token.
"""

import asyncio
import os
import subprocess
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

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")           # a GitHub Personal Access Token (repo scope)
GITHUB_REPO = os.environ.get("GITHUB_REPO")              # e.g. "Anesu550/synthetics-bot"


def push_results_to_github():
    """Commits and pushes trades.db, trade_log.csv, and candle_cache/ back
    to the GitHub repo, the same way GitHub Actions' workflow file used to
    do automatically. Railway has no equivalent built-in step, so this
    replicates it explicitly using a Personal Access Token over HTTPS.

    If GITHUB_TOKEN / GITHUB_REPO aren't set, this is skipped with a clear
    warning rather than crashing the whole bot -- results still exist
    locally inside the running container even if this step is misconfigured,
    so a missing credential shouldn't take down the trading logic itself.
    """
    if not GITHUB_TOKEN or not GITHUB_REPO:
        print("  [git-push] SKIPPED -- GITHUB_TOKEN and/or GITHUB_REPO not set. "
              "Results are NOT being saved back to your repo. Set both env vars "
              "on Railway to fix this.")
        return

    remote_url = f"https://x-access-token:{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"

    try:
        # Configure identity (harmless if already set; needed on a fresh container)
        subprocess.run(["git", "config", "--global", "user.name", "railway-bot"], check=True, capture_output=True)
        subprocess.run(["git", "config", "--global", "user.email", "bot@users.noreply.github.com"], check=True, capture_output=True)

        # Point the remote at the authenticated URL (idempotent -- safe to re-run every cycle)
        subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=True, capture_output=True)

        subprocess.run(["git", "add", "trades.db", "trade_log.csv", "candle_cache/"], check=True, capture_output=True)

        # Nothing to commit is a normal, expected outcome most cycles (no new signals/trades)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], capture_output=True)
        if diff.returncode == 0:
            print("  [git-push] Nothing changed since last push -- skipping commit.")
            return

        commit_msg = f"Bot run: {datetime.now(timezone.utc).isoformat()}"
        subprocess.run(["git", "commit", "-m", commit_msg], check=True, capture_output=True)
        push = subprocess.run(["git", "push", "origin", "HEAD:main"], check=True, capture_output=True, text=True)
        print(f"  [git-push] Pushed results to {GITHUB_REPO} successfully.")
    except subprocess.CalledProcessError as e:
        print(f"  !! [git-push] FAILED: {e}")
        print(f"     stdout: {e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout}")
        print(f"     stderr: {e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr}")


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

    # Verify this container actually has a git repo to push to -- fail loudly
    # at startup rather than silently failing every push for the next 3 months.
    git_dir_check = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True)
    if git_dir_check.returncode != 0:
        print("  !! WARNING: this container has no .git directory. Results CANNOT "
              "be pushed to GitHub from here -- trade_log.csv will only exist "
              "inside this Railway container and will be LOST on redeploy. "
              "This needs fixing before trusting the forward test.")
    elif not GITHUB_TOKEN or not GITHUB_REPO:
        print("  !! WARNING: GITHUB_TOKEN / GITHUB_REPO not set -- see push_results_to_github() "
              "docstring. Results will NOT be saved to your repo until these are set.")
    else:
        print(f"  [git-push] Configured to push results to {GITHUB_REPO} after every cycle.")

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
            push_results_to_github()   # <-- the missing step: save results back to your repo
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
