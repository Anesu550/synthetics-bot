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


def ensure_git_repo():
    """Turns Railway's container filesystem into a real git repo, so
    push_results_to_github() has something to actually push from.

    Railway's build (Nixpacks) copies your repo's files into the container
    as a plain snapshot -- it does NOT preserve `.git` history. Confirmed
    directly: `git rev-parse --is-inside-work-tree` failed on a real run.
    So there is no git repo to push from at all until we create one here.

    Approach: `git init` in place (the files are already correctly laid
    out from the build), point `origin` at the authenticated GitHub URL,
    and make one initial commit if needed so future pushes have a base to
    diff against. This does NOT re-download anything -- it just turns the
    existing files into a proper git working tree.
    """
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return False  # push_results_to_github() will print its own warning later

    already_repo = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True)
    if already_repo.returncode == 0:
        print("  [git-init] Container already has a .git directory -- skipping init.")
        return True

    print("  [git-init] No .git directory found -- initializing one now...")
    remote_url = f"https://x-access-token:{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"

    try:
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(["git", "config", "--global", "--add", "safe.directory", "*"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "railway-bot"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "bot@users.noreply.github.com"], check=True, capture_output=True)
        subprocess.run(["git", "remote", "add", "origin", remote_url], check=True, capture_output=True)

        # Fetch the real branch so we start from the actual repo history
        # (not a disconnected fresh history), then reset our working files
        # on top of it -- this keeps everything already on GitHub intact.
        fetch = subprocess.run(["git", "fetch", "origin", "main"], capture_output=True, text=True)
        if fetch.returncode == 0:
            subprocess.run(["git", "branch", "-M", "main"], check=True, capture_output=True)
            subprocess.run(["git", "reset", "origin/main"], check=True, capture_output=True)  # adopt remote history, keep local files as-is (working tree untouched)
            print("  [git-init] Adopted existing GitHub history successfully.")
        else:
            print(f"  !! [git-init] fetch failed, starting fresh history instead: {fetch.stderr}")
            subprocess.run(["git", "branch", "-M", "main"], check=True, capture_output=True)

        return True
    except subprocess.CalledProcessError as e:
        print(f"  !! [git-init] FAILED: {e}")
        return False


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
    print("  ########## GIT-PUSH ATTEMPT STARTING ##########")

    if not GITHUB_TOKEN or not GITHUB_REPO:
        print("  [git-push] SKIPPED -- GITHUB_TOKEN and/or GITHUB_REPO not set. "
              "Results are NOT being saved back to your repo. Set both env vars "
              "on Railway to fix this.")
        print("  ########## GIT-PUSH ATTEMPT FINISHED (skipped) ##########")
        return

    try:
        add_result = subprocess.run(["git", "add", "trades.db", "trade_log.csv", "candle_cache/"],
                                     capture_output=True, text=True)
        print(f"  [git-push] git add returncode={add_result.returncode} "
              f"stdout={add_result.stdout!r} stderr={add_result.stderr!r}")
        add_result.check_returncode()  # raise now if it actually failed, AFTER we've already printed the details

        # Nothing to commit is a normal, expected outcome most cycles (no new signals/trades)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], capture_output=True)
        if diff.returncode == 0:
            print("  [git-push] Nothing changed since last push -- skipping commit.")
            print("  ########## GIT-PUSH ATTEMPT FINISHED (nothing to push) ##########")
            return

        commit_msg = f"Bot run: {datetime.now(timezone.utc).isoformat()}"
        commit_result = subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True, text=True)
        print(f"  [git-push] git commit returncode={commit_result.returncode} "
              f"stdout={commit_result.stdout!r} stderr={commit_result.stderr!r}")
        commit_result.check_returncode()

        push_result = subprocess.run(["git", "push", "origin", "HEAD:main"], capture_output=True, text=True)
        print(f"  [git-push] git push returncode={push_result.returncode} "
              f"stdout={push_result.stdout!r} stderr={push_result.stderr!r}")
        push_result.check_returncode()

        print(f"  [git-push] SUCCESS -- pushed results to {GITHUB_REPO}.")
        print("  ########## GIT-PUSH ATTEMPT FINISHED (success) ##########")
    except subprocess.CalledProcessError as e:
        print(f"  !! [git-push] FAILED at a specific step -- see the returncode/stdout/stderr "
              f"printed just above this line for exactly which command failed and why.")
        print("  ########## GIT-PUSH ATTEMPT FINISHED (FAILED) ##########")
    except Exception as e:
        print(f"  !! [git-push] UNEXPECTED ERROR (not a git command failure): {type(e).__name__}: {e}")
        print("  ########## GIT-PUSH ATTEMPT FINISHED (FAILED) ##########")


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

    # Turn this container into a real git repo (see ensure_git_repo docstring
    # for why this is necessary on Railway specifically) before attempting
    # any pushes.
    git_ready = ensure_git_repo()
    if not git_ready and (not GITHUB_TOKEN or not GITHUB_REPO):
        print("  !! WARNING: GITHUB_TOKEN / GITHUB_REPO not set -- results will NOT "
              "be saved to your repo until these are set on Railway.")
    elif not git_ready:
        print("  !! WARNING: git repo setup failed -- see [git-init] error above. "
              "Results will NOT be saved to your repo until this is fixed.")
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
    while True:
        try:
            asyncio.run(run_forever())
        except BaseException as e:
            # Last-resort guard: run_forever() already has its own per-cycle
            # try/except, and run_trading_pass/run_monitor_pass now catch
            # BaseException per-symbol too. This outermost layer exists
            # purely so that if something STILL escapes all of that (a
            # genuinely unexpected crash), the process restarts itself
            # in-place after a short pause instead of relying on Railway's
            # container restart (which loses in-memory state and re-triggers
            # the whole slow first-history-build sequence every time).
            import traceback
            print(f"\n!!!! FATAL: run_forever() crashed entirely: {type(e).__name__}: {e}")
            traceback.print_exc()
            print("!!!! Restarting in 10 seconds...\n")
            import time
            time.sleep(10)
