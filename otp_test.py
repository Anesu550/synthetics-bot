"""
otp_test.py -- Isolated diagnostic for Deriv's NEW API flow (confirmed via
their own documentation and support this session):
  1. GET  https://api.derivws.com/trading/v1/options/accounts
  2. POST https://api.derivws.com/trading/v1/options/accounts/{accountId}/otp
  3. Connect to the returned wss:// URL directly (OTP handles auth)
  4. Try a basic message and print the FULL raw response

This does NOT attempt to place a real trade. It only tests connectivity and
shows us exactly what Deriv sends back at each step, since documentation
did not confirm whether Multiplier contracts (MULTUP/MULTDOWN) are
supported through this specific endpoint.
"""
import asyncio
import json
import os
import requests
import websockets

TOKEN = os.environ.get("DERIV_API_TOKEN")
APP_ID = os.environ.get("DERIV_APP_ID", "1089")
BASE = "https://api.derivws.com"

async def main():
    if not TOKEN:
        print("!! DERIV_API_TOKEN not set")
        return

    headers = {"Authorization": f"Bearer {TOKEN}", "Deriv-App-ID": APP_ID}

    print("--- Step 1: GET /trading/v1/options/accounts ---")
    resp = requests.get(f"{BASE}/trading/v1/options/accounts", headers=headers)
    print(f"Status: {resp.status_code}")
    print(resp.text)

    if resp.status_code != 200:
        print("\nCan't proceed to OTP step without a successful account list. Stopping here.")
        return

    accounts_data = resp.json()
    accounts = accounts_data.get("data", accounts_data) if isinstance(accounts_data, dict) else accounts_data
    if isinstance(accounts, list) and accounts:
        account_id = accounts[0].get("account_id") or accounts[0].get("id")
    else:
        print("\nCouldn't find an account_id in the response above -- inspect it manually.")
        return

    print(f"\nUsing account_id: {account_id}")

    print("\n--- Step 2: POST /trading/v1/options/accounts/{account_id}/otp ---")
    resp2 = requests.post(f"{BASE}/trading/v1/options/accounts/{account_id}/otp", headers=headers)
    print(f"Status: {resp2.status_code}")
    print(resp2.text)

    if resp2.status_code != 200:
        print("\nCan't proceed to WebSocket step without a successful OTP. Stopping here.")
        return

    ws_url = resp2.json().get("data", {}).get("url")
    print(f"\nWebSocket URL received (otp hidden): {ws_url.split('?')[0]}?otp=***HIDDEN***" if ws_url else "No URL in response!")

    if not ws_url:
        return

    print("\n--- Step 3: Connect to the WebSocket URL ---")
    async with websockets.connect(ws_url) as ws:
        print("Connected successfully!")
        await ws.send(json.dumps({"ping": 1}))
        resp3 = await ws.recv()
        print("Response to ping:")
        print(resp3)

if __name__ == "__main__":
    asyncio.run(main())
