"""
debug_auth.py -- Isolated diagnostic. Tests authorize() against BOTH app_id=1
and app_id=1089, printing the FULL raw response from Deriv for each so we
get real evidence instead of guessing. Never prints the token itself.
"""
import asyncio
import json
import os
import websockets

TOKEN = os.environ.get("DERIV_API_TOKEN")

async def test_app_id(app_id):
    uri = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    print(f"\n--- Testing app_id={app_id} ---")
    try:
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({"authorize": TOKEN}))
            resp = json.loads(await ws.recv())
            # print everything except the echoed-back token itself
            if "echo_req" in resp and "authorize" in resp.get("echo_req", {}):
                resp["echo_req"]["authorize"] = "***HIDDEN***"
            print(json.dumps(resp, indent=2))
    except Exception as e:
        print(f"Connection-level error: {e}")

async def main():
    if not TOKEN:
        print("!! DERIV_API_TOKEN not set")
        return
    print(f"Token length: {len(TOKEN)} chars (value hidden)")
    await test_app_id(1)
    await test_app_id(1089)

if __name__ == "__main__":
    asyncio.run(main())
