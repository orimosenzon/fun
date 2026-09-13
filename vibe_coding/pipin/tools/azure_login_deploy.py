#!/usr/bin/env python3
"""כניסה ל-Azure ופריסת מודל תמונות, בפקודה אחת.

למה זה קיים: ה-refresh token של Azure פג אחרי 90 יום ללא שימוש, ובלעדיו
אי אפשר לפרוס מודלים. az CLI לא מותקן כאן, ולכן הסקריפט עושה device-code
flow בעצמו מול ה-client id הציבורי של ה-CLI, ומעדכן את אותו מטמון טוקנים
ש-ai_azure קורא ממנו.

קוד ההתחברות תקף 15 דקות בלבד, ולכן הוא מונפק רק כשמריצים — לא מראש.

    python3 tools/azure_login_deploy.py                 # gpt-image-2
    python3 tools/azure_login_deploy.py --model dall-e-3
    python3 tools/azure_login_deploy.py --skip-login    # אם כבר מחוברים
"""

import argparse
import json
import pathlib
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import ai_azure

CLI_CLIENT_ID = "04b07795-8ddb-461a-bbee-02f9e1bf7b46"
TENANT_URL = "https://login.microsoftonline.com/organizations/oauth2/v2.0"
CACHE = pathlib.Path.home() / ".azure" / "msal_token_cache.json"


def device_login() -> None:
    body = urllib.parse.urlencode({
        "client_id": CLI_CLIENT_ID,
        "scope": "https://management.azure.com/.default offline_access",
    }).encode()
    flow = json.load(urllib.request.urlopen(
        urllib.request.Request(f"{TENANT_URL}/devicecode", data=body)))

    print()
    print("  פתח:  " + flow["verification_uri"])
    print("  קוד:  " + flow["user_code"])
    print(f"  (תקף {flow['expires_in'] // 60} דקות)")
    print()
    print("ממתין לאישור…", flush=True)

    deadline = time.time() + flow["expires_in"]
    poll_body = urllib.parse.urlencode({
        "client_id": CLI_CLIENT_ID,
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        "device_code": flow["device_code"],
    }).encode()

    while time.time() < deadline:
        time.sleep(flow.get("interval", 5))
        try:
            token = json.load(urllib.request.urlopen(
                urllib.request.Request(f"{TENANT_URL}/token", data=poll_body)))
            break
        except urllib.error.HTTPError as e:
            err = json.loads(e.read().decode())
            if err.get("error") in ("authorization_pending", "slow_down"):
                continue
            raise SystemExit(f"ההתחברות נכשלה: {err.get('error_description', '')[:200]}")
    else:
        raise SystemExit("הקוד פג לפני שאושר. הרץ שוב.")

    if not CACHE.exists():
        raise SystemExit(f"{CACHE} לא קיים — צריך az login אחד ידני כדי ליצור אותו.")
    shutil.copy(CACHE, CACHE.with_suffix(".json.bak"))
    data = json.loads(CACHE.read_text(encoding="utf-8"))
    key = list(data["RefreshToken"].keys())[0]
    data["RefreshToken"][key]["secret"] = token["refresh_token"]
    CACHE.write_text(json.dumps(data), encoding="utf-8")
    print("מחובר. הטוקן עודכן.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-image-2")
    ap.add_argument("--skip-login", action="store_true")
    args = ap.parse_args()

    if not args.skip_login:
        device_login()

    print(f"פורס {args.model}…", flush=True)
    ai_azure.deploy(args.model)

    # הפריסה אסינכרונית בצד של Azure; מחכים שהמודל יענה בפועל
    import azure_images
    for attempt in range(12):
        if azure_images.available_model(force=True) == args.model:
            break
        time.sleep(10)
        print(f"  עדיין עולה… ({(attempt + 1) * 10} שניות)", flush=True)
    else:
        print("נפרס, אבל עוד לא עונה. נסה שוב בעוד דקה.")
        return

    print(f"מוכן: {args.model}")
    print("פרוס עכשיו:", ai_azure.deployments())
    print(f"נוצל מהקרדיט: ${ai_azure.spend()} מתוך $1000")


if __name__ == "__main__":
    main()
