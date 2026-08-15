#!/usr/bin/env python3
"""Where in the world did pardespedia.info get visitors from, over a date range.

The GA4 web UI only draws a world map in the Realtime report, which covers the
last 30 minutes. For any longer window it gives a table — so this pulls the
same numbers through the Data API and prints the full list, longest first.

One-time setup (interactive, opens a browser):

    gcloud auth application-default login \\
      --scopes=https://www.googleapis.com/auth/analytics.readonly,\\
https://www.googleapis.com/auth/cloud-platform

    pip3 install --user google-analytics-data google-analytics-admin

Usage:
    python3 ga_geo.py [--days 7] [--property 123456789] [--csv out.csv]

With no --property the script lists the GA4 properties the account can see and
uses the only one, or asks you to pick when there are several.
"""
import argparse
import csv
import sys


def die(msg, hint=""):
    print(msg, file=sys.stderr)
    if hint:
        print(hint, file=sys.stderr)
    sys.exit(1)


def load_clients():
    try:
        from google.analytics.data_v1beta import BetaAnalyticsDataClient
        from google.analytics.admin import AnalyticsAdminServiceClient
    except ImportError:
        die("חסרות הספריות של GA4.",
            "pip3 install --user google-analytics-data google-analytics-admin")
    return BetaAnalyticsDataClient, AnalyticsAdminServiceClient


def find_property(admin_cls):
    admin = admin_cls()
    props = []
    for acct in admin.list_account_summaries():
        for p in acct.property_summaries:
            props.append((p.property.split("/")[-1], p.display_name))
    if not props:
        die("לא נמצאו נכסי GA4 לחשבון הזה.",
            "ודא שהחשבון שאיתו הרצת gcloud auth יש לו גישה לנכס.")
    if len(props) > 1:
        print("נמצאו כמה נכסים, בחר אחד עם --property:", file=sys.stderr)
        for pid, name in props:
            print(f"  {pid}  {name}", file=sys.stderr)
        sys.exit(2)
    print(f"נכס: {props[0][1]} ({props[0][0]})", file=sys.stderr)
    return props[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--property")
    ap.add_argument("--csv")
    args = ap.parse_args()

    data_cls, admin_cls = load_clients()
    from google.analytics.data_v1beta.types import (
        DateRange, Dimension, Metric, RunReportRequest, OrderBy)

    prop = args.property or find_property(admin_cls)
    client = data_cls()

    req = RunReportRequest(
        property=f"properties/{prop}",
        date_ranges=[DateRange(start_date=f"{args.days}daysAgo", end_date="today")],
        dimensions=[Dimension(name="country"), Dimension(name="city")],
        metrics=[Metric(name="activeUsers"), Metric(name="sessions"),
                 Metric(name="screenPageViews")],
        order_bys=[OrderBy(desc=True,
                           metric=OrderBy.MetricOrderBy(metric_name="activeUsers"))],
        limit=1000,
    )
    resp = client.run_report(req)

    rows = [(r.dimension_values[0].value, r.dimension_values[1].value,
             int(r.metric_values[0].value), int(r.metric_values[1].value),
             int(r.metric_values[2].value)) for r in resp.rows]

    by_country = {}
    for country, _city, users, sessions, views in rows:
        agg = by_country.setdefault(country, [0, 0, 0])
        agg[0] += users
        agg[1] += sessions
        agg[2] += views

    print(f"\n=== {args.days} הימים האחרונים, לפי מדינה ===")
    print(f"{'משתמשים':>9} {'הפעלות':>8} {'צפיות':>7}  מדינה")
    for country, (users, sessions, views) in sorted(
            by_country.items(), key=lambda kv: -kv[1][0]):
        print(f"{users:>9} {sessions:>8} {views:>7}  {country}")
    print(f"\nסה\"כ {sum(v[0] for v in by_country.values())} משתמשים "
          f"מ-{len(by_country)} מדינות, {len(rows)} ערים.")

    print(f"\n=== לפי עיר ===")
    print(f"{'משתמשים':>9} {'הפעלות':>8}  עיר, מדינה")
    for country, city, users, sessions, _views in rows:
        print(f"{users:>9} {sessions:>8}  {city or '(לא ידוע)'}, {country}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["מדינה", "עיר", "משתמשים", "הפעלות", "צפיות"])
            w.writerows(rows)
        print(f"\nנכתב אל {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
