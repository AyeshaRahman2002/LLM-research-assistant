# src/data_analysis/trend_forecast.py
import argparse
import json
from pathlib import Path

import pandas as pd

OUT = Path("results/forecast.json")


def naive_seasonal_last(df, col, period=7, horizon=14):
    y = df[col].dropna().values
    if len(y) < period:
        return []
    base = y[-period:]
    return base.tolist() * (horizon // period + 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        required=True,
        help="CSV with a date/time column named 'date' and a numeric 'y' column",
    )
    ap.add_argument("--horizon", type=int, default=14)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    if "date" not in df.columns or "y" not in df.columns:
        print("[skip] need 'date' and 'y' columns")
        exit(0)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    fc = naive_seasonal_last(
        df, "y", period=min(14, max(2, len(df) // 4)), horizon=args.horizon
    )[: args.horizon]
    start = df["date"].max()
    dates = pd.date_range(start=start, periods=len(fc) + 1, freq="D")[1:]
    payload = [
        {"date": d.strftime("%Y-%m-%d"), "forecast": float(v)}
        for d, v in zip(dates, fc)
    ]
    OUT.write_text(
        json.dumps(
            {"model": "naive_seasonal", "horizon": args.horizon, "points": payload},
            indent=2,
        )
    )
    print(f"DONE forecast -> {OUT}")
