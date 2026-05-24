"""
시총 2000 종목 전체기간 시그널 백필
=================================
2020-04-06 ~ 2026-05-22 전체기간, 2000종목, 4프리셋.
multiprocessing.Pool 사용.
"""

import os
import sys
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

OHLCV = None
CODE_INFO = None
PRESETS_CFG = None


def init_worker():
    global OHLCV, CODE_INFO, PRESETS_CFG
    from config import SignalParams, ScoreWeights
    from rules import PRESETS

    _W_MAP = {
        "s1": "s1_box_breakout", "s2": "s2_volume_surge", "s3": "s3_long_candle",
        "s4": "s4_ma_alignment", "s5": "s5_near_high", "s6": "s6_no_overheating",
        "s7": "s7_pullback_setup", "s8": "s8_demand_continuity",
        "s9": "s9_longterm_ma_breakout", "s10": "s10_relative_strength",
        "s11": "s11_gap_ma_confluence", "s12": "s12_pattern_quality",
    }
    with open("cache/ohlcv_2020-01-01_2026-05-23.pkl", "rb") as f:
        OHLCV = pickle.load(f)
    snapshot = pd.read_parquet("cache/market_snapshot.parquet")
    snapshot = snapshot.sort_values("MarketCap", ascending=False).head(2000)
    CODE_INFO = snapshot.set_index("Code")[["Name", "Market"]].to_dict("index")
    PRESETS_CFG = {}
    for pname in ["default", "box_breakout", "habarocell", "pullback"]:
        p = PRESETS[pname]
        wd = {_W_MAP.get(k, k): v for k, v in p["weights"].items()}
        PRESETS_CFG[pname] = (ScoreWeights(**wd), SignalParams())


def signals_for_date(target_str):
    from scanner import compute_signals, total_score
    target = pd.Timestamp(target_str)
    rows = []
    for code, bars in OHLCV.items():
        if code not in CODE_INFO: continue
        info = CODE_INFO[code]
        sub = bars[bars.index <= target]
        if len(sub) < 260: continue
        if (target - sub.index[-1]).days > 4: continue
        actual_date = sub.index[-1]

        for pname, (sw, sp) in PRESETS_CFG.items():
            try:
                sig = compute_signals(sub, sp)
            except Exception:
                continue
            if not sig.get("valid", False): continue
            score = total_score(sig, sw, name=info["Name"])
            if score < 40: continue
            row = {"Date": actual_date, "Code": code,
                   "Name": info["Name"], "Market": info["Market"],
                   "Score": float(score),
                   "Close": float(sub["Close"].iloc[-1]),
                   "preset": pname}
            row.update(sig)
            rows.append(row)
    return rows


def main():
    print(f"[2000-종목 전체기간 백필] CPU: {cpu_count()}", flush=True)
    START = pd.Timestamp("2021-04-06")  # OHLCV 260일+ 보장
    END = pd.Timestamp("2026-05-22")
    dates = [str(d) for d in pd.bdate_range(START, END)]
    print(f"  영업일: {len(dates)}", flush=True)

    t0 = time.time()
    all_rows = []
    workers = 6  # 메모리 절약 (OHLCV 큼)
    print(f"  workers: {workers}", flush=True)

    with Pool(processes=workers, initializer=init_worker) as pool:
        for i, rows in enumerate(pool.imap_unordered(signals_for_date, dates, chunksize=2), 1):
            all_rows.extend(rows)
            if i % 50 == 0:
                el = time.time() - t0
                rate = i/el if el>0 else 0
                eta = (len(dates)-i)/rate/60 if rate>0 else 0
                print(f"  [{i}/{len(dates)}] 누적 {len(all_rows):,} | {rate:.2f}d/s | ETA {eta:.1f}분", flush=True)
                # 중간 저장 (빈 처리)
                if i % 200 == 0 and len(all_rows) > 0:
                    df_mid = pd.DataFrame(all_rows)
                    if "Date" in df_mid.columns:
                        df_mid["Date"] = pd.to_datetime(df_mid["Date"])
                        df_mid.to_parquet("cache/_signals_2000_partial.parquet", index=False)

    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        print("[경고] 0건", flush=True); return
    df["Date"] = pd.to_datetime(df["Date"])
    out = Path("cache") / "signals_2000_2020-04_2026-05.parquet"
    df.to_parquet(out, index=False)
    print(f"\n[완료] {len(df):,}건 → {out.name}", flush=True)
    print(f"  일자수: {df['Date'].nunique()}, 평균 {len(df)/df['Date'].nunique():.1f}건/일", flush=True)
    print(f"  Market: {df['Market'].value_counts().to_dict()}", flush=True)
    print(f"  소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
