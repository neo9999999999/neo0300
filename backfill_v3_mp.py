"""
백필 V3 - multiprocessing.Pool 사용
================================
일자별 작업을 process pool로 병렬 처리.
hang 회피 + 빠른 속도.
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

# 전역 (worker 초기화 시 로드)
OHLCV = None
CODE_INFO = None
PRESETS_CFG = None  # {preset: (sw, sp)}


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
    print(f"[MP 백필] CPU: {cpu_count()}, workers={cpu_count()-1}", flush=True)
    START = pd.Timestamp("2025-08-25")
    END = pd.Timestamp("2026-05-22")
    dates = [str(d) for d in pd.bdate_range(START, END)]
    print(f"  영업일: {len(dates)}", flush=True)

    t0 = time.time()
    all_rows = []
    with Pool(processes=max(2, cpu_count()-1), initializer=init_worker) as pool:
        for i, rows in enumerate(pool.imap_unordered(signals_for_date, dates, chunksize=2), 1):
            all_rows.extend(rows)
            if i % 10 == 0:
                el = time.time() - t0
                rate = i/el if el>0 else 0
                eta = (len(dates)-i)/rate/60 if rate>0 else 0
                print(f"  [{i}/{len(dates)}] 누적 {len(all_rows):,} | {rate:.2f}d/s | ETA {eta:.1f}분", flush=True)

    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        print("[경고] 0건", flush=True); return
    df["Date"] = pd.to_datetime(df["Date"])
    out = Path("cache") / "backfill_v3_2025-08-25_2026-05-22.parquet"
    df.to_parquet(out, index=False)
    print(f"\n[완료] {len(df):,}건 → {out.name}", flush=True)
    print(f"  일자수: {df['Date'].nunique()}, 평균 {len(df)/df['Date'].nunique():.1f}건/일", flush=True)
    print(f"  소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
