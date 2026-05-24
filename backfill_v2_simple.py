"""
백필 V2 - 직렬 + 단순화 (hang 회피)
=================================
2025-08-23 ~ 2026-05-18 시그널 백필 (default 프리셋만).
직렬 처리 (no thread/process), 진행 상황 자주 출력.
"""

import sys
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from config import SignalParams, ScoreWeights
from scanner import compute_signals, total_score
from rules import PRESETS

CACHE = Path("cache")

# PRESETS의 s1..s12 → ScoreWeights 풀네임 매핑
_W_MAP = {
    "s1": "s1_box_breakout", "s2": "s2_volume_surge", "s3": "s3_long_candle",
    "s4": "s4_ma_alignment", "s5": "s5_near_high", "s6": "s6_no_overheating",
    "s7": "s7_pullback_setup", "s8": "s8_demand_continuity",
    "s9": "s9_longterm_ma_breakout", "s10": "s10_relative_strength",
    "s11": "s11_gap_ma_confluence", "s12": "s12_pattern_quality",
}


def get_preset(name):
    p = PRESETS[name]
    weights_dict = {_W_MAP.get(k, k): v for k, v in p["weights"].items()}
    return ScoreWeights(**weights_dict), SignalParams()


def main():
    print("[로드]", flush=True)
    with open(CACHE / "ohlcv_2020-01-01_2026-05-23.pkl", "rb") as f:
        OHLCV = pickle.load(f)
    snapshot = pd.read_parquet(CACHE / "market_snapshot.parquet")
    code_info = snapshot.set_index("Code")[["Name", "Market"]].to_dict("index")
    print(f"  종목: {len(OHLCV)}, snapshot: {len(snapshot)}", flush=True)

    # 4 프리셋
    preset_keys = ["default", "box_breakout", "habarocell", "pullback"]
    presets = {k: get_preset(k) for k in preset_keys}

    # 백필 기간
    START = pd.Timestamp("2025-08-25")
    END = pd.Timestamp("2026-05-22")
    all_dates = pd.bdate_range(START, END)
    print(f"  영업일: {len(all_dates)}일 ({START.date()} ~ {END.date()})", flush=True)

    all_rows = []
    t0 = time.time()

    for di, target in enumerate(all_dates, 1):
        # 그 일자 시그널
        for code, bars in OHLCV.items():
            if code not in code_info: continue
            info = code_info[code]
            sub = bars[bars.index <= target]
            if len(sub) < 260: continue
            if (target - sub.index[-1]).days > 4: continue
            actual_date = sub.index[-1]

            for pname, (sw, sp) in presets.items():
                try:
                    sig = compute_signals(sub, sp)
                except Exception:
                    continue
                if not sig.get("valid", False): continue
                score = total_score(sig, sw, name=info["Name"])
                if score < 40: continue
                all_rows.append({
                    "Date": actual_date, "Code": code,
                    "Name": info["Name"], "Market": info["Market"],
                    "Score": float(score),
                    "Close": float(sub["Close"].iloc[-1]),
                    "preset": pname,
                    **sig,
                })

        if di % 10 == 0 or di == len(all_dates):
            elapsed = time.time() - t0
            rate = di / elapsed if elapsed > 0 else 0
            eta = (len(all_dates) - di) / rate / 60 if rate > 0 else 0
            print(f"  [{di}/{len(all_dates)}] {target.date()} | 누적 {len(all_rows):,} | "
                  f"{rate:.1f} d/s | ETA {eta:.1f}분", flush=True)
            # 중간 저장
            if di % 30 == 0:
                df_mid = pd.DataFrame(all_rows)
                if not df_mid.empty:
                    df_mid["Date"] = pd.to_datetime(df_mid["Date"])
                    df_mid.to_parquet(CACHE / "backfill_v2_partial.parquet", index=False)

    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        print("[경고] 시그널 0건", flush=True)
        return
    df["Date"] = pd.to_datetime(df["Date"])
    out = CACHE / "backfill_v2_2025-08-25_2026-05-22.parquet"
    df.to_parquet(out, index=False)
    print(f"\n[완료] {len(df):,}건 → {out.name}", flush=True)
    print(f"  일자수: {df['Date'].nunique()}, 평균 {len(df)/df['Date'].nunique():.1f}건/일", flush=True)
    print(f"  Market: {df['Market'].value_counts().to_dict()}", flush=True)
    print(f"  소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
