"""
2025-08-23 ~ 2026-05-22 시그널 백필
=================================
walk_forward 전체 재실행은 오래 걸려서, OHLCV에서 직접 매일 시그널 계산.

각 영업일에 대해:
- OHLCV 끝점을 그 일자로 자르고
- 4 프리셋 ensemble로 시그널 추출
- chart_feats 형식으로 누적

결과: cache/backfill_signals_2025-08-23_2026-05-22.parquet
이후 candidates_enriched 확장 → today_picks 적용
"""

import sys
import time
import pickle
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import FilterConfig, ScoreWeights, SignalParams, RecommendConfig
from scanner import compute_signals, total_score
from rules import PRESETS


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
    sw = ScoreWeights(**weights_dict)
    sp = SignalParams()
    return sw, sp, p.get("min_score", 40)

CACHE = Path("cache")

# 1) 시총 상위 500 종목 OHLCV 로드
with open(CACHE / "ohlcv_2020-01-01_2026-05-23.pkl", "rb") as f:
    OHLCV = pickle.load(f)
snapshot = pd.read_parquet(CACHE / "market_snapshot.parquet")
print(f"종목수: {len(OHLCV)}, snapshot: {len(snapshot)}")

# Code -> Name, Market
code_info = snapshot.set_index("Code")[["Name", "Market"]].to_dict("index")


# 2) 백필 기간 영업일 추출
START = pd.Timestamp("2025-08-25")
END = pd.Timestamp("2026-05-22")
all_dates = pd.bdate_range(START, END)
print(f"백필 영업일: {len(all_dates)}일")


# 3) 일자별 시그널 생성 함수
def signals_for_date(target_date, preset_keys):
    """target_date 종가 기준 시그널 계산. 4 프리셋 모두 OK 면 ensemble.
       반환: list of dict (one per code per preset)
    """
    rows = []
    for code, bars in OHLCV.items():
        if code not in code_info:
            continue
        info = code_info[code]
        # target_date까지의 OHLCV
        sub = bars[bars.index <= target_date]
        if len(sub) < 260:
            continue
        # 마지막 데이터가 target_date의 4영업일 이내여야 (휴장일 허용)
        if (target_date - sub.index[-1]).days > 4:
            continue
        actual_date = sub.index[-1]

        for preset_name in preset_keys:
            try:
                sw, sp, min_score = get_preset(preset_name)
                sig = compute_signals(sub, sp)
            except Exception as e:
                continue
            if not sig.get("valid", False):
                continue
            score = total_score(sig, sw, name=info["Name"])
            if score < 40:
                continue
            rows.append({
                "Date": actual_date,
                "Code": code, "Name": info["Name"], "Market": info["Market"],
                "Score": float(score),
                "Close": float(sub["Close"].iloc[-1]),
                "preset": preset_name,
                **sig,
            })
    return rows


# 4) 백필 - 멀티스레드 일자별
preset_keys = ["default", "box_breakout", "habarocell", "pullback"]
print(f"\n프리셋: {preset_keys}")

all_rows = []
t0 = time.time()


def worker(d):
    return signals_for_date(d, preset_keys)


with ThreadPoolExecutor(max_workers=4) as ex:
    futures = {ex.submit(worker, d): d for d in all_dates}
    for i, fut in enumerate(as_completed(futures), 1):
        rows = fut.result()
        all_rows.extend(rows)
        if i % 20 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(all_dates) - i) / rate / 60
            print(f"  [{i}/{len(all_dates)}] 누적 시그널 {len(all_rows):,}, "
                  f"속도 {rate:.1f} d/s, ETA {eta:.1f}분")


print(f"\n시그널 총: {len(all_rows):,}건, 소요 {(time.time()-t0)/60:.1f}분")

if all_rows:
    df = pd.DataFrame(all_rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df.to_parquet(CACHE / "backfill_signals_2025-08-23_2026-05-22.parquet", index=False)
    print(f"\n저장: cache/backfill_signals_2025-08-23_2026-05-22.parquet")
    print(f"  일별 평균 {len(df)/df['Date'].nunique():.1f}건")
    print(f"  Market 분포: {df['Market'].value_counts().to_dict()}")
    print(f"  날짜 범위: {df['Date'].min().date()} ~ {df['Date'].max().date()}")
else:
    print("시그널 없음 (OHLCV 끝점 확인 필요)")
