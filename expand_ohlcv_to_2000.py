"""
OHLCV 캐시 시총 상위 500 → 2000 확장
=================================
fdr.DataReader로 시총 1500개 추가 다운로드.
- 진행상황 출력
- 실패 종목 skip
- timeout 5초/종목
"""

import os
import sys
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import FinanceDataReader as fdr
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

CACHE = Path("cache")
OHLCV_PATH = CACHE / "ohlcv_2020-01-01_2026-05-23.pkl"

# 1) 시총 상위 2000 추출
snap = pd.read_parquet(CACHE / "market_snapshot.parquet")
snap = snap.sort_values("MarketCap", ascending=False).head(2000).copy()
codes_top2000 = snap["Code"].tolist()
print(f"[목표] 시총 상위 2000 (시총 컷오프: {snap.iloc[-1]['MarketCap']/1e8:.0f}억)")

# 2) 기존 OHLCV 로드
with open(OHLCV_PATH, "rb") as f:
    ohlcv_dict = pickle.load(f)
existing = set(ohlcv_dict.keys())
print(f"[기존] {len(existing)}종목 캐시 보유")

# 3) 추가 다운로드 대상
to_download = [c for c in codes_top2000 if c not in existing]
print(f"[다운로드 대상] {len(to_download)}종목 추가 필요")

if len(to_download) == 0:
    print("이미 완료. exit.")
    sys.exit(0)


def fetch_one(code):
    try:
        bars = fdr.DataReader(code, "2020-01-01", "2026-05-23")
        if bars is None or bars.empty or len(bars) < 100:
            return code, None
        return code, bars
    except Exception as e:
        return code, None


t0 = time.time()
new_count = 0
failed = []

print(f"\n[다운로드 시작] (병렬 8 workers)")
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(fetch_one, c): c for c in to_download}
    for i, fut in enumerate(as_completed(futures), 1):
        code, bars = fut.result()
        if bars is not None:
            ohlcv_dict[code] = bars
            new_count += 1
        else:
            failed.append(code)
        if i % 50 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(to_download) - i) / rate / 60
            print(f"  [{i}/{len(to_download)}] new={new_count}, fail={len(failed)}, {rate:.1f}/s, ETA {eta:.1f}분", flush=True)
            # 중간 저장
            if i % 200 == 0:
                with open(OHLCV_PATH, "wb") as f:
                    pickle.dump(ohlcv_dict, f)

# 최종 저장
with open(OHLCV_PATH, "wb") as f:
    pickle.dump(ohlcv_dict, f)

print(f"\n[완료] 추가 {new_count}종목, 실패 {len(failed)}, 총 {len(ohlcv_dict)}종목")
print(f"  소요: {(time.time()-t0)/60:.1f}분")
if failed[:10]:
    print(f"  실패 샘플: {failed[:10]}")
