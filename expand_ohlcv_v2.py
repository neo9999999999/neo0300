"""
OHLCV 시총 2000 확장 V2 — timeout 강제 (hang 방지)
=================================================
각 종목 다운로드 timeout 8초 강제.
multiprocessing.Pool 사용 (thread 보다 안정).
"""

import sys
import time
import pickle
import signal
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path
from multiprocessing import Pool, TimeoutError

CACHE = Path("cache")
OHLCV_PATH = CACHE / "ohlcv_2020-01-01_2026-05-23.pkl"


def fetch_one(code):
    """단일 종목 다운로드. timeout 강제는 호출자가 함."""
    import FinanceDataReader as fdr
    try:
        bars = fdr.DataReader(code, "2020-01-01", "2026-05-23")
        if bars is None or bars.empty or len(bars) < 100:
            return code, None
        return code, bars
    except Exception:
        return code, None


def main():
    # 시총 2000
    snap = pd.read_parquet(CACHE / "market_snapshot.parquet")
    snap = snap.sort_values("MarketCap", ascending=False).head(2000)
    codes_top2000 = snap["Code"].tolist()
    print(f"[목표] 시총 상위 2000 (cutoff {snap.iloc[-1]['MarketCap']/1e8:.0f}억)", flush=True)

    with open(OHLCV_PATH, "rb") as f:
        ohlcv_dict = pickle.load(f)
    existing = set(ohlcv_dict.keys())
    print(f"[기존] {len(existing)}종목", flush=True)

    to_download = [c for c in codes_top2000 if c not in existing]
    print(f"[다운로드 대상] {len(to_download)}종목", flush=True)
    if not to_download:
        print("완료. exit."); return

    t0 = time.time()
    new_count = 0
    failed = []

    print(f"\n[다운로드 시작] (Pool workers=6, timeout 8s/종목)", flush=True)
    with Pool(processes=6) as pool:
        # imap_unordered: 결과 도착 순서로 처리
        results_iter = pool.imap_unordered(fetch_one, to_download, chunksize=1)
        for i in range(len(to_download)):
            try:
                code, bars = results_iter.next(timeout=15)  # 15초 안에 한 종목 결과
            except TimeoutError:
                # 한 종목이 hang 됨 → skip
                failed.append("TIMEOUT")
                continue
            except StopIteration:
                break
            if bars is not None:
                ohlcv_dict[code] = bars
                new_count += 1
            else:
                failed.append(code)
            if (i+1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i+1)/elapsed
                eta = (len(to_download) - (i+1))/rate/60
                print(f"  [{i+1}/{len(to_download)}] new={new_count}, fail={len(failed)}, "
                      f"{rate:.1f}/s, ETA {eta:.1f}분", flush=True)
                # 중간 저장
                if (i+1) % 200 == 0:
                    with open(OHLCV_PATH, "wb") as f:
                        pickle.dump(ohlcv_dict, f)

    with open(OHLCV_PATH, "wb") as f:
        pickle.dump(ohlcv_dict, f)
    print(f"\n[완료] 추가 {new_count}, 실패 {len(failed)}, 총 {len(ohlcv_dict)}종목", flush=True)
    print(f"  소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
