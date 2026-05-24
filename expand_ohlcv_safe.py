"""
OHLCV 시총 2000 안전 다운로드 V3
==============================
signal.SIGALRM으로 종목당 8초 hard timeout. 단일 프로세스 직렬.
- 1500종목 × 6초/평균 = 약 1.5시간
- hang 종목은 강제 abort 후 다음
- 50종목마다 중간 저장 (재시작 가능)
"""

import sys
import time
import pickle
import signal
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import FinanceDataReader as fdr
from pathlib import Path

CACHE = Path("cache")
OHLCV_PATH = CACHE / "ohlcv_2020-01-01_2026-05-23.pkl"


class TimeoutException(Exception): pass

def _timeout_handler(signum, frame):
    raise TimeoutException()


def fetch_with_timeout(code, timeout=8):
    """signal.SIGALRM으로 hard timeout."""
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        bars = fdr.DataReader(code, "2020-01-01", "2026-05-23")
        signal.alarm(0)
        if bars is None or bars.empty or len(bars) < 100:
            return None
        return bars
    except TimeoutException:
        return "TIMEOUT"
    except Exception:
        signal.alarm(0)
        return None


def main():
    snap = pd.read_parquet(CACHE / "market_snapshot.parquet")
    snap = snap.sort_values("MarketCap", ascending=False).head(2000)
    codes_top2000 = snap["Code"].tolist()
    print(f"[목표] 시총 2000 (cutoff {snap.iloc[-1]['MarketCap']/1e8:.0f}억)", flush=True)

    with open(OHLCV_PATH, "rb") as f:
        ohlcv_dict = pickle.load(f)
    existing = set(ohlcv_dict.keys())
    print(f"[기존] {len(existing)}종목", flush=True)

    to_download = [c for c in codes_top2000 if c not in existing]
    print(f"[다운로드 대상] {len(to_download)}", flush=True)
    if not to_download:
        print("완료"); return

    t0 = time.time()
    new_count = 0
    timeout_count = 0
    fail_count = 0

    for i, code in enumerate(to_download, 1):
        result = fetch_with_timeout(code, timeout=8)
        if isinstance(result, str) and result == "TIMEOUT":
            timeout_count += 1
        elif result is None:
            fail_count += 1
        else:
            ohlcv_dict[code] = result
            new_count += 1

        if i % 50 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(to_download) - i) / rate / 60
            print(f"  [{i}/{len(to_download)}] new={new_count}, timeout={timeout_count}, "
                  f"fail={fail_count}, {rate:.1f}/s, ETA {eta:.1f}분", flush=True)
            # 중간 저장
            with open(OHLCV_PATH, "wb") as f:
                pickle.dump(ohlcv_dict, f)

    with open(OHLCV_PATH, "wb") as f:
        pickle.dump(ohlcv_dict, f)
    print(f"\n[완료] new={new_count}, timeout={timeout_count}, fail={fail_count}, 총 {len(ohlcv_dict)}", flush=True)
    print(f"  소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
