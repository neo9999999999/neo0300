"""
9개 프리셋의 enriched trades를 사전 계산해서 parquet 캐시 저장.
한 번 돌리면 Streamlit에서 즉시 로드 (재계산 없음).
"""
import warnings
warnings.filterwarnings("ignore")

import pickle
import time
import sys
from pathlib import Path
import pandas as pd

from backtest_helpers import enrich_trades

CACHE = Path("cache")
SRC = CACHE / "wf_full_2020-01-01_2026-05-21_u1000.pkl"


def main():
    if not SRC.exists():
        print(f"❌ {SRC} 없음. walk_forward 먼저 실행 필요.", flush=True)
        sys.exit(1)

    print(f"=== 사전 enrich 시작: {pd.Timestamp.now()} ===", flush=True)
    with open(SRC, "rb") as f:
        all_trades = pickle.load(f)

    print(f"프리셋 {len(all_trades)}개 발견", flush=True)
    t0 = time.time()

    for i, (key, df) in enumerate(all_trades.items(), 1):
        out = CACHE / f"enriched_{key}.parquet"
        if out.exists():
            print(f"[{i}/{len(all_trades)}] {key} — 이미 캐시 존재 ✓", flush=True)
            continue

        if df is None or df.empty:
            print(f"[{i}/{len(all_trades)}] {key} — 빈 데이터프레임 skip", flush=True)
            continue

        print(f"[{i}/{len(all_trades)}] {key} enrich 시작 ({len(df):,}행)", flush=True)
        t1 = time.time()
        try:
            enriched = enrich_trades(df, with_forward=True)
            enriched.to_parquet(out)
            print(f"  → {out.name} 저장 ({time.time()-t1:.0f}초)", flush=True)
        except Exception as e:
            print(f"  ❌ 오류: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print(f"\n=== 완료 (총 {(time.time()-t0)/60:.1f}분) ===", flush=True)


if __name__ == "__main__":
    main()
