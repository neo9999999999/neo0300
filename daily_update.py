"""
매일 자동 업데이트 — KRX 데이터 캐시 갱신.

[실행]
python3 daily_update.py

[자동화 (GitHub Actions)]
.github/workflows/daily.yml 에 등록됨 (매일 한국 시간 16:00)

[자동화 (로컬 cron)]
0 16 * * 1-5 cd /path/to/neo0300 && /path/to/venv/bin/python3 daily_update.py && git add cache/ && git commit -m "auto: $(date +%Y-%m-%d) 캐시 갱신" && git push
"""
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import FinanceDataReader as fdr


CACHE = Path("cache")
CACHE.mkdir(exist_ok=True)


def update_market_snapshot():
    """KRX 종목 마스터 + 시세 갱신."""
    print(f"[{datetime.now()}] KRX 종목 마스터 다운로드 중...")
    kospi = fdr.StockListing("KOSPI").assign(Market="KOSPI")
    kosdaq = fdr.StockListing("KOSDAQ").assign(Market="KOSDAQ")
    df = pd.concat([kospi, kosdaq], ignore_index=True)
    df = df.rename(columns={"ChagesRatio": "ChangeRatio", "Marcap": "MarketCap"})
    out = CACHE / "market_snapshot.parquet"
    df.to_parquet(out, index=False)
    print(f"  ✓ {out} ({len(df)}건)")

    meta = {
        "updated_at": datetime.now().isoformat(),
        "n_stocks": len(df),
        "kospi": len(kospi),
        "kosdaq": len(kosdaq),
    }
    with open(CACHE / "market_snapshot_meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  ✓ meta saved")
    return df


def update_ohlcv_cache(top_n: int = 500, end_date: Optional[str] = None,
                         start_date: str = "2020-01-01"):
    """시총 상위 N 종목의 OHLCV pkl 갱신."""
    snapshot_path = CACHE / "market_snapshot.parquet"
    if not snapshot_path.exists():
        print("  ⚠ market_snapshot.parquet 없음 — 먼저 update_market_snapshot 실행")
        return None

    df = pd.read_parquet(snapshot_path)
    df = df.sort_values("MarketCap", ascending=False).head(top_n)
    codes = df["Code"].tolist()
    print(f"\n[{datetime.now()}] 시총 상위 {top_n} OHLCV 갱신 중...")

    end_date = end_date or datetime.now().strftime("%Y-%m-%d")
    out_path = CACHE / f"ohlcv_{start_date}_{end_date}.pkl"

    # 기존 pkl 로드 후 증분 업데이트
    import pickle
    if out_path.exists():
        with open(out_path, "rb") as f:
            ohlcv_dict = pickle.load(f)
        print(f"  기존 {len(ohlcv_dict)}종목 로드")
    else:
        ohlcv_dict = {}

    # 신규/최신화
    new_count = 0
    updated_count = 0
    failed = []
    for i, code in enumerate(codes, 1):
        try:
            existing = ohlcv_dict.get(code)
            if existing is not None and not existing.empty:
                last_date = existing.index.max()
                next_date = last_date + pd.Timedelta(days=1)
                if next_date >= pd.Timestamp(end_date):
                    continue
                start = next_date.strftime("%Y-%m-%d")
            else:
                start = start_date
            new_data = fdr.DataReader(code, start, end_date)
            if new_data is None or new_data.empty:
                continue
            if existing is not None:
                merged = pd.concat([existing, new_data])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                ohlcv_dict[code] = merged
                updated_count += 1
            else:
                ohlcv_dict[code] = new_data
                new_count += 1
        except Exception as e:
            failed.append((code, str(e)[:80]))
        if i % 50 == 0:
            print(f"  진행: {i}/{len(codes)} (new {new_count} / updated {updated_count} / fail {len(failed)})")

    # 저장
    with open(out_path, "wb") as f:
        pickle.dump(ohlcv_dict, f)
    print(f"  ✓ {out_path} ({len(ohlcv_dict)}종목)")
    print(f"  새 종목: {new_count} · 업데이트: {updated_count} · 실패: {len(failed)}")
    return out_path


def main():
    print("=" * 60)
    print(f"KRX 데이터 자동 업데이트 시작 · {datetime.now()}")
    print("=" * 60)

    try:
        update_market_snapshot()
    except Exception as e:
        print(f"❌ market_snapshot 실패: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        update_ohlcv_cache(top_n=500)
    except Exception as e:
        print(f"⚠ ohlcv 캐시 갱신 실패 (계속): {e}")
        traceback.print_exc()

    print(f"\n[{datetime.now()}] 완료")


if __name__ == "__main__":
    main()
