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


def update_fundamentals_current():
    """현재 시점 PER/PBR/시총 갱신 (네이버 모바일 API)."""
    from collect_fundamentals import fetch_current, parse_num
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import json as _json

    cd = CACHE / "_signal_codes.json"
    if not cd.exists():
        print("  ⚠ _signal_codes.json 없음 — skip")
        return
    with open(cd) as f:
        codes = _json.load(f)
    print(f"\n[{datetime.now()}] 펀더멘털(현재) {len(codes)}종목 갱신 중...")
    rows = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(fetch_current, c): c for c in codes}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                rows.append(r)
    df = pd.DataFrame(rows)
    for c in ["PER", "EPS", "PBR", "BPS", "추정PER", "추정EPS",
              "배당수익률", "주당배당금", "외인소진율", "시총", "52주 최고", "52주 최저"]:
        if c in df.columns:
            df[c + "_num"] = df[c].apply(parse_num)
    df.to_parquet(CACHE / "fundamentals_current.parquet", index=False)
    print(f"  ✓ fundamentals_current.parquet ({len(df)}종목)")


def update_supply_demand_incremental():
    """수급 데이터 증분 갱신 (최근 1페이지만)."""
    from collect_supply_demand import fetch_code
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import json as _json

    cd = CACHE / "_signal_codes.json"
    sd_path = CACHE / "supply_demand.parquet"
    if not cd.exists():
        print("  ⚠ _signal_codes.json 없음 — skip")
        return
    with open(cd) as f:
        codes = _json.load(f)

    print(f"\n[{datetime.now()}] 수급 증분 갱신 {len(codes)}종목 (최근 1페이지=20거래일)...")
    existing = pd.read_parquet(sd_path) if sd_path.exists() else pd.DataFrame()

    new_rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_code, c, 1): c for c in codes}
        for fut in as_completed(futs):
            df_c = fut.result()
            if not df_c.empty:
                new_rows.append(df_c)
    if new_rows:
        merged = pd.concat([existing] + new_rows, ignore_index=True)
        merged = merged.drop_duplicates(subset=["Code", "Date"])
        merged.to_parquet(sd_path, index=False)
        print(f"  ✓ supply_demand.parquet ({len(merged):,}건)")


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

    # B+C: 펀더멘털/수급 증분 갱신
    try:
        update_fundamentals_current()
    except Exception as e:
        print(f"⚠ 펀더멘털 갱신 실패 (계속): {e}")
        traceback.print_exc()

    try:
        update_supply_demand_incremental()
    except Exception as e:
        print(f"⚠ 수급 갱신 실패 (계속): {e}")
        traceback.print_exc()

    # 오늘의 추천 빌드
    try:
        from live_filter import build_today_picks
        build_today_picks(top_n=20)
    except Exception as e:
        print(f"⚠ 추천 빌드 실패 (계속): {e}")
        traceback.print_exc()

    print(f"\n[{datetime.now()}] 완료")


if __name__ == "__main__":
    main()
