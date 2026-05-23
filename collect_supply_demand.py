"""
수급 데이터 (외국인/기관 순매수) 수집기
=====================================

네이버 금융 frgn.naver 페이지에서 일별 외국인/기관 순매수량 추출.
- 종목당 페이지 1~80 (~ 1600거래일 = 2019~2026)
- 동시 8개 병렬
- 결과: cache/supply_demand.parquet  (Date, Code, Foreign_NetBuy, Inst_NetBuy, IndividualNet, ...)
"""

import json
import time
import requests
import pandas as pd
from io import StringIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib3.exceptions import InsecureRequestWarning
import urllib3
urllib3.disable_warnings(InsecureRequestWarning)


CACHE = Path("cache")
OUT_PATH = CACHE / "supply_demand.parquet"
CHECKPOINT = CACHE / "_sd_checkpoint.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://finance.naver.com/",
}

PAGES_PER_CODE = 80   # 페이지 1~80 (약 1600거래일)
MAX_WORKERS = 8       # 동시 8개
SLEEP_BETWEEN = 0.05  # 호출간 약간의 sleep
RETRY = 2


def fetch_page(code: str, page: int) -> pd.DataFrame:
    url = f"https://finance.naver.com/item/frgn.naver?code={code}&page={page}"
    for attempt in range(RETRY):
        try:
            r = requests.get(url, headers=HEADERS, timeout=8)
            r.raise_for_status()
            tables = pd.read_html(StringIO(r.text))
            if len(tables) < 4:
                return pd.DataFrame()
            t = tables[3]
            # MultiIndex 컬럼 평탄화
            t.columns = [c[1] if isinstance(c, tuple) else c for c in t.columns]
            t = t.dropna(how="all")
            # 필요 컬럼: 날짜, 종가, 전일비, 등락률, 거래량, 기관 순매매량, 외국인 순매매량
            if "날짜" not in t.columns:
                return pd.DataFrame()
            return t
        except Exception:
            if attempt < RETRY - 1:
                time.sleep(0.5)
            else:
                return pd.DataFrame()
    return pd.DataFrame()


def fetch_code(code: str, pages: int = PAGES_PER_CODE) -> pd.DataFrame:
    rows = []
    for p in range(1, pages + 1):
        t = fetch_page(code, p)
        if t.empty:
            # 빈 페이지 만나면 그 종목 끝
            if p > 5:
                break
            continue
        # 컬럼 정리
        cols_map = {}
        for c in t.columns:
            cs = str(c).strip()
            cols_map[c] = cs
        t = t.rename(columns=cols_map)
        # 표준화
        keep = {}
        for orig, std in [("날짜", "Date"), ("종가", "Close"),
                          ("등락률", "ChangeRatio"),
                          ("거래량", "Volume"),
                          ("기관", "Inst_NetBuy"),
                          ("외국인", "Foreign_NetBuy")]:
            if orig in t.columns:
                keep[orig] = std
        if "날짜" not in t.columns:
            continue
        # 기관/외국인 컬럼이 같은 이름으로 중복될 수 있음 — 첫번째 (순매매량) 사용
        # 실제: 컬럼이 [날짜, 종가, 전일비, 등락률, 거래량, 기관(순매매량), 외국인(순매매량), 보유주수, 보유율]
        # MultiIndex 평탄화시 같은 단어가 중복될 수 있음
        # 수동 매핑:
        cols = list(t.columns)
        # 위치 기반 매핑
        # 보통: 0=날짜 1=종가 2=전일비 3=등락률 4=거래량 5=기관 6=외국인 7=보유주수 8=보유율
        if len(cols) >= 7:
            t2 = pd.DataFrame()
            t2["Date"] = t.iloc[:, 0]
            t2["Close"] = t.iloc[:, 1]
            t2["Inst_NetBuy"] = t.iloc[:, 5]
            t2["Foreign_NetBuy"] = t.iloc[:, 6]
            t2["Code"] = code
            rows.append(t2)
        time.sleep(SLEEP_BETWEEN)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    # 정제
    df = df.dropna(subset=["Date", "Close"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    # 숫자 변환
    for c in ["Close", "Inst_NetBuy", "Foreign_NetBuy"]:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", "").str.replace(" ", "").str.replace("--", "0"),
            errors="coerce")
    df = df.drop_duplicates(subset=["Code", "Date"])
    return df


def load_checkpoint():
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return set(json.load(f).get("done", []))
    return set()


def save_checkpoint(done):
    with open(CHECKPOINT, "w") as f:
        json.dump({"done": list(done)}, f)


def main():
    # 시그널 종목 로드
    with open(CACHE / "_signal_codes.json") as f:
        codes = json.load(f)
    print(f"[수급수집] 총 {len(codes)}종목")

    done = load_checkpoint()
    print(f"  체크포인트: {len(done)}/{len(codes)} 완료")

    # 기존 결과 로드
    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        print(f"  기존 결과: {len(existing):,}건")
    else:
        existing = pd.DataFrame()

    remaining = [c for c in codes if c not in done]
    print(f"  남은 종목: {len(remaining)}")

    new_rows = []
    cnt = 0
    failed = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_code, c): c for c in remaining}
        for fut in as_completed(futures):
            c = futures[fut]
            try:
                df = fut.result()
                if not df.empty:
                    new_rows.append(df)
                    done.add(c)
                    cnt += 1
                    if cnt % 20 == 0:
                        elapsed = time.time() - t0
                        rate = cnt / elapsed if elapsed > 0 else 0
                        eta = (len(remaining) - cnt) / rate if rate > 0 else 0
                        print(f"  [{cnt}/{len(remaining)}] {c} rows={len(df)} rate={rate:.1f}/s ETA={eta/60:.1f}min")
                        # 중간 저장
                        if new_rows:
                            partial = pd.concat([existing] + new_rows, ignore_index=True)
                            partial = partial.drop_duplicates(subset=["Code", "Date"])
                            partial.to_parquet(OUT_PATH, index=False)
                            save_checkpoint(done)
                else:
                    failed.append(c)
                    done.add(c)
            except Exception as e:
                failed.append(c)
                print(f"  ERR {c}: {e}")

    # 최종 저장
    if new_rows:
        all_df = pd.concat([existing] + new_rows, ignore_index=True)
        all_df = all_df.drop_duplicates(subset=["Code", "Date"])
        all_df.to_parquet(OUT_PATH, index=False)
        save_checkpoint(done)
        print(f"\n[완료] 총 {len(all_df):,}건 → {OUT_PATH}")
    print(f"[실패] {len(failed)}종목: {failed[:10]}")
    print(f"[총시간] {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
