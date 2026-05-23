"""
펀더멘털 수집기
==============

네이버 모바일 API에서 종목별:
- 분기 매출/영업이익/순이익/지배순이익/영업이익률/순이익률/ROE/EPS/BPS/PER/PBR
- 연간 동일
- 현재 시점 PER/PBR/EPS/BPS/시총/외인소진율

결과: cache/fundamentals_quarter.parquet, cache/fundamentals_annual.parquet, cache/fundamentals_current.parquet
"""

import json
import time
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

CACHE = Path("cache")
HEADERS = {"User-Agent": "Mozilla/5.0"}

MAX_WORKERS = 10
RETRY = 2


def fetch_finance(code: str, ft: str):
    """ft: 'annual' or 'quarter'. Returns list of dict rows."""
    for _ in range(RETRY):
        try:
            r = requests.get(
                f"https://m.stock.naver.com/api/stock/{code}/finance/{ft}",
                headers=HEADERS, timeout=8,
            )
            if r.status_code != 200:
                time.sleep(0.3); continue
            d = r.json()
            fi = d.get("financeInfo", {})
            rows = fi.get("rowList", [])
            results = []
            for row in rows:
                title = row.get("title", "").strip()
                cols = row.get("columns", {})
                if isinstance(cols, dict):
                    for period, v in cols.items():
                        val = v.get("value") if isinstance(v, dict) else v
                        if val is None or val in ("", "-"):
                            continue
                        results.append({
                            "Code": code,
                            "Item": title,
                            "Period": period,
                            "Value": val,
                            "Type": ft,
                        })
            return results
        except Exception:
            time.sleep(0.3)
    return []


def fetch_current(code: str):
    """현재 시점 PER/PBR/EPS/BPS/시총 등."""
    for _ in range(RETRY):
        try:
            r = requests.get(
                f"https://m.stock.naver.com/api/stock/{code}/integration",
                headers=HEADERS, timeout=8,
            )
            if r.status_code != 200:
                time.sleep(0.3); continue
            d = r.json()
            out = {"Code": code, "Name": d.get("stockName")}
            for ti in d.get("totalInfos", []):
                k = ti.get("key")
                v = ti.get("value")
                out[k] = v
            return out
        except Exception:
            time.sleep(0.3)
    return None


def parse_num(s):
    if s is None:
        return None
    s = str(s).replace(",", "").replace("배", "").replace("원", "").replace("%", "").strip()
    if "조" in s and "억" in s:
        # 시총
        parts = s.split("조")
        try:
            jo = float(parts[0])
            rest = parts[1].split("억")[0]
            uk = float(rest) if rest else 0
            return jo * 1e12 + uk * 1e8
        except Exception:
            return None
    if "조" in s:
        try: return float(s.replace("조", "")) * 1e12
        except: return None
    if "억" in s:
        try: return float(s.replace("억", "")) * 1e8
        except: return None
    if "백만" in s:
        try: return float(s.replace("백만", "")) * 1e6
        except: return None
    try:
        return float(s)
    except:
        return None


def main():
    with open(CACHE / "_signal_codes.json") as f:
        codes = json.load(f)
    print(f"[펀더수집] {len(codes)}종목")

    # 현재 시점
    print("\n[1/3] 현재 시점 PER/PBR/시총...")
    t0 = time.time()
    current_rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_current, c): c for c in codes}
        for i, fut in enumerate(as_completed(futures)):
            r = fut.result()
            if r:
                current_rows.append(r)
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(codes)}] elapsed={time.time()-t0:.0f}s")
    cur_df = pd.DataFrame(current_rows)
    # 숫자화
    for c in ["PER", "EPS", "PBR", "BPS", "추정PER", "추정EPS",
              "배당수익률", "주당배당금", "외인소진율", "시총", "52주 최고", "52주 최저"]:
        if c in cur_df.columns:
            cur_df[c + "_num"] = cur_df[c].apply(parse_num)
    cur_df.to_parquet(CACHE / "fundamentals_current.parquet", index=False)
    print(f"  ✓ current: {len(cur_df)}종목, {time.time()-t0:.0f}s")

    # 연간
    print("\n[2/3] 연간 재무...")
    t0 = time.time()
    annual_rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_finance, c, "annual"): c for c in codes}
        for i, fut in enumerate(as_completed(futures)):
            rows = fut.result()
            annual_rows.extend(rows)
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(codes)}] elapsed={time.time()-t0:.0f}s")
    ann_df = pd.DataFrame(annual_rows)
    if not ann_df.empty:
        ann_df["ValueNum"] = ann_df["Value"].apply(parse_num)
    ann_df.to_parquet(CACHE / "fundamentals_annual.parquet", index=False)
    print(f"  ✓ annual: {len(ann_df):,}건, {time.time()-t0:.0f}s")

    # 분기
    print("\n[3/3] 분기 재무...")
    t0 = time.time()
    q_rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_finance, c, "quarter"): c for c in codes}
        for i, fut in enumerate(as_completed(futures)):
            rows = fut.result()
            q_rows.extend(rows)
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(codes)}] elapsed={time.time()-t0:.0f}s")
    q_df = pd.DataFrame(q_rows)
    if not q_df.empty:
        q_df["ValueNum"] = q_df["Value"].apply(parse_num)
    q_df.to_parquet(CACHE / "fundamentals_quarter.parquet", index=False)
    print(f"  ✓ quarter: {len(q_df):,}건, {time.time()-t0:.0f}s")

    print("\n[완료]")


if __name__ == "__main__":
    main()
