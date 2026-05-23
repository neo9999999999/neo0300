"""
OOS 검증 + 전체 종목 리스트 출력
==============================

1. Train(2020-04 ~ 2023-12)으로 회피 보강 룰 + 강화 룰 발굴
2. Test(2024-01 ~ 2025-08)에서 검증
3. 슈퍼위너/100%+ 농도 변화 측정
4. 주1건/일1건 시뮬레이션
5. 전체 종목 리스트 (년/월/일별) 출력

산출:
- cache/oos_results.csv          - 시뮬레이션 결과 요약
- cache/final_candidates.csv     - 최종 통과 전체 종목 리스트 (대표만 X)
- cache/year_month_returns.csv   - 년/월별 수익률
- FINAL_GUIDE.md                 - 최종 가이드
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

CACHE = Path("cache")

# enriched 로드 (analyze_super_winner.py 결과)
ENRICHED = CACHE / "candidates_enriched.parquet"
if not ENRICHED.exists():
    raise SystemExit(f"먼저 analyze_super_winner.py 실행 필요 → {ENRICHED}")

cand = pd.read_parquet(ENRICHED)
cand["Date"] = pd.to_datetime(cand["Date"])
print(f"[로드] {len(cand):,}건")

# Train / Test
train = cand[cand["Date"] < "2024-01-01"].copy()
test = cand[cand["Date"] >= "2024-01-01"].copy()
print(f"  Train: {len(train):,}건 (2020-04~2023-12)")
print(f"  Test:  {len(test):,}건 (2024-01~)")


# -------- 회피 보강 룰 정의 (Train에서 발굴된 것 적용) --------
# analyze_super_winner.py 결과 보고 수동 선정 → 일단 후보 모두 적용해보고 best 선택
def apply_extra_avoid(df, version="v2"):
    """v2: 부분 분석 결과 - SW 손실 적은 룰만 채택 (전수 분석 기반)
       v3: v2 + 강화 포함 룰 (P9 외인+기관 동시 순매수만 남김)
    """
    d = df.copy()
    if version == "v2":
        # 부분 분석 결과 채택된 룰만 (SW 손실 적은 것):
        # - X7 외국인 20일 누적 매수 하위 10% (제외 192건, 제외 SW만 3.13%)
        # - X13 강하락추세+52주高-50%↓ (제외 36건, SW 2.78%)
        mask = pd.Series(False, index=d.index)
        if "For_20d" in d.columns and d["For_20d"].notna().sum() > 100:
            mask |= d["For_20d"] < d["For_20d"].quantile(0.10)
        mask |= (d["slope60"] <= -2) & (d["pos_252_high"] <= -50)
        return d[~mask].copy()
    elif version == "v3":
        # v3: v2 + 포함 강화 룰 P9 (외인+기관 동시 순매수) — 100%+, 50%+ 농도 ↑
        d2 = apply_extra_avoid(d, "v2")
        if "For_20d" in d2.columns and d2["For_20d"].notna().sum() > 100:
            d2 = d2[(d2["For_20d"] > 0) & (d2["Inst_20d"] > 0)].copy()
        return d2
    elif version == "v4":
        # v4: v2 + P2 (slope60 ≥ 0.5 상승추세) — SW 농도 ↑
        d2 = apply_extra_avoid(d, "v2")
        d2 = d2[d2["slope60"] >= 0.5].copy()
        return d2
    return d


def stats(df, label):
    if len(df) == 0:
        return {}
    return {
        "label": label, "n": len(df),
        "SW(200%+)": (df["peak_180d"] >= 200).sum(),
        "SW%": (df["peak_180d"] >= 200).mean() * 100,
        "100%+": (df["peak_180d"] >= 100).sum(),
        "100%+_%": (df["peak_180d"] >= 100).mean() * 100,
        "50%+": (df["peak_180d"] >= 50).sum(),
        "50%+_%": (df["peak_180d"] >= 50).mean() * 100,
        "평균peak": df["peak_180d"].mean(),
        "평균ret180": df["ret_180d"].mean(),
        "승률": (df["ret_180d"] > 0).mean() * 100,
    }


# -------- 농도 비교 --------
print("\n" + "=" * 100)
print("회피 보강 효과 - 풀의 슈퍼위너 농도 변화")
print("=" * 100)

cmp_rows = []
for label, df_train, df_test in [("전체", cand, None)]:
    pass

# Train
cmp_rows.append({"set": "Train", **stats(train, "회피6")})
cmp_rows.append({"set": "Train", **stats(apply_extra_avoid(train, "v2"), "회피6+v2")})
cmp_rows.append({"set": "Train", **stats(apply_extra_avoid(train, "v3"), "회피6+v3_P9")})
cmp_rows.append({"set": "Train", **stats(apply_extra_avoid(train, "v4"), "회피6+v4_slope")})

# Test (OOS)
cmp_rows.append({"set": "Test_OOS", **stats(test, "회피6")})
cmp_rows.append({"set": "Test_OOS", **stats(apply_extra_avoid(test, "v2"), "회피6+v2")})
cmp_rows.append({"set": "Test_OOS", **stats(apply_extra_avoid(test, "v3"), "회피6+v3_P9")})
cmp_rows.append({"set": "Test_OOS", **stats(apply_extra_avoid(test, "v4"), "회피6+v4_slope")})

cmp_df = pd.DataFrame(cmp_rows)
print(cmp_df.to_string(index=False))
cmp_df.to_csv(CACHE / "avoid_v2v3_comparison.csv", index=False)


# -------- 시뮬레이션: 주1건 / 일1건 / 풀가동 --------
def simulate(df, mode, capital0=10_000_000, alloc_pct=0.10, sort_by="Amount asc"):
    """
    mode: 'weekly_1', 'daily_1', 'unlimited', 'slots_N'
    sort_by: 'Score desc/asc', 'Amount asc/desc', 'Code'
    """
    df = df.dropna(subset=["sell_close"]).sort_values("Date").copy()
    if mode == "weekly_1":
        df["bucket"] = df["Date"].dt.strftime("%Y-%U")
    elif mode == "daily_1":
        df["bucket"] = df["Date"].dt.strftime("%Y-%m-%d")
    else:
        df["bucket"] = None

    cash = capital0
    pending_sells = []
    trades = []
    skipped = 0

    if mode in ("weekly_1", "daily_1"):
        for b, g in df.groupby("bucket"):
            if sort_by == "Score asc":
                pick = g.sort_values("Score").iloc[0]
            elif sort_by == "Amount asc":
                pick = g.sort_values("Amount").iloc[0]
            elif sort_by == "Score desc":
                pick = g.sort_values("Score", ascending=False).iloc[0]
            else:
                pick = g.sort_values("Code").iloc[0]
            dt = pick["Date"]
            # 매도 처리
            still = []
            for sd, val in pending_sells:
                if sd <= dt: cash += val
                else: still.append((sd, val))
            pending_sells = still
            alloc = capital0 * alloc_pct
            if cash < alloc:
                skipped += 1; continue
            qty = alloc / pick["Close"]
            cash -= alloc
            pending_sells.append((pick["sell_date"], qty * pick["sell_close"]))
            trades.append({
                "buy_date": pick["Date"], "code": pick["Code"], "name": pick["Name"],
                "buy_price": pick["Close"], "alloc": alloc,
                "sell_date": pick["sell_date"], "sell_close": pick["sell_close"],
                "ret_180d": pick["ret_180d"], "peak_180d": pick["peak_180d"],
            })
    else:
        # unlimited or slots_N
        slot_n = None
        if mode.startswith("slots_"):
            slot_n = int(mode.split("_")[1])
        for _, r in df.iterrows():
            dt = r["Date"]
            still = []
            for sd, val in pending_sells:
                if sd <= dt: cash += val
                else: still.append((sd, val))
            pending_sells = still
            if slot_n is not None and len(pending_sells) >= slot_n:
                skipped += 1; continue
            alloc = capital0 * alloc_pct
            if cash < alloc:
                skipped += 1; continue
            qty = alloc / r["Close"]
            cash -= alloc
            pending_sells.append((r["sell_date"], qty * r["sell_close"]))
            trades.append({
                "buy_date": r["Date"], "code": r["Code"], "name": r["Name"],
                "buy_price": r["Close"], "alloc": alloc,
                "sell_date": r["sell_date"], "sell_close": r["sell_close"],
                "ret_180d": r["ret_180d"], "peak_180d": r["peak_180d"],
            })
    for sd, val in pending_sells:
        cash += val
    t = pd.DataFrame(trades)
    if len(t) == 0:
        return {"n": 0}, t
    pnl = (t["sell_close"] / t["buy_price"] - 1) * t["alloc"]
    final = capital0 + pnl.sum()
    return {
        "n": len(t), "skip": skipped,
        "SW_n": int((t["peak_180d"] >= 200).sum()),
        "SW_rate": (t["peak_180d"] >= 200).mean() * 100,
        "100+_n": int((t["peak_180d"] >= 100).sum()),
        "100+_rate": (t["peak_180d"] >= 100).mean() * 100,
        "50+_n": int((t["peak_180d"] >= 50).sum()),
        "50+_rate": (t["peak_180d"] >= 50).mean() * 100,
        "avg_peak": t["peak_180d"].mean(),
        "avg_ret": t["ret_180d"].mean(),
        "winrate": (t["ret_180d"] > 0).mean() * 100,
        "final": final,
        "return_pct": (final / capital0 - 1) * 100,
    }, t


# Test 기간 시뮬레이션
print("\n" + "=" * 100)
print("OOS Test (2024-01~) 시뮬레이션 — 회피 보강 효과 검증")
print("=" * 100)

results = []
all_trades = {}
for filter_label, filter_fn in [
    ("회피6", lambda d: d),
    ("회피6+v2", lambda d: apply_extra_avoid(d, "v2")),
    ("회피6+v3_P9", lambda d: apply_extra_avoid(d, "v3")),
    ("회피6+v4_slope", lambda d: apply_extra_avoid(d, "v4")),
]:
    pool = filter_fn(test)
    for mode in ["weekly_1", "daily_1", "slots_5", "slots_10"]:
        r, t = simulate(pool, mode, sort_by="Amount asc")
        if r.get("n", 0) == 0:
            continue
        r["filter"] = filter_label
        r["mode"] = mode
        results.append(r)
        all_trades[(filter_label, mode)] = t

res_df = pd.DataFrame(results)
cols = ["filter", "mode", "n", "skip", "SW_n", "SW_rate", "100+_n", "100+_rate",
        "50+_n", "50+_rate", "avg_peak", "avg_ret", "winrate", "final", "return_pct"]
res_df = res_df[cols]
print(res_df.to_string(index=False))
res_df.to_csv(CACHE / "oos_simulation_results.csv", index=False)

# 전체 기간 시뮬레이션 (Train+Test)
print("\n" + "=" * 100)
print("전체기간 (2020-04~) 시뮬레이션")
print("=" * 100)
full_results = []
for filter_label, filter_fn in [
    ("회피6", lambda d: d),
    ("회피6+v2", lambda d: apply_extra_avoid(d, "v2")),
    ("회피6+v3_P9", lambda d: apply_extra_avoid(d, "v3")),
    ("회피6+v4_slope", lambda d: apply_extra_avoid(d, "v4")),
]:
    pool = filter_fn(cand)
    for mode in ["weekly_1", "daily_1", "slots_5", "slots_10", "slots_20"]:
        r, t = simulate(pool, mode, sort_by="Amount asc")
        if r.get("n", 0) == 0:
            continue
        r["filter"] = filter_label
        r["mode"] = mode
        full_results.append(r)

full_df = pd.DataFrame(full_results)
full_df = full_df[cols]
print(full_df.to_string(index=False))
full_df.to_csv(CACHE / "full_simulation_results.csv", index=False)


# -------- 최종 종목 리스트 (전체) --------
print("\n" + "=" * 100)
print("최종 종목 리스트 출력")
print("=" * 100)

# 회피6+v2 적용한 전체 풀
final_pool = apply_extra_avoid(cand, "v2").sort_values("Date")
# 출력 컬럼 정리
out_cols = ["Date", "Code", "Name", "Market", "Close", "Score", "Amount",
            "chart_pattern", "past_60", "past_120", "past_240",
            "pos_252_high", "slope60", "drawdown60",
            "peak_180d", "ret_180d", "sell_date", "sell_close"]
extra = [c for c in ["For_5d", "Inst_5d", "For_20d", "Inst_20d",
                     "PER_num", "PBR_num", "시총_num"] if c in final_pool.columns]
out_cols.extend(extra)
out_cols = [c for c in out_cols if c in final_pool.columns]
final_pool_out = final_pool[out_cols].copy()
# 년/월 추가
final_pool_out["Year"] = final_pool_out["Date"].dt.year
final_pool_out["Month"] = final_pool_out["Date"].dt.month
final_pool_out["YYYYMM"] = final_pool_out["Date"].dt.strftime("%Y-%m")
final_pool_out.to_csv(CACHE / "final_candidates_all.csv", index=False)
print(f"  → cache/final_candidates_all.csv ({len(final_pool_out):,}건)")


# -------- 년/월별 수익률 (주1건 거래대금↓) --------
_, best_trades = simulate(apply_extra_avoid(cand, "v2"), "weekly_1", sort_by="Amount asc")
best_trades["YYYYMM"] = pd.to_datetime(best_trades["buy_date"]).dt.strftime("%Y-%m")
best_trades["return_pct"] = (best_trades["sell_close"] / best_trades["buy_price"] - 1) * 100
month_stat = best_trades.groupby("YYYYMM").agg(
    n=("code", "count"),
    avg_ret=("ret_180d", "mean"),
    avg_peak=("peak_180d", "mean"),
    SW_n=("peak_180d", lambda x: (x >= 200).sum()),
    w100_n=("peak_180d", lambda x: (x >= 100).sum()),
    w50_n=("peak_180d", lambda x: (x >= 50).sum()),
).reset_index()
month_stat.to_csv(CACHE / "year_month_returns.csv", index=False)
print(f"  → cache/year_month_returns.csv ({len(month_stat)}개월)")
print()
print(month_stat.to_string(index=False))

best_trades.to_csv(CACHE / "best_trades_weekly_v2.csv", index=False)
print(f"\n  → cache/best_trades_weekly_v2.csv ({len(best_trades):,}건)")

print("\n[완료]")
