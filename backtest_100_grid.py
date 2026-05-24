"""
100가지 백테스트 그리드 서치
==========================

필터 10가지 × 정렬 10가지 = 100조합
× 2모드 (매일 1건, 주 2건) = 200 시뮬

각 모드별 Top3 + 전체 매수 종목 리스트 출력
"""

import pandas as pd
import numpy as np
from pathlib import Path

CACHE = Path("cache")

# enriched 로드 (analyze_super_winner.py 결과)
cand = pd.read_parquet(CACHE / "candidates_enriched.parquet")
cand["Date"] = pd.to_datetime(cand["Date"])
cand = cand.dropna(subset=["sell_close", "peak_180d"]).copy()
print(f"[로드] {len(cand):,}건")


# ============== 필터 10가지 ==============
def filter_pool(df, fid):
    d = df.copy()
    if fid == "F01_회피6_기본":
        return d
    elif fid == "F02_slope_05":
        return d[d["slope60"] >= 0.5]
    elif fid == "F03_slope_10":
        return d[d["slope60"] >= 1.0]
    elif fid == "F04_외인상위25":
        if "For_20d" in d.columns:
            q = d["For_20d"].quantile(0.75)
            return d[d["For_20d"] > q]
        return d
    elif fid == "F05_slope05_For양":
        d = d[d["slope60"] >= 0.5]
        if "For_20d" in d.columns:
            d = d[d["For_20d"] > 0]
        return d
    elif fid == "F06_외인기관동시매수":
        if "For_20d" in d.columns and "Inst_20d" in d.columns:
            return d[(d["For_20d"] > 0) & (d["Inst_20d"] > 0)]
        return d
    elif fid == "F07_눌림목30_10":
        return d[(d["pos_252_high"] >= -30) & (d["pos_252_high"] <= -10)]
    elif fid == "F08_안정추세60":
        return d[(d["past_60"] >= -10) & (d["past_60"] <= 30)]
    elif fid == "F09_slope05_과열X":
        d = d[d["slope60"] >= 0.5]
        return d[(d["past_120"] >= -20) & (d["past_120"] <= 60)]
    elif fid == "F10_KOSDAQ만":
        return d[d["Market"] == "KOSDAQ"]
    return d


FILTERS = [
    "F01_회피6_기본", "F02_slope_05", "F03_slope_10", "F04_외인상위25",
    "F05_slope05_For양", "F06_외인기관동시매수", "F07_눌림목30_10",
    "F08_안정추세60", "F09_slope05_과열X", "F10_KOSDAQ만",
]

# ============== 정렬 10가지 ==============
def sort_pool(df, sid):
    if sid == "S01_거래대금낮":
        return df.sort_values("Amount", ascending=True)
    elif sid == "S02_거래대금높":
        return df.sort_values("Amount", ascending=False)
    elif sid == "S03_점수낮":
        return df.sort_values("Score", ascending=True)
    elif sid == "S04_점수높":
        return df.sort_values("Score", ascending=False)
    elif sid == "S05_변동성낮":
        return df.sort_values("vol60", ascending=True)
    elif sid == "S06_변동성높":
        return df.sort_values("vol60", ascending=False)
    elif sid == "S07_slope강한":
        return df.sort_values("slope60", ascending=False)
    elif sid == "S08_외인매수강":
        if "For_20d" in df.columns:
            return df.sort_values("For_20d", ascending=False, na_position="last")
        return df.sort_values("Code")
    elif sid == "S09_52주고점근접":
        return df.sort_values("pos_252_high", ascending=False)
    elif sid == "S10_랜덤":
        return df.sample(frac=1, random_state=42)
    return df


SORTS = [
    "S01_거래대금낮", "S02_거래대금높", "S03_점수낮", "S04_점수높",
    "S05_변동성낮", "S06_변동성높", "S07_slope강한", "S08_외인매수강",
    "S09_52주고점근접", "S10_랜덤",
]


# ============== 시뮬레이터 ==============
def simulate(df, mode, capital0=10_000_000, alloc_pct=0.10):
    """
    mode:
      'daily_1': 매일 가장 좋은 시그널 1건
      'weekly_2': 주 2건
      'weekly_3': 주 3건
    """
    df = df.copy()
    if mode == "daily_1":
        df["bucket"] = df["Date"].dt.strftime("%Y-%m-%d")
        n_per_bucket = 1
    elif mode == "weekly_2":
        df["bucket"] = df["Date"].dt.strftime("%Y-%U")
        n_per_bucket = 2
    elif mode == "weekly_3":
        df["bucket"] = df["Date"].dt.strftime("%Y-%U")
        n_per_bucket = 3
    else:
        raise ValueError(mode)

    cash = capital0
    pending_sells = []  # (sell_date, value)
    trades = []
    skipped = 0

    # bucket 별로 정렬은 이미 호출자가 한 상태
    for b, g in df.groupby("bucket", sort=True):
        picks = g.head(n_per_bucket)
        dt = picks["Date"].iloc[0]
        # 매도 처리
        still = []
        for sd, val in pending_sells:
            if sd <= dt:
                cash += val
            else:
                still.append((sd, val))
        pending_sells = still
        for _, p in picks.iterrows():
            alloc = capital0 * alloc_pct
            if cash < alloc:
                skipped += 1
                continue
            qty = alloc / p["Close"]
            cash -= alloc
            pending_sells.append((p["sell_date"], qty * p["sell_close"]))
            trades.append({
                "buy_date": p["Date"], "code": p["Code"], "name": p["Name"],
                "Market": p["Market"],
                "buy_price": p["Close"], "alloc": alloc,
                "sell_date": p["sell_date"], "sell_close": p["sell_close"],
                "ret_180d": p["ret_180d"], "peak_180d": p["peak_180d"],
            })
    for sd, val in pending_sells:
        cash += val

    t = pd.DataFrame(trades)
    if len(t) == 0:
        return {"n": 0}, t

    pnl = (t["sell_close"] / t["buy_price"] - 1) * t["alloc"]
    final = capital0 + pnl.sum()
    return {
        "n": len(t),
        "skip": skipped,
        "SW_n": int((t["peak_180d"] >= 200).sum()),
        "SW_rate": (t["peak_180d"] >= 200).mean() * 100,
        "w100_n": int((t["peak_180d"] >= 100).sum()),
        "w100_rate": (t["peak_180d"] >= 100).mean() * 100,
        "w50_n": int((t["peak_180d"] >= 50).sum()),
        "w50_rate": (t["peak_180d"] >= 50).mean() * 100,
        "avg_peak": t["peak_180d"].mean(),
        "avg_ret": t["ret_180d"].mean(),
        "winrate": (t["ret_180d"] > 0).mean() * 100,
        "final": final,
        "ret_pct": (final / capital0 - 1) * 100,
        "profit": final - capital0,
    }, t


# ============== 그리드 서치 ==============
def run_grid(cand, mode, label_for_print):
    print(f"\n{'='*100}")
    print(f"GRID: 필터 10 × 정렬 10 = 100조합 · 모드={mode} ({label_for_print})")
    print('='*100)

    rows = []
    all_trades = {}
    total = len(FILTERS) * len(SORTS)
    i = 0
    for fid in FILTERS:
        pool_f = filter_pool(cand, fid)
        if len(pool_f) == 0:
            continue
        for sid in SORTS:
            i += 1
            pool_fs = sort_pool(pool_f, sid)
            r, t = simulate(pool_fs, mode)
            if r.get("n", 0) == 0:
                continue
            r["filter"] = fid
            r["sort"] = sid
            r["mode"] = mode
            r["combo"] = f"{fid}__{sid}"
            rows.append(r)
            all_trades[r["combo"]] = t
            if i % 20 == 0:
                print(f"  [{i}/{total}] {fid}/{sid} → n={r['n']} ret={r['ret_pct']:.1f}%")
    return rows, all_trades


# OOS Test 기간 (2024-01~)
test = cand[cand["Date"] >= "2024-01-01"].copy()
full = cand.copy()

# ========= OOS 기간 백테스트 (1000만 자본 → 2024-2026 결과) =========
print("\n\n############### OOS TEST 2024-2026 ###############")
oos_d, oos_d_trades = run_grid(test, "daily_1", "매일 1건")
oos_w2, oos_w2_trades = run_grid(test, "weekly_2", "주 2건")
oos_w3, oos_w3_trades = run_grid(test, "weekly_3", "주 3건")

# ========= 전체기간 (2020-04~) =========
print("\n\n############### FULL 2020-2026 ###############")
full_d, full_d_trades = run_grid(full, "daily_1", "매일 1건")
full_w2, full_w2_trades = run_grid(full, "weekly_2", "주 2건")
full_w3, full_w3_trades = run_grid(full, "weekly_3", "주 3건")


# ========= Top3 추출 =========
def show_top3(rows, label, all_trades):
    df = pd.DataFrame(rows)
    if len(df) == 0:
        print(f"[{label}] no rows")
        return df, []
    # 수익률 기준
    top_ret = df.sort_values("ret_pct", ascending=False).head(3)
    # 슈퍼위너 기준
    top_sw = df.sort_values("SW_rate", ascending=False).head(3)
    # 100%+ 기준
    top_100 = df.sort_values("w100_rate", ascending=False).head(3)

    print(f"\n=========== TOP3 - {label} (수익률 기준) ===========")
    cols = ["filter", "sort", "n", "skip", "SW_n", "SW_rate",
            "w100_n", "w100_rate", "w50_n", "w50_rate",
            "avg_peak", "avg_ret", "winrate", "final", "ret_pct", "profit"]
    print(top_ret[cols].to_string(index=False))

    print(f"\n=========== TOP3 - {label} (슈퍼위너 비율) ===========")
    print(top_sw[cols].to_string(index=False))

    print(f"\n=========== TOP3 - {label} (100%+ 비율) ===========")
    print(top_100[cols].to_string(index=False))

    top_combos = list(top_ret["combo"]) + list(top_sw["combo"]) + list(top_100["combo"])
    return df, list(set(top_combos))


# 출력 + 저장
all_summary = []

dfs = [
    (oos_d,  "OOS_daily_1",  oos_d_trades),
    (oos_w2, "OOS_weekly_2", oos_w2_trades),
    (oos_w3, "OOS_weekly_3", oos_w3_trades),
    (full_d, "FULL_daily_1", full_d_trades),
    (full_w2, "FULL_weekly_2", full_w2_trades),
    (full_w3, "FULL_weekly_3", full_w3_trades),
]

top_combos_save = {}
for rows, label, trades in dfs:
    df, top_combos = show_top3(rows, label, trades)
    df["set"] = label
    all_summary.append(df)
    top_combos_save[label] = top_combos
    # Top3 trades 저장
    df_ret = df.sort_values("ret_pct", ascending=False).head(3)
    for rank, (_, r) in enumerate(df_ret.iterrows(), 1):
        combo = r["combo"]
        t = trades[combo]
        out_path = CACHE / f"grid_{label}_TOP{rank}_{combo}_trades.csv"
        t.to_csv(out_path, index=False)

summary = pd.concat(all_summary, ignore_index=True)
summary.to_csv(CACHE / "grid_100_summary.csv", index=False)
print(f"\n[저장] cache/grid_100_summary.csv ({len(summary)}조합 결과)")

# Top combos 저장
import json
with open(CACHE / "grid_100_top_combos.json", "w", encoding="utf-8") as f:
    json.dump(top_combos_save, f, ensure_ascii=False, indent=2)


# ============ 최고 조합 전체 풀 리스트 (2020-2026) ============
print("\n\n=========== 최고 추천방식 (OOS 기준) 의 풀 전체 리스트 출력 ===========")
# OOS 매일1건 Top1
oos_d_df = pd.DataFrame(oos_d)
best_oos_d = oos_d_df.sort_values("ret_pct", ascending=False).iloc[0]
print(f"OOS 매일1건 최고: {best_oos_d['filter']} / {best_oos_d['sort']} → ret {best_oos_d['ret_pct']:.1f}%")

# OOS 주2건 Top1
oos_w2_df = pd.DataFrame(oos_w2)
best_oos_w2 = oos_w2_df.sort_values("ret_pct", ascending=False).iloc[0]
print(f"OOS 주2건 최고:  {best_oos_w2['filter']} / {best_oos_w2['sort']} → ret {best_oos_w2['ret_pct']:.1f}%")

# 두 best 조합의 풀 전체 (2020-2026)
def export_pool(filter_id, sort_id, label):
    pool = filter_pool(cand, filter_id)
    pool = sort_pool(pool, sort_id)
    cols = ["Date", "Code", "Name", "Market", "Close", "Amount", "Score",
            "chart_pattern", "past_60", "past_120", "pos_252_high",
            "slope60", "drawdown60", "peak_180d", "ret_180d",
            "sell_date", "sell_close"]
    extra = [c for c in ["For_5d", "Inst_5d", "For_20d", "Inst_20d",
                          "PER_num", "PBR_num", "시총_num"] if c in pool.columns]
    cols.extend(extra)
    cols = [c for c in cols if c in pool.columns]
    out = pool[cols].copy()
    out["Year"] = out["Date"].dt.year
    out["YYYYMM"] = out["Date"].dt.strftime("%Y-%m")
    out_path = CACHE / f"BEST_POOL_{label}.csv"
    out.to_csv(out_path, index=False)
    print(f"  → {out_path} ({len(out):,}건)")

export_pool(best_oos_d["filter"], best_oos_d["sort"], "daily_1_OOS_TOP1")
export_pool(best_oos_w2["filter"], best_oos_w2["sort"], "weekly_2_OOS_TOP1")
export_pool(best_oos_d["filter"], best_oos_d["sort"], "daily_1_FULL")  # 전체 풀 (2020-2026)

# 매일1건 Top3 매수 종목 합본
print("\n=========== Top3 매수 종목 합본 ===========")
def export_trades_top3(rows, label, trades):
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return
    df_top3 = df.sort_values("ret_pct", ascending=False).head(3)
    all_picks = []
    for rank, (_, r) in enumerate(df_top3.iterrows(), 1):
        t = trades[r["combo"]].copy()
        t["rank"] = rank
        t["combo"] = r["combo"]
        t["set"] = label
        all_picks.append(t)
    out = pd.concat(all_picks, ignore_index=True)
    out["YYYYMM"] = pd.to_datetime(out["buy_date"]).dt.strftime("%Y-%m")
    out_path = CACHE / f"BEST_TRADES_{label}.csv"
    out.to_csv(out_path, index=False)
    print(f"  → {out_path} ({len(out):,}건)")

export_trades_top3(full_d, "FULL_daily_1", full_d_trades)
export_trades_top3(full_w2, "FULL_weekly_2", full_w2_trades)
export_trades_top3(full_w3, "FULL_weekly_3", full_w3_trades)
export_trades_top3(oos_d, "OOS_daily_1", oos_d_trades)
export_trades_top3(oos_w2, "OOS_weekly_2", oos_w2_trades)

print("\n[완료]")
