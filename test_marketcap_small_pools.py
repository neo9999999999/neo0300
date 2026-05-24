"""
B: 시총 100/150/200/300/600 비교 (더 작은 풀이 더 좋은지)
C: 시총 300 + weekly_var에서 StrongScore 가중치 그리드 튜닝
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from train_rf_loss_model import FEATURES, prepare_X

CACHE = Path("cache")
ALLOC = 100_000


sigs = pd.read_parquet(CACHE / "signals_2000_enriched.parquet")
sigs["Date"] = pd.to_datetime(sigs["Date"])
sigs = sigs.dropna(subset=["peak_180d", "ret_180d", "Amount"]).copy()
sigs["Year"] = sigs["Date"].dt.year

snap = pd.read_parquet(CACHE / "market_snapshot.parquet")
snap_sorted = snap.sort_values("MarketCap", ascending=False).reset_index(drop=True)


def get_codes(top_n):
    return set(snap_sorted.head(top_n)["Code"])


def train_models(train_df):
    available = [f for f in FEATURES if f in train_df.columns]
    X, _ = prepare_X(train_df, features=available)
    models = {}
    for tgt_name, y in [
        ("loss", (train_df["ret_180d"]<=-20).astype(int)),
        ("sw", (train_df["peak_180d"]>=200).astype(int)),
        ("100plus", (train_df["peak_180d"]>=100).astype(int)),
        ("50plus", (train_df["peak_180d"]>=50).astype(int)),
    ]:
        clf = RandomForestClassifier(n_estimators=120, max_depth=6, min_samples_leaf=20,
                                       class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X, y)
        models[tgt_name] = clf
    return models, available


def predict_score(test_df, models, features, w_sw=3.0, w_100=1.5, w_50=1.0, w_loss=2.0):
    X, _ = prepare_X(test_df, features=features)
    p_loss = models["loss"].predict_proba(X)[:, 1]
    p_sw = models["sw"].predict_proba(X)[:, 1]
    p_100 = models["100plus"].predict_proba(X)[:, 1]
    p_50 = models["50plus"].predict_proba(X)[:, 1]
    test_df["StrongScore"] = p_sw*w_sw + p_100*w_100 + p_50*w_50 - p_loss*w_loss
    return test_df


def simulate(pool, mode):
    pool = pool.dropna(subset=["sell_close"]).copy()
    if mode == "daily_3":
        pool = pool.sort_values("Amount")
        pool["bucket"] = pool["Date"].dt.strftime("%Y-%m-%d")
        picks = pool.groupby("bucket").head(3)
    elif mode == "daily_must":
        pool = pool.sort_values("StrongScore", ascending=False)
        pool["bucket"] = pool["Date"].dt.strftime("%Y-%m-%d")
        picks = pool.groupby("bucket").head(1)
    elif mode == "weekly_var":
        pool = pool.sort_values("StrongScore", ascending=False)
        pool["bucket"] = pool["Date"].dt.strftime("%Y-%U")
        top5 = pool.groupby("bucket").head(5)
        threshold = top5["StrongScore"].quantile(0.40)
        def select_var(g):
            g = g.sort_values("StrongScore", ascending=False).reset_index(drop=True)
            base = g.head(3)
            extra = g.iloc[3:5]
            extra = extra[extra["StrongScore"] >= threshold]
            return pd.concat([base, extra])
        picks = top5.groupby("bucket", group_keys=False).apply(select_var)
    picks = picks.drop_duplicates(["Date", "Code"])
    if len(picks) == 0: return None
    invest = len(picks) * ALLOC
    profit = ((picks["sell_close"]/picks["Close"] - 1) * ALLOC).sum()
    return {
        "n": len(picks),
        "SW": int((picks["peak_180d"]>=200).sum()),
        "100+": int((picks["peak_180d"]>=100).sum()),
        "50+": int((picks["peak_180d"]>=50).sum()),
        "손절": int((picks["ret_180d"]<=-20).sum()),
        "SW률%": round((picks["peak_180d"]>=200).mean()*100, 1),
        "손절률%": round((picks["ret_180d"]<=-20).mean()*100, 1),
        "투자만": invest/1e4,
        "수익만": round(profit/1e4),
        "수익률%": round(profit/invest*100, 1),
    }


windows = [
    {"tr": (2021,2022), "te":2023},
    {"tr": (2021,2023), "te":2024},
    {"tr": (2021,2024), "te":2025},
    {"tr": (2021,2025), "te":2026},
]


# ============ B: 시총 100/150/200 + 비교 ============
print("="*100)
print("B: 시총 100/150/200/300/600 비교")
print("="*100)

B_results = []
B_pools = {}

for top_n in [100, 150, 200, 300, 600]:
    codes = get_codes(top_n)
    pool_sigs = sigs[sigs["Code"].isin(codes)].copy()
    B_pools[top_n] = pool_sigs
    cutoff = snap_sorted.iloc[top_n-1]["MarketCap"] / 1e8
    print(f"\n=== 시총 {top_n} (cutoff {cutoff:,.0f}억, 시그널 {len(pool_sigs):,}) ===")

    for w in windows:
        train = pool_sigs[(pool_sigs["Year"]>=w["tr"][0])&(pool_sigs["Year"]<=w["tr"][1])].copy()
        test = pool_sigs[pool_sigs["Year"]==w["te"]].copy()
        if len(train)<300 or len(test)<30: continue
        models, features = train_models(train)
        test = predict_score(test, models, features)
        for mode in ["daily_3", "daily_must", "weekly_var"]:
            r = simulate(test, mode)
            if r is None: continue
            r["풀"] = top_n; r["year"] = w["te"]; r["모드"] = mode
            B_results.append(r)
            print(f"  [{w['te']} {mode}] n={r['n']}, SW={r['SW']}({r['SW률%']}%), 손절={r['손절']}({r['손절률%']}%), 수익={r['수익률%']}%")

B_df = pd.DataFrame(B_results)
B_df.to_csv(CACHE / "B_marketcap_pools.csv", index=False)


# 누적 종합 비교
print("\n" + "="*100)
print("B 누적 4년 비교")
print("="*100)
print(f"\n{'풀':6s}{'모드':12s}{'매수':>7s}{'SW':>5s}{'SW%':>6s}{'손절':>5s}{'손절%':>7s}{'투자(만)':>10s}{'수익(만)':>10s}{'수익률':>8s}")
for top_n in [100, 150, 200, 300, 600]:
    for mode in ["daily_3", "daily_must", "weekly_var"]:
        sub = B_df[(B_df["풀"]==top_n)&(B_df["모드"]==mode)]
        if len(sub)==0: continue
        n = sub["n"].sum()
        sw = sub["SW"].sum()
        loser = sub["손절"].sum()
        inv = sub["투자만"].sum()
        prof = sub["수익만"].sum()
        print(f"top{top_n:<3d}{mode:12s}{n:>7d}{sw:>5d}{sw/n*100:>5.1f}%{loser:>5d}{loser/n*100:>6.1f}%"
              f"{inv:>10,.0f}{prof:>10,.0f}{prof/inv*100:>+7.1f}%")


# ============ C: 시총 300 + weekly_var 가중치 그리드 ============
print("\n" + "="*100)
print("C: 시총 300 + weekly_var — StrongScore 가중치 그리드")
print("="*100)

pool300 = B_pools[300]
print(f"풀: 시총 300, 시그널 {len(pool300):,}")

# 그리드: w_sw × w_100 × w_loss
grid = []
for w_sw in [2.0, 3.0, 5.0, 7.0]:
    for w_100 in [1.0, 1.5, 2.0]:
        for w_loss in [1.0, 2.0, 3.0, 5.0]:
            grid.append((w_sw, w_100, 1.0, w_loss))

print(f"\n총 {len(grid)}가지 조합 시뮬...")
C_results = []
for w_sw, w_100, w_50, w_loss in grid:
    total = {"n":0,"SW":0,"손절":0,"투자만":0,"수익만":0}
    for w in windows:
        train = pool300[(pool300["Year"]>=w["tr"][0])&(pool300["Year"]<=w["tr"][1])].copy()
        test = pool300[pool300["Year"]==w["te"]].copy()
        if len(train)<300 or len(test)<30: continue
        models, features = train_models(train)
        test = predict_score(test, models, features, w_sw, w_100, w_50, w_loss)
        r = simulate(test, "weekly_var")
        if r is None: continue
        for k in total: total[k] += r[k]
    if total["n"]==0: continue
    C_results.append({
        "w_sw": w_sw, "w_100": w_100, "w_50": w_50, "w_loss": w_loss,
        "n": total["n"], "SW": total["SW"], "손절": total["손절"],
        "SW률%": round(total["SW"]/total["n"]*100, 1),
        "손절률%": round(total["손절"]/total["n"]*100, 1),
        "투자만": total["투자만"], "수익만": total["수익만"],
        "수익률%": round(total["수익만"]/total["투자만"]*100, 1) if total["투자만"]>0 else 0,
    })

C_df = pd.DataFrame(C_results)
C_df.to_csv(CACHE / "C_strongscore_grid.csv", index=False)

# 상위 10개 (수익률 기준)
print("\n[Top 10 가중치 조합 - 수익률 기준]")
print(C_df.sort_values("수익률%", ascending=False).head(10).to_string(index=False))

# 상위 10개 (SW률 기준)
print("\n[Top 10 가중치 조합 - SW률 기준]")
print(C_df.sort_values("SW률%", ascending=False).head(10).to_string(index=False))

# 상위 10개 (손절률 낮은 순)
print("\n[Top 10 가중치 조합 - 손절률 낮은 순]")
print(C_df.sort_values("손절률%", ascending=True).head(10).to_string(index=False))

print(f"\n[저장] cache/B_marketcap_pools.csv + C_strongscore_grid.csv")
