"""
시총 300/600 검증 (3 모드)
========================
- 매일 3건 (거래대금↓ TOP 3)
- 매일 필수매수 1건 (StrongScore TOP 1)
- 주 가변 3-5건 (StrongScore > threshold + 상한 5건)

년도별 수익률 비교.
"""

import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from train_rf_loss_model import FEATURES, prepare_X

CACHE = Path("cache")
ALLOC = 100_000


# 1) signals_2000_enriched 로드 + 시총 필터
sigs = pd.read_parquet(CACHE / "signals_2000_enriched.parquet")
sigs["Date"] = pd.to_datetime(sigs["Date"])
sigs = sigs.dropna(subset=["peak_180d", "ret_180d", "Amount"]).copy()
print(f"전체 시그널 (시총 2000): {len(sigs):,}건")

snap = pd.read_parquet(CACHE / "market_snapshot.parquet")
snap_sorted = snap.sort_values("MarketCap", ascending=False).reset_index(drop=True)

top300_codes = set(snap_sorted.head(300)["Code"])
top600_codes = set(snap_sorted.head(600)["Code"])
print(f"시총 300위 컷오프: {snap_sorted.iloc[299]['MarketCap']/1e8:.0f}억")
print(f"시총 600위 컷오프: {snap_sorted.iloc[599]['MarketCap']/1e8:.0f}억")


def filter_pool(pool, codes_set):
    return pool[pool["Code"].isin(codes_set)].copy()


# 시그널 필터
sigs300 = filter_pool(sigs, top300_codes)
sigs600 = filter_pool(sigs, top600_codes)
sigs300["Year"] = sigs300["Date"].dt.year
sigs600["Year"] = sigs600["Date"].dt.year
sigs300["is_loser"] = (sigs300["ret_180d"]<=-20).astype(int)
sigs300["is_sw"] = (sigs300["peak_180d"]>=200).astype(int)
sigs600["is_loser"] = (sigs600["ret_180d"]<=-20).astype(int)
sigs600["is_sw"] = (sigs600["peak_180d"]>=200).astype(int)
print(f"\n시총 300 시그널: {len(sigs300):,}건 (년도별 평균 {len(sigs300)/4:.0f}건)")
print(f"시총 600 시그널: {len(sigs600):,}건 (년도별 평균 {len(sigs600)/4:.0f}건)")


# 2) Walk-forward RF 학습 + 매수 시뮬
windows = [
    {"tr": (2021, 2022), "te": 2023},
    {"tr": (2021, 2023), "te": 2024},
    {"tr": (2021, 2024), "te": 2025},
    {"tr": (2021, 2025), "te": 2026},
]


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
        clf = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=20,
                                       class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X, y)
        models[tgt_name] = clf
    y_reg = np.clip(train_df["peak_180d"].values, -50, 500)
    reg = RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=15,
                                  random_state=42, n_jobs=-1)
    reg.fit(X, y_reg)
    models["peak_reg"] = reg
    return models, available


def predict(test_df, models, features):
    X, _ = prepare_X(test_df, features=features)
    p_loss = models["loss"].predict_proba(X)[:, 1]
    p_sw = models["sw"].predict_proba(X)[:, 1]
    p_100 = models["100plus"].predict_proba(X)[:, 1]
    p_50 = models["50plus"].predict_proba(X)[:, 1]
    peak = models["peak_reg"].predict(X)
    test_df["p_loss"] = p_loss
    test_df["p_sw"] = p_sw
    test_df["p_100plus"] = p_100
    test_df["p_50plus"] = p_50
    test_df["peak_pred"] = peak
    test_df["StrongScore"] = p_sw*3.0 + p_100*1.5 + p_50*1.0 - p_loss*2.0
    return test_df


def simulate(pool, mode, threshold_score=None):
    """매수 시뮬 - 종목당 10만"""
    pool = pool.dropna(subset=["sell_close"]).copy()
    if mode == "daily_3":
        pool = pool.sort_values("Amount")
        pool["bucket"] = pool["Date"].dt.strftime("%Y-%m-%d")
        picks = pool.groupby("bucket").head(3)
    elif mode == "daily_must":
        # 매일 StrongScore 1등만
        pool = pool.sort_values("StrongScore", ascending=False)
        pool["bucket"] = pool["Date"].dt.strftime("%Y-%m-%d")
        picks = pool.groupby("bucket").head(1)
    elif mode == "weekly_var":
        # 주 가변 3-5건 (StrongScore > 0 이면 매수, 최대 5건)
        pool = pool.sort_values("StrongScore", ascending=False)
        pool["bucket"] = pool["Date"].dt.strftime("%Y-%U")
        # 주별 상위 5건만 후보
        top5 = pool.groupby("bucket").head(5)
        # StrongScore > threshold만 매수 (단 최소 3건 보장)
        if threshold_score is None:
            threshold_score = top5["StrongScore"].quantile(0.40)
        # 각 주에 대해: top 3 무조건 매수, 4-5등은 score > th일 때만
        def select_var(g):
            g = g.sort_values("StrongScore", ascending=False).reset_index(drop=True)
            # 1-3등 무조건
            base = g.head(3)
            # 4-5등 임계값 통과시
            extra = g.iloc[3:5]
            extra = extra[extra["StrongScore"] >= threshold_score]
            return pd.concat([base, extra])
        picks = top5.groupby("bucket", group_keys=False).apply(select_var)
    else:
        raise ValueError(mode)

    picks = picks.drop_duplicates(subset=["Date", "Code"])
    if len(picks) == 0:
        return None
    invest = len(picks) * ALLOC
    profit = ((picks["sell_close"]/picks["Close"] - 1) * ALLOC).sum()
    return {
        "n": len(picks),
        "익절": int((picks["ret_180d"]>0).sum()),
        "손절": int((picks["ret_180d"]<=-20).sum()),
        "SW": int((picks["peak_180d"]>=200).sum()),
        "100+": int((picks["peak_180d"]>=100).sum()),
        "50+": int((picks["peak_180d"]>=50).sum()),
        "SW률%": round((picks["peak_180d"]>=200).mean()*100, 1),
        "100+%": round((picks["peak_180d"]>=100).mean()*100, 1),
        "50+%": round((picks["peak_180d"]>=50).mean()*100, 1),
        "손절률%": round((picks["ret_180d"]<=-20).mean()*100, 1),
        "투자만": invest/1e4,
        "수익만": round(profit/1e4),
        "수익률%": round(profit/invest*100, 1),
    }, picks


# 3) walk-forward 시뮬레이션 (시총 300 + 시총 600)
all_results = []
all_picks_list = []

for label, sigs_pool in [("시총300", sigs300), ("시총600", sigs600)]:
    print(f"\n{'='*100}")
    print(f"{label} walk-forward OOS")
    print('='*100)
    for w in windows:
        train = sigs_pool[(sigs_pool["Year"]>=w["tr"][0])&(sigs_pool["Year"]<=w["tr"][1])].copy()
        test = sigs_pool[sigs_pool["Year"]==w["te"]].copy()
        if len(train)<500 or len(test)<50:
            print(f"\n[{label} {w['tr']}→{w['te']}] 데이터 부족 (Train {len(train)}, Test {len(test)})"); continue
        print(f"\n[{label} {w['tr'][0]}-{w['tr'][1]} → Test {w['te']}] Train {len(train):,} / Test {len(test):,}")
        models, features = train_models(train)
        test = predict(test, models, features)

        for mode in ["daily_3", "daily_must", "weekly_var"]:
            r, picks = simulate(test, mode)
            if r is None: continue
            r["year"] = w["te"]; r["풀"] = label; r["모드"] = mode
            all_results.append(r)
            picks["year"] = w["te"]; picks["풀"] = label; picks["모드"] = mode
            all_picks_list.append(picks)
            print(f"  [{mode}] n={r['n']}, SW={r['SW']}({r['SW률%']}%), 100+={r['100+']}, "
                  f"50+={r['50+']}, 손절={r['손절']}({r['손절률%']}%), 수익={r['수익률%']}%")

res_df = pd.DataFrame(all_results)
res_df.to_csv(CACHE / "test300_600_results.csv", index=False)

# 4) 종합 출력 - 년도별 + 풀별 + 모드별
print("\n" + "="*120)
print("종합 년도별 수익률 비교")
print("="*120)

pivot = res_df.pivot_table(
    index=["풀", "모드"],
    columns="year",
    values="수익률%",
    aggfunc="sum"
)
print("\n[년도별 수익률 (%) - 매수당 평균]")
print(pivot)

pivot_n = res_df.pivot_table(
    index=["풀", "모드"],
    columns="year",
    values="n",
    aggfunc="sum"
)
print("\n[년도별 매수 수]")
print(pivot_n)

pivot_sw = res_df.pivot_table(
    index=["풀", "모드"],
    columns="year",
    values="SW률%",
    aggfunc="sum"
)
print("\n[년도별 SW률 (%)]")
print(pivot_sw)

pivot_loss = res_df.pivot_table(
    index=["풀", "모드"],
    columns="year",
    values="손절률%",
    aggfunc="sum"
)
print("\n[년도별 손절률 (%)]")
print(pivot_loss)

# 누적
print("\n" + "="*100)
print("누적 (4년 OOS) — 모드별 종합")
print("="*100)
for pool_label in ["시총300", "시총600"]:
    print(f"\n[{pool_label}]")
    for mode in ["daily_3", "daily_must", "weekly_var"]:
        sub = res_df[(res_df["풀"]==pool_label)&(res_df["모드"]==mode)]
        if len(sub)==0: continue
        n = sub["n"].sum()
        sw = sub["SW"].sum()
        loser = sub["손절"].sum()
        invest = sub["투자만"].sum()
        profit = sub["수익만"].sum()
        print(f"  [{mode}]")
        print(f"    매수 {n:,}, SW {sw} ({sw/n*100:.1f}%), 손절 {loser} ({loser/n*100:.1f}%)")
        print(f"    투자 {invest:,.0f}만 → 수익 {profit:+,.0f}만 ({profit/invest*100:+.1f}%)")

# 매수 종목 리스트
all_picks_df = pd.concat(all_picks_list, ignore_index=True)
all_picks_df = all_picks_df.sort_values("Date", ascending=False)
all_picks_df.to_csv(CACHE / "test300_600_picks.csv", index=False)
print(f"\n[저장] cache/test300_600_results.csv + test300_600_picks.csv ({len(all_picks_df):,}건)")
