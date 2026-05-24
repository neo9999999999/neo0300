"""
2020-04 ~ 2026-05 전체 walk-forward (시총 300 + ★ 강력매수 주 3건)
=============================================================
가능한 전체 데이터로 walk-forward. 시그널 풀 2개 통합.
"""

import warnings
warnings.filterwarnings("ignore")
import json, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from train_rf_loss_model import FEATURES, prepare_X, add_pre_features
from recommendation_grader import assign_grades

CACHE = Path("cache")
ALLOC = 100_000

print("[1] 시그널 풀 통합")
# 시총 500 기반 풀 (2020-04 ~ 2025-08)
old = pd.read_parquet(CACHE / "candidates_enriched_full.parquet")
old["Date"] = pd.to_datetime(old["Date"])
print(f"  candidates_enriched_full (시총 500): {len(old):,}건 ({old['Date'].min().date()} ~ {old['Date'].max().date()})")

# 시총 2000 기반 풀 (2021-04 ~ 2026-05)
new = pd.read_parquet(CACHE / "signals_2000_enriched.parquet")
new["Date"] = pd.to_datetime(new["Date"])
print(f"  signals_2000_enriched: {len(new):,}건 ({new['Date'].min().date()} ~ {new['Date'].max().date()})")

# 시총 300 필터
snap = pd.read_parquet(CACHE / "market_snapshot.parquet")
top300 = set(snap.sort_values("MarketCap", ascending=False).head(300)["Code"])
old300 = old[old["Code"].isin(top300)].copy()
new300 = new[new["Code"].isin(top300)].copy()
print(f"  시총 300 필터: old {len(old300):,}, new {len(new300):,}")

# 통합 (날짜 중복 제거 - new 우선)
old300 = old300[old300["Date"] < "2021-04-01"].copy()  # new가 2021-04부터라 old는 그 전까지만
print(f"  old (2020-04~2021-03): {len(old300):,}")

# 컬럼 정리 (양쪽 공통)
common_cols = list(set(old300.columns) & set(new300.columns))
combined = pd.concat([old300[common_cols], new300[common_cols]], ignore_index=True)
combined = combined.dropna(subset=["peak_180d","ret_180d","Amount"]).copy()
combined = combined.sort_values(["Date","Code"]).reset_index(drop=True)
combined["Year"] = combined["Date"].dt.year
print(f"  통합 풀 (시총 300, 전체): {len(combined):,}건 ({combined['Date'].min().date()} ~ {combined['Date'].max().date()})")
print(f"  년도별: {combined.groupby('Year').size().to_dict()}")


# 시계열 특성 추가 (없는 종목만)
if "pre_5d_max_high_ratio" not in combined.columns:
    print("[2] 시계열 특성 추가 (없는 부분)")
    combined = add_pre_features(combined)


# Walk-forward (2020 데이터 학습 후 2021 검증, ..., 누적 학습)
windows = [
    {"tr_end": 2021, "te": 2022},
    {"tr_end": 2022, "te": 2023},
    {"tr_end": 2023, "te": 2024},
    {"tr_end": 2024, "te": 2025},
    {"tr_end": 2025, "te": 2026},
]


def train_all(train_df):
    available = [f for f in FEATURES if f in train_df.columns]
    X, _ = prepare_X(train_df, features=available)
    models = {}
    for name, y in [
        ("loss", (train_df["ret_180d"]<=-20).astype(int)),
        ("sw", (train_df["peak_180d"]>=200).astype(int)),
        ("100plus", (train_df["peak_180d"]>=100).astype(int)),
        ("50plus", (train_df["peak_180d"]>=50).astype(int)),
    ]:
        clf = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=20,
                                       class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X, y)
        models[name] = clf
    return models, available


def predict(test_df, models, features):
    X, _ = prepare_X(test_df, features=features)
    test_df["p_loss"] = models["loss"].predict_proba(X)[:, 1]
    test_df["p_sw"] = models["sw"].predict_proba(X)[:, 1]
    test_df["p_100plus"] = models["100plus"].predict_proba(X)[:, 1]
    test_df["p_50plus"] = models["50plus"].predict_proba(X)[:, 1]
    test_df["StrongScore"] = test_df["p_sw"]*3.0 + test_df["p_100plus"]*1.5 + test_df["p_50plus"]*1.0 - test_df["p_loss"]*2.0
    return test_df


# 시뮬레이션
all_picks = []
year_results = []
for w in windows:
    train = combined[combined["Year"] <= w["tr_end"]].copy()
    test = combined[combined["Year"] == w["te"]].copy()
    if len(train)<500 or len(test)<30: continue
    print(f"\n[Walk] ~{w['tr_end']} → {w['te']} | Train {len(train):,} / Test {len(test):,}")
    models, features = train_all(train)
    test = predict(test, models, features)
    test = assign_grades(test, scope="day")
    test["Year"] = w["te"]

    # ★ 강력매수만, 주 3건
    strong = test[test["등급"]=="★ 강력매수"].copy()
    strong = strong.sort_values("StrongScore", ascending=False)
    strong["week"] = strong["Date"].dt.strftime("%Y-%U")
    weekly3 = strong.groupby("week").head(3).drop_duplicates(["Date","Code"])

    if len(weekly3)==0:
        print(f"  ★ 강력매수 0건"); continue

    invest = len(weekly3)*ALLOC
    profit = ((weekly3["sell_close"]/weekly3["Close"] - 1)*ALLOC).sum()
    sw = (weekly3["peak_180d"]>=200).sum()
    w100 = (weekly3["peak_180d"]>=100).sum()
    w50 = (weekly3["peak_180d"]>=50).sum()
    loser = (weekly3["ret_180d"]<=-20).sum()
    print(f"  매수 {len(weekly3)}건, SW {sw}({sw/len(weekly3)*100:.1f}%), "
          f"100+ {w100}, 50+ {w50}, 손절 {loser}({loser/len(weekly3)*100:.1f}%), "
          f"수익 {profit/1e4:+,.0f}만 ({profit/invest*100:+.1f}%)")
    year_results.append({
        "year": w["te"], "매수": len(weekly3),
        "SW": sw, "SW%": round(sw/len(weekly3)*100, 1),
        "100+": w100, "50+": w50,
        "손절": loser, "손절%": round(loser/len(weekly3)*100, 1),
        "투자만": invest/1e4, "수익만": round(profit/1e4),
        "수익률%": round(profit/invest*100, 1),
    })
    all_picks.append(weekly3)


# 누적
print("\n" + "="*100)
print("2020~2026 전체 walk-forward (시총 300 + ★ 강력매수 주 3건)")
print("="*100)
yr_df = pd.DataFrame(year_results)
print("\n[년도별]")
print(yr_df.to_string(index=False))

total = pd.concat(all_picks, ignore_index=True)
n = len(total)
sw_n = (total["peak_180d"]>=200).sum()
w100_n = (total["peak_180d"]>=100).sum()
w50_n = (total["peak_180d"]>=50).sum()
loss_n = (total["ret_180d"]<=-20).sum()
inv = n*ALLOC
prof = ((total["sell_close"]/total["Close"] - 1)*ALLOC).sum()

print(f"\n[전체 누적 5년 (2022~2026)]")
print(f"  매수: {n}건")
print(f"  슈퍼위너 (200%+): {sw_n} ({sw_n/n*100:.1f}%)")
print(f"  100%+: {w100_n} ({w100_n/n*100:.1f}%)")
print(f"  50%+: {w50_n} ({w50_n/n*100:.1f}%)")
print(f"  손절 (-20%↓): {loss_n} ({loss_n/n*100:.1f}%)")
print(f"  투자 {inv/1e4:,.0f}만 → 수익 {prof/1e4:+,.0f}만 ({prof/inv*100:+.1f}%)")

# 저장
total.to_csv(CACHE / "FINAL_매수_2020-2026.csv", index=False)
yr_df.to_csv(CACHE / "FINAL_년도별_2020-2026.csv", index=False)
print(f"\n[저장] cache/FINAL_매수_2020-2026.csv + FINAL_년도별_2020-2026.csv")
