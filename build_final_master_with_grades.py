"""
최종 마스터 종목 리스트 + 등급 표기
=================================
시총 300 풀 walk-forward OOS 결과를 등급/태그 부여하여 출력.
"""

import pickle
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from train_rf_loss_model import FEATURES, prepare_X
from recommendation_grader import assign_grades, add_predicted_returns

CACHE = Path("cache")
ALLOC = 100_000

sigs = pd.read_parquet(CACHE / "signals_2000_enriched.parquet")
sigs["Date"] = pd.to_datetime(sigs["Date"])
sigs = sigs.dropna(subset=["peak_180d", "ret_180d", "Amount"]).copy()
sigs["Year"] = sigs["Date"].dt.year
snap = pd.read_parquet(CACHE / "market_snapshot.parquet")
top300 = set(snap.sort_values("MarketCap", ascending=False).head(300)["Code"])
pool300 = sigs[sigs["Code"].isin(top300)].copy()
print(f"시총 300 풀: {len(pool300):,}")


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
    y_reg = np.clip(train_df["peak_180d"].values, -50, 500)
    reg = RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=15,
                                  random_state=42, n_jobs=-1)
    reg.fit(X, y_reg)
    models["peak_reg"] = reg
    return models, available


def predict_all(test_df, models, features):
    X, _ = prepare_X(test_df, features=features)
    test_df["p_loss"] = models["loss"].predict_proba(X)[:, 1]
    test_df["p_sw"] = models["sw"].predict_proba(X)[:, 1]
    test_df["p_100plus"] = models["100plus"].predict_proba(X)[:, 1]
    test_df["p_50plus"] = models["50plus"].predict_proba(X)[:, 1]
    test_df["peak_pred"] = models["peak_reg"].predict(X)
    test_df["StrongScore"] = test_df["p_sw"]*3.0 + test_df["p_100plus"]*1.5 + test_df["p_50plus"]*1.0 - test_df["p_loss"]*2.0
    return test_df


# Walk-forward로 OOS 예측 → 등급 부여
windows = [
    {"tr":(2021,2022), "te":2023},
    {"tr":(2021,2023), "te":2024},
    {"tr":(2021,2024), "te":2025},
    {"tr":(2021,2025), "te":2026},
]

all_graded = []
for w in windows:
    train = pool300[(pool300["Year"]>=w["tr"][0])&(pool300["Year"]<=w["tr"][1])].copy()
    test = pool300[pool300["Year"]==w["te"]].copy()
    if len(train)<300 or len(test)<30: continue
    print(f"\n=== {w['tr']} → {w['te']} ===")
    models, features = train_all(train)
    test = predict_all(test, models, features)
    test = assign_grades(test, scope="day")
    test = add_predicted_returns(test)
    all_graded.append(test)

graded = pd.concat(all_graded, ignore_index=True)
print(f"\n전체 OOS 등급 부여: {len(graded):,}건")


# 등급별 통계
print("\n[등급별 통계]")
for grade in ["★ 강력매수", "○ 추천", "- 관망", "⚠️ 손절위험"]:
    sub = graded[graded["등급"]==grade]
    if len(sub)==0: continue
    sw_rate = (sub["peak_180d"]>=200).mean()*100
    w100_rate = (sub["peak_180d"]>=100).mean()*100
    w50_rate = (sub["peak_180d"]>=50).mean()*100
    loss_rate = (sub["ret_180d"]<=-20).mean()*100
    avg_peak = sub["peak_180d"].mean()
    print(f"  {grade}: {len(sub):,}건 | SW {sw_rate:.1f}% | 100+ {w100_rate:.1f}% | "
          f"50+ {w50_rate:.1f}% | 손절 {loss_rate:.1f}% | 평균peak {avg_peak:.0f}%")


# 가능성 태그별 통계
print("\n[가능성 태그별 실제 결과]")
for tag in ["🏆 슈퍼위너 강력후보", "⭐ 슈퍼위너후보", "💯 100%+ 가능", "📈 50%+ 가능", "🔻 손절 주의"]:
    sub = graded[graded["가능성태그"].str.contains(tag, regex=False, na=False)]
    if len(sub)==0: continue
    sw_rate = (sub["peak_180d"]>=200).mean()*100
    w100_rate = (sub["peak_180d"]>=100).mean()*100
    w50_rate = (sub["peak_180d"]>=50).mean()*100
    loss_rate = (sub["ret_180d"]<=-20).mean()*100
    print(f"  {tag}: {len(sub):,}건 | 실제 SW {sw_rate:.1f}% | 100+ {w100_rate:.1f}% | "
          f"50+ {w50_rate:.1f}% | 손절 {loss_rate:.1f}%")


# 저장 - 마스터 종목 리스트
out_cols = ["Date", "등급", "가능성태그", "예상수익률",
            "Code", "Name", "Market", "Close", "Amount", "Score",
            "StrongScore", "peak_pred",
            "p_sw", "p_100plus", "p_50plus", "p_loss",
            "ret_180d", "peak_180d", "sell_close", "sell_date",
            "chart_pattern", "past_60", "past_120", "past_240",
            "pos_252_high", "slope60", "drawdown60"]
out_cols = [c for c in out_cols if c in graded.columns]
graded_out = graded[out_cols].copy()
graded_out["Year"] = graded_out["Date"].dt.year
graded_out["YYYYMM"] = graded_out["Date"].dt.strftime("%Y-%m")

# 확률 % 변환
for c in ["p_sw", "p_100plus", "p_50plus", "p_loss"]:
    if c in graded_out.columns:
        graded_out[c+"%"] = (graded_out[c]*100).round(1)
graded_out["peak_pred"] = graded_out["peak_pred"].round(1)
graded_out["StrongScore"] = graded_out["StrongScore"].round(2)
graded_out["ret_180d"] = graded_out["ret_180d"].round(1)
graded_out["peak_180d"] = graded_out["peak_180d"].round(1)

# 최신순
graded_out = graded_out.sort_values("Date", ascending=False)

# 전체 CSV
graded_out.to_csv(CACHE / "MASTER_등급포함_시총300_2023-2026.csv", index=False)
print(f"\n[저장] cache/MASTER_등급포함_시총300_2023-2026.csv ({len(graded_out):,}건)")

# 등급별 분리 CSV
for grade, fname in [
    ("★ 강력매수", "MASTER_강력매수_시총300.csv"),
    ("○ 추천", "MASTER_추천_시총300.csv"),
    ("⚠️ 손절위험", "MASTER_손절위험_시총300.csv"),
]:
    sub = graded_out[graded_out["등급"]==grade]
    sub.to_csv(CACHE / fname, index=False)
    print(f"  {fname}: {len(sub):,}건")


# 주별 추천 (주에 3-5건)
print("\n[주별 매수 가능 종목 분포]")
graded["YearWeek"] = graded["Date"].dt.strftime("%Y-%U")
weekly_stat = graded.groupby("YearWeek").agg(
    n=("Code", "count"),
    strong_n=("등급", lambda x: (x=="★ 강력매수").sum()),
    추천_n=("등급", lambda x: (x=="○ 추천").sum()),
    위험_n=("등급", lambda x: (x=="⚠️ 손절위험").sum()),
).reset_index()
print(f"평균 시그널/주: {weekly_stat['n'].mean():.1f}")
print(f"평균 ★강력매수/주: {weekly_stat['strong_n'].mean():.1f}")
print(f"평균 ○추천/주: {weekly_stat['추천_n'].mean():.1f}")
print(f"평균 ⚠️위험/주: {weekly_stat['위험_n'].mean():.1f}")
weekly_stat.to_csv(CACHE / "MASTER_주별분포.csv", index=False)
