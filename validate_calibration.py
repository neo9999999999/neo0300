"""
모델 확률 vs 실제 OOS 적중률 검증
================================
walk-forward로 calibrated 모델 학습 + 매일 🔥 슈퍼강력 TOP 1 매수
각 매수 종목의 예측 확률 vs 180일 후 실제 결과 비교
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from train_rf_loss_model import FEATURES, prepare_X

CACHE = Path("cache")
ALLOC = 100_000

sigs = pd.read_parquet(CACHE / "signals_2000_enriched.parquet")
sigs["Date"] = pd.to_datetime(sigs["Date"])
sigs = sigs.dropna(subset=["peak_180d","ret_180d","Amount"]).copy()
sigs["Year"] = sigs["Date"].dt.year
snap = pd.read_parquet(CACHE / "market_snapshot.parquet")
top300 = set(snap.sort_values("MarketCap", ascending=False).head(300)["Code"])
pool = sigs[sigs["Code"].isin(top300)].copy()
print(f"풀: {len(pool):,}건")

windows = [
    {"tr_end": 2022, "te": 2023},
    {"tr_end": 2023, "te": 2024},
    {"tr_end": 2024, "te": 2025},
    {"tr_end": 2025, "te": 2026},
]

TARGETS = {
    "loss":    lambda d: (d["ret_180d"]<=-20).astype(int),
    "sw":      lambda d: (d["peak_180d"]>=200).astype(int),
    "100plus": lambda d: (d["peak_180d"]>=100).astype(int),
    "50plus":  lambda d: (d["peak_180d"]>=50).astype(int),
    "30plus":  lambda d: (d["peak_180d"]>=30).astype(int),
    "10plus":  lambda d: (d["peak_180d"]>=10).astype(int),
}


def train(train_df, available):
    X, _ = prepare_X(train_df, features=available)
    models = {}
    for name, fn in TARGETS.items():
        y = fn(train_df)
        base = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=20,
                                       random_state=42, n_jobs=-1)
        clf = CalibratedClassifierCV(base, method="isotonic", cv=3, n_jobs=-1)
        clf.fit(X, y)
        models[name] = clf
    return models


def predict_window(test_df, models, available):
    X, _ = prepare_X(test_df, features=available)
    for k in ["loss","sw","100plus","50plus","30plus","10plus"]:
        test_df[f"p_{k}"] = models[k].predict_proba(X)[:, 1]
    test_df["SuperScore"] = test_df["p_sw"]*5 + test_df["p_100plus"]*2 + test_df["p_50plus"]*1 - test_df["p_loss"]*3
    return test_df


# Walk-forward
print("\n[Walk-forward calibration]")
all_picks = []
for w in windows:
    train_df = pool[pool["Year"]<=w["tr_end"]].copy()
    test_df = pool[pool["Year"]==w["te"]].copy()
    if len(train_df)<500 or len(test_df)<30: continue
    available = [f for f in FEATURES if f in train_df.columns]
    print(f"  ~{w['tr_end']} → {w['te']} | Train {len(train_df):,} / Test {len(test_df):,}")
    models = train(train_df, available)
    test_df = predict_window(test_df, models, available)
    test_df["window"] = w["te"]
    all_picks.append(test_df)

predicted = pd.concat(all_picks, ignore_index=True)

# 매일 🔥 슈퍼강력 (상위 5%) TOP 1
predicted["_pct"] = predicted.groupby(predicted["Date"].dt.strftime("%Y-%m-%d"))["SuperScore"].rank(pct=True)
strong = predicted[predicted["_pct"] >= 0.95].copy()
# 매일 TOP 1
strong_top1 = strong.sort_values(["Date","SuperScore"], ascending=[True, False]).groupby(strong["Date"].dt.strftime("%Y-%m-%d")).head(1)
print(f"\n[매일 🔥 슈퍼강력 TOP 1 매수: {len(strong_top1)}건]")


# 확률 vs 실제 결과 비교 ===========
print("\n" + "="*100)
print("예측 확률 vs 실제 OOS 적중률 (5년 매일 TOP 1)")
print("="*100)

# 평균 예측 vs 실제
sw_pred = strong_top1["p_sw"].mean() * 100
sw_actual = (strong_top1["peak_180d"]>=200).mean() * 100
p100_pred = strong_top1["p_100plus"].mean() * 100
p100_actual = (strong_top1["peak_180d"]>=100).mean() * 100
p50_pred = strong_top1["p_50plus"].mean() * 100
p50_actual = (strong_top1["peak_180d"]>=50).mean() * 100
loss_pred = strong_top1["p_loss"].mean() * 100
loss_actual = (strong_top1["ret_180d"]<=-20).mean() * 100

print(f"\n[전체 평균 - 모든 타깃]")
print(f"{'타깃':<16s}{'평균 예측':>12s}{'실제 적중':>12s}{'차이':>10s}{'평가':>12s}")
metrics = [
    ("🏆 슈퍼위너 (peak ≥200%)", "p_sw", lambda d: (d["peak_180d"]>=200).mean()*100),
    ("💯 100%+ (peak ≥100%)", "p_100plus", lambda d: (d["peak_180d"]>=100).mean()*100),
    ("📈 50%+ (peak ≥50%)", "p_50plus", lambda d: (d["peak_180d"]>=50).mean()*100),
    ("📊 30%+ (peak ≥30%)", "p_30plus", lambda d: (d["peak_180d"]>=30).mean()*100),
    ("✅ 10%+ (peak ≥10%)", "p_10plus", lambda d: (d["peak_180d"]>=10).mean()*100),
    ("❌ 손절 (ret ≤-20%)", "p_loss", lambda d: (d["ret_180d"]<=-20).mean()*100),
]
for label, pcol, act_fn in metrics:
    if pcol not in strong_top1.columns: continue
    pred = strong_top1[pcol].mean()*100
    act = act_fn(strong_top1)
    diff = act - pred
    if pcol == "p_loss":
        ev = "✅ 보수적" if diff < 0 else "⚠️ 과대"
    else:
        ev = "✅ 정확" if abs(diff)<3 else ("📈 보수적" if diff>0 else "📉 낙관적")
    print(f"{label:<22s}{pred:>10.1f}%{act:>10.1f}%{diff:>+8.1f}%p   {ev}")

# 모든 타깃 확률 분위별 검증
bins_default = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.0]
bins_high = [0, 0.20, 0.40, 0.60, 0.70, 0.80, 0.90, 1.0]

calib_targets = [
    ("🏆 슈퍼위너", "p_sw",      lambda d: (d["peak_180d"]>=200).mean()*100, bins_default),
    ("💯 100%+",  "p_100plus", lambda d: (d["peak_180d"]>=100).mean()*100, bins_default),
    ("📈 50%+",   "p_50plus",  lambda d: (d["peak_180d"]>=50).mean()*100, [0,0.2,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
    ("📊 30%+",   "p_30plus",  lambda d: (d["peak_180d"]>=30).mean()*100, [0,0.3,0.5,0.6,0.7,0.8,0.9,1.0]),
    ("✅ 10%+",   "p_10plus",  lambda d: (d["peak_180d"]>=10).mean()*100, [0,0.5,0.7,0.8,0.85,0.9,0.95,1.0]),
    ("❌ 손절",    "p_loss",    lambda d: (d["ret_180d"]<=-20).mean()*100, bins_default),
]

for tgt_label, pcol, act_fn, bins in calib_targets:
    if pcol not in strong_top1.columns: continue
    print(f"\n[확률 분위별 - {tgt_label}]")
    print(f"  {'예측 범위':<18s}{'종목수':>7s}{'평균 예측':>11s}{'실제 적중률':>12s}")
    strong_top1[f"_bin_{pcol}"] = pd.cut(strong_top1[pcol], bins=bins, duplicates="drop")
    for bin_, g in strong_top1.groupby(f"_bin_{pcol}"):
        if len(g) == 0: continue
        pred = g[pcol].mean()*100
        act = act_fn(g)
        print(f"  {str(bin_):<18s}{len(g):>7d}{pred:>10.1f}%{act:>11.1f}%")

# 년도별 — 모든 타깃 비교
strong_top1["Year"] = strong_top1["Date"].dt.year
print("\n[년도별 - 예측 vs 실제]")
print(f"  {'년도':<6s}{'매수':>5s}{'  SW 예측/실제':>17s}{'  100+ 예측/실제':>20s}{'  50+ 예측/실제':>20s}{'  10+ 예측/실제':>20s}{'  손절 예측/실제':>20s}{'수익률':>10s}")
for y, g in strong_top1.groupby("Year"):
    n = len(g)
    invest = n*ALLOC
    profit = ((g["sell_close"]/g["Close"]-1)*ALLOC).sum()
    sw_p = g["p_sw"].mean()*100
    sw_a = (g["peak_180d"]>=200).mean()*100
    p100_p = g["p_100plus"].mean()*100
    p100_a = (g["peak_180d"]>=100).mean()*100
    p50_p = g["p_50plus"].mean()*100
    p50_a = (g["peak_180d"]>=50).mean()*100
    p10_p = g["p_10plus"].mean()*100 if "p_10plus" in g.columns else 0
    p10_a = (g["peak_180d"]>=10).mean()*100
    l_p = g["p_loss"].mean()*100
    l_a = (g["ret_180d"]<=-20).mean()*100
    print(f"  {int(y):<6d}{n:>5d}{sw_p:>9.1f}%/{sw_a:>5.1f}%{p100_p:>11.1f}%/{p100_a:>5.1f}%{p50_p:>11.1f}%/{p50_a:>5.1f}%{p10_p:>11.1f}%/{p10_a:>5.1f}%{l_p:>11.1f}%/{l_a:>5.1f}%{profit/invest*100:>+9.1f}%")

# 저장
strong_top1.to_csv(CACHE / "VALIDATION_calibration.csv", index=False)
print(f"\n[저장] cache/VALIDATION_calibration.csv ({len(strong_top1)}건)")
