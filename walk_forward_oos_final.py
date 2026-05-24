"""
Walk-Forward OOS 검증 (진짜 검증)
==============================
매년 forward로 검증:
- 2020-2022 학습 → 2023 매수 추천 → 실제 결과
- 2020-2023 학습 → 2024 매수 추천 → 실제 결과
- 2020-2024 학습 → 2025 매수 추천 → 실제 결과
- 2020-2025 학습 → 2026 매수 추천 (미래 결과 부분 가능)

각 forward year마다:
- RF 모델 학습
- Test year에서 시그널 발생 종목 → 매일 거래대금↓ 3건 추천
- 슈퍼위너/100%/50% 적중률, 손절률, 누적 수익률

종목당 10만 매수, 180일 보유.
"""

import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from train_rf_loss_model import add_pre_features, prepare_X, FEATURES

CACHE = Path("cache")
ALLOC = 100_000


print("[로드]")
df = pd.read_parquet(CACHE / "candidates_enriched_full.parquet")
df["Date"] = pd.to_datetime(df["Date"])
print(f"  전체: {len(df):,}건 ({df['Date'].min().date()} ~ {df['Date'].max().date()})")
df["Year"] = df["Date"].dt.year

# 시계열 특성 + RF 학습용 X
df = add_pre_features(df)
df["is_loser"] = (df["ret_180d"] <= -20).astype(int)
df["is_sw"] = (df["peak_180d"] >= 200).astype(int)
df["is_100plus"] = (df["peak_180d"] >= 100).astype(int)
df["is_50plus"] = (df["peak_180d"] >= 50).astype(int)


# Walk-Forward 윈도우 설계
forward_windows = [
    {"train_start": 2020, "train_end": 2022, "test_year": 2023},
    {"train_start": 2020, "train_end": 2023, "test_year": 2024},
    {"train_start": 2020, "train_end": 2024, "test_year": 2025},
    {"train_start": 2020, "train_end": 2025, "test_year": 2026},  # 미래 (일부만 결과)
]

all_year_results = []
all_picks = []

for w in forward_windows:
    print(f"\n{'='*100}")
    print(f"Walk {w['train_start']}-{w['train_end']} → Test {w['test_year']}")
    print('='*100)

    train = df[(df["Year"] >= w["train_start"]) & (df["Year"] <= w["train_end"]) & df["ret_180d"].notna()].copy()
    test = df[df["Year"] == w["test_year"]].copy()
    print(f"  Train: {len(train):,}건 (loser {train['is_loser'].sum()})")
    print(f"  Test:  {len(test):,}건 (ret_180d 확정: {test['ret_180d'].notna().sum():,})")

    if len(train) < 500 or len(test) < 50:
        print("  데이터 부족, skip"); continue

    # RF 학습
    available = [f for f in FEATURES if f in train.columns]
    X_tr, _ = prepare_X(train, features=available)
    X_te, _ = prepare_X(test, features=available)
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=20,
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_tr, train["is_loser"])
    test["RF확률"] = rf.predict_proba(X_te)[:, 1]
    # Train 분위 80% 임계값
    th20 = np.quantile(rf.predict_proba(X_tr)[:, 1], 0.80)
    test["RF위험"] = (test["RF확률"] >= th20).astype(int)
    print(f"  Train th20={th20:.4f}")

    # 매수 시뮬레이션 - 매일 거래대금↓ 3건 (회피X & RF안전 비교)
    def sim(pool, label):
        pool = pool.dropna(subset=["sell_close", "ret_180d"]).copy()
        if len(pool) == 0: return None
        pool = pool.sort_values("Amount")
        pool["bucket"] = pool["Date"].dt.strftime("%Y-%m-%d")
        picks = pool.groupby("bucket").head(3).drop_duplicates(subset=["Date","Code"])
        if len(picks) == 0: return None
        invest = len(picks) * ALLOC
        profit = ((picks["sell_close"]/picks["Close"] - 1) * ALLOC).sum()
        return {
            "year": w["test_year"], "set": label,
            "매수": len(picks),
            "익절": int((picks["ret_180d"]>0).sum()),
            "손절": int((picks["ret_180d"]<=-20).sum()),
            "SW(200%+)": int((picks["peak_180d"]>=200).sum()),
            "SW률%": round((picks["peak_180d"]>=200).mean()*100, 1),
            "100+률%": round((picks["peak_180d"]>=100).mean()*100, 1),
            "50+률%": round((picks["peak_180d"]>=50).mean()*100, 1),
            "손절률%": round((picks["ret_180d"]<=-20).mean()*100, 1),
            "투자만": invest/1e4,
            "수익만": round(profit/1e4),
            "수익률%": round(profit/invest*100, 1),
        }, picks

    res_naive, picks_naive = sim(test, "회피X")
    rf_safe = test[test["RF위험"]==0].copy()
    res_rf, picks_rf = sim(rf_safe, "RF안전")

    for r in [res_naive, res_rf]:
        if r: all_year_results.append(r)
    if picks_naive is not None:
        picks_naive["set"] = "회피X"; picks_naive["year"] = w["test_year"]
        all_picks.append(picks_naive)
    if picks_rf is not None:
        picks_rf["set"] = "RF안전"; picks_rf["year"] = w["test_year"]
        all_picks.append(picks_rf)

    print(f"\n  [회피X 매일3건]")
    print(f"    {res_naive}")
    print(f"  [RF안전 매일3건]")
    print(f"    {res_rf}")


# ============ 종합 결과 ============
print("\n" + "="*100)
print("Walk-Forward 종합 (각 년도 별도 OOS 검증)")
print("="*100)

res_df = pd.DataFrame(all_year_results)
print(res_df.to_string(index=False))

print(f"\n{'='*100}")
print("종합 누적 (자본 1억 + 종목당 10만)")
print('='*100)

for set_label in ["회피X", "RF안전"]:
    s = res_df[res_df["set"]==set_label]
    total_n = s["매수"].sum()
    total_invest = s["투자만"].sum()
    total_profit = s["수익만"].sum()
    total_sw = s["SW(200%+)"].sum()
    total_loser = s["손절"].sum()
    total_win = s["익절"].sum()
    print(f"\n[{set_label}] 매일3건 매수 누적 (2023~2026 OOS)")
    print(f"  매수: {total_n:,}건")
    print(f"  익절: {total_win:,}건 ({total_win/total_n*100:.1f}%)")
    print(f"  손절: {total_loser:,}건 ({total_loser/total_n*100:.1f}%)")
    print(f"  슈퍼위너: {total_sw}건 ({total_sw/total_n*100:.1f}%)")
    print(f"  투자: {total_invest:,.0f}만원")
    print(f"  수익: {total_profit:+,.0f}만원")
    print(f"  수익률: {total_profit/total_invest*100:+.1f}%")
    print(f"  최종자본: {total_invest+total_profit:,.0f}만원 (자본 1억 시작 가정)")


res_df.to_csv(CACHE / "WALK_FORWARD_OOS_final.csv", index=False)
all_picks_df = pd.concat(all_picks, ignore_index=True) if all_picks else pd.DataFrame()
if len(all_picks_df):
    all_picks_df["Year"] = all_picks_df["Date"].dt.year
    all_picks_df.to_csv(CACHE / "WALK_FORWARD_OOS_picks.csv", index=False)
print(f"\n[저장] cache/WALK_FORWARD_OOS_final.csv + WALK_FORWARD_OOS_picks.csv")
