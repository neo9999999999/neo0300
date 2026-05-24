"""
진짜 OOS 검증 (Data Leakage 제거)
================================
Train: 2020-04 ~ 2023-12 (45개월)
Test:  2024-01 ~ 2026-05 (29개월)

검증 흐름:
1. Train 데이터만으로 RF 손절예측 모델 학습
2. Train 데이터만으로 회피 룰 발굴 (cutoff 결정)
3. Test 시그널에 Train-학습 모델 적용 → 매일 매수 추천
4. Test 실제 결과로 OOS 성과 측정:
   - 슈퍼위너/100%/50% 농도가 풀에 비해 얼마나 높아졌는가
   - 손절률이 얼마나 낮아졌는가
   - 매일 1건 vs 주 3건 비교

비교 군:
- A: 회피X + 거래대금↓ (baseline)
- B: 회피6 (Train 발굴)
- C: 회피6 + RF (Train 학습)
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
print("[로드]")
master = pd.read_parquet(CACHE / "candidates_enriched_full.parquet")
master["Date"] = pd.to_datetime(master["Date"])
master = master.dropna(subset=["peak_180d", "ret_180d"]).copy()
print(f"  전체: {len(master):,}건 ({master['Date'].min().date()} ~ {master['Date'].max().date()})")

# 통합 마스터에 이미 결과 있음
master["is_loser"] = (master["ret_180d"] <= -20).astype(int)
master["is_sw"] = (master["peak_180d"] >= 200).astype(int)
master["is_100plus"] = (master["peak_180d"] >= 100).astype(int)
master["is_50plus"] = (master["peak_180d"] >= 50).astype(int)


# Train / Test 분리
TRAIN_END = "2024-01-01"
train = master[master["Date"] < TRAIN_END].copy()
test = master[master["Date"] >= TRAIN_END].copy()
print(f"  Train: {len(train):,}건 (2020-04 ~ 2023-12)")
print(f"  Test:  {len(test):,}건 (2024-01 ~ 2026-05)")
print(f"  Train 손절률: {train['is_loser'].mean()*100:.1f}%, SW률: {train['is_sw'].mean()*100:.1f}%")
print(f"  Test 손절률:  {test['is_loser'].mean()*100:.1f}%, SW률: {test['is_sw'].mean()*100:.1f}%")


# ===== 1) Train 으로만 RF 학습 =====
print("\n[1] Train(2020~2023) 으로 RF 학습")
train2 = add_pre_features(train.copy()) if "pre_5d_max_high_ratio" not in train.columns else train.copy()
test2 = add_pre_features(test.copy()) if "pre_5d_max_high_ratio" not in test.columns else test.copy()

available_features = [f for f in FEATURES if f in train2.columns]
print(f"   사용 특성: {len(available_features)}개")

X_train, _ = prepare_X(train2, features=available_features)
X_test, _ = prepare_X(test2, features=available_features)

rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=20,
                             class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, train2["is_loser"])

# Train 예측 (in-sample)
train2["RF확률"] = rf.predict_proba(X_train)[:, 1]
# Test 예측 (OOS!)
test2["RF확률"] = rf.predict_proba(X_test)[:, 1]

# Train에서 임계값 결정
TH20 = float(np.quantile(train2["RF확률"], 0.80))
TH30 = float(np.quantile(train2["RF확률"], 0.70))
print(f"   Train th20={TH20:.4f}, th30={TH30:.4f}")

# Test에 임계값 적용
test2["RF위험"] = (test2["RF확률"] >= TH20).astype(int)
print(f"   Test 위험 분류: {test2['RF위험'].sum()}/{len(test2)} ({test2['RF위험'].mean()*100:.1f}%)")


# ===== 2) RF 위험 분류별 실제 손절률 검증 (이게 핵심!) =====
print("\n[2] OOS - RF 위험 예측 vs 실제 손절 (Test 데이터)")
test2["prob_decile"] = pd.qcut(test2["RF확률"], 10, labels=False, duplicates='drop')
print(test2.groupby("prob_decile").agg(
    n=("Code", "count"),
    실제손절률=("is_loser", lambda x: f"{x.mean()*100:.1f}%"),
    SW률=("is_sw", lambda x: f"{x.mean()*100:.1f}%"),
    평균peak=("peak_180d", "mean"),
    평균ret=("ret_180d", "mean"),
).round(1).to_string())


# ===== 3) 매수 시뮬레이션 (Test 기간만) =====
print("\n[3] OOS 매수 시뮬레이션 (Test 2024-2026)")

def simulate_picks(df, mode, alloc=100_000):
    """매일1건 또는 주3건"""
    df = df.dropna(subset=["sell_close"]).sort_values("Amount").copy()
    if mode == "daily_1":
        df["bucket"] = df["Date"].dt.strftime("%Y-%m-%d")
        n_per = 1
    elif mode == "weekly_3":
        df = df[df["Market"] == "KOSDAQ"].copy()
        df["bucket"] = df["Date"].dt.strftime("%Y-%U")
        n_per = 3
    picks = df.groupby("bucket").head(n_per)
    picks = picks.drop_duplicates(subset=["Date", "Code"])
    if len(picks) == 0: return None
    invest = len(picks) * alloc
    profit = ((picks["sell_close"] / picks["Close"] - 1) * alloc).sum()
    return {
        "n": len(picks),
        "익절": int((picks["ret_180d"] > 0).sum()),
        "손절(<=-20%)": int((picks["ret_180d"] <= -20).sum()),
        "SW": int((picks["peak_180d"] >= 200).sum()),
        "100%+": int((picks["peak_180d"] >= 100).sum()),
        "50%+": int((picks["peak_180d"] >= 50).sum()),
        "투자만": invest/1e4,
        "수익만": round(profit/1e4),
        "수익률%": round(profit/invest*100, 1),
        "승률%": round((picks["ret_180d"]>0).mean()*100, 1),
        "SW률%": round((picks["peak_180d"]>=200).mean()*100, 1),
        "100+%": round((picks["peak_180d"]>=100).mean()*100, 1),
        "50+%": round((picks["peak_180d"]>=50).mean()*100, 1),
        "손절%": round((picks["ret_180d"]<=-20).mean()*100, 1),
    }


# 회피 6 룰 함수 (Train에서 발견한 그대로)
def apply_avoid6(df):
    d = df.copy()
    x1 = (d["chart_pattern"] == "pullback_recovery") & (d["slope60"] <= -1) & (d["pos_252_high"] <= -40)
    x2 = (d["Market"] == "KOSPI") & (d["past_120"] <= -20) & (d["pos_252_high"] <= -40)
    x3 = (d["s12"] >= 80) & (d["new_high_252"] == 1) & (d["past_120"] >= 50)
    x4 = d["past_240"] >= 100
    x5 = d["past_240"] >= 150
    x6 = d["Amount"] >= 3000e8
    return d[~(x1 | x2 | x3 | x4 | x5 | x6).fillna(False)].copy()


# 3가지 풀
test2["chart_pattern"] = test2.get("chart_pattern", "mixed").fillna("mixed")
test2["new_high_252"] = test2.get("new_high_252", 0).fillna(0)
test2["s12"] = test2.get("s12", 0).fillna(0)

A = test2.copy()                                      # 회피 X
B = apply_avoid6(test2)                               # 회피6 (Train)
C = B[B["RF위험"] == 0].copy()                        # 회피6 + RF (Train)
print(f"  A (회피X): {len(A):,}")
print(f"  B (회피6): {len(B):,}")
print(f"  C (회피6+RF): {len(C):,}")

results = []
for label, pool in [("A: 회피X", A), ("B: 회피6", B), ("C: 회피6+RF(Train)", C)]:
    for mode in ["daily_1", "weekly_3"]:
        r = simulate_picks(pool, mode)
        if r:
            r["군"] = label; r["모드"] = mode
            results.append(r)

res = pd.DataFrame(results)
cols = ["군","모드","n","익절","손절(<=-20%)","SW","100%+","50%+","투자만","수익만","수익률%","승률%","SW률%","100+%","50+%","손절%"]
print("\n" + res[cols].to_string(index=False))


# ===== 4) 풀 농도 vs 실제 매수 농도 (OOS 검증의 핵심) =====
print("\n[4] OOS 풀 농도 vs 매수 농도 (RF 효과 진짜?)")
for label, pool in [("회피X 전체 풀", A), ("회피6 풀", B), ("회피6+RF 풀", C)]:
    if len(pool) == 0: continue
    print(f"  [{label}] n={len(pool):,}, "
          f"손절률={pool['is_loser'].mean()*100:.1f}%, "
          f"SW률={pool['is_sw'].mean()*100:.1f}%, "
          f"100+률={pool['is_100plus'].mean()*100:.1f}%, "
          f"50+률={pool['is_50plus'].mean()*100:.1f}%")


# ===== 5) 매일 3건 추천에서 슈퍼위너/100%/50% 적중률 =====
print("\n[5] 매일 시그널 중 거래대금↓ 3건 추천 시 적중률 (Test 기간)")

def daily_top3(df, label):
    df = df.dropna(subset=["sell_close"]).copy()
    df = df.sort_values("Amount")
    df["bucket"] = df["Date"].dt.strftime("%Y-%m-%d")
    top3 = df.groupby("bucket").head(3).drop_duplicates(subset=["Date","Code"])
    print(f"\n  [{label}] 매일 3건 추천 ({len(top3):,}건)")
    print(f"    슈퍼위너 적중: {top3['is_sw'].sum()}건 / {len(top3)} = {top3['is_sw'].mean()*100:.1f}%")
    print(f"    100%+ 적중:   {top3['is_100plus'].sum()}건 / {len(top3)} = {top3['is_100plus'].mean()*100:.1f}%")
    print(f"    50%+ 적중:    {top3['is_50plus'].sum()}건 / {len(top3)} = {top3['is_50plus'].mean()*100:.1f}%")
    print(f"    손절(-20%↓): {top3['is_loser'].sum()}건 / {len(top3)} = {top3['is_loser'].mean()*100:.1f}%")
    invest = len(top3) * 10
    profit = ((top3["sell_close"]/top3["Close"] - 1) * 10).sum()
    print(f"    종목당 10만 매수 → 투자 {invest:,}만, 수익 {profit:+,.0f}만 (수익률 {profit/invest*100:+.1f}%)")
    return top3

t_a = daily_top3(A, "A: 회피X")
t_b = daily_top3(B, "B: 회피6 (Train룰 적용)")
t_c = daily_top3(C, "C: 회피6+RF (Train학습 모델)")


# ===== 6) 매수 종목 농도 vs 풀 농도 (기준 효과 검증) =====
print("\n[6] 매수 3건이 풀 평균보다 얼마나 좋은 종목인가 (selection skill)")
for label, pool, picks in [
    ("A: 회피X", A, t_a),
    ("B: 회피6", B, t_b),
    ("C: 회피6+RF", C, t_c),
]:
    if len(pool)==0 or len(picks)==0: continue
    print(f"\n  [{label}]")
    print(f"    풀 SW률 {pool['is_sw'].mean()*100:.1f}% → 매수 SW률 {picks['is_sw'].mean()*100:.1f}% "
          f"(배율 {picks['is_sw'].mean()/pool['is_sw'].mean():.2f}배)")
    print(f"    풀 100+률 {pool['is_100plus'].mean()*100:.1f}% → 매수 100+률 {picks['is_100plus'].mean()*100:.1f}% "
          f"(배율 {picks['is_100plus'].mean()/pool['is_100plus'].mean():.2f}배)")
    print(f"    풀 손절률 {pool['is_loser'].mean()*100:.1f}% → 매수 손절률 {picks['is_loser'].mean()*100:.1f}% "
          f"(배율 {picks['is_loser'].mean()/pool['is_loser'].mean():.2f}배)")


# 저장
res.to_csv(CACHE / "TRUE_OOS_results.csv", index=False)
test2[["Date","Code","Name","Market","Close","ret_180d","peak_180d","is_loser","is_sw","RF확률","RF위험"]].to_csv(
    CACHE / "TRUE_OOS_test_with_rf.csv", index=False)
print(f"\n[저장]")
print(f"  cache/TRUE_OOS_results.csv")
print(f"  cache/TRUE_OOS_test_with_rf.csv ({len(test2):,}건)")
